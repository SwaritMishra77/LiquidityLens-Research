from __future__ import annotations
from collections import Counter
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from .financials import FactStore, FEATURES, feature_vector, days
from .io import dump_json, sha256, utc
from .sec import load_json_gz, filing_available_at

def build_dataset(root, config):
    root=Path(root); out=root/"data/processed"; out.mkdir(parents=True,exist_ok=True)
    universe=json.loads((root/"data/raw/universe_resolved.json").read_text())
    rows,seqs,proofs,company_audits=[],[],[],[]
    cutoff=utc(config["data_cutoff"])+pd.Timedelta(days=1)
    for company in universe:
        rejected=Counter(); ticker=company["ticker"]
        if company["status"]!="ok":
            company_audits.append({"ticker":ticker,"status":company["status"],"samples":0}); continue
        cik=company["cik"]; fact_path=root/f"data/raw/facts/{cik}.json.gz"
        store=FactStore(load_json_gz(fact_path))
        filings=json.loads((root/f"data/raw/filings/{cik}.json").read_text())
        originals={}
        for f in filings:
            end=f.get("reportDate","")
            if f["form"] not in {"10-K","10-Q"} or not end or end<"2012-01-01": continue
            at=filing_available_at(f)
            if at>=cutoff: continue
            if end not in originals or at<filing_available_at(originals[end]): originals[end]=f
        obs=[]
        for end,f in sorted(originals.items()):
            values,prov=store.snapshot(end,f["filingDate"],f["accessionNumber"])
            obs.append({"end":end,"filing":f,"at":filing_available_at(f),"values":values,"provenance":prov})
        count_before=len(rows)
        for i,current in enumerate(obs):
            at=current["at"]; f=current["filing"]; end=current["end"]
            if at<utc(config["prediction_start"]) or at>=utc(config["prediction_end"])+pd.Timedelta(days=1): continue
            if i<config["lookback"]-1: rejected["insufficient_history"]+=1; continue
            if i+2>=len(obs): rejected["future_labels_unavailable"]+=1; continue
            nxt,second=obs[i+1:i+3]
            if not (65<=days(end,nxt["end"])-1<=130 and 65<=days(nxt["end"],second["end"])-1<=130):
                rejected["future_quarter_gap"]+=1; continue
            lead=(utc(nxt["end"])-at).days
            if lead<config["minimum_economic_lead_days"]:
                rejected["economic_lead_too_short"]+=1; continue
            v,yv,zv=current["values"],nxt["values"],second["values"]
            if not (math.isfinite(v["cfo"]) and v["revenue"]>0 and math.isfinite(yv["cfo"]) and yv["revenue"]>0
                    and math.isfinite(zv["cfo"]) and zv["revenue"]>0):
                rejected["core_fact_or_label_missing"]+=1; continue
            history=obs[i-config["lookback"]+1:i+1]
            if any(not 65<=(utc(b["end"])-utc(a["end"])).days<=130 for a,b in zip(history,history[1:])):
                rejected["history_quarter_gap"]+=1; continue
            vectors,history_proof=[],[]
            for h in history:
                # Historical values reconstructed as of this prediction, including only then-known restatements.
                hv,hp=store.snapshot(h["end"],f["filingDate"],f["accessionNumber"])
                vectors.append(feature_vector(hv)); history_proof.append({"end":h["end"],"facts":hp})
            arr=np.asarray(vectors,float)
            sample_id=f"{cik}_{f['accessionNumber']}"
            row={"sample_id":sample_id,"ticker":ticker,"company":company.get("name",ticker),"sector":company["sector"],"cik":cik,
                 "accession":f["accessionNumber"],"primary_document":f.get("primaryDocument",""),"form":f["form"],
                 "report_end":end,"filing_date":f["filingDate"],"prediction_at":at.isoformat(),
                 "target_end":nxt["end"],"target_available_at":nxt["at"].isoformat(),
                 "stress_end":second["end"],"stress_available_at":max(nxt["at"],second["at"]).isoformat(),
                 "label_available_at":max(nxt["at"],second["at"]).isoformat(),"economic_lead_days":lead,
                 "y_margin":yv["cfo"]/yv["revenue"],"y_stress":int(yv["cfo"]+zv["cfo"]<0),
                 "current_margin":v["cfo"]/v["revenue"],"target_cfo":yv["cfo"],"target_revenue":yv["revenue"],
                 "second_cfo":zv["cfo"],"second_revenue":zv["revenue"],
                 "source_url":f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{f['accessionNumber'].replace('-','')}/{f.get('primaryDocument','')}"}
            rows.append(row); seqs.append(arr)
            proofs.append({"sample_id":sample_id,"forecast_filing_date":f["filingDate"],"prediction_at":at.isoformat(),
                           "history":history_proof,"target":nxt["provenance"],"second_target":second["provenance"]})
        company_audits.append({"ticker":ticker,"cik":cik,"status":"ok","report_periods":len(obs),"samples":len(rows)-count_before,
                              "rejections":dict(rejected),"facts_file_sha256":sha256(fact_path),
                              "missing_core_periods":[{"end":o["end"],"filing_date":o["filing"]["filingDate"],
                                  "missing":[k for k in ["revenue","cfo"] if not math.isfinite(o["values"][k]) or (k=="revenue" and o["values"][k]<=0)]}
                                  for o in obs if not math.isfinite(o["values"]["cfo"]) or not o["values"]["revenue"]>0]})
        print(ticker,"samples",len(rows)-count_before,dict(rejected),flush=True)
    if not rows: raise ValueError("No usable examples; inspect normalization and source coverage")
    frame=pd.DataFrame(rows)
    order=np.argsort(frame["sample_id"].to_numpy()); frame=frame.iloc[order].reset_index(drop=True)
    sequences=np.stack(seqs)[order]
    frame.to_csv(out/"samples.csv",index=False,float_format="%.12g")
    np.savez_compressed(out/"sequences.npz",sequence=sequences,sample_id=frame.sample_id.to_numpy(dtype=str),feature_names=np.asarray(FEATURES))
    with (out/"provenance.jsonl").open("w",encoding="utf-8") as f:
        for j in order: f.write(json.dumps(proofs[j],allow_nan=False)+"\n")
    dump_json(out/"feature_schema.json",{"features":FEATURES,"lookback":config["lookback"],"order":"oldest_to_newest",
        "missing_policy":"NaN retained; preprocessing fit on training examples only",
        "target":"next standalone-quarter operating cash flow / next standalone-quarter revenue",
        "stress_proxy":"negative sum of operating cash flow in the next two standalone quarters; not insolvency/default"})
    audit={"samples":len(frame),"companies":company_audits,"usable_companies":int(frame.ticker.nunique()),
           "date_start":frame.prediction_at.min(),"date_end":frame.prediction_at.max(),
           "missing_feature_fraction":{n:float(np.isnan(sequences[:,:,j]).mean()) for j,n in enumerate(FEATURES)},
           "economic_lead_days_min":int(frame.economic_lead_days.min()),"stress_prevalence":float(frame.y_stress.mean()),
           "target_margin_range":[float(frame.y_margin.min()),float(frame.y_margin.max())],
           "limitations":["Purposive pilot; not a historical investable universe", "US-GAAP USD standard tags only; custom tags omitted",
               "Current Company Facts archive filtered by filing dates; not a contemporaneously collected archive",
               "Missing primary data and irregular quarters rejected; selection bias remains",
               "Financing subsidiaries and changes in corporate perimeter require sensitivity analysis",
               "Conservative filing date plus two days; intraday prediction not supported"]}
    dump_json(root/"reports/data_audit.json",audit)
    dump_json(out/"dataset_manifest.json",{"config":config,"files":{p.name:sha256(p) for p in [out/"samples.csv",out/"sequences.npz",out/"provenance.jsonl",out/"feature_schema.json"]}})
    return frame

def audit_provenance(root):
    root=Path(root); errors=[]; facts=0; unknown=set();ledger={}
    universe=json.loads((root/"data/raw/universe_resolved.json").read_text())
    for company in universe:
        if company["status"]!="ok":continue
        for filing in json.loads((root/f"data/raw/filings/{company['cik']}.json").read_text()):
            ledger[filing["accessionNumber"]]=filing_available_at(filing)
    with (root/"data/processed/provenance.jsonl").open() as f:
        for line in f:
            row=json.loads(line)
            for period in row["history"]:
                for field,prov in period["facts"].items():
                    for fact in prov.get("facts",[]):
                        facts+=1
                        if fact["filed"]>row["forecast_filing_date"] or fact["end"]>period["end"]:
                            errors.append({"sample_id":row["sample_id"],"field":field,"fact":fact})
                        if fact["accn"] not in ledger:unknown.add(fact["accn"])
                        elif ledger[fact["accn"]]>pd.Timestamp(row["prediction_at"]):
                            errors.append({"sample_id":row["sample_id"],"field":field,"reason":"source filing not yet available","fact":fact})
    report={"checked_feature_facts":facts,"violations":errors,"unknown_source_accessions":sorted(unknown),
            "publication_ledger_accessions":len(ledger),"passed":not errors and not unknown}
    dump_json(root/"reports/availability_audit.json",report)
    if errors: raise AssertionError(f"{len(errors)} feature availability violations")
    if unknown:raise AssertionError(f"{len(unknown)} source accession publication times could not be verified")
    return report

def load_dataset(root):
    root=Path(root)
    frame=pd.read_csv(root/"data/processed/samples.csv",dtype={"cik":str})
    archive=np.load(root/"data/processed/sequences.npz",allow_pickle=False)
    if not np.array_equal(archive["sample_id"],frame.sample_id.to_numpy(str)): raise ValueError("Sequence/row alignment mismatch")
    return frame,archive["sequence"],archive["feature_names"].tolist()

