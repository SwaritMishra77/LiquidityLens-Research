"""Independent inline-XBRL source checks for a deterministic pilot sample.

This is an independent byte-level extraction check, not a human audit of the
economic meaning of every table or context.
"""
from __future__ import annotations
import gzip,json,re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from .io import dump_json

def local_name(node): return node.name.split(":")[-1].lower() if node.name else ""

def parse_inline_facts(payload):
    soup=BeautifulSoup(payload,"html.parser");contexts={};units={}
    for node in soup.find_all(lambda t:local_name(t)=="context"):
        if node.find(lambda t:local_name(t) in {"explicitmember","typedmember"}):continue
        period=node.find(lambda t:local_name(t)=="period")
        if period is None:continue
        vals={local_name(t):t.get_text(strip=True) for t in period.find_all(True)}
        contexts[node.get("id")]=vals
    for node in soup.find_all(lambda t:local_name(t)=="unit"):
        measures=node.find_all(lambda t:local_name(t)=="measure")
        if len(measures)==1:units[node.get("id")]=measures[0].get_text(strip=True).split(":")[-1]
    facts=[]
    for node in soup.find_all(lambda t:local_name(t)=="nonfraction"):
        name=node.get("name",""); context=contexts.get(node.get("contextref")); unit=units.get(node.get("unitref"))
        if not name.startswith("us-gaap:") or context is None or unit!="USD":continue
        text=node.get_text("",strip=True).replace(",","").replace("$","").replace("\u00a0","").strip()
        if text in {"—","–","-",""}:value=0.
        else:
            try:value=float(text.strip("()"))
            except ValueError:continue
            if text.startswith("("):value=-abs(value)
        if node.get("sign")=="-":value=-abs(value)
        value*=10**int(node.get("scale","0"))
        end=context.get("enddate",context.get("instant"))
        facts.append({"tag":name.split(":",1)[1],"start":context.get("startdate"),"end":end,"value":value,"context":node.get("contextref")})
    return facts

def reconstruct(prov):
    fs=prov["facts"];method=prov["method"]
    if method in {"direct_quarter","instant"}:return float(fs[0]["val"])
    if method=="ytd_difference":return float(fs[0]["val"])-float(fs[1]["val"])
    if method=="components_sum":return sum(float(f["val"]) for f in fs)
    if method=="ytd_difference_components":
        end=max(f["end"] for f in fs)
        return sum(float(f["val"])*(1 if f["end"]==end else -1) for f in fs)
    raise ValueError(method)

def reconcile_sources(root,periods=30):
    root=Path(root);provenance=[json.loads(s) for s in (root/"data/processed/provenance.jsonl").read_text().splitlines()]
    frame=pd.read_csv(root/"data/processed/samples.csv").set_index("sample_id")
    # Stable shuffled priority and fully cached sources. Coverage selection is reported explicitly.
    candidates=[r for r in provenance if "2019-01-01"<=r["prediction_at"]<"2025-01-01"]
    rng=np.random.default_rng(42);rng.shuffle(candidates);cache={};results=[];skipped=0;unsupported=[]
    for r in candidates:
        measures={"current_revenue":r["history"][-1]["facts"]["revenue"],"current_cfo":r["history"][-1]["facts"]["cfo"],
                  "target_revenue":r["target"]["revenue"],"target_cfo":r["target"]["cfo"]}
        primitive=[f for p in measures.values() for f in p.get("facts",[])]
        if any(not (root/f"data/raw/html/{f['accn']}.html.gz").exists() for f in primitive):skipped+=1;continue
        checks=[];unsupported_accn=set()
        for label,prov in measures.items():
            for f in prov["facts"]:
                accn=f["accn"]
                if accn not in cache:
                    with gzip.open(root/f"data/raw/html/{accn}.html.gz","rb") as g:cache[accn]=parse_inline_facts(g.read())
                if not cache[accn]:unsupported_accn.add(accn)
                matches=[x for x in cache[accn] if x["tag"]==prov["tag"] and x["end"]==f["end"] and x.get("start")==f.get("start")]
                # Component reconciliations have different underlying tag names; identify a numeric/context match for each component.
                if prov["tag"]=="consolidated_cfo_reconciled":
                    tags={"NetCashProvidedByUsedInOperatingActivities","NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
                          "CashProvidedByUsedInOperatingActivitiesDiscontinuedOperations","CashProvidedByUsedInDiscontinuedOperationsOperatingActivities"}
                    matches=[x for x in cache[accn] if x["tag"] in tags and x["end"]==f["end"] and x.get("start")==f.get("start")]
                found=[x for x in matches if np.isclose(x["value"],float(f["val"]),rtol=1e-12,atol=.01)]
                checks.append({"measure":label,"source_accession":accn,"end":f["end"],"expected":f["val"],
                    "source_tag":found[0]["tag"] if found else None,"status":"matched" if found else "not_matched"})
        if unsupported_accn:
            unsupported.append({"sample_id":r["sample_id"],"accessions":sorted(unsupported_accn),"reason":"No supported non-dimensional USD inline-XBRL facts; external XBRL instances not parsed"})
            continue
        target_margin=reconstruct(measures["target_cfo"])/reconstruct(measures["target_revenue"])
        arithmetic=bool(np.isclose(target_margin,float(frame.loc[r["sample_id"],"y_margin"]),rtol=1e-10,atol=1e-12))
        results.append({"sample_id":r["sample_id"],"ticker":frame.loc[r["sample_id"],"ticker"],"prediction_at":r["prediction_at"],
                        "primitive_checks":checks,"target_arithmetic_passed":arithmetic,"passed":all(x["status"]=="matched" for x in checks) and arithmetic})
        print("source check",len(results),results[-1]["ticker"],results[-1]["passed"],flush=True)
        if len(results)>=periods:break
    report={"sampled_periods":len(results),"passed_periods":sum(r["passed"] for r in results),
        "primitive_facts_checked":sum(len(r["primitive_checks"]) for r in results),"companies":len({r["ticker"] for r in results}),
        "skipped_uncached_source_candidates":skipped,"unsupported_format_candidates":unsupported,
        "selection":"seed 42, 2019–2024 forecasts with fully cached current/target core source facts and supported inline-XBRL format",
        "limitations":"Independent inline-XBRL extraction; not a human accounting review or a representative universe-wide validation", "results":results}
    dump_json(root/"reports/source_reconciliation.json",report)
    return report

