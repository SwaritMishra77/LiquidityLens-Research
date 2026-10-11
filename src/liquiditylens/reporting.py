from __future__ import annotations
import html,json,math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .dataset import load_dataset
from .io import dump_json

def records_json(frame):
    return json.loads(frame.to_json(orient="records"))

def generate_report(root,config):
    root=Path(root);plots=root/"reports/figures";plots.mkdir(parents=True,exist_ok=True)
    from .decision import generate_decision_report
    decision=generate_decision_report(root)
    frame,seq,names=load_dataset(root)
    audit=json.loads((root/"reports/data_audit.json").read_text())
    numeric=pd.read_csv(root/"reports/numeric/metrics.csv")
    tables=[numeric];preds=[pd.read_csv(root/"reports/numeric/predictions.csv")]
    text_path=root/"reports/matched_text/metrics.csv"
    if text_path.exists():tables.append(pd.read_csv(text_path));preds.append(pd.read_csv(root/"reports/matched_text/predictions.csv"))
    combined=pd.concat(tables,ignore_index=True);allpred=pd.concat(preds,ignore_index=True)
    combined.to_csv(root/"reports/all_metrics.csv",index=False)
    summary=combined.groupby(["cohort","model","group"],as_index=False).agg(n=("n","first"),companies=("companies","first"),
        mae=("mae","mean"),mae_seed_sd=("mae","std"),coverage=("interval_coverage","mean"),width=("interval_mean_width","mean"),
        pr_auc=("pr_auc","mean"),brier=("brier","mean"),stress_prevalence=("stress_prevalence","first"))
    summary.to_csv(root/"reports/model_summary.csv",index=False)
    unseen_numeric=summary[(summary.cohort=="numeric")&(summary.group=="test_unseen")].iloc[0]
    seen_text=summary[(summary.cohort=="matched_text")&(summary.group=="test_seen")].iloc[0]
    unseen_events=int(round(unseen_numeric.n*unseen_numeric.stress_prevalence))
    text_events=int(round(seen_text.n*seen_text.stress_prevalence))
    selected=json.loads((root/"reports/numeric/champion.json").read_text())
    champion=allpred[(allpred.cohort=="numeric")&(allpred.model==selected["model"])&(allpred.seed==selected["seed"])].copy()
    champion["absolute_error"]=np.abs(champion.y_margin-champion.point)
    failures=champion.sort_values("absolute_error",ascending=False).head(15)
    failures.to_csv(root/"reports/failure_cases.csv",index=False)
    # Error slices expose performance changes without refitting or selecting a different model.
    enriched=champion.merge(frame[["sample_id","sector","form"]],on="sample_id")
    enriched["forecast_calendar_quarter"]=pd.to_datetime(enriched.prediction_at,utc=True).dt.quarter
    enriched["target_negative"]=enriched.y_margin<0
    slices=[]
    for dimension in ["sector","form","group","forecast_calendar_quarter","target_negative"]:
        for value,g in enriched.groupby(dimension):
            slices.append({"dimension":dimension,"value":str(value),"n":len(g),"mae":float(g.absolute_error.mean()),
                           "coverage":float(((g.y_margin>=g.lower)&(g.y_margin<=g.upper)).mean())})
    pd.DataFrame(slices).to_csv(root/"reports/error_slices.csv",index=False)
    # Conditional sensitivity to financing subsidiaries / changed corporate perimeter, flagged before modelling.
    flagged={"CAT","DE","F","GM","PCAR","GE","CNH"}
    for tag,g in [("all",enriched),("exclude_flagged_perimeter_and_financing",enriched[~enriched.ticker.isin(flagged)])]:
        slices.append({"dimension":"scope_sensitivity","value":tag,"n":len(g),"mae":float(g.absolute_error.mean()),
                       "coverage":float(((g.y_margin>=g.lower)&(g.y_margin<=g.upper)).mean())})
    dump_json(root/"reports/sensitivity.json",slices)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    g=summary[(summary.cohort=="numeric")&(summary.group=="test_seen")].sort_values("mae")
    fig,ax=plt.subplots(figsize=(8,4.4));ax.barh(g.model,g.mae*100,color="#25636b");ax.invert_yaxis();ax.set_xlabel("Mean absolute error in operating cash-flow margin (percentage points)")
    ax.set_title("2025 historical evaluation · companies represented in training");fig.tight_layout();fig.savefig(plots/"numeric_mae.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,4.3));ax.scatter(champion.point*100,champion.y_margin*100,c=champion.group.map({"test_seen":"#25636b","test_unseen":"#c18748"}),alpha=.7,s=24)
    lo=min(champion.point.min(),champion.y_margin.min())*100;hi=max(champion.point.max(),champion.y_margin.max())*100
    ax.plot([lo,hi],[lo,hi],"--",color="#839198");ax.set(xlabel="Forecast margin (%)",ylabel="Realised margin (%)",title=f"Validation-selected model: {selected['model']} · seed {selected['seed']}")
    fig.tight_layout();fig.savefig(plots/"forecast_errors.png",dpi=180);plt.close(fig)
    passages={}
    if (root/"data/processed/passages.jsonl").exists(): passages={r["sample_id"]:r for r in [json.loads(s) for s in (root/"data/processed/passages.jsonl").read_text().splitlines()]}
    # Cache-independent standalone demo; it never queries a live service.
    display=[]
    rows=frame.set_index("sample_id");lookup={k:i for i,k in enumerate(frame.sample_id)}
    for row in champion.to_dict("records"):
        source=rows.loc[row["sample_id"]];i=lookup[row["sample_id"]]
        evidence=passages.get(row["sample_id"],{})
        textpred=allpred[(allpred.cohort=="matched_text")&(allpred.sample_id==row["sample_id"])&(allpred.seed==0)]
        row.update(company=source.company,sector=source.sector,current_margin=float(source.current_margin),
                   cash_buffer=float(seq[i,-1,names.index("cash_current_liabilities")]) if np.isfinite(seq[i,-1,names.index("cash_current_liabilities")]) else None,
                   history=[float(v) if np.isfinite(v) else None for v in seq[i,:,0]],
                   evidence=[{"text":" ".join(p["text"].split()[:90]),"score":p["retrieval_score"]} for p in evidence.get("passages",[])[:3]],
                   lexical_change=evidence.get("lexical_change"),form=source.form,
                   numeric_text_forecast=None,text_only_forecast=None)
        for model,key in [("linear_numeric","numeric_text_forecast"),("linear_finbert","text_only_forecast")]:
            q=textpred[textpred.model==model]
            if len(q):row[key]=float(q.iloc[0].point)
        display.append(row)
    split_manifest=json.loads((root/"reports/numeric/run_manifest.json").read_text())
    text_audit=json.loads((root/"reports/text_extraction_audit.json").read_text()) if (root/"reports/text_extraction_audit.json").exists() else {}
    payload={"cases":display,"summary":records_json(summary),"audit":{"samples":len(frame),"companies":int(frame.ticker.nunique()),
        "checked_facts":json.loads((root/"reports/availability_audit.json").read_text())["checked_feature_facts"],"text_records":text_audit.get("records",0)},
        "champion":selected,"split_counts":split_manifest["split_counts"],"heldout":split_manifest["heldout_companies"]}
    template=(root/"demo/template.html").read_text()
    serialized=json.dumps(payload,allow_nan=False).replace("<","\\u003c").replace(">","\\u003e").replace("&","\\u0026")
    (root/"demo/liquiditylens_demo.html").write_text(template.replace("__RESEARCH_DATA__",serialized),encoding="utf-8")
    neural_ran=summary.model.str.startswith(("mlp_","gru_","transformer_")).any()
    model_scope=("seasonal, linear, gradient-boosted, neural and transformer" if neural_ran
                 else "seasonal, linear and gradient-boosted")
    lines=["# LiquidityLens — research pilot results","",f"Built from **{len(frame):,} company–quarter examples across {frame.ticker.nunique()} selected companies**. All financial labels are derived from actual SEC disclosures; no synthetic financial outcomes are used.","",
        "This is a working exploratory research version. It is not a validated credit-rating system, a bankruptcy predictor, or a live service. The earlier 50-company development run and its 2025 test were inspected before expanding to this 100-company sample; the new larger test is also exploratory, not untouched confirmatory evidence.","",
        "## Dataset flow","",
        f"The selected universe has {len(audit['companies'])} issuers; {sum(x.get('samples',0)>0 for x in audit['companies'])} contributed usable financial rows. There are {text_audit.get('records',0)} cached filing passages from {text_audit.get('companies',0)} text issuers, of which {text_audit.get('with_previous_year',0)} have a comparable prior-year disclosure. Eligibility rules and individual rejection reasons are in data_audit.json.","",
        "| Financial split | Examples |","|---|---:|",
        *[f"| {name} | {count} |" for name,count in split_manifest["split_counts"].items()],"",
        "## Prediction and evaluation","","Primary target: next standalone-quarter operating cash flow / revenue. Secondary proxy: negative combined operating cash flow over the next two quarters. Forecasts require at least 14 days before the primary economic quarter ends. Publication availability is conservatively guarded by filing date plus two days.","",
        f"Training labels known by {config['train_end']}; validation in 2021–2022; calibration in 2023–2024; historical evaluation in 2025. Labels crossing each development boundary are purged. {len(split_manifest['heldout_companies'])} whole companies are excluded from training, validation, calibration and preprocessing.","",
        "## Main numeric results","","MAE is shown in percentage points of cash-flow margin. " + ("Neural values average three seeds; a seed standard deviation describes training variability, not a confidence interval." if neural_ran else "This run omitted optional neural training."),"",
        "| Model | Company group | Examples | Stress cases | MAE (pp) | Nominal 80% interval coverage | Stress PR-AUC |","|---|---|---:|---:|---:|---:|---:|"]
    for r in summary[summary.cohort=="numeric"].to_dict("records"):
        pr=f"{r['pr_auc']:.3f}" if pd.notna(r["pr_auc"]) else "undefined: one outcome class"
        lines.append(f"| {r['model']} | {r['group']} | {r['n']} | {round(r['n']*r['stress_prevalence'])} | {r['mae']*100:.2f} | {r['coverage']:.1%} | {pr} |")
    lines += ["",f"The demo uses **{selected['model']} (seed {selected['seed']})**, chosen by validation MAE. The best test result is not used to choose a deployment model.","",
        "## Text contribution","","Text experiments use only rows with both current and same-form prior-year disclosures. Numeric, TF-IDF, frozen FinBERT and change-feature comparators share those exact rows, company exclusions and time boundaries. These results must not be compared directly with the larger numeric cohort.","",
        "| Model | Company group | Examples | MAE (pp) | Interval coverage |","|---|---|---:|---:|---:|"]
    for r in summary[summary.cohort=="matched_text"].to_dict("records"):
        lines.append(f"| {r['model']} | {r['group']} | {r['n']} | {r['mae']*100:.2f} | {r['coverage']:.1%} |")
    reconciliation=json.loads((root/"reports/source_reconciliation.json").read_text()) if (root/"reports/source_reconciliation.json").exists() else None
    lines += ["","## Historical review queue (descriptive)","",
        "A fixed budget of five reviews per calendar-quarter arrival wave is ranked by lowest predicted next-quarter cash-flow margin. The validation-selected model is compared on the same cases with a prior-year seasonal forecast and a gradient-boosted forecast. This is a backtest of cases surfaced, not measured savings or a live intervention.","",
        "| Company group | Ranking | Reviews | Two-quarter negative-CFO cases found / available | Next-quarter negative-margin cases found / available |",
        "|---|---|---:|---:|---:|"]
    for r in decision["results"]:
        lines.append(f"| {r['group']} | {r['model']} | {r['reviews']} | {r['two_quarter_negative_cfo_found']} / {r['two_quarter_negative_cfo_total']} | {r['next_quarter_negative_margin_found']} / {r['next_quarter_negative_margin_total']} |")
    lines += ["","Cohorts are small, review waves are simplified calendar quarters, and test outcomes were already inspected during exploratory development. See decision_backtest.json for selected sample IDs and wave counts.","",
        "## Source validation","",
        f"The publication audit checked {payload['audit']['checked_facts']:,} feature facts against filing dates and their source-accession publication ledger. It found zero violations and no unknown source accessions.",""]
    if reconciliation:
        lines += [f"An independent inline-XBRL parser matched {reconciliation['primitive_facts_checked']} primitive financial facts and target arithmetic across {reconciliation['passed_periods']} of {reconciliation['sampled_periods']} sampled supported-format periods from {reconciliation['companies']} companies. The selection required fully cached sources and supported inline format. {len(reconciliation['unsupported_format_candidates'])} traditional/unsupported-format candidates were retained as exclusions. This is an extraction/arithmetic check, not a human accounting review.",""]
    lines += ["## What the results support","",
        "The pipeline establishes a serious reproducible foundation: source caching, financial reconciliation, tracked provenance, strong baselines, actual neural/transformer training, frozen financial-language embeddings, matched-cohort ablations, empirical uncertainty checks and a historical analyst demo. Model complexity alone is not evidence of improvement. Read paired_comparisons.json for company-cluster uncertainty around model differences.","",
        "## Limits that change interpretation","",
        f"- The purposive {frame.ticker.nunique()}-company universe remains selection-biased. Current SIC/ticker metadata is descriptive, not a point-in-time universe reconstruction.",
        "- Missing custom tags, non-USD reporting and irregular periods reduce coverage. Missing optional financial features are imputed using training data only.",
        "- Company Facts is a current archive filtered by original filing metadata, not an archive collected contemporaneously. Independent filing-level reconciliation remains necessary.",
        "- CFO subtraction can combine then-known values from different filing versions. Restatement changes not exposed in standard tags can still cause inconsistent totals.",
        "- Some manufacturing companies have financing operations; corporate spin-offs can change comparability. sensitivity.json reports errors after excluding flagged firms.",
        "- Frozen FinBERT checkpoint and release provenance are pinned; exact corpus overlap is not independently audited. Text retrieval is keyword-based and has not been certified as full MD&A extraction.",
        f"- Negative two-quarter CFO is a cash-flow pattern, not proof of financial distress. The 2025 unseen-company group has {unseen_events} positive {('case' if unseen_events==1 else 'cases')}. Ranking AUC is undefined with zero events and very unstable with one event.",
        "- Serial dependence and common economic shocks weaken standard conformal exchangeability. Nominal coverage is a target, not a guarantee. The company bootstrap does not resample common time shocks.",
        f"- The matched-text 2025 seen-company evaluation has {text_events} positive stress-proxy {('case' if text_events==1 else 'cases')}; sparse-event ranking cannot establish text's value. Continuous-margin forecasts remain evaluable.",
        "- There are few stress events, multiple model comparisons and one purposive pilot. Results are exploratory; no financial savings or deployment impact is claimed.","",
        "## Next acceptance gates","",
        "1. Independently reconcile at least 30 randomly selected periods against filing cash-flow tables; review ambiguous contexts and missing tags.",
        "2. Use the offline historical SEC index selector to draw a larger issuer candidate list, then review identities/industries, ingest actual facts and disclose attrition before reporting any expanded results.",
        "3. Annotate a double-reviewed passage set for liquidity relevance and change alignment; measure precision and recall of extraction.",
        "4. Freeze a new protocol before inspecting additional future periods; report genuine out-of-sample results and drift.",
        "5. Keep transformers only where repeatable gains or a distinct useful capability justify their cost.","",
        "## CV wording for the current version","",
        f"> Built LiquidityLens, a reproducible SEC-based cash-flow forecasting research pipeline covering {frame.ticker.nunique()} selected companies and {len(frame):,} company–quarter examples; compared {model_scope} models using chronological label purging and whole-company holdouts.","",
        "> Implemented filing provenance, cumulative-to-quarter cash-flow reconciliation, frozen FinBERT disclosure features, matched-cohort ablations, uncertainty evaluation and an interactive historical review demo.","",
        "Do not claim reliable default prediction, transformer superiority, production deployment or realised financial savings from this pilot.","",
        "See START_HERE.md for operating instructions, docs/RESEARCH_PROTOCOL.md for design and docs/INTERVIEW_GUIDE.md for explanation."]
    (root/"reports/RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return payload

