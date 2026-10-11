"""Descriptive historical review-queue backtest on already-frozen predictions.

Ranking uses predictions only. Outcomes are inspected after the chosen IDs
are fixed; no dollar benefit, prevented loss or prospective utility is implied.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .io import dump_json


def select_reviews(rows, budget=5):
    if budget < 1: raise ValueError("Review budget must be positive")
    required={"sample_id","prediction_at","point"}
    if not required.issubset(rows): raise ValueError(f"Missing ranking columns: {required-set(rows)}")
    if rows.sample_id.duplicated().any(): raise ValueError("Duplicate sample IDs in a model's test group")
    ranked=rows.copy()
    ranked["wave"]=pd.to_datetime(ranked.prediction_at,utc=True).dt.strftime("%Y-Q")+(
        (pd.to_datetime(ranked.prediction_at,utc=True).dt.quarter).astype(str))
    # Rank the least forecast operating cash-flow margin. Stable ID breaks ties,
    # independent of labels or input row order.
    ranked=ranked.sort_values(["wave","point","sample_id"],kind="stable")
    return ranked.groupby("wave",sort=True).head(budget).copy()


def review_backtest(predictions, selected, budget=5):
    models=["seasonal","boosted_numeric",selected["model"]]
    outcomes=[]
    for group in ["test_seen","test_unseen"]:
        group_rows=predictions[predictions.group==group]
        if group_rows.empty: continue
        relevant={}
        for model in dict.fromkeys(models):
            seed=selected["seed"] if model==selected["model"] else 0
            relevant[model]=group_rows[(group_rows.model==model)&(group_rows.seed==seed)].copy()
        reference=relevant["seasonal"].set_index("sample_id")
        ids=set(reference.index)
        if not ids or any(set(f.sample_id)!=ids for f in relevant.values()):
            raise ValueError("Review backtest requires identical sample IDs across models")
        for model,rows in relevant.items():
            picked=select_reviews(rows,budget)
            base=reference.loc[picked.sample_id]
            if not (base.y_stress.to_numpy()==picked.y_stress.to_numpy()).all():
                raise ValueError("Outcome mismatch between matched model rows")
            if not (base.y_margin.to_numpy()==picked.y_margin.to_numpy()).all():
                raise ValueError("Margin mismatch between matched model rows")
            outcomes.append({"group":group,"model":model,"budget_per_calendar_quarter":budget,
                "available_cases":len(rows),"available_waves":int(picked.wave.nunique()),
                "reviews":len(picked),"two_quarter_negative_cfo_found":int(picked.y_stress.sum()),
                "two_quarter_negative_cfo_total":int(reference.y_stress.sum()),
                "next_quarter_negative_margin_found":int((picked.y_margin<0).sum()),
                "next_quarter_negative_margin_total":int((reference.y_margin<0).sum()),
                "selected_sample_ids":picked.sample_id.tolist(),
                "by_wave":[{"wave":wave,"available":int((select_reviews(rows,len(rows)).wave==wave).sum()),
                            "reviewed":len(w),"negative_cfo":int(w.y_stress.sum()),
                            "negative_margin":int((w.y_margin<0).sum())}
                           for wave,w in picked.groupby("wave",sort=True)]})
    return {"policy":"For each calendar-quarter arrival wave, review up to five lowest predicted "
            "next-quarter CFO margins. This is retrospective selection on frozen test predictions.",
            "selection":"Validation-selected champion plus prespecified seasonal and tree comparators; "
            "same observations, fixed review budget, deterministic ID ties.",
            "limitations":"Purposively selected firms; repeated quarters and few events. "
            "Quarterly calendar grouping is a simplified staffing assumption; filings arrive on different days. "
            "This evaluates case-finding counts, not prevented losses, time saved or real analyst impact.",
            "results":outcomes}


def generate_decision_report(root, budget=5):
    root=Path(root)
    pred=pd.read_csv(root/"reports/numeric/predictions.csv")
    selected=json.loads((root/"reports/numeric/champion.json").read_text())
    report=review_backtest(pred,selected,budget)
    dump_json(root/"reports/decision_backtest.json",report)
    return report
