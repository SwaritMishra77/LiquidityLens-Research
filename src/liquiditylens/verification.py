from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .dataset import load_dataset,audit_provenance
from .io import dump_json,sha256

def verify_artifacts(root):
    root=Path(root);checks=[]
    frame,seq,names=load_dataset(root)
    assert frame.sample_id.is_unique
    assert np.isfinite(frame.y_margin).all()
    assert frame.y_stress.isin([0,1]).all()
    assert (pd.to_datetime(frame.target_end,utc=True)>pd.to_datetime(frame.prediction_at,utc=True)).all()
    checks.append({"check":"unique rows, finite labels, future economic target","passed":True,"rows":len(frame)})
    checks.append({"check":"feature publication provenance","passed":audit_provenance(root)["passed"]})
    manifest=json.loads((root/"data/processed/dataset_manifest.json").read_text())
    for name,digest in manifest["files"].items():
        assert sha256(root/"data/processed"/name)==digest,name
    checks.append({"check":"dataset file hashes","passed":True})
    for directory in sorted((root/"reports").glob("*")):
        membership=directory/"split_membership.csv"
        if not membership.exists() or directory.name.startswith("development_"):continue
        r=json.loads((directory/"run_manifest.json").read_text());cfg=r["config"]
        splits=pd.read_csv(membership)
        assert splits.sample_id.is_unique
        unseen=set(r["heldout_companies"])
        assert not set(splits[splits.split.isin(["train","validation","calibration"])].ticker)&unseen
        for split,bound in [("train","train_end"),("validation","validation_end"),("calibration","calibration_end")]:
            dates=pd.to_datetime(splits[splits.split==split].label_available_at,utc=True)
            assert (dates<pd.Timestamp(cfg[bound],tz="UTC")+pd.Timedelta(days=1)).all()
        predictions=pd.read_csv(directory/"predictions.csv")
        assert np.isfinite(predictions[["point","lower","upper","stress_probability"]].to_numpy()).all()
        assert (predictions.lower<=predictions.upper).all()
        assert predictions.stress_probability.between(0,1).all()
        expected=set(splits[splits.split.isin(["test_seen","test_unseen"])].sample_id)
        for (model,seed),g in predictions.groupby(["model","seed"]):
            assert set(g.sample_id)==expected,(directory.name,model,seed)
        checks.append({"check":directory.name+" split isolation, label availability, prediction completeness","passed":True})
    text_path=root/"data/processed/passages.jsonl"
    if text_path.exists():
        records={r["sample_id"]:r for r in [json.loads(s) for s in text_path.read_text().splitlines()]}
        for r in records.values():
            if r["previous_sample_id"]:
                assert r["previous_available_at"]<r["prediction_at"]
                assert r["previous_sample_id"] in records
        checks.append({"check":"prior-year text availability","passed":True,"records":len(records)})
    report={"passed":True,"checks":checks}
    dump_json(root/"reports/verification.json",report);print(json.dumps(report,indent=2));return report

