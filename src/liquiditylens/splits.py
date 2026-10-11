from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from .io import utc

def company_holdout(tickers, seed, fraction):
    names=sorted(set(tickers),key=lambda n:hashlib.sha256(f"{seed}:{n}".encode()).hexdigest())
    return set(names[:max(1,round(len(names)*fraction))])

def historical_splits(frame, config, heldout=None):
    heldout=heldout if heldout is not None else company_holdout(frame.ticker,config["seed"],config["unseen_company_fraction"])
    at=pd.to_datetime(frame.prediction_at,utc=True)
    known=pd.to_datetime(frame.label_available_at,utc=True)
    seen=~frame.ticker.isin(heldout)
    train_end=utc(config["train_end"])+pd.Timedelta(days=1)
    val_end=utc(config["validation_end"])+pd.Timedelta(days=1)
    cal_end=utc(config["calibration_end"])+pd.Timedelta(days=1)
    test_end=utc(config["test_end"])+pd.Timedelta(days=1)
    splits={
        "train":seen & (at<train_end) & (known<train_end),
        "validation":seen & (at>=train_end) & (at<val_end) & (known<val_end),
        "calibration":seen & (at>=val_end) & (at<cal_end) & (known<cal_end),
        "test_seen":seen & (at>=cal_end) & (at<test_end),
        "test_unseen":~seen & (at>=cal_end) & (at<test_end),
    }
    indices={k:np.flatnonzero(v.to_numpy()) for k,v in splits.items()}
    used=np.concatenate(list(indices.values()))
    if len(set(used))!=len(used): raise AssertionError("Overlapping split rows")
    if set(frame.iloc[indices["train"]].ticker)&heldout: raise AssertionError("Held-out company in training")
    return indices,heldout

