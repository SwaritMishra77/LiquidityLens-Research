from __future__ import annotations
import hashlib, json, os, platform, sys
from pathlib import Path
import numpy as np
import yaml

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def dump_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str), encoding="utf-8")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def environment():
    import importlib.metadata
    names = ["numpy", "pandas", "scikit-learn", "scipy", "torch", "transformers", "beautifulsoup4"]
    versions = {}
    for name in names:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "thread_environment": {k: os.getenv(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]}}

def utc(value):
    import pandas as pd
    return pd.Timestamp(value, tz="UTC") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert("UTC")

