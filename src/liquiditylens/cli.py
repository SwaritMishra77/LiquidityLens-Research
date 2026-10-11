from __future__ import annotations
import argparse, json
from pathlib import Path
from .io import load_config

def main():
    parser=argparse.ArgumentParser(description="LiquidityLens reproducible research pipeline")
    parser.add_argument("command",choices=["ingest","build","audit","reconcile","fetch-text","process-text","encode","experiment","rolling","report","verify","all"])
    parser.add_argument("--root",default=".")
    parser.add_argument("--config",default="configs/pilot.yaml")
    parser.add_argument("--cohort",choices=["numeric","matched_text"],default="numeric")
    parser.add_argument("--no-deep",action="store_true")
    parser.add_argument("--offline",action="store_true",help="Use only locally cached transformer weights")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args(); root=Path(args.root).resolve()
    cfg=load_config(root/args.config)
    cfg["offline"]=args.offline
    def run(cmd):
        if cmd=="ingest":
            from .sec import ingest
            return ingest(root,root/"configs/universe.csv",workers=args.workers)
        if cmd=="build":
            from .dataset import build_dataset
            return build_dataset(root,cfg)
        if cmd=="audit":
            from .dataset import audit_provenance
            return audit_provenance(root)
        if cmd=="fetch-text":
            from .sec import fetch_texts
            return fetch_texts(root,cfg,workers=args.workers)
        if cmd=="process-text":
            from .text import process_texts
            return process_texts(root,cfg)
        if cmd=="encode":
            from .text import encode_texts
            return encode_texts(root,cfg)
        if cmd=="experiment":
            from .experiments import run_experiments
            return run_experiments(root,cfg,args.cohort,not args.no_deep)
        if cmd=="rolling":
            from .experiments import rolling_baselines
            return rolling_baselines(root,cfg)
        if cmd=="reconcile":
            from .reconciliation import reconcile_sources
            return reconcile_sources(root)
        if cmd=="report":
            from .reporting import generate_report
            return generate_report(root,cfg)
        if cmd=="verify":
            from .verification import verify_artifacts
            return verify_artifacts(root)
    if args.command=="all":
        for cmd in ["ingest","build","audit","fetch-text","process-text","encode"]: run(cmd)
        run("reconcile")
        from .experiments import run_experiments
        for cohort in ["numeric","matched_text"]: run_experiments(root,cfg,cohort,not args.no_deep)
        run("rolling")
        run("report"); run("verify")
    else: run(args.command)

if __name__=="__main__": main()

