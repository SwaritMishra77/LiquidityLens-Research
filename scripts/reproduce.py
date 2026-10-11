"""Rebuild from packaged SEC caches; default reuses pinned, cached embeddings."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path

def verify_cached_embeddings(root,config):
    from liquiditylens.io import load_config
    cfg=load_config(root/config)
    manifest=json.loads((root/"data/processed/encoder_manifest.json").read_text())
    if manifest["model"]!=cfg["text_model"] or manifest["revision"]!=cfg["text_revision"]:
        raise ValueError("Frozen text encoder is inconsistent with the current config; re-encode")
    for filename,expected in [("passages.jsonl",manifest["input_sha256"]),
                              ("text_embeddings.npz",manifest["output_sha256"])]:
        path=root/"data/processed"/filename
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected: raise ValueError(f"Text cache hash mismatch: {filename}; re-encode")

def main():
    p=argparse.ArgumentParser();p.add_argument("--reencode",action="store_true");p.add_argument("--no-deep",action="store_true");args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
    # Editable installation is recommended; this also supports a source checkout.
    env["PYTHONPATH"]=str(root/"src")
    def run(command,*extra):
        cmd=[sys.executable,"-m","liquiditylens.cli",command,*extra]
        print("Running",command,*extra,flush=True);subprocess.run(cmd,cwd=root,env=env,check=True)
    run("build");run("audit");run("reconcile")
    if args.reencode:run("process-text");run("encode")
    else: verify_cached_embeddings(root,Path("configs/pilot.yaml"))
    extra=["--no-deep"] if args.no_deep else []
    for cohort in ["numeric","matched_text"]:run("experiment","--cohort",cohort,*extra)
    run("rolling");run("report");run("verify")

if __name__=="__main__":main()

