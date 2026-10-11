"""Build a checksum-verified portable research snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

EXCLUDE={"CHECKSUMS.sha256","PACKAGE_MANIFEST.json"}
SKIP_PARTS={"__pycache__",".pytest_cache"}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve()
    files=sorted(p for p in root.rglob("*") if p.is_file()
                 and p.name not in EXCLUDE
                 and not any(part in SKIP_PARTS for part in p.relative_to(root).parts)
                 and not p.name.endswith((".pyc",".tmp",".log"))
                 and p.resolve()!=output)
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root/"PACKAGE_MANIFEST.json").write_text(json.dumps({
        "project":"LiquidityLens","version":"0.2.0","files":len(files),"file_hashes":hashes
    },indent=2)+"\n")
    (root/"CHECKSUMS.sha256").write_text("".join(f"{digest}  {name}\n" for name,digest in hashes.items()))
    with ZipFile(output,"w",ZIP_DEFLATED,compresslevel=6) as bundle:
        for p in files+[root/"PACKAGE_MANIFEST.json",root/"CHECKSUMS.sha256"]:
            bundle.write(p,"liquiditylens/"+str(p.relative_to(root)))
    print(json.dumps({"path":str(output),"files":len(files)+2,
                      "bytes":output.stat().st_size,
                      "sha256":hashlib.sha256(output.read_bytes()).hexdigest()}))

if __name__=="__main__":main()
