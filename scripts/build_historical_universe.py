"""Offline candidate selection after downloading SEC quarterly master.idx files."""
import argparse
from pathlib import Path
from liquiditylens.universe import build_candidates

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("index_directory",type=Path,help="Directory containing downloaded SEC master.idx files")
    parser.add_argument("--output",type=Path,default=Path("reports/historical_universe"))
    parser.add_argument("--max-candidates",type=int,default=300)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    files=sorted(args.index_directory.rglob("master.idx"))
    if not files: parser.error("No master.idx files found")
    print(build_candidates(files,args.output,seed=args.seed,max_candidates=args.max_candidates))

if __name__=="__main__": main()
