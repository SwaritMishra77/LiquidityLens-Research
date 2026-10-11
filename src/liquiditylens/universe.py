"""Construct a dated issuer candidate list from downloaded SEC master indexes.

The index lists filings and CIKs, but neither a reliable historical ticker nor
the sector classification this project's current feature schema requires.
Candidates therefore need an explicit reviewed manifest before ingestion.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read_master_index(path):
    """Yield original annual/quarterly filing rows from an SEC master.idx file."""
    started = False
    with Path(path).open(encoding="latin-1") as source:
        for line in source:
            line = line.strip()
            if not started:
                started = line == "CIK|Company Name|Form Type|Date Filed|Filename"
                continue
            parts = line.split("|")
            if len(parts) != 5 or parts[2] not in {"10-K", "10-Q"}:
                continue
            cik, name, form, filed, location = parts
            if cik.isdigit() and len(filed) == 10 and location:
                yield (cik.zfill(10), name, form, filed, location)
    if not started:
        raise ValueError(f"Missing SEC master index header: {path}")


def build_candidates(index_paths, output_dir, *, seed=42, max_candidates=300,
                     first_year=2013, last_year=2014):
    if not index_paths or max_candidates < 1 or first_year > last_year:
        raise ValueError("Supply indexes, a positive candidate limit and ordered years")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    filings = defaultdict(list); sources = []
    for path in sorted(map(Path, index_paths)):
        payload = path.read_bytes()
        sources.append({"filename":path.name,"sha256":hashlib.sha256(payload).hexdigest()})
        for record in read_master_index(path):
            if first_year <= int(record[3][:4]) <= last_year:
                filings[record[0]].append(record)
    eligible = []
    for cik, entries in filings.items():
        # Require an annual filing and two quarterly filings in EACH year.
        counts = {year:{"10-K":0,"10-Q":0} for year in range(first_year,last_year+1)}
        for _,_,form,filed,_ in entries:
            counts[int(filed[:4])][form] += 1
        if any(c["10-K"] < 1 or c["10-Q"] < 2 for c in counts.values()):
            continue
        first = min(entries,key=lambda x:(x[3],x[4]))
        eligible.append({"cik":cik,"historical_name":first[1],
                         "annual_filings":sum(c["10-K"] for c in counts.values()),
                         "quarterly_filings":sum(c["10-Q"] for c in counts.values()),
                         "first_filed":first[3],
                         "selection_hash":hashlib.sha256(f"{seed}:{cik}".encode()).hexdigest()})
    selected = sorted(eligible,key=lambda x:(x["selection_hash"],x["cik"]))[:max_candidates]
    with (output/"candidates.csv").open("w",newline="",encoding="utf-8") as target:
        writer=csv.DictWriter(target,fieldnames=list(selected[0]) if selected else
                              ["cik","historical_name","annual_filings","quarterly_filings","first_filed","selection_hash"])
        writer.writeheader();writer.writerows(selected)
    with (output/"universe_review_template.csv").open("w",newline="",encoding="utf-8") as target:
        writer=csv.DictWriter(target,fieldnames=["ticker","sector","rationale","cik"])
        writer.writeheader()
        for entry in selected:
            writer.writerow({"ticker":"","sector":"","rationale":"","cik":entry["cik"]})
    audit={"sources":sources,"first_year":first_year,"last_year":last_year,
           "eligibility":"each calendar year: >=1 original 10-K and >=2 original 10-Q filing entries",
           "seed":seed,"eligible_ciks":len(eligible),"selected_ciks":len(selected),
           "post_selection_work":"Review issuer identity, historical ticker, industry/perimeter and sector. "
             "Fill the blank universe manifest before SEC ingestion; report missing facts, ceased filers "
             "and unusable rows without quietly replacing them. Historical listing membership is not established."}
    (output/"selection_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    return audit
