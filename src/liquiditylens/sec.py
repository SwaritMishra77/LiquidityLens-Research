"""Rate-limited official SEC ingestion with content-addressed provenance.

No paid APIs, impersonation, or synthetic replacement for missing filings.
"""
from __future__ import annotations
import csv, gzip, hashlib, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import pandas as pd
from .io import dump_json, utc

class SECClient:
    def __init__(self, root, user_agent=None, rate=2.0, timeout=45):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT")
        if not self.user_agent or len(self.user_agent) < 12:
            raise ValueError("Set SEC_USER_AGENT to an honest project identity and your contact email.")
        self.rate, self.timeout = rate, timeout
        self.lock = threading.Lock(); self.last = 0.0

    def fetch(self, url, relative, refresh=False):
        path = self.root / relative
        if path.exists() and not refresh:
            with gzip.open(path, "rb") as f: return f.read()
        error = None
        for attempt in range(3):
            with self.lock:
                delay = 1 / self.rate - (time.monotonic() - self.last)
                if delay > 0: time.sleep(delay)
                self.last = time.monotonic()
            try:
                req = Request(url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "identity"})
                with urlopen(req, timeout=self.timeout) as response:
                    payload = response.read()
                if not payload or (b"Request Rate Threshold Exceeded" in payload[:5000]):
                    raise RuntimeError("Empty response or SEC access-denied page")
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".tmp")
                with tmp.open("wb") as target:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as f: f.write(payload)
                tmp.replace(path)
                dump_json(str(path) + ".meta.json", {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "sha256_uncompressed": hashlib.sha256(payload).hexdigest(), "bytes_uncompressed": len(payload),
                    "user_agent": self.user_agent, "http_status": 200})
                return payload
            except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as e:
                error = e
                # Do not retry policy-denied requests or missing documents.
                if isinstance(e, HTTPError) and e.code in (403, 404): break
                if attempt < 2: time.sleep(2 ** attempt)
        raise RuntimeError(f"Could not fetch {url}: {error}")

    def json(self, url, relative, **kwargs):
        return json.loads(self.fetch(url, relative, **kwargs))

def filing_available_at(record):
    # Conservative date-only guard; avoids relying on ambiguous SEC timestamp timezone conventions.
    conservative = utc(record["filingDate"]) + pd.Timedelta(days=2)
    accepted = pd.to_datetime(record.get("acceptanceDateTime"), utc=True, errors="coerce")
    return max(conservative, accepted) if pd.notna(accepted) else conservative

def records_from_columns(columns):
    if not columns: return []
    n = len(columns.get("accessionNumber", []))
    return [{k: v[i] for k,v in columns.items() if isinstance(v,list) and len(v)==n} for i in range(n)]

# Explicit CIKs retain selected delisted/renamed filers instead of silently dropping them from current ticker lookup.
LEGACY_CIKS = {"JWN": "0000072333", "WBA": "0001618921", "GPS": "0000039911"}

def ingest(root, universe_path, workers=4):
    root = Path(root); client = SECClient(root / "data/raw")
    tickers = client.json("https://www.sec.gov/files/company_tickers.json", "company_tickers.json.gz")
    lookup = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in tickers.values()}
    universe = list(csv.DictReader(open(universe_path, encoding="utf-8")))
    def one(company):
        ticker = company["ticker"]
        # Historical-universe manifests retain the CIK even if a ticker later
        # disappears or changes. An explicit CIK must never be overridden by a
        # present-day ticker lookup.
        supplied = str(company.get("cik", "")).strip()
        cik = supplied.zfill(10) if supplied.isdigit() else LEGACY_CIKS.get(ticker) or lookup.get(ticker)
        if not cik: return {**company, "status": "unresolved", "error": "Not in current ticker lookup; supply explicit CIK"}
        try:
            sub = client.json(f"https://data.sec.gov/submissions/CIK{cik}.json", f"submissions/{cik}.json.gz")
            facts = client.json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", f"facts/{cik}.json.gz")
            filings = records_from_columns(sub.get("filings", {}).get("recent", {}))
            for part in sub.get("filings", {}).get("files", []):
                if str(part.get("filingTo", "9999")) < "2012-01-01": continue
                historical = client.json("https://data.sec.gov/submissions/" + part["name"], "submissions/"+part["name"]+".gz")
                filings.extend(records_from_columns(historical))
            filings = sorted({r["accessionNumber"]: r for r in filings if r.get("form") in {"10-K","10-Q","10-K/A","10-Q/A"}}.values(),
                             key=lambda x:(x["filingDate"],x["accessionNumber"]))
            dump_json(root / f"data/raw/filings/{cik}.json", filings)
            return {**company,"cik":cik,"name":sub.get("name"),"sic":sub.get("sic"),"status":"ok",
                    "filings":len(filings),"fact_tags":len(facts.get("facts",{}).get("us-gaap",{})),
                    "ticker_resolution":"manifest_cik" if supplied.isdigit() else "explicit_legacy_cik" if ticker in LEGACY_CIKS else "current_lookup"}
        except Exception as e: return {**company,"cik":cik,"status":"failed","error":str(e)}
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one,c) for c in universe]):
            r=future.result(); results.append(r); print(r["ticker"],r["status"],flush=True)
    results.sort(key=lambda r: next(i for i,c in enumerate(universe) if c["ticker"]==r["ticker"]))
    dump_json(root / "data/raw/universe_resolved.json", results)
    dump_json(root / "reports/ingestion_audit.json", {"companies":results,"successful":sum(x["status"]=="ok" for x in results),
             "universe_policy":"purposive pilot, not representative; contains explicit legacy filers but remains selection-biased"})
    return results

def load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f: return json.load(f)

def fetch_texts(root, config, workers=4):
    root=Path(root); client=SECClient(root/"data/raw")
    samples=pd.read_csv(root/"data/processed/samples.csv",dtype={"cik":str})
    selected=samples[(samples.ticker.isin(config["text_companies"])) & (samples.prediction_at>=config["text_start"])]
    jobs=selected[["cik","accession","primary_document","source_url"]].drop_duplicates("accession").to_dict("records")
    def one(row):
        try:
            payload=client.fetch(row["source_url"],f"html/{row['accession']}.html.gz")
            if b"<html" not in payload[:10000].lower() and b"<body" not in payload[:10000].lower():
                raise ValueError("Expected HTML filing")
            return {"accession":row["accession"],"status":"ok","bytes":len(payload)}
        except Exception as e: return {"accession":row["accession"],"status":"failed","error":str(e)}
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(one,row) for row in jobs]),1):
            results.append(f.result())
            if i%20==0: print(f"text {i}/{len(jobs)}",flush=True)
    dump_json(root/"reports/text_download_audit.json",{"requested":len(jobs),"results":sorted(results,key=lambda x:x["accession"])})
    return results
