import csv
import gzip
import json
import hashlib
import pandas as pd
import pytest
from liquiditylens.universe import build_candidates
from liquiditylens.decision import select_reviews, review_backtest
from liquiditylens.text import process_texts
from scripts.reproduce import verify_cached_embeddings


def test_historical_index_selection_excludes_incomplete_and_is_order_stable(tmp_path):
    header="Description\nCIK|Company Name|Form Type|Date Filed|Filename\n"
    paths=[]
    for year in (2013,2014):
        for q in range(1,5):
            rows=[]
            for cik in ("11","22","33"):
                if cik=="33" and year==2014 and q in (2,3): continue
                form="10-K" if q==1 else "10-Q"
                rows.append(f"{cik}|Original {cik}|{form}|{year}-0{q}-01|edgar/{cik}/{year}{q}.txt")
            path=tmp_path/f"{year}_{q}"/"master.idx";path.parent.mkdir()
            path.write_text(header+"\n".join(rows)+"\n")
            paths.append(path)
    result=build_candidates(paths,tmp_path/"out",max_candidates=2)
    assert result["eligible_ciks"]==2 and result["selected_ciks"]==2
    first=(tmp_path/"out"/"candidates.csv").read_bytes()
    build_candidates(list(reversed(paths)),tmp_path/"out",max_candidates=2)
    assert (tmp_path/"out"/"candidates.csv").read_bytes()==first
    with (tmp_path/"out"/"universe_review_template.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==2 and all(not row["ticker"] and not row["sector"] for row in rows)
    assert len(json.loads((tmp_path/"out"/"selection_audit.json").read_text())["sources"])==8


def test_review_queue_only_uses_predictions_to_rank():
    rows=pd.DataFrame({"sample_id":["c","b","a","d"],"prediction_at":["2025-02-01"]*3+["2025-05-01"],
                       "point":[.2,-.1,-.1,.3],"y_stress":[0,0,1,0],"y_margin":[.1,.1,-.2,.2]})
    selected=select_reviews(rows,1)
    assert selected.sample_id.tolist()==["a","d"]
    changed=rows.copy();changed["y_stress"]=1-changed.y_stress;changed["y_margin"]=999
    assert select_reviews(changed.sample(frac=1,random_state=3),1).sample_id.tolist()==["a","d"]
    with pytest.raises(ValueError):select_reviews(pd.concat([rows,rows.head(1)]),1)


def test_backtest_rejects_nonmatching_cohorts():
    base=pd.DataFrame({"sample_id":["a","b"],"prediction_at":["2025-02-01"]*2,
                       "point":[-.1,.2],"y_stress":[1,0],"y_margin":[-.1,.2],
                       "group":["test_seen"]*2,"model":["seasonal"]*2,"seed":[0]*2})
    other=base.copy();other["model"]="boosted_numeric";other.loc[0,"sample_id"]="c"
    champion=base.copy();champion["model"]="mlp_numeric";champion["seed"]=73
    with pytest.raises(ValueError,match="identical sample IDs"):
        review_backtest(pd.concat([base,other,champion]),{"model":"mlp_numeric","seed":73})


def test_passage_cache_reuses_only_verified_unchanged_html(tmp_path):
    processed=tmp_path/"data/processed"; processed.mkdir(parents=True)
    source=tmp_path/"data/raw/html";source.mkdir(parents=True)
    pd.DataFrame([{"sample_id":"cik_accession","accession":"accession","ticker":"T",
                   "form":"10-Q","report_end":"2023-03-31",
                   "prediction_at":"2023-05-01","source_url":"https://example.invalid"}]).to_csv(processed/"samples.csv",index=False)
    path=source/"accession.html.gz"
    first="<html><p>"+("Liquidity and working capital were stable through the reporting period. "*9)+"</p></html>"
    with gzip.open(path,"wb") as f:f.write(first.encode())
    record=process_texts(tmp_path,{"max_passages":4})[0]
    assert record["html_sha256"]==hashlib.sha256(first.encode()).hexdigest()
    assert json.loads((tmp_path/"reports/text_extraction_audit.json").read_text())["newly_parsed"]==1
    process_texts(tmp_path,{"max_passages":4})
    assert json.loads((tmp_path/"reports/text_extraction_audit.json").read_text())["reused_verified_passages"]==1
    changed=first.replace("stable","challenging")
    with gzip.open(path,"wb") as f:f.write(changed.encode())
    updated=process_texts(tmp_path,{"max_passages":4})[0]
    assert updated["html_sha256"]!=record["html_sha256"]
    assert json.loads((tmp_path/"reports/text_extraction_audit.json").read_text())["newly_parsed"]==1


def test_reproduction_rejects_stale_text_cache(tmp_path):
    (tmp_path/"configs").mkdir();(tmp_path/"data/processed").mkdir(parents=True)
    (tmp_path/"configs/pilot.yaml").write_text("text_model: pinned/model\ntext_revision: '1234567890123456789012345678901234567890'\n")
    paths=[tmp_path/"data/processed/passages.jsonl",tmp_path/"data/processed/text_embeddings.npz"]
    for path in paths:path.write_bytes(b"original")
    (tmp_path/"data/processed/encoder_manifest.json").write_text(json.dumps({
        "model":"pinned/model","revision":"1234567890123456789012345678901234567890",
        "input_sha256":hashlib.sha256(b"original").hexdigest(),
        "output_sha256":hashlib.sha256(b"original").hexdigest()}))
    verify_cached_embeddings(tmp_path,__import__("pathlib").Path("configs/pilot.yaml"))
    paths[0].write_bytes(b"changed")
    with pytest.raises(ValueError,match="passages.jsonl"):
        verify_cached_embeddings(tmp_path,__import__("pathlib").Path("configs/pilot.yaml"))
