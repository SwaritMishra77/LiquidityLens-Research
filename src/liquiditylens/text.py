from __future__ import annotations
from collections import Counter
import gzip, hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from .io import dump_json, sha256

KEYWORDS = {"liquidity":4,"working capital":4,"cash flow":3,"cash flows":3,"operating cash":4,
            "accounts receivable":3,"collections":2,"inventory":2,"financing":2,"credit facility":3,
            "covenant":4,"refinancing":3,"debt maturity":3,"capital resources":3,"cash requirements":3}

def extract_passages(html, max_passages=4, words=150):
    soup=BeautifulSoup(html,"html.parser")
    for node in soup.find_all(["script","style","noscript","ix:header","ix:hidden"]): node.decompose()
    # Remove numeric tables, but retain narrative layout tables.
    for table in soup.find_all("table"):
        text=table.get_text(" ",strip=True)
        digits=sum(c.isdigit() for c in text)
        if len(text)>0 and digits/len(text)>.09: table.decompose()
    blocks=[]
    for node in soup.find_all(["p","div","li"]):
        if node.find(["p","div","li"]): continue
        s=re.sub(r"\s+"," ",node.get_text(" ",strip=True)).strip()
        if len(s.split())>=30: blocks.append(s)
    if not blocks:
        text=re.sub(r"\s+"," ",soup.get_text(" ",strip=True)); tokens=text.split()
        blocks=[" ".join(tokens[i:i+words]) for i in range(0,len(tokens),words)]
    candidates=[]; seen=set()
    for block in blocks:
        tokens=block.split()
        for start in range(0,len(tokens),words):
            chunk=" ".join(tokens[start:start+words]); lower=chunk.lower()
            if len(chunk.split())<25 or chunk in seen: continue
            seen.add(chunk)
            score=sum(weight for term,weight in KEYWORDS.items() if term in lower)
            if score>0: candidates.append({"text":chunk,"retrieval_score":score})
    candidates.sort(key=lambda p:(-p["retrieval_score"],p["text"]))
    return candidates[:max_passages],{"blocks":len(blocks),"candidate_passages":len(candidates),"selected":min(len(candidates),max_passages)}

def token_set(s): return set(re.findall(r"[a-z]{3,}",s.lower()))

def lexical_change(current, previous):
    a,b=token_set(current),token_set(previous)
    return 1-len(a&b)/max(1,len(a|b))

def process_texts(root, config):
    root=Path(root); frame=pd.read_csv(root/"data/processed/samples.csv",dtype={"cik":str})
    old_path=root/"data/processed/passages.jsonl"
    cached={r["sample_id"]:r for r in (json.loads(s) for s in old_path.read_text().splitlines())} if old_path.exists() else {}
    records={}; failures=[]; reused=0; parsed=0
    for row in frame.to_dict("records"):
        path=root/f"data/raw/html/{row['accession']}.html.gz"
        if not path.exists(): continue
        with gzip.open(path,"rb") as f: html=f.read()
        digest=hashlib.sha256(html).hexdigest()
        prior=cached.get(row["sample_id"])
        if (prior and prior.get("html_sha256")==digest
                and len(prior.get("passages",[]))==min(prior.get("extraction_audit",{}).get("candidate_passages",-1),config["max_passages"])):
            passages,audit=prior["passages"],prior["extraction_audit"];reused+=1
        else:
            passages,audit=extract_passages(html,config["max_passages"]);parsed+=1
        if not passages:
            failures.append({"sample_id":row["sample_id"],"reason":"no relevant passages"}); continue
        record={"sample_id":row["sample_id"],"accession":row["accession"],"ticker":row["ticker"],
                "form":row["form"],"report_end":row["report_end"],"prediction_at":row["prediction_at"],
                "source_url":row["source_url"],"html_sha256":digest,"passages":passages,"extraction_audit":audit}
        records[row["sample_id"]]=record
    for ticker,g in frame.groupby("ticker"):
        for row in g.sort_values("prediction_at").to_dict("records"):
            record=records.get(row["sample_id"])
            if record is None: continue
            earlier=[records[r["sample_id"]] for r in g.to_dict("records") if r["sample_id"] in records
                     and r["prediction_at"]<row["prediction_at"] and r["form"]==row["form"]
                     and 300<=(pd.Timestamp(row["report_end"])-pd.Timestamp(r["report_end"])).days<=430]
            # Compare nearest same-form prior-year disclosure; no future document can be selected.
            prior=min(earlier,key=lambda r:abs((pd.Timestamp(row["report_end"])-pd.Timestamp(r["report_end"])).days-365)) if earlier else None
            record["previous_sample_id"]=prior["sample_id"] if prior else None
            record["previous_available_at"]=prior["prediction_at"] if prior else None
            current=" ".join(p["text"] for p in record["passages"])
            previous=" ".join(p["text"] for p in prior["passages"]) if prior else ""
            record["lexical_change"]=lexical_change(current,previous) if prior else None
    ordered=[records[k] for k in sorted(records)]
    with (root/"data/processed/passages.jsonl").open("w",encoding="utf-8") as f:
        for r in ordered: f.write(json.dumps(r)+"\n")
    dump_json(root/"reports/text_extraction_audit.json",{"records":len(records),"companies":len({r["ticker"] for r in ordered}),
        "reused_verified_passages":reused,"newly_parsed":parsed,
        "with_previous_year":sum(r["previous_sample_id"] is not None for r in ordered),"failures":failures,
        "method":"keyword retrieval, not verified MD&A section identification; selected passages require manual QA"})
    return ordered

def encode_texts(root, config, batch_size=16):
    import torch
    from transformers import AutoTokenizer, AutoModel
    root=Path(root); path=root/"data/processed/passages.jsonl"
    records=[json.loads(s) for s in path.read_text().splitlines()]
    if not records: raise ValueError("No passages to encode")
    revision=config.get("text_revision")
    if not revision or len(revision)!=40: raise ValueError("Pin a 40-character model commit before encoding")
    torch.set_num_threads(2); torch.manual_seed(config["seed"])
    tokenizer=AutoTokenizer.from_pretrained(config["text_model"],revision=revision,trust_remote_code=False,local_files_only=config.get("offline",False))
    model=AutoModel.from_pretrained(config["text_model"],revision=revision,trust_remote_code=False,local_files_only=config.get("offline",False))
    device="cuda" if torch.cuda.is_available() else "cpu"; model.to(device); model.eval()
    texts=[p["text"] for r in records for p in r["passages"]]
    batches=[]
    with torch.inference_mode():
        for start in range(0,len(texts),batch_size):
            inputs=tokenizer(texts[start:start+batch_size],padding=True,truncation=True,max_length=config["max_tokens"],return_tensors="pt")
            inputs={k:v.to(device) for k,v in inputs.items()}
            hidden=model(**inputs).last_hidden_state
            mask=inputs["attention_mask"].unsqueeze(-1)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
            batches.append(pooled.cpu().numpy())
            if start%(batch_size*10)==0: print(f"encoder {start}/{len(texts)} passages",flush=True)
    vectors=np.concatenate(batches); cursor=0; means=[]
    for r in records:
        n=len(r["passages"]); means.append(vectors[cursor:cursor+n].mean(0)); cursor+=n
    means=np.asarray(means,np.float32); lookup={r["sample_id"]:i for i,r in enumerate(records)}
    previous=np.full_like(means,np.nan); extra=[]
    for i,r in enumerate(records):
        j=lookup.get(r["previous_sample_id"])
        if j is not None:
            if records[j]["prediction_at"]>=r["prediction_at"]: raise AssertionError("Future text in change feature")
            previous[i]=means[j]
        extra.append([r["lexical_change"] if r["lexical_change"] is not None else np.nan,
                      float(j is not None),float(len(r["passages"]))])
    np.savez_compressed(root/"data/processed/text_embeddings.npz",sample_id=np.asarray([r["sample_id"] for r in records]),
                        current=means,previous=previous,extra=np.asarray(extra,np.float32))
    dump_json(root/"data/processed/encoder_manifest.json",{"model":config["text_model"],"revision":revision,
        "pooling":"attention-mask mean of final hidden states, then equal passage mean","max_tokens":config["max_tokens"],
        "records":len(records),"passages":len(texts),"device":device,"embedding_dimension":means.shape[1],
        "input_sha256":sha256(path),"output_sha256":sha256(root/"data/processed/text_embeddings.npz"),
        "weight_provenance":"Pinned public 2022 repository revision; corpus overlap not independently audited",
        "interpretation":"financial language representations; not liquidity labels or ground-truth sentiment"})
    return means

