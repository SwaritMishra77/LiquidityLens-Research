from __future__ import annotations
import json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, brier_score_loss
from .dataset import load_dataset
from .io import dump_json, environment, sha256
from .metrics import evaluate, finite_quantile, paired_company_bootstrap
from .models import SequencePreprocessor, flat_features, fit_neural
from .splits import historical_splits

def linear_fit(x,frame,split):
    tr,va=split["train"],split["validation"]; y=frame.y_margin.to_numpy(); yc=frame.y_stress.to_numpy()
    best=None
    for alpha in [1.,10.,100.]:
        reg=Ridge(alpha=alpha).fit(x[tr],y[tr]); score=mean_absolute_error(y[va],reg.predict(x[va]))
        if best is None or score<best[0]: best=(score,reg,alpha)
    if len(np.unique(yc[tr]))<2: raise ValueError("Training split has only one stress class")
    cbest=None
    for c in [.01,.1,1.]:
        clf=LogisticRegression(C=c,max_iter=2000,solver="lbfgs").fit(x[tr],yc[tr])
        score=brier_score_loss(yc[va],clf.predict_proba(x[va])[:,1])
        if cbest is None or score<cbest[0]: cbest=(score,clf,c)
    return best[1].predict(x),cbest[1].predict_proba(x)[:,1],{"ridge_alpha":best[2],"logistic_C":cbest[2]},(best[1],cbest[1])

def tree_fit(x,frame,split,seed):
    tr,va=split["train"],split["validation"]; y=frame.y_margin.to_numpy(); yc=frame.y_stress.to_numpy()
    best=None; cbest=None
    grid=[{"max_leaf_nodes":7,"l2_regularization":5.},{"max_leaf_nodes":15,"l2_regularization":1.}]
    for params in grid:
        reg=HistGradientBoostingRegressor(max_iter=150,learning_rate=.05,min_samples_leaf=15,early_stopping=False,random_state=seed,**params).fit(x[tr],y[tr])
        score=mean_absolute_error(y[va],reg.predict(x[va]))
        if best is None or score<best[0]: best=(score,reg,params)
        clf=HistGradientBoostingClassifier(max_iter=150,learning_rate=.05,min_samples_leaf=15,early_stopping=False,random_state=seed,**params).fit(x[tr],yc[tr])
        score=brier_score_loss(yc[va],clf.predict_proba(x[va])[:,1])
        if cbest is None or score<cbest[0]: cbest=(score,clf,params)
    quantiles=[]; quantile_models=[]
    for q in [.1,.9]:
        reg=HistGradientBoostingRegressor(loss="quantile",quantile=q,max_iter=150,learning_rate=.05,min_samples_leaf=15,
            early_stopping=False,random_state=seed,**best[2]).fit(x[tr],y[tr])
        quantiles.append(reg.predict(x)); quantile_models.append(reg)
    lower=np.minimum(*quantiles); upper=np.maximum(*quantiles)
    return best[1].predict(x),cbest[1].predict_proba(x)[:,1],{"regression":best[2],"classification":cbest[2]},(best[1],cbest[1],*quantile_models),(lower,upper)

def calibrate_probabilities(raw,labels,ids):
    p=np.clip(raw,1e-6,1-1e-6); logits=np.log(p/(1-p)).reshape(-1,1)
    counts=np.bincount(labels[ids].astype(int),minlength=2)
    if len(ids)<20 or min(counts)<5:
        return p,{"status":"not_calibrated","reason":"fewer than 20 rows or fewer than 5 examples of either class","class_counts":counts.tolist()},None
    calibrator=LogisticRegression(C=.1,max_iter=1000).fit(logits[ids],labels[ids])
    return calibrator.predict_proba(logits)[:,1],{"status":"platt","rows":len(ids)},calibrator

def interval(point,y,ids,alpha,bounds=None):
    if bounds is None:
        q=finite_quantile(np.abs(y[ids]-point[ids]),alpha)
        return point-q,point+q,{"method":"absolute residual split conformal","q":q,"rows":len(ids)}
    lower,upper=bounds
    scores=np.maximum.reduce([lower[ids]-y[ids],y[ids]-upper[ids],np.zeros(len(ids))])
    q=finite_quantile(scores,alpha)
    return lower-q,upper+q,{"method":"nonshrinking conformalized quantile interval","q":q,"rows":len(ids)}

def align_embeddings(root,frame):
    a=np.load(Path(root)/"data/processed/text_embeddings.npz",allow_pickle=False)
    lookup={k:i for i,k in enumerate(a["sample_id"])}
    keep=np.asarray([i for i,k in enumerate(frame.sample_id) if k in lookup and np.isfinite(a["previous"][lookup[k]]).all()])
    j=np.asarray([lookup[frame.iloc[i].sample_id] for i in keep])
    return keep,a["current"][j],a["current"][j]-a["previous"][j],a["extra"][j]

def text_features(root,frame,current,change,extra,split,artifact_dir):
    tr=split["train"]
    n=min(16,len(tr)-1,current.shape[1]); pca=PCA(n_components=n,svd_solver="full").fit(current[tr])
    pcac=PCA(n_components=min(8,len(tr)-1,change.shape[1]),svd_solver="full").fit(change[tr])
    now=pca.transform(current); delta=pcac.transform(change)
    scalar=np.nan_to_num(extra,nan=0.)
    s=StandardScaler().fit(now[tr]); sc=StandardScaler().fit(np.column_stack([delta,scalar])[tr])
    now=s.transform(now); delta=sc.transform(np.column_stack([delta,scalar]))
    records={r["sample_id"]:r for r in [json.loads(line) for line in (Path(root)/"data/processed/passages.jsonl").read_text().splitlines()]}
    texts=[" ".join(p["text"] for p in records[k]["passages"]) for k in frame.sample_id]
    vectorizer=TfidfVectorizer(max_features=4000,min_df=2,ngram_range=(1,2),sublinear_tf=True)
    sparse_train=vectorizer.fit_transform([texts[i] for i in tr])
    svd=TruncatedSVD(n_components=min(32,len(tr)-1,sparse_train.shape[1]-1),random_state=42).fit(sparse_train)
    tfidf=svd.transform(vectorizer.transform(texts)); st=StandardScaler().fit(tfidf[tr]); tfidf=st.transform(tfidf)
    with open(artifact_dir/"text_preprocessing.pkl","wb") as f: pickle.dump((pca,pcac,s,sc,vectorizer,svd,st),f)
    return now,delta,tfidf

def run_experiments(root,config,cohort="numeric",deep=True):
    root=Path(root); frame,raw,feature_names=load_dataset(root)
    _,heldout=historical_splits(frame,config)
    if cohort=="matched_text":
        keep,current,change,extra=align_embeddings(root,frame); frame=frame.iloc[keep].reset_index(drop=True); raw=raw[keep]
    split,_=historical_splits(frame,config,heldout)
    counts={k:len(v) for k,v in split.items()}
    if min(counts[k] for k in ["train","validation","calibration","test_seen"])<10:
        raise ValueError(f"Insufficient cohort coverage: {counts}")
    run_name=cohort+("_"+config["run_name"] if config.get("run_name") else "")
    artifact_dir=root/"artifacts"/run_name; artifact_dir.mkdir(parents=True,exist_ok=True)
    out=root/"reports"/run_name; out.mkdir(parents=True,exist_ok=True)
    pre=SequencePreprocessor().fit(raw[split["train"]]); sequence=pre.transform(raw); x=flat_features(sequence,frame.sector)
    dump_json(artifact_dir/"numeric_preprocessing.json",pre.parameters())
    dump_json(out/"run_manifest.json",{"config":config,"environment":environment(),"split_counts":counts,
        "heldout_companies":sorted(heldout),"cohort":cohort,"features":feature_names,
        "dataset_sha256":sha256(root/"data/processed/samples.csv"),"code_sha256":{p.name:sha256(p) for p in sorted((root/"src/liquiditylens").glob("*.py"))},
        "frozen_at":pd.Timestamp.now(tz="UTC").isoformat(),"interpretation":"exploratory pilot; not preregistered external validation"})
    split_rows=pd.concat([frame.iloc[ids][["sample_id","ticker","prediction_at","label_available_at"]].assign(split=name) for name,ids in split.items()])
    split_rows.to_csv(out/"split_membership.csv",index=False)
    y=frame.y_margin.to_numpy(); yc=frame.y_stress.to_numpy(); results=[]; predictions=[]; logs=[]
    def record(name,seed,point,prob,details,bounds=None,objects=None):
        calibrated,prob_details,calibrator=calibrate_probabilities(prob,yc,split["calibration"])
        lower,upper,int_details=interval(point,y,split["calibration"],config["interval_alpha"],bounds)
        details={**details,"probability_calibration":prob_details,"interval":int_details}
        details["probability_calibration_parameters"]={"coefficient":calibrator.coef_.tolist(),"intercept":calibrator.intercept_.tolist()} if calibrator is not None else None
        # Persist every model's validation score, never select from test metrics.
        details["validation_mae"]=float(mean_absolute_error(y[split["validation"]],point[split["validation"]]))
        if objects is not None:
            with open(artifact_dir/f"{name}_{seed}.pkl","wb") as f: pickle.dump({"models":objects,"calibrator":calibrator,"interval":int_details},f)
        for group in ["test_seen","test_unseen"]:
            ids=split[group]
            if not len(ids): continue
            r={"cohort":cohort,"model":name,"seed":seed,"group":group,**evaluate(frame.iloc[ids],point[ids],calibrated[ids],lower[ids],upper[ids])}
            results.append(r)
            pred=frame.iloc[ids][["sample_id","ticker","prediction_at","report_end","target_end","economic_lead_days","y_margin","y_stress","source_url"]].copy()
            pred=pred.assign(cohort=cohort,model=name,seed=seed,group=group,point=point[ids],stress_probability=calibrated[ids],
                             raw_stress_probability=prob[ids],lower=lower[ids],upper=upper[ids]); predictions.append(pred)
        logs.append({"model":name,"seed":seed,**details}); print(cohort,name,seed,"done",flush=True)
    prevalence=float(yc[split["train"]].mean()); constant=np.full(len(frame),prevalence)
    seasonal=raw[:,-4,0].copy(); fallback=float(np.nanmedian(y[split["train"]])); seasonal=np.nan_to_num(seasonal,nan=fallback)
    record("seasonal",0,seasonal,constant,{"classifier":"training prevalence"})
    record("persistence",0,np.nan_to_num(raw[:,-1,0],nan=fallback),constant,{"classifier":"training prevalence"})
    p,c,d,models=linear_fit(x,frame,split); record("linear_numeric",0,p,c,d,objects=models)
    p,c,d,models,bounds=tree_fit(x,frame,split,config["seed"]); record("boosted_numeric",0,p,c,d,bounds,models)
    if deep:
        for kind in ["mlp","gru","transformer"]:
            for seed in config["seeds"]:
                p,c,d=fit_neural(kind,sequence,frame,split,config,seed,artifact_dir/f"{kind}_numeric_{seed}.pt")
                record(f"{kind}_numeric",seed,p,c,d)
    if cohort=="matched_text":
        now,delta,tfidf=text_features(root,frame,current,change,extra,split,artifact_dir)
        features={"linear_tfidf":tfidf,"linear_numeric_tfidf":np.column_stack([x,tfidf]),
                  "linear_finbert":now,"linear_numeric_finbert":np.column_stack([x,now]),
                  "linear_numeric_finbert_change":np.column_stack([x,now,delta])}
        for name,z in features.items():
            p,c,d,models=linear_fit(z,frame,split); record(name,0,p,c,d,objects=models)
        if deep:
            for name,z in [("gru_finbert",now),("gru_finbert_change",np.column_stack([now,delta]))]:
                for seed in config["seeds"]:
                    p,c,d=fit_neural("gru",sequence,frame,split,config,seed,artifact_dir/f"{name}_{seed}.pt",text=z)
                    record(name,seed,p,c,d)
    metrics=pd.DataFrame(results); pred=pd.concat(predictions,ignore_index=True)
    metrics.to_csv(out/"metrics.csv",index=False,float_format="%.12g"); pred.to_csv(out/"predictions.csv",index=False,float_format="%.12g")
    dump_json(out/"training_details.json",logs)
    # Primary comparison uses seed-mean predictions, retaining company clustering.
    comparison=[]
    for group,g in pred.groupby("group"):
        averaged=g.groupby(["sample_id","model"],as_index=False).agg(point=("point","mean"),ticker=("ticker","first"),y_margin=("y_margin","first"))
        reference=averaged[averaged.model=="boosted_numeric"].set_index("sample_id")
        for name,candidate in averaged.groupby("model"):
            candidate=candidate.set_index("sample_id").reindex(reference.index)
            comparison.append({"group":group,"model":name,"reference":"boosted_numeric",
                **paired_company_bootstrap(reference.reset_index(),reference.point.to_numpy(),candidate.point.to_numpy(),config["seed"],config["bootstrap_replicates"])})
    dump_json(out/"paired_comparisons.json",comparison)
    # Champion is selected by validation before evaluating the held-out groups; no test ranking is used.
    champion=min(logs,key=lambda r:r["validation_mae"])
    dump_json(out/"champion.json",{"selection_metric":"validation MAE only","model":champion["model"],"seed":champion["seed"],"validation_mae":champion["validation_mae"]})
    return metrics

def rolling_baselines(root,config):
    """Annual rolling-origin sensitivity. Overlapping folds are not independent replications."""
    tables=[]
    for year in [2023,2024,2025]:
        cfg=dict(config,train_end=f"{year-5}-12-31",validation_end=f"{year-3}-12-31",
                 calibration_end=f"{year-1}-12-31",test_end=f"{year}-12-31",run_name=f"rolling_{year}")
        metrics=run_experiments(root,cfg,cohort="numeric",deep=False)
        tables.append(metrics.assign(test_year=year))
    pd.concat(tables).to_csv(Path(root)/"reports/rolling_baselines.csv",index=False)


