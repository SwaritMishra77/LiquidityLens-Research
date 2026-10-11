from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, average_precision_score, roc_auc_score, brier_score_loss, log_loss

def finite_quantile(scores, alpha):
    scores=np.asarray(scores,float)
    if len(scores)==0: raise ValueError("Calibration set is empty")
    # Conservative finite-sample order statistic; n too small cannot support finite valid coverage.
    rank=math.ceil((len(scores)+1)*(1-alpha))
    if rank>len(scores): raise ValueError("Too few calibration rows for the requested interval level")
    return float(np.sort(scores)[rank-1])

def interval_metrics(y, lower, upper):
    return {"interval_coverage":float(np.mean((y>=lower)&(y<=upper))),"interval_mean_width":float(np.mean(upper-lower))}

def classification_metrics(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6); y=np.asarray(y,int)
    order=np.argsort(-p,kind="stable"); k=max(1,math.ceil(.1*len(y)))
    both=len(np.unique(y))==2
    return {"stress_prevalence":float(y.mean()),"pr_auc":float(average_precision_score(y,p)) if both else None,
            "roc_auc":float(roc_auc_score(y,p)) if both else None,"brier":float(brier_score_loss(y,p)),
            "log_loss":float(log_loss(y,p,labels=[0,1])),"precision_top_10pct":float(y[order[:k]].mean()),"alerts":k}

def evaluate(frame, point, probability, lower, upper):
    y=frame.y_margin.to_numpy(); out={"n":len(frame),"companies":int(frame.ticker.nunique()),
        "mae":float(mean_absolute_error(y,point)),"rmse":float(np.sqrt(mean_squared_error(y,point)))}
    out.update(interval_metrics(y,lower,upper)); out.update(classification_metrics(frame.y_stress.to_numpy(),probability))
    errors=pd.DataFrame({"ticker":frame.ticker.to_numpy(),"ae":np.abs(y-point)})
    out["company_macro_mae"]=float(errors.groupby("ticker").ae.mean().mean())
    return out

def paired_company_bootstrap(frame, reference, contender, seed=42, replicates=400):
    errors=pd.DataFrame({"ticker":frame.ticker.to_numpy(),"delta":np.abs(frame.y_margin.to_numpy()-contender)-np.abs(frame.y_margin.to_numpy()-reference)})
    delta=errors.groupby("ticker").delta.mean().to_numpy()
    if len(delta)<2: return {"companies":len(delta),"mean_delta":float(delta.mean()),"ci95":None}
    rng=np.random.default_rng(seed)
    boot=np.mean(rng.choice(delta,size=(replicates,len(delta)),replace=True),axis=1)
    return {"companies":len(delta),"mean_delta":float(delta.mean()),"ci95":[float(v) for v in np.quantile(boot,[.025,.975])],
            "method":"paired company-cluster bootstrap of company-mean absolute-error differences; common time shocks not resampled"}

