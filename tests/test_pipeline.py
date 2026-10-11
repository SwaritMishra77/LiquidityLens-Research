import numpy as np
import pandas as pd
import pytest
from liquiditylens.io import load_config
from liquiditylens.models import SequencePreprocessor
from liquiditylens.sec import filing_available_at
from liquiditylens.splits import historical_splits
from liquiditylens.metrics import finite_quantile, classification_metrics
from liquiditylens.experiments import calibrate_probabilities
from liquiditylens.text import extract_passages,lexical_change

def test_publication_guard_is_conservative():
    at=filing_available_at({"filingDate":"2023-05-01","acceptanceDateTime":"2023-05-01T20:00:00Z"})
    assert at==pd.Timestamp("2023-05-03",tz="UTC")

def test_later_actual_acceptance_is_retained():
    at=filing_available_at({"filingDate":"2023-05-01","acceptanceDateTime":"2023-05-04T20:00:00Z"})
    assert at==pd.Timestamp("2023-05-04T20:00:00Z")

def test_labels_are_purged_and_companies_disjoint():
    config={"seed":42,"unseen_company_fraction":.2,"train_end":"2021-12-31","validation_end":"2022-12-31","calibration_end":"2023-12-31","test_end":"2025-12-31"}
    f=pd.DataFrame({"ticker":["A","A","B","A","B"],"prediction_at":["2021-05-01","2021-12-01","2021-05-01","2024-05-01","2024-05-01"],
                    "label_available_at":["2021-11-01","2022-06-01","2021-11-01","2024-11-01","2024-11-01"]})
    s,h=historical_splits(f,config,heldout={"B"})
    assert s["train"].tolist()==[0]
    assert s["test_unseen"].tolist()==[4]
    assert s["test_seen"].tolist()==[3]

def test_test_values_do_not_change_preprocessing():
    train=np.array([[[1.,np.nan],[3.,4.]]]); test=np.array([[[1000.,999.]]])
    p=SequencePreprocessor().fit(train); before=p.parameters(); p.transform(test)
    assert p.parameters()==before
    assert p.median.tolist()==[2.,4.]
    assert p.transform(train)[0,0,3]==1.

def test_finite_sample_interval_order_statistic():
    assert finite_quantile(np.arange(1,10),.2)==8
    with pytest.raises(ValueError): finite_quantile([1],.1)

def test_single_class_metrics_are_explicit():
    r=classification_metrics(np.zeros(10),np.full(10,.1))
    assert r["pr_auc"] is None and r["roc_auc"] is None

def test_too_few_events_do_not_fit_calibration():
    p=np.full(20,.1); y=np.array([1]*2+[0]*18)
    out,info,model=calibrate_probabilities(p,y,np.arange(20))
    assert info["status"]=="not_calibrated" and model is None
    assert np.array_equal(out,p)

def test_passages_ignore_script_and_include_relevant_narrative():
    text="Our liquidity and working capital resources support cash requirements. "+"Operations and collections are reviewed regularly by management. "*10
    html=f"<html><script>secret injected liquidity</script><p>{text}</p></html>"
    p,a=extract_passages(html)
    assert p and "secret" not in p[0]["text"]

def test_lexical_change_has_clear_endpoints():
    assert lexical_change("liquidity cash collections","liquidity cash collections")==0
    assert lexical_change("liquidity cash","inventory operations")==1

def test_neural_shapes_and_finite_gradients():
    torch=pytest.importorskip("torch")
    from liquiditylens.models import make_network
    for kind in ["mlp","gru","transformer"]:
        model=make_network(kind,20,8,32,text_size=16)
        out=model(torch.randn(3,8,20),torch.randn(3,16)); assert out.shape==(3,2)
        out.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

