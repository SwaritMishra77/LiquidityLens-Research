import math
from liquiditylens.financials import FactStore

TOTAL="NetCashProvidedByUsedInOperatingActivities"
CONT="NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"
DISC="CashProvidedByUsedInOperatingActivitiesDiscontinuedOperations"

def fact(start,end,value,filed="2023-11-01",accn="a",form="10-Q"):
    return {"start":start,"end":end,"val":value,"filed":filed,"accn":accn,"form":form}

def store(tags):
    return FactStore({"facts":{"us-gaap":{tag:{"units":{"USD":values}} for tag,values in tags.items()}}})

def test_quarterly_cashflow_is_not_cumulative():
    s=store({TOTAL:[fact("2023-01-01","2023-03-31",100,"2023-05-01"),fact("2023-01-01","2023-06-30",160,"2023-08-01")]})
    m=s.consolidated_cfo("2023-06-30","2023-08-02")
    assert m.value==60
    assert m.provenance["method"]=="ytd_difference"

def test_fourth_quarter_is_annual_minus_nine_months():
    s=store({TOTAL:[fact("2023-01-01","2023-09-30",250),fact("2023-01-01","2023-12-31",390,"2024-02-01",form="10-K")]})
    assert s.consolidated_cfo("2023-12-31","2024-02-02").value==140

def test_future_restatement_cannot_leak():
    s=store({TOTAL:[fact("2023-01-01","2023-03-31",100,"2023-05-01","old"),fact("2023-01-01","2023-03-31",999,"2024-05-01","new")]})
    assert s.consolidated_cfo("2023-03-31","2023-06-01").value==100
    assert s.consolidated_cfo("2023-03-31","2024-06-01").value==999

def test_fiscal_start_must_match_for_subtraction():
    s=store({TOTAL:[fact("2023-01-02","2023-03-31",100,"2023-05-01"),fact("2023-01-01","2023-06-30",160,"2023-08-01")]})
    assert s.consolidated_cfo("2023-06-30","2023-08-02") is None

def test_conflicting_contexts_are_not_averaged():
    s=store({TOTAL:[fact("2023-01-01","2023-03-31",100,"2023-05-01"),fact("2023-01-01","2023-03-31",200,"2023-05-01")]})
    assert s.consolidated_cfo("2023-03-31","2023-06-01") is None

def test_continuing_alone_is_not_consolidated_cashflow():
    s=store({CONT:[fact("2023-01-01","2023-03-31",100,"2023-05-01")]})
    assert s.consolidated_cfo("2023-03-31","2023-06-01") is None

def test_explicit_continuing_and_discontinued_sum():
    s=store({CONT:[fact("2023-01-01","2023-03-31",100,"2023-05-01")],DISC:[fact("2023-01-01","2023-03-31",-15,"2023-05-01")]})
    assert s.consolidated_cfo("2023-03-31","2023-06-01").value==85

def test_tag_transition_with_reconciled_prior_cumulative_total():
    s=store({CONT:[fact("2023-01-01","2023-09-30",200)],DISC:[fact("2023-01-01","2023-09-30",20)],
             TOTAL:[fact("2023-01-01","2023-12-31",300,"2024-02-01",form="10-K")]})
    assert s.consolidated_cfo("2023-12-31","2024-02-02").value==80

def test_retail_seventeen_week_quarter_supported():
    s=store({TOTAL:[fact("2022-08-29","2023-05-07",700,"2023-06-01"),fact("2022-08-29","2023-09-03",1000,"2023-10-01")]})
    assert s.consolidated_cfo("2023-09-03","2023-10-02").value==300

def test_missing_currency_is_not_reinterpreted_as_usd():
    s=FactStore({"facts":{"us-gaap":{TOTAL:{"units":{"EUR":[fact("2023-01-01","2023-03-31",100)]}}}}})
    assert s.consolidated_cfo("2023-03-31","2023-12-01") is None

