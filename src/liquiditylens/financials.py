"""Conservative USD US-GAAP normalization with explicit fact provenance.

Values are selected by filing date, never by the latest returned frame. A
standalone quarter may be direct or the difference of matching-start YTD
facts from the SAME tag and currency. Unsupported calendars are rejected.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import math

TAGS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "income": ["NetIncomeLoss", "ProfitLoss"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "inventory": ["InventoryNet"],
}
FEATURES = ["cfo_margin", "cash_assets", "cash_current_liabilities", "current_ratio",
            "receivables_sales", "inventory_sales", "capex_sales", "net_margin", "log_revenue", "log_assets"]

def days(start, end): return (date.fromisoformat(end) - date.fromisoformat(start)).days + 1

@dataclass
class Measure:
    value: float
    provenance: dict

class FactStore:
    def __init__(self, companyfacts):
        self.buckets = {}
        self.by_tag_start = {}
        for tag, obj in companyfacts.get("facts", {}).get("us-gaap", {}).items():
            for fact in obj.get("units", {}).get("USD", []):
                if fact.get("form") not in {"10-Q","10-K","10-Q/A","10-K/A"}: continue
                if not all(k in fact for k in ["end","filed","accn","val"]): continue
                if not math.isfinite(float(fact["val"])): continue
                self.buckets.setdefault((tag,fact["end"]), []).append(fact)
                if fact.get("start"):
                    self.by_tag_start.setdefault((tag,fact["start"]),[]).append(fact)

    def candidates(self, tag, end, asof):
        return [f for f in self.buckets.get((tag,end),[]) if f["filed"]<=asof]

    @staticmethod
    def choose(facts, current_accn=None):
        if not facts: return None
        best=max(facts,key=lambda f:(f["filed"], f["accn"]==current_accn, f["accn"]))
        tied=[f for f in facts if (f["filed"],f["accn"])==(best["filed"],best["accn"])]
        # Ambiguous entity/context values are not silently averaged.
        if len({float(f["val"]) for f in tied})>1: return None
        return best

    def instant(self, aliases, end, asof, current_accn=None):
        for tag in aliases:
            f=self.choose([f for f in self.candidates(tag,end,asof) if "start" not in f],current_accn)
            if f:
                return Measure(float(f["val"]),{"tag":tag,"unit":"USD","method":"instant","facts":[f]})
        return None

    def quarter(self, aliases, end, asof, current_accn=None):
        for tag in aliases:
            fs=[f for f in self.candidates(tag,end,asof) if f.get("start") and 65<=days(f["start"],end)<=380]
            direct=[f for f in fs if 65<=days(f["start"],end)<=130]
            # Group contexts before choosing; direct quarter only, not a mixed YTD context.
            starts=sorted({f["start"] for f in direct},reverse=True)
            for start in starts:
                f=self.choose([f for f in direct if f["start"]==start],current_accn)
                if f: return Measure(float(f["val"]),{"tag":tag,"unit":"USD","method":"direct_quarter","facts":[f]})
            if not fs: continue
            # A six/nine/twelve-month total needs a preceding cumulative total with identical fiscal start.
            for start in sorted({f["start"] for f in fs}):
                current=self.choose([f for f in fs if f["start"]==start],current_accn)
                if not current or days(start,end)<=130: continue
                previous=[]
                for candidate in self.by_tag_start.get((tag,start),[]):
                    if candidate["end"]>=end: continue
                    gap=(date.fromisoformat(end)-date.fromisoformat(candidate["end"])).days
                    if not 65<=gap<=130: continue
                    if candidate["filed"]<=asof: previous.append(candidate)
                if not previous: continue
                previous_end=max(f["end"] for f in previous)
                prior=self.choose([f for f in previous if f["end"]==previous_end],current_accn)
                if prior:
                    return Measure(float(current["val"])-float(prior["val"]),
                        {"tag":tag,"unit":"USD","method":"ytd_difference","quarter_start_after":previous_end,"facts":[current,prior]})
        return None

    def consolidated_cfo(self, end, asof, current_accn=None):
        total="NetCashProvidedByUsedInOperatingActivities"
        continuing="NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"
        discontinued=["CashProvidedByUsedInOperatingActivitiesDiscontinuedOperations",
                      "CashProvidedByUsedInDiscontinuedOperationsOperatingActivities"]
        direct=self.quarter([total],end,asof,current_accn)
        if direct: return direct
        def cumulative(period_end,start):
            f=self.choose([f for f in self.candidates(total,period_end,asof) if f.get("start")==start],current_accn)
            if f: return float(f["val"]),[f]
            c=self.choose([f for f in self.candidates(continuing,period_end,asof) if f.get("start")==start],current_accn)
            if not c: return None
            for tag in discontinued:
                d=self.choose([f for f in self.candidates(tag,period_end,asof) if f.get("start")==start],current_accn)
                if d: return float(c["val"])+float(d["val"]),[c,d]
            # Absence of a discontinued-operations tag is not evidence of zero.
            return None
        candidates=self.candidates(total,end,asof)+self.candidates(continuing,end,asof)
        starts=sorted({f["start"] for f in candidates if f.get("start") and 65<=days(f["start"],end)<=380},reverse=True)
        for start in starts:
            current=cumulative(end,start)
            if current is None: continue
            if days(start,end)<=130:
                return Measure(current[0],{"tag":"consolidated_cfo_reconciled","unit":"USD","method":"components_sum","facts":current[1]})
            ends=sorted({f["end"] for tag in [total,continuing] for f in self.by_tag_start.get((tag,start),[])
                         if f["filed"]<=asof and f["end"]<end and 65<=days(f["end"],end)-1<=130},reverse=True)
            for prior_end in ends:
                prior=cumulative(prior_end,start)
                if prior:
                    return Measure(current[0]-prior[0],{"tag":"consolidated_cfo_reconciled","unit":"USD",
                        "method":"ytd_difference_components","quarter_start_after":prior_end,"facts":current[1]+prior[1]})
        return None

    def snapshot(self, end, asof, accession=None):
        values,provenance={},{}
        for key,tags in TAGS.items():
            m=self.consolidated_cfo(end,asof,accession) if key=="cfo" else (self.quarter if key in {"revenue","capex","income"} else self.instant)(tags,end,asof,accession)
            values[key]=m.value if m else math.nan
            provenance[key]=m.provenance if m else {"missing":True,"aliases":tags}
        return values,provenance

def safe_ratio(a,b):
    return a/b if math.isfinite(a) and math.isfinite(b) and b>0 else math.nan

def feature_vector(v):
    return [safe_ratio(v["cfo"],v["revenue"]),safe_ratio(v["cash"],v["assets"]),
            safe_ratio(v["cash"],v["current_liabilities"]),safe_ratio(v["current_assets"],v["current_liabilities"]),
            safe_ratio(v["receivables"],v["revenue"]),safe_ratio(v["inventory"],v["revenue"]),
            safe_ratio(v["capex"],v["revenue"]),safe_ratio(v["income"],v["revenue"]),
            math.log1p(v["revenue"]) if v["revenue"]>0 else math.nan,
            math.log1p(v["assets"]) if v["assets"]>0 else math.nan]

