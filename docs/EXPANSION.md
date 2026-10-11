# Expansion record: 50 to 100 issuers

This release extends the 50-company pilot to 100 manually screened US SEC filers
(50 retail/consumer commerce and 50 manufacturing/industrial), with the same
2015–2025 prediction window, target definitions and model families. The first
50 tickers remain in `configs/universe_v0_1.csv`; the complete selected list
is `configs/universe.csv`. Every selected additional issuer has cached SEC
submissions and Company Facts, with a request timestamp and payload hash in
`data/raw/`. There are **3,412 usable company-quarter rows, including 1,599
from the added 50 issuers**; all 100 issuers supply at least one row.

Selection was manual and sector-focused. Three initially considered additions
were replaced *before financial outcome construction* after SEC SIC metadata
showed apparel manufacturing or a marketplace outside the intended retail
screen. The final three replacements were HVT, FND and PETS. Current SIC is a
sector sanity check, not a historical membership signal. Business models and
accounting perimeter remain heterogeneous, and only US-GAAP USD standard
facts are supported. All exclusions and failed/partial periods are recorded in
`reports/data_audit.json`. This expansion does **not** cure selection bias or
produce a representative set of troubled issuers.

The old 2025 test was inspected before this expansion. Changing the company
list also changes the SHA-based 20% company holdout and training set; an
apparent change in MAE cannot be attributed to more data alone. The original
metrics and manifest are retained in `reports/pilot_v0_1/`; models and raw
sources remain in the previous release archive. Results in this release are
exploratory, with model selection on validation MAE and no post hoc choice of
the best test-scoring model.

The text cohort was expanded from 12 to 24 selected issuers. Only examples
with both the current and a comparable prior-year form enter the matched-text
test. The total financial cohort and this conditional text cohort must be
reported separately. A fixed-budget review backtest counts cases flagged on
the same test rows; it makes no claim about financial savings or analyst time.

An offline EDGAR master-index selector is supplied in
`scripts/build_historical_universe.py`. It uses only filing activity in
2013–2014 and records each source hash, deterministic CIK selection and blank
identity/sector review fields. Dated issuer candidates still need accounting
screening, then a full build/retrain/attrition report before inclusion in any
benchmark. A truly new, untouched period and human accounting review remain
open research gates.
