# Research protocol

## Decision problem

An analyst has limited time to review company cash-flow deterioration. The system should estimate future operating cash-flow margin, quantify empirical forecast uncertainty, identify a predefined two-quarter cash-flow pattern and supply auditable evidence for review. It supports investigation; it does not autonomously approve lending, issue ratings or establish fraud.

The useful business outputs are a review queue, a forecast range, source passages and clear knowledge of what is missing. Any future claim of analyst time saved needs a measured user study against a realistic baseline.

## Research questions

1. Can historical accounting variables improve on same-quarter-last-year and persistence forecasts?
2. Does a compact neural history model improve on regularized linear models and strong trees?
3. Does disclosure text add information beyond those financial histories on an identical sample?
4. Do prior-year disclosure changes add information beyond the current text representation?
5. Do forecast rankings, interval coverage and model gains remain stable across time and entirely new companies?
6. Can evidence and model disagreement help an analyst investigate cases without presenting associations as causal explanations?

These are hypotheses. Negative results are retained. Multiple comparisons and a purposive pilot prevent interpreting isolated wins as strong general evidence.

## Population and source limitations

The initial universe was 25 retailer and 25 manufacturing/distribution tickers. The expanded sample includes 50 in each manually assigned sector, including explicit legacy CIK mappings for selected ceased/renamed filers. This remains a feasibility sample, with retailer boundary cases (e.g. pharmacy and online pet commerce) and industrial financing arms. It is not a historical index constituency or a representative distressed-company sample. The original universe manifest and earlier results are retained.

Current ticker and SIC information is used for descriptive metadata, not predictive features or historical membership. The manually assigned sector categories include boundary cases and financing subsidiaries; the manifest flags those decisions.

The current Company Facts archive includes standard non-custom US-GAAP USD facts. It omits many custom tags and cannot fully substitute for original XBRL instance contexts. Historical facts are selected using filing dates and source accessions, never by the latest frame. This limits observable restatement leakage but does not prove that all API extraction was available contemporaneously.

## Observation, timing and targets

- One observation is an original 10-K/10-Q reporting period, anchored at the earliest eligible original filing for that period.
- Availability is the later of supplied acceptance time and filing date plus two days in UTC. This sacrifices speed to avoid timezone/date ambiguity; no intraday claims are made.
- Prediction dates are 2015–2025. Labels are observed only if published before the fixed 2026-06-30 data cutoff.
- The primary economic quarter must end at least 14 days after the prediction timestamp. Rows with ended/too-short target quarters are rejected.
- Primary target: next standalone quarter's consolidated operating cash flow divided by revenue.
- Secondary target: negative sum of consolidated CFO over the next two standalone quarters.
- Both targets must be observed, so requiring the auxiliary label also selects against companies with missing future reports. This informative attrition is a limitation, not an assumption of random missingness.

Quarter gaps of 65–130 days support ordinary quarters and 16/17-week retail fourth quarters. Flows longer than a quarter require a matching-fiscal-start cumulative difference. Unsupported calendar changes are rejected.

## Accounting rules

Direct standalone flow facts are preferred. Otherwise Q2/Q3/Q4 are cumulative current total minus an eligible preceding cumulative total with the same fiscal start and definition. A consolidated CFO missing under the total tag is reconstructed only when continuing and discontinued components are both explicitly disclosed; an absent discontinued tag is never treated as zero.

Fact conflicts within the same period/source are rejected instead of averaged. Units must be USD. Revenue must be positive for margin labels. Current revenue and CFO, next revenue and CFO and second-quarter revenue and CFO are required. Other features retain missingness.

Separate disclosure versions can appear in the subtraction. Their filing dates are recorded, but an annual restatement may change prior-quarter values in ways the API does not expose. Human reconciliation and original instance-context handling remain important next stages.

Eight-quarter input histories are reconstructed as of each prediction filing date. All feature facts must have been filed by that date. First-published target values are used at the future original report, rather than latest restated target values.

## Features

The ten quarter-level features are CFO/revenue, cash/assets, cash/current liabilities, current ratio, receivables/revenue, inventory/revenue, capex/revenue, net income/revenue, log revenue and log assets. Fiscal flows and stock measures have different meanings; features are signals, not canonical credit ratios.

Training-only median imputation, scaling and explicit missing indicators prevent fitting preprocessing to held-out companies. A fixed ±8 standardized clipping bound stabilizes extreme inputs; targets remain untrimmed and errors on extreme outcomes are reported. Industry metadata contributes a single manually assigned sector indicator to classical flat-feature models; neural models use the financial history.

## Disclosure processing and pretrained provenance

The pipeline removes executable markup, excludes numerically dense tables and retrieves up to four 150-word passages using explicit liquidity/working-capital terms. This is passage retrieval, not a certified MD&A section extractor. It can miss relevant language, select repeated boilerplate or truncate important context.

The comparator is the nearest same-form prior-year filing within 300–430 days. Future text cannot be used. Lexical change is Jaccard distance over word sets, not a validated measure of sentiment or risk.

The financial encoder is `yiyanghkust/finbert-pretrain`, pinned to `88ab954a39ea6d3ce2b62cff086dd5ad1172c664`, a public 2022 repository revision. It is frozen; final hidden states are pooled with an attention-mask mean and then an equal passage mean. Inputs truncate at 256 tokens. The checkpoint and pooling are recorded; corpus overlap has not been independently audited.

PCA is fit on training embeddings only. Current representation, prior-year embedding difference and lexical scalars form the change ablation. TF-IDF vocabulary and dimensionality reduction also fit training rows only. Matched-text experiments require both current and comparable prior-year text, reducing coverage and selecting mature filers.

## Model controls

Classical baselines: same-quarter-last-year, last-quarter persistence, Ridge/logistic regression and histogram gradient boosting. Tree structure and linear regularization use a small predeclared validation grid. Tree quantiles use 0.1/0.9 bounds, reordered if crossing occurs.

Deep comparators: two-hidden-stage MLP, one-layer GRU and a two-layer temporal transformer. Neural models predict margin and auxiliary stress jointly using scaled Smooth-L1 plus 0.3 times binary cross-entropy. AdamW, clipping, deterministic CPU training, three fixed seeds and chronological validation early stopping are recorded. Neural text fusion compares a GRU with current FinBERT and with FinBERT plus change features.

Encoder attention is not used as a causal explanation. Retrieved passages are evidence of what was disclosed, not proof that a passage caused the prediction.

## Evaluation

Main boundaries: train through 2020, validate over 2021–2022, calibrate over 2023–2024, evaluate over 2025. Future labels must be published before each development boundary. Therefore the usable end of validation/calibration is shorter than its calendar window; two-year windows help retain seasonal coverage.

Twenty complete companies are assigned by a stable SHA-256 ordering, before preprocessing. They appear only in the unseen-company evaluation. Numeric-text comparisons reuse that same holdout set and exact sample IDs. Numerical and matched-text cohort scores are not interchangeable.

Three annual rolling-origin baseline folds evaluate 2023, 2024 and 2025, each with preceding development/calibration windows. Overlapping training histories mean these folds are sensitivity analyses, not independent experiments.

Point metrics: MAE, RMSE and equal-company macro MAE. Classification metrics: PR-AUC, ROC-AUC, Brier, log loss and aggregate top-decile precision. Aggregate top-decile precision is descriptive, not a measured prospective operating policy. Single-class ranking metrics are left undefined.

Probability calibration uses a separate Platt fit only when calibration has at least 20 rows and five examples of each class. Otherwise probabilities are left uncalibrated with an explicit reason. Classification is sparse, and model scores should not be interpreted as reliable credit probabilities.

Intervals use finite-sample residual order statistics or nonshrinking conformalized tree quantile intervals. Nominal 80% coverage is evaluated empirically. Serial dependence, cross-company shocks and drift weaken standard exchangeability; no distribution-free coverage guarantee is claimed.

Paired bootstraps resample company-level mean errors, averaging neural seed predictions. They do not capture common time-shock uncertainty or correct for multiple model comparisons. Reporting includes raw seed metrics, split memberships, subgroup errors, financing/perimeter sensitivity and source-linked failure cases.

## Integrity and development history

This is exploratory development. An initial numeric run was inspected, then accounting reconciliation and validation/calibration coverage were improved. The 50-company 2025 test was inspected before the 100-company expansion and a matched-text coverage increase. The earlier run is preserved. No reported 2025 test is an untouched external confirmatory sample. A new protocol must be frozen before examining additional future outcomes.

No deployment savings, fraud detections, company defaults or causal intervention effects are generated. Synthetic fixture numbers are confined to unit tests. The benchmark financial examples and labels use SEC data.

