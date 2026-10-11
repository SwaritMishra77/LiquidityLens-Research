# LiquidityLens — research pilot results

Built from **3,412 company–quarter examples across 100 selected companies**. All financial labels are derived from actual SEC disclosures; no synthetic financial outcomes are used.

This is a working exploratory research version. It is not a validated credit-rating system, a bankruptcy predictor, or a live service. The earlier 50-company development run and its 2025 test were inspected before expanding to this 100-company sample; the new larger test is also exploratory, not untouched confirmatory evidence.

## Dataset flow

The selected universe has 100 issuers; 100 contributed usable financial rows. There are 835 cached filing passages from 24 text issuers, of which 729 have a comparable prior-year disclosure. Eligibility rules and individual rejection reasons are in data_audit.json.

| Financial split | Examples |
|---|---:|
| train | 1042 |
| validation | 460 |
| calibration | 466 |
| test_seen | 309 |
| test_unseen | 77 |

## Prediction and evaluation

Primary target: next standalone-quarter operating cash flow / revenue. Secondary proxy: negative combined operating cash flow over the next two quarters. Forecasts require at least 14 days before the primary economic quarter ends. Publication availability is conservatively guarded by filing date plus two days.

Training labels known by 2020-12-31; validation in 2021–2022; calibration in 2023–2024; historical evaluation in 2025. Labels crossing each development boundary are purged. 20 whole companies are excluded from training, validation, calibration and preprocessing.

## Main numeric results

MAE is shown in percentage points of cash-flow margin. Neural values average three seeds; a seed standard deviation describes training variability, not a confidence interval.

| Model | Company group | Examples | Stress cases | MAE (pp) | Nominal 80% interval coverage | Stress PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| boosted_numeric | test_seen | 309 | 16 | 4.62 | 80.9% | 0.462 |
| boosted_numeric | test_unseen | 77 | 1 | 4.29 | 81.8% | 1.000 |
| gru_numeric | test_seen | 309 | 16 | 6.87 | 80.8% | 0.092 |
| gru_numeric | test_unseen | 77 | 1 | 6.05 | 85.7% | 0.036 |
| linear_numeric | test_seen | 309 | 16 | 4.96 | 80.6% | 0.337 |
| linear_numeric | test_unseen | 77 | 1 | 4.07 | 88.3% | 1.000 |
| mlp_numeric | test_seen | 309 | 16 | 5.51 | 82.1% | 0.111 |
| mlp_numeric | test_unseen | 77 | 1 | 4.84 | 82.3% | 0.510 |
| persistence | test_seen | 309 | 16 | 10.60 | 77.3% | 0.052 |
| persistence | test_unseen | 77 | 1 | 9.17 | 83.1% | 0.013 |
| seasonal | test_seen | 309 | 16 | 4.95 | 84.8% | 0.052 |
| seasonal | test_unseen | 77 | 1 | 5.07 | 80.5% | 0.013 |
| transformer_numeric | test_seen | 309 | 16 | 6.38 | 79.4% | 0.151 |
| transformer_numeric | test_unseen | 77 | 1 | 5.93 | 82.7% | 0.704 |

The demo uses **mlp_numeric (seed 73)**, chosen by validation MAE. The best test result is not used to choose a deployment model.

## Text contribution

Text experiments use only rows with both current and same-form prior-year disclosures. Numeric, TF-IDF, frozen FinBERT and change-feature comparators share those exact rows, company exclusions and time boundaries. These results must not be compared directly with the larger numeric cohort.

| Model | Company group | Examples | MAE (pp) | Interval coverage |
|---|---|---:|---:|---:|
| boosted_numeric | test_seen | 68 | 5.73 | 77.9% |
| boosted_numeric | test_unseen | 28 | 5.96 | 71.4% |
| gru_finbert | test_seen | 68 | 7.47 | 82.8% |
| gru_finbert | test_unseen | 28 | 7.48 | 84.5% |
| gru_finbert_change | test_seen | 68 | 7.73 | 83.3% |
| gru_finbert_change | test_unseen | 28 | 8.19 | 76.2% |
| gru_numeric | test_seen | 68 | 7.33 | 81.4% |
| gru_numeric | test_unseen | 28 | 7.87 | 83.3% |
| linear_finbert | test_seen | 68 | 7.87 | 82.4% |
| linear_finbert | test_unseen | 28 | 7.57 | 85.7% |
| linear_numeric | test_seen | 68 | 5.90 | 79.4% |
| linear_numeric | test_unseen | 28 | 4.00 | 89.3% |
| linear_numeric_finbert | test_seen | 68 | 5.81 | 77.9% |
| linear_numeric_finbert | test_unseen | 28 | 4.34 | 92.9% |
| linear_numeric_finbert_change | test_seen | 68 | 5.75 | 80.9% |
| linear_numeric_finbert_change | test_unseen | 28 | 4.77 | 92.9% |
| linear_numeric_tfidf | test_seen | 68 | 5.90 | 79.4% |
| linear_numeric_tfidf | test_unseen | 28 | 4.78 | 92.9% |
| linear_tfidf | test_seen | 68 | 7.86 | 80.9% |
| linear_tfidf | test_unseen | 28 | 7.96 | 78.6% |
| mlp_numeric | test_seen | 68 | 6.46 | 81.4% |
| mlp_numeric | test_unseen | 28 | 6.48 | 83.3% |
| persistence | test_seen | 68 | 10.77 | 79.4% |
| persistence | test_unseen | 28 | 11.11 | 78.6% |
| seasonal | test_seen | 68 | 5.16 | 86.8% |
| seasonal | test_unseen | 28 | 6.13 | 60.7% |
| transformer_numeric | test_seen | 68 | 7.62 | 79.9% |
| transformer_numeric | test_unseen | 28 | 6.93 | 88.1% |

## Historical review queue (descriptive)

A fixed budget of five reviews per calendar-quarter arrival wave is ranked by lowest predicted next-quarter cash-flow margin. The validation-selected model is compared on the same cases with a prior-year seasonal forecast and a gradient-boosted forecast. This is a backtest of cases surfaced, not measured savings or a live intervention.

| Company group | Ranking | Reviews | Two-quarter negative-CFO cases found / available | Next-quarter negative-margin cases found / available |
|---|---|---:|---:|---:|
| test_seen | seasonal | 20 | 4 / 16 | 6 / 34 |
| test_seen | boosted_numeric | 20 | 4 / 16 | 8 / 34 |
| test_seen | mlp_numeric | 20 | 5 / 16 | 9 / 34 |
| test_unseen | seasonal | 20 | 1 / 1 | 5 / 5 |
| test_unseen | boosted_numeric | 20 | 1 / 1 | 4 / 5 |
| test_unseen | mlp_numeric | 20 | 1 / 1 | 5 / 5 |

Cohorts are small, review waves are simplified calendar quarters, and test outcomes were already inspected during exploratory development. See decision_backtest.json for selected sample IDs and wave counts.

## Source validation

The publication audit checked 294,662 feature facts against filing dates and their source-accession publication ledger. It found zero violations and no unknown source accessions.

An independent inline-XBRL parser matched 177 primitive financial facts and target arithmetic across 30 of 30 sampled supported-format periods from 19 companies. The selection required fully cached sources and supported inline format. 4 traditional/unsupported-format candidates were retained as exclusions. This is an extraction/arithmetic check, not a human accounting review.

## What the results support

The pipeline establishes a serious reproducible foundation: source caching, financial reconciliation, tracked provenance, strong baselines, actual neural/transformer training, frozen financial-language embeddings, matched-cohort ablations, empirical uncertainty checks and a historical analyst demo. Model complexity alone is not evidence of improvement. Read paired_comparisons.json for company-cluster uncertainty around model differences.

## Limits that change interpretation

- The purposive 100-company universe remains selection-biased. Current SIC/ticker metadata is descriptive, not a point-in-time universe reconstruction.
- Missing custom tags, non-USD reporting and irregular periods reduce coverage. Missing optional financial features are imputed using training data only.
- Company Facts is a current archive filtered by original filing metadata, not an archive collected contemporaneously. Independent filing-level reconciliation remains necessary.
- CFO subtraction can combine then-known values from different filing versions. Restatement changes not exposed in standard tags can still cause inconsistent totals.
- Some manufacturing companies have financing operations; corporate spin-offs can change comparability. sensitivity.json reports errors after excluding flagged firms.
- Frozen FinBERT checkpoint and release provenance are pinned; exact corpus overlap is not independently audited. Text retrieval is keyword-based and has not been certified as full MD&A extraction.
- Negative two-quarter CFO is a cash-flow pattern, not proof of financial distress. The 2025 unseen-company group has 1 positive case. Ranking AUC is undefined with zero events and very unstable with one event.
- Serial dependence and common economic shocks weaken standard conformal exchangeability. Nominal coverage is a target, not a guarantee. The company bootstrap does not resample common time shocks.
- The matched-text 2025 seen-company evaluation has 1 positive stress-proxy case; sparse-event ranking cannot establish text's value. Continuous-margin forecasts remain evaluable.
- There are few stress events, multiple model comparisons and one purposive pilot. Results are exploratory; no financial savings or deployment impact is claimed.

## Next acceptance gates

1. Independently reconcile at least 30 randomly selected periods against filing cash-flow tables; review ambiguous contexts and missing tags.
2. Use the offline historical SEC index selector to draw a larger issuer candidate list, then review identities/industries, ingest actual facts and disclose attrition before reporting any expanded results.
3. Annotate a double-reviewed passage set for liquidity relevance and change alignment; measure precision and recall of extraction.
4. Freeze a new protocol before inspecting additional future periods; report genuine out-of-sample results and drift.
5. Keep transformers only where repeatable gains or a distinct useful capability justify their cost.

## CV wording for the current version

> Built LiquidityLens, a reproducible SEC-based cash-flow forecasting research pipeline covering 100 selected companies and 3,412 company–quarter examples; compared seasonal, linear, gradient-boosted, neural and transformer models using chronological label purging and whole-company holdouts.

> Implemented filing provenance, cumulative-to-quarter cash-flow reconciliation, frozen FinBERT disclosure features, matched-cohort ablations, uncertainty evaluation and an interactive historical review demo.

Do not claim reliable default prediction, transformer superiority, production deployment or realised financial savings from this pilot.

See START_HERE.md for operating instructions, docs/RESEARCH_PROTOCOL.md for design and docs/INTERVIEW_GUIDE.md for explanation.
