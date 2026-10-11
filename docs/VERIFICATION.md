# Verification record for the source release

This release uses the original 100-company `LiquidityLens_v0_2.zip` as its
historical evidence and data snapshot. The earlier claim that this archive
contained no source was incorrect: it contains `src/liquiditylens/` (16 Python
modules), four original test modules, four scripts, source caches, model
artifacts, detailed reports and the demo. This repository makes the source
visible at the top level and improves its packaging; it does not invent a new
set of financial outcomes.

The new `configs/universe.csv` pins all 100 CIKs to the identities in the
verified source cache. This removes dependence on a mutable ticker lookup for
future fresh ingestion. The original config is preserved inside the snapshot.

## Checks performed

| Check | Observed result |
|---|---|
| Archive integrity | 2,540 listed SHA-256 checksums matched the archived file bytes. |
| Dataset manifest | `samples.csv`, `sequences.npz`, `provenance.jsonl`, `feature_schema.json` matched the packaged hashes. |
| Encoder manifest | Cached `passages.jsonl` and `text_embeddings.npz` matched the pinned manifest hashes. This verifies the files, not the pretrained model's original training data. |
| Cached financial rebuild | Rebuilding from the 100 issuers' cached SEC facts produced byte-identical copies of all four processed financial files, including 3,412 rows and approximately 95 MB of feature/label provenance. |
| Publication ledger | `liquiditylens audit` completed without violations; the archived report lists 294,662 checked feature facts and zero unknown accessions. |
| Numeric comparisons | Eight classical model/group rows from the original 26-row report were rerun: MAE, coverage and PR-AUC differences were exactly zero at stored CSV precision. |
| Matched-text comparisons | Eighteen classical model/group rows from the original 48-row report were rerun on the same cohort: MAE, coverage and PR-AUC differences were exactly zero at stored CSV precision. |
| Annual rolling baselines | All 24 model/group/year rows across 2023, 2024 and 2025 were rerun; MAE and interval coverage matched the archived CSV exactly at stored precision. |
| Fixture tests | Ten accounting reconstruction tests, three importer tests, and one pinned-manifest test passed; the importer tests cover successful import, tampered data and unsafe ZIP paths. |

The classical rerun included the seasonal and persistence forecasts, Ridge,
gradient boosting, TF-IDF, frozen embedding and text-change linear comparisons.
The text experiment reused the packaged frozen embeddings; it did not fetch
FinBERT weights or generate new embeddings.

## Limits of these checks

- PyTorch, transformers, Beautiful Soup and pytest were unavailable in the
  current execution environment and package downloads were blocked. Neural
  training, fresh text extraction, source HTML reconciliation, the original
  pytest suite, and the end-to-end report generator were not rerun here. The
  repository CI installs the core and test dependencies and runs the fixture
  suite; a full local run needs the documented deep extras.
- The original report's 30-period inline-XBRL reconciliation is a cached
  experimental result, not a human verification of every SEC fact or company.
- 2025 results were inspected during earlier development. They are useful for
  investigating this pilot, but should never be called an untouched test.
- The selected universe is not a representative cross-section of distressed
  issuers. Only one stress case occurred in the unseen-company 2025 set, so
  classification ranking statistics there are unstable.

## Next research gates

1. Run the full dependency-locked pipeline including three-seed neural models
   on a compatible machine, then compare the generated hashes and tables.
2. Have a person independently review cash-flow calculations and disclosure
   relevance on a blinded sample of original filings.
3. Freeze a protocol before obtaining additional future periods or an
   independently selected issuer cohort; report attrition and negative results.

