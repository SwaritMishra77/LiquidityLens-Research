# Development decisions

1. Start with public US-GAAP filings rather than an expensive transcript API. India transfer is a later data-comparability study, not an assumed free extension.
2. Use a continuous next-quarter CFO margin target and a separately named negative two-quarter CFO proxy. Do not substitute “default” or “bankruptcy” for those labels.
3. Guard publication time conservatively and require an economic lead. This distinguishes forecasting a still-running quarter from predicting a report after its economic period has ended.
4. Preserve unsupported periods and missingness in audits. Do not convert absent financial components to zero.
5. During source inspection, add explicit continuing/discontinued CFO reconciliation and support 17-week retail quarters. Preserve the original numeric development results.
6. Extend validation and calibration from one to two calendar years after noticing that publication purging removed late-year labels and weakened seasonal coverage. Final historical evaluation is 2025; the development history is disclosed.
7. Pin the financial pretraining checkpoint and freeze it. Keep TF-IDF as a comparator so transformer-specific value can be tested.
8. Require identical examples for disclosure ablations; expose the smaller sample and missing-text coverage instead of comparing unrelated scores.
9. Avoid calibrating probability scores when too few outcomes exist. Do not generate synthetic stress cases to make AUC computable.
10. Choose the demo model on validation MAE and retain all tested architectures, including those that lose. Selection on test scores would overstate performance.
11. Keep the demo historical and source-backed. New-filing inference, live updates and decision impact require separate acceptance tests.
12. Independently parse a supported-format original-filing sample. Traditional pre-inline filings require separate XBRL instance support and are reported as unsupported, not quietly counted as matched.

