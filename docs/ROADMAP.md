# Acceptance gates and growth roadmap

The code implements the pilot architecture. The following gates determine when stronger research, business-impact and CV claims become justified. These are remaining work, not completed achievements.

| Gate | Work | Acceptance evidence |
|---|---|---|
| Accounting validity | Human-review at least 30 random periods, including annual differences, retail calendars, negative CFO and reconciled components | Original table, calculations, reviewer notes and resolutions; source extraction checks alone are insufficient |
| Context completeness | Handle custom tags and original standalone XBRL instances; distinguish unsupported formats from failed reconciliations | Documented alias/context mappings and independently reviewed examples |
| Historical population | Build issuer candidates from dated filing indexes; document industry exclusions, ceased filers, mergers and future-report attrition | Reproducible universe construction and cohort-flow report |
| Disclosure extraction | Double-review a passage set for liquidity relevance and align comparable sections across years | Annotator agreement, relevance precision/recall and an extraction error taxonomy |
| Incremental model value | Compare text and advanced models on identical examples; examine data size and sector-specific gains | Repeated temporal results, compute costs and useful gains beyond strong baselines |
| Uncertainty quality | Check intervals and calibration by season, sector and unfamiliar firms | Coverage/width/reliability results with sufficient outcome counts |
| Fresh evaluation | Freeze models, data rules and metrics before inspecting a new period | Timestamped protocol and a genuinely new evaluation; no retrospective model selection |
| Analyst usefulness | Ask consenting finance/analytics reviewers to complete defined review tasks | Time, quality and error measurements versus a stated baseline |
| Reusable delivery | Add safe new-filing inference, refresh validation and monitoring | Reproducible daily/quarterly refresh, clear abstention rules and versioned outputs |

## Useful extensions after the core gates

The offline `scripts/build_historical_universe.py` supplies deterministic dated-index candidates and a blank review manifest. It does not fetch extra financial facts, resolve issuer identities, fill sectors or turn candidates into valid examples. Those steps, plus documented exclusions, remain part of the historical-population gate.

- A sector-specific seasonality model or residual neural model that learns corrections to a strong seasonal forecast. Register this as a new experiment; do not silently replace the reported architecture.
- Additional liquidity targets such as cash-buffer deterioration, if accounting definitions and outcome availability can be defended.
- A small Indian-company transfer study, only after verifying accessible historical filings, cash-flow reporting frequency, Ind AS comparability and rights to use data.
- Historical economic covariates with original release dates and vintage values. Revised macro data can leak just as revised financial facts can.
- Robustness to missing text, publisher changes, filing format changes and a distribution shift in firms or industries.
- A separately evaluated passage-review assistant with evidence citations, measured extraction quality and explicit abstention. A generative summary should not substitute for a forecasting model or reference labels.

## Scope discipline

Do not add news, earnings-call transcripts, supplier graphs, reinforcement learning or large generative-model tuning before their data and evaluation paths are credible. Do not generate synthetic cash-flow labels to inflate sample size. Simulation can later assess an analyst-workflow policy, with assumptions and simulated outcomes clearly identified.

## Keeping it relevant through third year

Release a tested research snapshot, then add one substantive increment at a time: accounting coverage, a reviewed text set, a new temporal result or external workflow evidence. Preserve earlier results and write a clear change log. This makes the project a sustained body of work instead of a static dashboard.

