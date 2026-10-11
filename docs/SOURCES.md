# Primary sources and research references

Retrieved/verified during the 2026-10-10 work session; raw SEC request metadata retains actual retrieval times. API responses may contain later dated records than the configured study cutoff; those records are filtered out.

| Source | Use |
|---|---|
| https://www.sec.gov/search-filings/edgar-application-programming-interfaces | Official submissions and Company Facts endpoints, standard-tag coverage and bulk archives |
| https://www.sec.gov/about/developer-resources | SEC fair-access guidance; ingestion uses a shared two-request/second limiter |
| https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data | Historical quarterly full indexes and index format for the offline universe candidate selector |
| https://www.sec.gov/files/company_tickers.json | Current ticker-to-CIK lookup, supplemented by explicit selected legacy CIKs |
| https://data.sec.gov/submissions/ | Original filing metadata and historical submission shards |
| https://data.sec.gov/api/xbrl/companyfacts/ | USD US-GAAP financial facts, filtered by source filing date |
| https://www.sec.gov/Archives/edgar/data/ | Original filing HTML and inline-XBRL source checks |
| https://arxiv.org/abs/2006.08097 | Yang, Uy and Huang, FinBERT: a financial-language pretraining reference |
| https://huggingface.co/yiyanghkust/finbert-pretrain/tree/88ab954a39ea6d3ce2b62cff086dd5ad1172c664 | Exact frozen financial encoder revision used by this project |
| https://github.com/yya518/FinBERT | Publisher research code/reference |
| https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html | Official histogram gradient boosting and quantile regression documentation |
| https://pytorch.org/get-started/previous-versions/ | Official CPU PyTorch installation |

Model release provenance does not establish that its entire pretraining corpus has been independently audited. No external architecture's published benchmark scores are presented as this project's results.

