# Data snapshot

The SEC source cache and derived data are deliberately separate from Git. The
original `LiquidityLens_v0_2.zip` includes `data/raw`, `data/processed`, the
reported models and their run details. Put the ZIP beside this checkout and run
`python scripts/import_snapshot.py ../LiquidityLens_v0_2.zip`.

The importer verifies all 2,540 listed archive checksums, rejects unsafe ZIP
paths and symlinks, then extracts only `data/`. It never loads a model pickle.
The raw data total about 156 MB and the processed data about 104 MB.

The original 100-company universe is a selected research cohort, not an
investable historical universe. The frozen encoder embeddings are supplied in
the snapshot. Model weights for re-encoding are downloaded separately from
the pinned public checkpoint in `configs/pilot.yaml`.
