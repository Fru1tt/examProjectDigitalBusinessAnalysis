# Scripts

Pipeline entrypoints:

- `prepare_data.py`: load, validate, clean, and feature-engineer raw data into `data/processed/`.
- `analyze.py`: run core analysis and write intermediate/final tables.
- `make_outputs.py`: generate final figures/tables for reporting.

Run from project root:

```bash
python scripts/prepare_data.py
python scripts/analyze.py
python scripts/make_outputs.py
```

`prepare_data.py` outputs:

- `data/processed/shopping_behavior_clean.csv`
- `data/processed/prepare_data_report.json`

Optional input override:

```bash
python scripts/prepare_data.py --input "data/raw/online vs store shopping dataset.csv"
```
