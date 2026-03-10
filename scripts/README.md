# Scripts

Pipeline entrypoints:

- `prepare_data.py`: load, validate, clean, and feature-engineer raw data into `data/processed/`.
- `analyze.py`: train/evaluate models and write metrics, predictions, and model artifacts.
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

`analyze.py` outputs:

- `outputs/tables/model_metrics.csv`
- `outputs/tables/model_class_metrics.csv`
- `outputs/tables/modeling_summary.json`
- `outputs/tables/confusion_matrix_<model>.csv`
- `outputs/tables/feature_importance_<best_model>.csv`
- `outputs/tables/test_predictions_best_model.csv`
- `outputs/tables/test_predictions_with_probabilities.csv`
- `outputs/figures/confusion_matrix_<model>.png`
- `outputs/figures/feature_importance_<best_model>.png`

Optional input override:

```bash
python scripts/prepare_data.py --input "data/raw/online vs store shopping dataset.csv"
```
