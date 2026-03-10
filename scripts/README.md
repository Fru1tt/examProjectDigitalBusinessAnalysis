# Scripts

Pipeline entrypoints:

- `prepare_data.py`: load, validate, clean, and feature-engineer raw data into `data/processed/`.
- `analyze.py`: train/evaluate models and write metrics, predictions, and model artifacts.
- `make_outputs.py`: generate business-facing figures/tables for reporting and presentation.

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
- `outputs/tables/hybrid_threshold_tuning_logistic_regression_hybrid_tuned.csv`
- `outputs/tables/confusion_matrix_<model>.csv`
- `outputs/tables/feature_importance_<best_model>.csv`
- `outputs/tables/test_predictions_best_model.csv`
- `outputs/tables/test_predictions_with_probabilities.csv`
- `outputs/figures/confusion_matrix_<model>.png`
- `outputs/figures/feature_importance_<best_model>.png`
- `outputs/figures/hybrid_threshold_tuning_logistic_regression_hybrid_tuned.png`

`make_outputs.py` outputs:

- `outputs/tables/executive_kpi_summary.csv`
- `outputs/tables/executive_model_comparison.csv`
- `outputs/tables/executive_channel_distribution.csv`
- `outputs/tables/executive_predicted_channel_mix.csv`
- `outputs/tables/executive_segment_recommendations.csv`
- `outputs/tables/executive_top_drivers.csv`
- `outputs/tables/executive_outputs_manifest.json`
- `outputs/figures/executive_model_comparison.png`
- `outputs/figures/executive_channel_distribution.png`
- `outputs/figures/executive_predicted_channel_mix.png`
- `outputs/figures/executive_segment_online_heatmap.png`
- `outputs/figures/executive_top_drivers.png`

Optional input override:

```bash
python scripts/prepare_data.py --input "data/raw/online vs store shopping dataset.csv"
```
