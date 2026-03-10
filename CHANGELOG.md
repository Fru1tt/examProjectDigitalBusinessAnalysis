# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and semantic-style sections.

## [Unreleased]

### Added
- Initialized project structure for exam delivery:
  - `data/raw`, `data/processed`
  - `notebooks`
  - `scripts`
  - `outputs/figures`, `outputs/tables`
  - `docs`
- Added project setup files:
  - `.gitignore`
  - `README.md`
  - `requirements.txt`
- Added documentation templates and filled project-specific planning docs.
- Added raw dataset in `data/raw/online vs store shopping dataset.csv`.
- Added data preparation pipeline in `scripts/prepare_data.py`.
- Added modeling/evaluation pipeline in `scripts/analyze.py` with:
  - baseline model (`dummy_most_frequent`)
  - multinomial logistic regression
  - random forest classifier
  - stratified train/test split and class-level metrics
- Added automated output exports for modeling:
  - model metrics table
  - per-class metrics table
  - confusion matrices (CSV + PNG)
  - best-model prediction exports
  - best-model feature-importance export
- Added generated processed outputs:
  - `data/processed/shopping_behavior_clean.csv`
  - `data/processed/prepare_data_report.json`

### Changed
- Corrected initial folder nesting so structure is rooted directly at `examProject`.
- Updated proposal docs to reflect that the current CSV contains 25 columns.

## [2026-03-10]

### Added
- Repository initialized and changelog tracking started.
