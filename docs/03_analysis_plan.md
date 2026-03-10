# 03 Analysis Plan

## Research/Business Questions

1. Which customer characteristics best explain preference for online, in-store, or hybrid shopping?
2. How accurately can we predict channel preference at individual customer level?
3. How can model outputs support channel investment and marketing allocation decisions?

## Methods

- Descriptive analytics:
  summary statistics, class distribution, segment profiling, and pairwise relationship exploration.
- Diagnostic analytics:
  feature importance and class-level comparison to identify main preference drivers.
- Predictive/prescriptive analytics (if applicable):
  multi-class classification models (baseline + stronger model), followed by recommendation logic for segment-channel focus.

## Assumptions

- Assumption 1:
  Simulated behavioral variables are sufficiently realistic for demonstrating decision-support methodology.
- Assumption 2:
  Target labels (shopping preference) are internally consistent with predictor variables.

## Evaluation Strategy

- Metrics:
  macro F1, weighted F1, accuracy, per-class precision/recall, confusion matrix.
- Baseline/comparison:
  majority class baseline and simple interpretable model (e.g., multinomial logistic regression) compared against a non-linear model (e.g., random forest or gradient boosting).
- Robustness checks:
  stratified cross-validation, sensitivity to feature subsets, and basic error analysis by segment.

## Visualization Plan

- Executive-level charts:
  channel preference distribution, top feature drivers, segment-level prediction summary.
- Operational-level charts:
  confusion matrix, partial dependence/SHAP-style explanation chart (if implemented), and key variable distributions by class.
- Table outputs required:
  model comparison table, KPI table, top-driver ranking table, segment recommendation table.
