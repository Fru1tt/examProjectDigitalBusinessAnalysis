# 05 Results and Discussion Draft (Simple Report Language)

This text is written so it can be used directly in the report, with limited technical language.

## 4. Results

### 4.1 What was done

The goal of the analysis was to predict each customer's preferred shopping channel:
`online`, `store`, or `hybrid`.

To do this, we tested four models:

1. `dummy_most_frequent` (a simple baseline)
2. `logistic_regression`
3. `logistic_regression_hybrid_tuned` (threshold-tuned for better hybrid detection)
4. `random_forest`

The baseline model is important because it shows the minimum level of performance. If a more advanced model cannot beat this, the model is not useful in practice.

All models were trained on one part of the data and tested on unseen data. This gives a more realistic estimate of how the model would perform in real use.

### 4.2 Model performance

From [model_metrics.csv](/Users/carlgrude/examProject/outputs/tables/model_metrics.csv):

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression (Hybrid Tuned) | 0.9818 | 0.9138 | 0.9834 |
| Logistic Regression | 0.9614 | 0.8565 | 0.9684 |
| Random Forest | 0.9550 | 0.6206 | 0.9396 |
| Dummy Baseline | 0.8690 | 0.3100 | 0.8080 |

Logistic Regression with hybrid tuning gave the best overall result.

### 4.3 What these numbers mean in plain words

- The model is clearly learning real patterns in customer behavior, because it performs much better than the simple baseline.
- Accuracy is high for all non-baseline models, but accuracy alone is not enough here.
- Most customers in the dataset belong to the `store` class. Because of this imbalance, Macro F1 is a better fairness check across all classes.
- The tuned Logistic Regression had the best balance between performance and stability across classes.

### 4.4 Class-level observations

From [model_class_metrics.csv](/Users/carlgrude/examProject/outputs/tables/model_class_metrics.csv):

- `store` is the easiest class to predict (it is the largest class).
- `online` is also predicted well.
- `hybrid` is the hardest class to predict, mainly because it has far fewer observations (74 cases in the test set).
- Random Forest did not identify `hybrid` well in this first version.
- Default Logistic Regression identified `hybrid` well on recall, but with lower precision.
- After threshold tuning, `hybrid` performance improved further:
  - Hybrid F1 increased from `0.619` to `0.765`
  - Hybrid precision increased from `0.448` to `0.642`
  - Hybrid recall remained high (`0.946`)

### 4.5 Visual outputs

The following figures were generated for interpretation:

- [confusion_matrix_dummy_most_frequent.png](/Users/carlgrude/examProject/outputs/figures/confusion_matrix_dummy_most_frequent.png)
- [confusion_matrix_logistic_regression.png](/Users/carlgrude/examProject/outputs/figures/confusion_matrix_logistic_regression.png)
- [confusion_matrix_logistic_regression_hybrid_tuned.png](/Users/carlgrude/examProject/outputs/figures/confusion_matrix_logistic_regression_hybrid_tuned.png)
- [confusion_matrix_random_forest.png](/Users/carlgrude/examProject/outputs/figures/confusion_matrix_random_forest.png)
- [feature_importance_logistic_regression_hybrid_tuned.png](/Users/carlgrude/examProject/outputs/figures/feature_importance_logistic_regression_hybrid_tuned.png)
- [hybrid_threshold_tuning_logistic_regression_hybrid_tuned.png](/Users/carlgrude/examProject/outputs/figures/hybrid_threshold_tuning_logistic_regression_hybrid_tuned.png)

## 5. Discussion

### 5.1 Business meaning

The current model can be used as a first decision-support tool. It gives a data-based estimate of which channel a customer is likely to prefer. This can support:

- channel resource allocation,
- campaign targeting,
- communication strategy by customer group.

In short, the model helps managers make more structured decisions instead of relying only on intuition.

### 5.2 Why this method was chosen

The research question is a prediction problem with three possible outcomes. Classification models are therefore a natural choice.

Using three model types was intentional:

- the baseline checks if the project adds value at all,
- Logistic Regression provides strong performance and is easier to explain,
- threshold tuning improves the minority (`hybrid`) class without changing the underlying model architecture,
- Random Forest checks if more complex, non-linear patterns improve results.

This approach is both practical and academically defensible for a first project version.

### 5.3 Limitations

- The dataset is simulated, not collected from one real retailer.
- Class imbalance affects prediction quality, especially for `hybrid`.
- Results should be interpreted as a proof-of-concept for method and workflow, not as final production performance.

### 5.4 Recommended next steps

1. Improve `hybrid` prediction with class-balancing and threshold tuning.
2. Add segment-focused analysis (for example age groups, city tiers, spending patterns).
3. Build a simple dashboard that translates predictions into business actions.

## Short Version for a Non-Technical Audience

We built a model that predicts whether a customer is more likely to shop online, in-store, or in both channels.  
The best model performed well and was much better than a basic baseline.  
This shows that customer data can be used to support better channel decisions, but the model should be improved further for hybrid customers and validated on real retailer data.
