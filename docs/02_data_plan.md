# 02 Data Plan

## Data Sources

| Source | Owner | Access Method | Time Span | Notes |
|---|---|---|---|---|
| Online vs In-Store Shopping Behaviour Dataset (Kaggle) | Kaggle dataset publisher | CSV download | Cross-sectional snapshot | Simulated data, 11,789 rows, 26 features |

## Data Dictionary (Initial)

| Variable | Type | Meaning | Expected Quality Issues |
|---|---|---|---|
| Age | Numeric | Customer age | Outliers or unrealistic values |
| Monthly Income | Numeric | Monthly income level | Skewed distribution, potential extremes |
| Daily Internet Hours | Numeric | Time spent online daily | Self-report bias, range checks needed |
| Smartphone Usage Years | Numeric | Years using smartphones | Non-negative range validation |
| Social Media Hours | Numeric | Daily social media usage | Outliers, potential right skew |
| Online Payment Trust Score | Numeric | Trust in online payments | Scale consistency check |
| Tech Savvy Score | Numeric | Digital proficiency level | Scale consistency check |
| Monthly Online Orders | Numeric | Number of online purchases per month | High-count outliers |
| Monthly Store Visits | Numeric | Number of store visits per month | High-count outliers |
| Avg Online Spend | Numeric | Average online spending | Currency/scale and outliers |
| Avg Store Spend | Numeric | Average in-store spending | Currency/scale and outliers |
| Discount Sensitivity | Numeric | Response to discounting | Scale interpretation |
| Return Frequency | Numeric | Product return frequency | Right skew or sparse non-zero values |
| Need for Touch and Feel | Numeric | Preference for physical interaction with products | Scale interpretation |
| Environmental Awareness | Numeric | Sustainability orientation | Scale interpretation |
| Gender | Categorical | Reported gender | Category consistency |
| Shopping Preference | Categorical (target) | Online, In-store, Hybrid | Class imbalance |

## Data Quality Checks

- Completeness: quantify missing values per column; define imputation/drop rules.
- Consistency: standardize category labels and capitalization.
- Validity: enforce domain/range checks on behavioral and score variables.
- Uniqueness: verify no duplicate rows or duplicate synthetic IDs (if present).
- Timeliness: cross-sectional simulated data; no time refresh expected.

## Data Governance

- Sensitive data present? (Y/N): Potentially sensitive demographics and income-like fields.
- Personal data present? (Y/N): No direct identifiers observed in current description.
- Required anonymization: Not required beyond source dataset state; avoid adding identifiable joins.
- Storage and access controls: keep data in `data/raw/`, processed files in `data/processed/`, and restrict sharing to course/project context.

## Preprocessing Plan

- Cleaning steps:
  handle missing values, normalize category names, remove impossible numeric values, treat extreme outliers based on transparent rules.
- Feature engineering:
  ratio features (online vs store activity/spend), grouped behavioral bands, optional interaction terms for model improvement.
- Train/test split or time split strategy:
  stratified train/test split due to multi-class target (`shopping_preference`), with cross-validation in training.
