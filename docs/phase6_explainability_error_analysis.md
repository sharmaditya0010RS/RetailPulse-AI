# Phase 6 — Forecast Explainability and Error Analysis

## 1. Executive Summary

RetailPulse AI 2.0 is a synthetic retail analytics and demand-forecasting project.

Phase 6 explains the frozen Gradient Boosting champion selected in Phase 5 and analyzes its forecasting errors across stores, products, and weekdays.

Two complementary analyses were completed:

1. **Validation-set permutation importance** to understand which input features the champion relies on.
2. **Descriptive held-out test error analysis** to identify segments where forecast accuracy varies.

The champion was not retrained, and the held-out test set was not reused for model selection.

### Key Findings

- Champion held-out WAPE: **28.6666%**
- Rolling 7-day baseline held-out WAPE: **34.1410%**
- Relative WAPE improvement: **16.03%**
- Strongest validation permutation feature: **28-day rolling mean of sales**
- Champion outperformed the baseline in all **10 stores, 10 products, and 7 weekdays** analyzed.

All data and results are synthetic demonstrations, not measurements of a real retailer.

---

## 2. Forecasting Context

### 2.1 Forecasting Objective

Predict daily units sold for each store-product combination.

The forecasting dataset contains:

- 10 stores
- 10 products
- 100 store-product series
- 67,200 feature-engineered observations

### 2.2 Chronological Data Splits

| Split | Dates | Observations |
|---|---|---:|
| Train | 2024-01-29 to 2025-05-31 | 48,900 |
| Validation | 2025-06-01 to 2025-08-31 | 9,200 |
| Test | 2025-09-01 to 2025-11-30 | 9,100 |

The test period was evaluated once after champion selection was frozen.

The forecasting setup uses rolling one-day-ahead predictions with observed historical demand available to lag and rolling-window features. It is not a fixed-origin, multi-step forecast.

### 2.3 Frozen Champion

The champion is `gradient_boosting_v1`, implemented with scikit-learn's `HistGradientBoostingRegressor`.

Its selection was based on validation performance, not held-out test performance.

The frozen selection is recorded in:

`models/champion_selection.json`

The trained artifact is stored in:

`models/gradient_boosting_v1.joblib`

---

## 3. Explainability Methodology

### 3.1 Permutation Feature Importance

Permutation importance measures how much the validation-set forecasting error changes when a single input feature is shuffled.

For each feature:

1. Calculate the champion's original validation WAPE.
2. Randomly permute the feature values across validation observations.
3. Generate predictions with the unchanged trained model.
4. Recalculate validation WAPE.
5. Repeat the permutation three times.
6. Report the mean and standard deviation of the WAPE increase.

The original validation WAPE was **28.3538%**.

A larger positive increase indicates that disrupting a feature produces a larger deterioration in validation performance.

The analysis operates on 12 original input features, including numerical and categorical inputs, rather than treating individual one-hot-encoded columns as separate business features.

### 3.2 Permutation Settings

| Parameter | Value |
|---|---|
| Analysis split | Validation |
| Observations | 9,200 |
| Original features | 12 |
| Permutation repeats | 3 |
| Random seed | 42 |
| Error metric | WAPE |
| Importance unit | Percentage points |

### 3.3 Feature Importance Results

| Feature | Mean WAPE increase (pp) |
|---|---:|
| 28-day rolling mean | 32.2559 |
| Day of week | 5.4698 |
| 7-day rolling mean | 4.5324 |
| Product ID | 0.7257 |
| Previous-day sales lag | 0.6744 |
| Store ID | 0.3851 |
| 7-day sales lag | 0.3154 |
| 14-day rolling mean | 0.2573 |
| 14-day sales lag | 0.0753 |
| 28-day sales lag | 0.0681 |
| Month | 0.0037 |
| Weekend flag | 0.0000 |

The complete report includes the standard deviation across three permutations.

### 3.4 Interpretation

**Longer-term demand level matters most.**

The 28-day rolling mean produced the largest validation WAPE increase when shuffled, indicating strong reliance on the recent demand level.

**Weekly patterns contribute additional predictive information.**

Day of week and the 7-day rolling mean were the next most important features.

**Store and product identity provide contextual information.**

Their measured importance was positive but smaller than that of the strongest temporal features.

**Low permutation importance does not prove a feature is useless.**

Correlated features may substitute for one another. For example, multiple lag and rolling-window variables describe overlapping demand history.

Permutation importance is a model-dependence diagnostic, not evidence of causality.

---

## 4. Held-Out Error Analysis Methodology

### 4.1 Analysis Source

The error analysis reads previously saved held-out predictions from:

`reports/test/holdout_predictions.csv`

No model fitting or new holdout inference was performed.

Each model has 9,100 saved test predictions.

The analysis compares:

- `gradient_boosting_v1` — frozen champion
- `rolling_mean_7` — selected reference baseline

### 4.2 Evaluation Metrics

**Weighted Absolute Percentage Error (WAPE)**

WAPE measures total absolute forecast error relative to total actual units sold.

WAPE = 100 × sum(|prediction − actual|) / sum(actual)

Lower is better.

**Root Mean Squared Scaled Error (RMSSE)**

RMSSE measures squared forecast error scaled using the corresponding store-product series' training-period demand changes.

Lower is better.

Training-only scales are preserved to avoid using future data to define the scaling denominator.

**Forecast Bias**

Bias = 100 × sum(prediction − actual) / sum(actual)

Positive bias indicates aggregate overprediction. Negative bias indicates aggregate underprediction.

### 4.3 Segment Analysis

The saved predictions were grouped by:

1. Store ID
2. Product ID
3. Day of week

For each group, the analysis calculates WAPE, RMSSE, bias, observation count, actual units, and predicted units.

The champion and baseline are then compared using WAPE percentage-point and relative improvements.

The results were reconciled against the original saved global holdout metrics.

---

## 5. Held-Out Results

### 5.1 Overall Performance

| Metric | Gradient Boosting | Rolling 7-day mean |
|---|---:|---:|
| WAPE | 28.6666% | 34.1410% |
| RMSSE | 0.6267 | 0.7372 |
| Bias | +0.3821% | −0.9331% |

The champion achieved a **5.4744 percentage-point reduction in WAPE**, equivalent to approximately **16.03% relative improvement** over the baseline.

These values come from the original Phase 5 held-out evaluation, not from a new evaluation in Phase 6.

### 5.2 Store-Level Error

| Store | Champion WAPE | Baseline WAPE |
|---|---:|---:|
| S002 | 31.99% | 37.73% |
| S009 | 31.19% | 35.94% |
| S005 | 30.85% | 34.74% |
| S010 | 29.66% | 36.64% |
| S007 | 29.47% | 35.32% |
| S008 | 28.72% | 33.26% |
| S003 | 27.83% | 35.10% |
| S004 | 26.67% | 31.87% |
| S006 | 26.54% | 31.22% |
| S001 | 26.51% | 31.92% |

The champion outperformed the baseline in **all 10 stores**.

S002 had the highest champion WAPE at 31.99%, while S001 had the lowest at 26.51%.

The difference identifies variation in forecasting difficulty across synthetic stores; it does not establish the cause of that variation.

### 5.3 Product-Level Error

| Product | Champion WAPE | Baseline WAPE |
|---|---:|---:|
| P004 | 33.64% | 39.85% |
| P008 | 33.28% | 38.20% |
| P010 | 31.94% | 37.37% |
| P007 | 30.35% | 34.31% |
| P003 | 29.02% | 36.70% |
| P002 | 28.33% | 34.16% |
| P006 | 28.00% | 32.89% |
| P009 | 27.76% | 32.25% |
| P001 | 27.58% | 32.13% |
| P005 | 25.38% | 30.90% |

The champion outperformed the baseline in **all 10 products**.

P004 had the highest champion WAPE at 33.64%, while P005 had the lowest at 25.38%.

Product-level differences could motivate future investigation of demand volatility, stockouts, promotions, or other relevant variables. The current analysis does not demonstrate which factors explain those differences.

### 5.4 Day-of-Week Error

| Day | Champion WAPE | Baseline WAPE |
|---|---:|---:|
| Tuesday | 30.21% | 35.37% |
| Wednesday | 29.68% | 34.28% |
| Monday | 29.29% | 35.49% |
| Sunday | 29.06% | 34.78% |
| Friday | 28.45% | 33.42% |
| Thursday | 28.40% | 32.53% |
| Saturday | 26.42% | 33.28% |

The champion outperformed the baseline on **all seven weekdays**.

Tuesday had the highest champion WAPE, while Saturday had the lowest.

The observed weekday differences are descriptive properties of this synthetic test period.

---

## 6. Business Interpretation

### 6.1 Demand Planning

The importance of the 28-day rolling mean suggests that recent demand history is central to the champion's predictions.

For a real retail deployment, reliable historical sales records would therefore be important for maintaining forecast quality.

### 6.2 Store-Level Monitoring

The store-level report identifies S002, S009, and S005 as the highest-WAPE stores in this test period.

These segments are candidates for operational review, but the results alone do not establish whether inventory constraints, demand volatility, or other factors caused the higher errors.

### 6.3 Product-Level Monitoring

P004, P008, and P010 show the highest product-level champion WAPE.

In a production setting, similar reports could help analysts prioritize investigation of forecasting reliability across product categories.

### 6.4 Weekday Monitoring

Tuesday and Wednesday have the highest weekday WAPE for the champion.

This finding could inform monitoring and reporting priorities, but it should not be treated as a causal explanation of weekday demand.

### 6.5 Decision-Making Boundary

Because the test set was already used for final evaluation, these segment findings are not used to change the selected model or its hyperparameters.

Any future model improvements should be evaluated using a newly designed chronological evaluation protocol with an untouched future holdout.

---

## 7. Limitations

### Synthetic Data

The dataset is generated rather than collected from a real retailer. The observed relationships and performance cannot be assumed to generalize to real commercial operations.

### Permutation Importance and Correlation

Lag and rolling-window features are correlated. Individual permutation importance values may understate or redistribute importance among related features.

### One-Step-Ahead Evaluation

The forecasting setup assumes observed recent demand is available at each prediction point. Its results do not establish performance for long-horizon, fixed-origin forecasts.

### Limited Test Horizon

The held-out test period spans September through November 2025. It does not demonstrate performance across every season or under future distribution shifts.

### Segment-Level Interpretation

Differences in WAPE do not establish their underlying causes. A segment with higher WAPE is not necessarily a segment with the greatest absolute business impact.

### No Causal Claims

The analysis identifies predictive dependencies and observed error patterns, not causal effects of stores, products, or weekdays.

---

## 8. Reproducibility and Governance

### Explainability Implementation

`src/analytics/forecast_explainability.py`

### Error Analysis Implementation

`src/analytics/forecast_error_analysis.py`

### Automated Tests

- `tests/test_forecast_explainability.py`
- `tests/test_forecast_error_analysis.py`

### Explainability Artifacts

- `reports/validation/gradient_boosting_permutation_importance.csv`
- `reports/validation/gradient_boosting_explainability_summary.json`

### Error Analysis Artifacts

- `reports/test/error_analysis/store_metrics.csv`
- `reports/test/error_analysis/store_comparison.csv`
- `reports/test/error_analysis/product_metrics.csv`
- `reports/test/error_analysis/product_comparison.csv`
- `reports/test/error_analysis/day_of_week_metrics.csv`
- `reports/test/error_analysis/day_of_week_comparison.csv`
- `reports/test/error_analysis/error_analysis_summary.json`

### Governance Controls

- Chronological train, validation, and test splits
- Champion frozen before test evaluation
- Original held-out evaluation preserved
- Validation-only permutation importance
- Descriptive-only held-out error analysis
- No retraining during Phase 6
- No test-based model selection
- Existing report files protected against accidental overwrites

### Quality Verification

At the completion of Phase 6 Step 6:

- Ruff: all checks passed
- Automated tests: **224 passed**

---

## 9. Phase 6 Conclusion

Phase 6 demonstrates how RetailPulse AI 2.0 moves beyond a single forecast-accuracy score.

The project now includes:

- Validation-based feature importance for the frozen champion
- Store-level, product-level, and weekday-level forecast diagnostics
- Reconciliation against the original held-out evaluation
- Automated tests for explainability and error analysis
- Clear separation between model selection and post-evaluation interpretation

The results support a transparent, reproducible demonstration of synthetic retail demand forecasting while preserving the integrity of the original chronological holdout evaluation.