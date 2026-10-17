# Phase 5 — Retail Demand Forecasting

## Project

**RetailPulse AI 2.0**

**Data source:** Fully synthetic retail transaction data.

**Objective:** Predict daily units sold for each store-product
combination using historical demand and calendar features.

This phase establishes a reproducible, validation-selected
forecasting model and evaluates it once on a reserved
chronological test period.

## 1. Dataset and forecasting unit

The original synthetic dataset contains:

- 70,000 daily store-product observations.
- 10 stores and 10 products.
- 100 store-product time series.
- Dates from 2024-01-01 through 2025-11-30.

The target is `units_sold`.

Each forecast corresponds to one store, one product,
and one calendar date.

The feature-building process removes the first 28 days
from each series because the longest lag requires
28 preceding observations.

The resulting forecasting dataset contains 67,200 rows.

## 2. Chronological data splits

| Split | Dates | Rows |
|---|---|---:|
| Training | 2024-01-29 to 2025-05-31 | 48,900 |
| Validation | 2025-06-01 to 2025-08-31 | 9,200 |
| Test | 2025-09-01 to 2025-11-30 | 9,100 |

The split boundaries are chronological and identical
across all 100 store-product series.

Validation was used for model comparison and champion
selection. Test data was reserved until after the
champion-selection record had been frozen.

## 3. Forecast features

Historical sales features:

- Sales lag 1 day.
- Sales lag 7 days.
- Sales lag 14 days.
- Sales lag 28 days.
- Shifted rolling mean over 7 days.
- Shifted rolling mean over 14 days.
- Shifted rolling mean over 28 days.

Calendar features:

- Day of week.
- Month.
- Weekend indicator.

Store and product identifiers are represented using
one-hot encoded categorical features.

Rolling statistics are shifted so that a prediction
does not use the target value from the same date.

The target, date, and split labels are excluded from
model input features.

## 4. Evaluation protocol

**Forecast horizon:** One day ahead.

**Evaluation style:** Rolling one-day-ahead predictions.

For each evaluated date, historical lag features may
contain actual sales from preceding dates, including
earlier dates in the evaluation period.

This protocol assumes recent actual sales observations
are available before each next-day prediction.

It is not a fixed-origin, 91-day-ahead forecast.

## 5. Evaluation metrics

### WAPE

Weighted Absolute Percentage Error:

WAPE = 100 × sum(abs(actual - predicted))
       / sum(actual)

Lower is better.

### RMSSE

Root Mean Squared Scaled Error:

RMSSE = sqrt(mean((actual - predicted)^2 / scale))

The scale is computed separately for each store-product
series using the mean squared first difference of
training-period sales only.

Lower is better.

### Bias percentage

Bias = 100 × sum(predicted - actual) / sum(actual)

Positive bias indicates aggregate overforecasting.
Negative bias indicates aggregate underforecasting.

All three reported metrics are computed on the
corresponding evaluation rows.

## 6. Candidate models

### Naive and statistical baselines

- Previous-day sales (`naive_1`).
- Previous-week sales (`seasonal_naive_7`).
- Shifted seven-day rolling mean (`rolling_mean_7`).

The rolling-mean baseline was the strongest baseline
on the validation period.

### Random Forest v1

A Random Forest regressor trained on the training split
using lagged sales, calendar, store, and product features.

### Gradient Boosting v1

A histogram-based gradient boosting regressor trained
on the same training split and feature set.

Configuration:

- Squared-error loss.
- Learning rate: 0.07.
- Maximum iterations: 180.
- Maximum leaf nodes: 31.
- Minimum samples per leaf: 30.
- L2 regularization: 1.0.
- Random state: 42.
- Early stopping disabled.

## 7. Validation results

All models below were evaluated on the same 9,200
validation observations.

| Model | WAPE (%) | RMSSE | Bias (%) |
|---|---:|---:|---:|
| Gradient Boosting v1 | 28.3538 | 0.6265 | +0.6228 |
| Random Forest v1 | 28.9108 | 0.6344 | +2.6529 |
| Rolling mean-7 | 33.5911 | 0.7334 | +0.4323 |

Gradient Boosting achieved the lowest validation WAPE
and RMSSE among the evaluated candidates.

It was selected as the champion using validation WAPE,
with validation RMSSE as the tie-breaker.

The selection was recorded in:

`models/champion_selection.json`

The frozen selection identifies:

`models/gradient_boosting_v1.joblib`

No test metrics were used to select the champion.

## 8. Final held-out test results

**Test dates:** 2025-09-01 through 2025-11-30.

**Observations per model:** 9,100.

| Model | WAPE (%) | RMSSE | Bias (%) |
|---|---:|---:|---:|
| Gradient Boosting v1 | 28.6666 | 0.6267 | +0.3821 |
| Rolling mean-7 | 34.1410 | 0.7372 | -0.9331 |

The frozen Gradient Boosting model reduced test WAPE
by approximately 16.03% relative to the rolling-mean
baseline.

Its WAPE increased from 28.3538% on validation to
28.6666% on test, a difference of +0.3128 percentage
points.

Its RMSSE changed from 0.6265 to 0.6267.

The saved Gradient Boosting model was evaluated without
retraining or hyperparameter tuning.

The baseline was included for diagnostic comparison,
not for selecting a new champion.

## 9. Leakage controls and reproducibility

The project implements the following safeguards:

1. Chronological train, validation, and test splits.
2. Lagged and shifted historical demand features.
3. No same-day target in model input features.
4. Training-only RMSSE scaling.
5. Model fitting on training rows only.
6. Validation-only champion selection.
7. Frozen champion-selection JSON before test evaluation.
8. Single held-out evaluation with overwrite protection.
9. Saved prediction and metric reconciliation tests.
10. Reproducible serialized model artifacts.

The final holdout evaluation uses the previously trained
model and its saved feature schema.

## 10. Saved artifacts

### Forecasting dataset

`data/processed/forecasting_dataset.csv`

### Model artifacts

- `models/random_forest_v1.joblib`
- `models/gradient_boosting_v1.joblib`
- `models/champion_selection.json`

### Validation reports

- `reports/validation/baseline_metrics.csv`
- `reports/validation/random_forest_v1_metrics.csv`
- `reports/validation/gradient_boosting_v1_metrics.csv`
- `reports/validation/phase5_model_comparison.csv`

### Test reports

- `reports/test/holdout_metrics.csv`
- `reports/test/holdout_predictions.csv`
- `reports/test/holdout_summary.json`

### Implementation modules

- `src/features/build_forecasting_dataset.py`
- `src/models/forecast_baselines.py`
- `src/models/train_random_forest.py`
- `src/models/train_gradient_boosting.py`
- `src/models/select_champion.py`
- `src/models/evaluate_holdout.py`

## 11. Testing and quality assurance

The completed Phase 5 implementation has:

- 191 passing automated tests.
- Clean Ruff checks.
- Saved-model prediction reproducibility checks.
- Validation and holdout metric reconciliation checks.
- Chronological boundary checks.
- Frozen-selection integrity checks.
- Nonnegative and finite prediction checks.

## 12. Limitations

**Synthetic data:** All results describe generated
retail behavior. They are not evidence of accuracy
on actual customer demand.

**One-day-ahead scope:** The model relies on recent
observed sales. Its results must not be interpreted
as a multi-month forecast generated at one origin date.

**Observed sales versus true demand:** Units sold may
be affected by inventory availability and stockouts.
Forecasting observed sales does not necessarily recover
unconstrained customer demand.

**Feature availability:** Future deployment would
require a reliable daily sales data pipeline and
consistent store-product identifiers.

**Temporal generalization:** One reserved test window
does not establish stability across all seasons,
economic conditions, or future retail environments.

**No production deployment claim:** This phase
evaluates an offline model on synthetic data.
Production monitoring, drift handling, and operational
forecast delivery are outside its scope.

## 13. Phase conclusion

Gradient Boosting v1 is the validation-selected,
frozen champion for RetailPulse AI 2.0.

It achieved:

- Validation WAPE: 28.3538%.
- Held-out test WAPE: 28.6666%.
- Held-out test RMSSE: 0.6267.
- Held-out test bias: +0.3821%.
- Relative held-out WAPE improvement over the
  rolling-mean baseline: approximately 16.03%.

The next phase will focus on model interpretation,
forecast error analysis, and business-facing insights,
while preserving the completed holdout evaluation.