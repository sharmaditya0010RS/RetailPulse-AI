# Phase 7 — Forecast Serving and Inventory Recommendations

## Executive Summary

Phase 7 connects the frozen RetailPulse AI 2.0 forecasting champion to an illustrative inventory replenishment workflow.

The pipeline uses synthetic demand observations through November 30, 2025, to generate one-day-ahead forecasts for December 1, 2025, across 100 store-product combinations.

Forecasts are combined with the latest inventory snapshot to estimate stock coverage, identify inventory risks, and recommend replenishment quantities.

The synthetic planning scenario suggests 4,108 replenishment units. These recommendations are not purchase orders or evidence of measured business savings.

## Architecture

Synthetic retail history
    |
    +-- build_next_day_features()
    |       |
    |       +-- load_champion()
    |       +-- predict_daily_demand()
    |
    +-- build_inventory_snapshot()
            |
            v
    generate_inventory_plan()
            |
            v
    recommend_replenishment()
            |
            v
    export_inventory_plan()
            |
            +-- CSV recommendations
            +-- JSON recommendations
            +-- JSON summary

## Main Modules

| Module | Responsibility |
|---|---|
| `src/models/forecast_inference.py` | Validate and serve the frozen champion |
| `src/analytics/inventory_recommendations.py` | Calculate inventory targets and replenishment suggestions |
| `src/analytics/forecast_inventory_pipeline.py` | Build next-day features and combine forecasts with inventory |
| `src/analytics/export_inventory_plan.py` | Produce reproducible CSV and JSON exports |

## Frozen Forecasting Champion

The selected champion is `gradient_boosting_v1`, using `HistGradientBoostingRegressor`.

The champion was selected using the validation split in Phase 5. The held-out test was evaluated once after model selection.

Phase 7 does not retrain the champion, change the selection, repeat the holdout evaluation, or tune the inventory policy against holdout results.

## Forecast Chronology

| Item | Date |
|---|---|
| Latest observed demand | 2025-11-30 |
| Inventory snapshot | 2025-11-30 |
| Forecast date | 2025-12-01 |
| Forecast horizon | One day |

The dataset contains 70,000 synthetic historical records across 100 store-product series, with 700 days per series.

Next-day feature generation requires 28 consecutive recent daily observations.

The model uses demand lags of 1, 7, 14, and 28 days; shifted rolling means over 7, 14, and 28 days; and calendar features.

All demand features for December 1 use observations from November 30 or earlier.

## Inventory Recommendation Formulas

Let:

- `D` = nonnegative predicted daily demand.
- `L` = supplier lead time in days.
- `S` = safety days, default 2.
- `I` = available closing stock.

Lead-time demand:

`lead_time_demand_units = D * L`

Safety stock:

`safety_stock_units = ceil(D * S)`

Target stock:

`target_stock_units = ceil(D * (L + S))`

Suggested replenishment:

`suggested_order_units = max(0, target_stock_units - I)`

Inventory coverage, when predicted demand is positive:

`coverage_days = I / D`

For zero predicted demand, coverage is internally infinite and exported as null.

The existing reorder point is reported separately:

`below_existing_reorder_point = I < reorder_point`

The reorder-point indicator does not override the forecast-based target.

## Inventory Status Rules

Statuses are assigned in priority order:

1. `STOCKOUT`: closing stock is zero.
2. `BELOW_LEAD_TIME_COVERAGE`: coverage is shorter than supplier lead time.
3. `REPLENISHMENT_SUGGESTED`: suggested order quantity is positive.
4. `SUFFICIENT_FOR_POLICY`: none of the above applies.

Statuses are mutually exclusive.

## December 1, 2025 Results

| Metric | Value |
|---|---:|
| Store-product combinations | 100 |
| Suggested replenishment units | 4,108 |
| Stockout | 15 |
| Below lead-time coverage | 27 |
| Replenishment suggested | 17 |
| Sufficient for policy | 41 |

Suggested replenishment units include recommendations from multiple inventory-status categories, not just the category named `REPLENISHMENT_SUGGESTED`.

## Export Artifacts

All exports are saved under `reports/planning/`.

- `inventory_recommendations_2025-12-01.csv`
- `inventory_recommendations_2025-12-01.json`
- `inventory_planning_summary_2025-12-01.json`

The record-level exports contain 100 store-product recommendations. The summary includes model provenance, dates, forecast horizon, aggregate inventory quantities, status counts, and methodology limitations.

Exports use deterministic store-product ordering and stable column ordering. Infinite coverage values become null in JSON.

## Reproduction

Run the following commands from the repository root with the virtual environment activated.

Forecast and inventory integration:

    python -m src.analytics.forecast_inventory_pipeline

Planning exports:

    python -m src.analytics.export_inventory_plan

Code quality:

    python -m ruff check config scripts src tests

Complete test suite:

    python -m pytest tests -q

## Test Results

At the Phase 7 documentation checkpoint:

| Test module | Passing tests |
|---|---:|
| Forecast inference | 23 |
| Inventory recommendations | 47 |
| Forecast-inventory integration | 26 |
| Inventory exports | 33 |
| Full project suite | 353 |

Ruff reported all checks passed.

## Assumptions and Limitations

### Synthetic data

The scenario uses generated retail data, not live operational records.

### Forecast horizon

The champion is a one-day-ahead forecasting model. The inventory policy multiplies its daily estimate across lead-time and safety-stock days, assuming constant demand.

This is not a validated multi-day forecast.

### Open purchase orders

Historical `units_ordered` and `units_received` fields do not establish which purchase orders remain outstanding.

The inventory position is therefore set to closing stock, without crediting unconfirmed inbound orders.

### Simplified inventory policy

The recommendations do not optimize supplier order constraints, case packs, minimum order quantities, service levels, budget, warehouse capacity, or ordering schedules.

Supplier reliability is retained as descriptive information rather than used to estimate probabilistic safety stock.

### Business impact

No reduction in stockouts, lost sales, or inventory costs has been demonstrated. The suggested quantities are illustrative decision-support outputs, not executed orders.

### Historical demonstration

December 1, 2025 is the forecast date for a reproducible historical scenario. This is not a live forecast for the current date.

## Phase 7 Deliverables

- [x] Frozen champion inference.
- [x] Forecast input validation.
- [x] Inventory recommendation heuristic.
- [x] Chronologically correct next-day features.
- [x] Forecast-to-inventory integration.
- [x] Reproducible CSV and JSON exports.
- [x] Automated tests.
- [x] Technical and business documentation.

## Next Phase

Phase 8 will present the project's historical analytics, forecasting results, and illustrative inventory recommendations through a portfolio-ready application or dashboard.

The frozen champion and original holdout evaluation remain unchanged.