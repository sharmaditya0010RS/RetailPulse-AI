"""Export reproducible synthetic next-day inventory planning artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.forecast_inventory_pipeline import (
    generate_inventory_plan,
    load_historical_data,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "reports" / "planning"

EXPORT_COLUMNS = [
    "date",
    "inventory_snapshot_date",
    "store_id",
    "product_id",
    "predicted_units",
    "closing_stock",
    "reorder_point",
    "supplier_lead_time_days",
    "supplier_reliability",
    "lead_time_demand_units",
    "safety_days",
    "safety_stock_units",
    "target_stock_units",
    "inventory_position_units",
    "suggested_order_units",
    "coverage_days",
    "below_existing_reorder_point",
    "stockout_flag",
    "inventory_status",
    "forecast_model",
    "planning_method",
    "data_source",
    "open_purchase_orders_included",
]

STATUS_ORDER = [
    "STOCKOUT",
    "BELOW_LEAD_TIME_COVERAGE",
    "REPLENISHMENT_SUGGESTED",
    "SUFFICIENT_FOR_POLICY",
]


def prepare_planning_export(plan):
    """Validate and select a stable, documented export schema."""
    if not isinstance(plan, pd.DataFrame) or plan.empty:
        raise ValueError("Planning export requires a nonempty DataFrame.")

    missing = [
        column for column in EXPORT_COLUMNS
        if column not in plan.columns
    ]

    if missing:
        raise ValueError(f"Missing planning export columns: {missing}")

    result = plan[EXPORT_COLUMNS].copy(deep=True)

    for column in ("date", "inventory_snapshot_date"):
        result[column] = pd.to_datetime(
            result[column],
            errors="raise",
        )

    if result[["date", "inventory_snapshot_date"]].isna().any().any():
        raise ValueError("Planning export dates cannot be missing.")

    if result["date"].nunique() != 1:
        raise ValueError("Planning export must contain one forecast date.")

    if result["inventory_snapshot_date"].nunique() != 1:
        raise ValueError("Planning export must contain one snapshot date.")

    if not (
        result["date"]
        == result["inventory_snapshot_date"] + pd.Timedelta(days=1)
    ).all():
        raise ValueError("Forecast must follow the inventory snapshot.")

    if result.duplicated(["store_id", "product_id"]).any():
        raise ValueError("Duplicate store-product planning records.")

    if result[["store_id", "product_id"]].isna().any().any():
        raise ValueError("Planning identifiers cannot be missing.")

    if not np.isfinite(
        result["predicted_units"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Planning predictions must be finite.")

    if (result["predicted_units"] < 0).any():
        raise ValueError("Planning predictions cannot be negative.")

    if (result["suggested_order_units"] < 0).any():
        raise ValueError("Suggested orders cannot be negative.")

    if not pd.api.types.is_integer_dtype(
        result["suggested_order_units"]
    ):
        raise ValueError("Suggested orders must be integers.")

    if not result["inventory_status"].isin(STATUS_ORDER).all():
        raise ValueError("Unknown inventory planning status.")

    if not result["forecast_model"].eq(
        "gradient_boosting_v1"
    ).all():
        raise ValueError("Unexpected planning forecast model.")

    if not result["data_source"].eq("synthetic").all():
        raise ValueError("Unexpected planning data source.")

    if result["open_purchase_orders_included"].any():
        raise ValueError("Unconfirmed open orders must be excluded.")

    return result.sort_values(
        ["store_id", "product_id"]
    ).reset_index(drop=True)


def build_planning_summary(plan):
    """Build a JSON-compatible aggregate summary."""
    prepared = prepare_planning_export(plan)

    forecast_date = prepared["date"].iloc[0].date().isoformat()
    snapshot_date = (
        prepared["inventory_snapshot_date"].iloc[0]
        .date()
        .isoformat()
    )

    status_counts = prepared["inventory_status"].value_counts()

    return {
        "project": "RetailPulse AI 2.0",
        "data_source": "synthetic",
        "forecast_model": "gradient_boosting_v1",
        "forecast_date": forecast_date,
        "inventory_snapshot_date": snapshot_date,
        "forecast_horizon_days": 1,
        "store_product_count": int(len(prepared)),
        "total_predicted_daily_units": float(
            prepared["predicted_units"].sum()
        ),
        "total_available_stock_units": int(
            prepared["closing_stock"].sum()
        ),
        "total_target_stock_units": int(
            prepared["target_stock_units"].sum()
        ),
        "total_suggested_order_units": int(
            prepared["suggested_order_units"].sum()
        ),
        "series_with_suggested_orders": int(
            prepared["suggested_order_units"].gt(0).sum()
        ),
        "inventory_status_counts": {
            status: int(status_counts.get(status, 0))
            for status in STATUS_ORDER
        },
        "safety_days": float(prepared["safety_days"].iloc[0]),
        "planning_method": "one_day_forecast_times_coverage_days",
        "open_purchase_orders_included": False,
        "limitations": [
            "Synthetic retail data; not a live operational forecast.",
            "One-day-ahead model; no validated multi-day forecasts.",
            "Lead-time demand assumes constant daily predicted demand.",
            "Safety days are illustrative, not service-level optimized.",
            "Unconfirmed outstanding purchase orders are excluded.",
            "Recommendations are decision support, not purchase orders.",
        ],
    }


def export_inventory_plan(plan, output_dir=OUTPUT_DIR):
    """Write deterministic CSV and JSON planning artifacts."""
    prepared = prepare_planning_export(plan)
    summary = build_planning_summary(prepared)

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    date_label = summary["forecast_date"]

    csv_path = directory / (
        f"inventory_recommendations_{date_label}.csv"
    )
    json_path = directory / (
        f"inventory_recommendations_{date_label}.json"
    )
    summary_path = directory / (
        f"inventory_planning_summary_{date_label}.json"
    )

    serializable = prepared.copy(deep=True)

    for column in ("date", "inventory_snapshot_date"):
        serializable[column] = (
            serializable[column].dt.strftime("%Y-%m-%d")
        )

    # JSON does not permit Infinity; zero-demand coverage is unknown
    # or unbounded, so export it as null instead.
    serializable["coverage_days"] = (
        serializable["coverage_days"]
        .replace([np.inf, -np.inf], np.nan)
    )

    records = json.loads(
        serializable.to_json(
            orient="records",
            double_precision=10,
        )
    )

    csv_export = prepared.copy(deep=True)
    csv_export["coverage_days"] = (
        csv_export["coverage_days"]
        .replace([np.inf, -np.inf], np.nan)
    )

    csv_export.to_csv(
        csv_path,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.10f",
    )

    json_path.write_text(
        json.dumps(
            records,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    return {
        "csv": csv_path,
        "json": json_path,
        "summary": summary_path,
    }


def main():
    """Generate and export the frozen-model planning scenario."""
    history = load_historical_data()
    plan = generate_inventory_plan(history)
    paths = export_inventory_plan(plan)
    summary = build_planning_summary(plan)

    print("FORECAST DATE:", summary["forecast_date"])
    print("INVENTORY DATE:", summary["inventory_snapshot_date"])
    print("MODEL:", summary["forecast_model"])
    print("SERIES:", summary["store_product_count"])
    print(
        "TOTAL SUGGESTED ORDER UNITS:",
        summary["total_suggested_order_units"],
    )

    print("\nEXPORTED FILES:")
    for label, path in paths.items():
        print(f"{label.upper()}: {path}")

    print("\nPASS: Reproducible inventory planning exports")


if __name__ == "__main__":
    main()