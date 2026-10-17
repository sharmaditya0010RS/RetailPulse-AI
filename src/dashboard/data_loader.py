"""Read-only access to saved RetailPulse dashboard artifacts."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CSV_ARTIFACTS = {
    "executive_kpis": "data/processed/bi/executive_kpis.csv",
    "monthly_performance": "data/processed/bi/monthly_performance.csv",
    "store_performance": "data/processed/bi/store_performance.csv",
    "category_performance": "data/processed/bi/category_performance.csv",
    "product_stockouts": "data/processed/bi/product_stockouts.csv",
    "holdout_metrics": "reports/test/holdout_metrics.csv",
    "holdout_predictions": "reports/test/holdout_predictions.csv",
    "permutation_importance": (
        "reports/validation/"
        "gradient_boosting_permutation_importance.csv"
    ),
    "inventory_recommendations": (
        "reports/planning/"
        "inventory_recommendations_2025-12-01.csv"
    ),
}

JSON_ARTIFACTS = {
    "executive_scorecard": (
        "data/processed/bi/executive_scorecard.json"
    ),
    "holdout_summary": "reports/test/holdout_summary.json",
    "planning_summary": (
        "reports/planning/"
        "inventory_planning_summary_2025-12-01.json"
    ),
    "champion_selection": "models/champion_selection.json",
}

REQUIRED_COLUMNS = {
    "executive_kpis": {
        "date_start",
        "date_end",
        "observations",
        "stores",
        "products",
        "days",
        "units_sold",
        "lost_units",
        "net_revenue",
        "estimated_gross_profit",
        "estimated_gross_margin_pct",
        "stockout_observation_rate_pct",
    },
    "monthly_performance": {
        "sales_month",
        "units_sold",
        "lost_units",
        "net_revenue",
        "estimated_gross_profit",
        "revenue_growth_mom_pct",
    },
    "store_performance": {
        "store_id",
        "region",
        "store_tier",
        "units_sold",
        "lost_units",
        "net_revenue",
        "stockout_observation_rate_pct",
    },
    "category_performance": {
        "category",
        "products",
        "units_sold",
        "lost_units",
        "net_revenue",
        "revenue_share_pct",
    },
    "product_stockouts": {
        "product_id",
        "category",
        "brand",
        "units_sold",
        "lost_units",
        "stockout_observation_rate_pct",
    },
    "holdout_metrics": {
        "model",
        "split",
        "observations",
        "wape_pct",
        "rmsse",
        "bias_pct",
    },
    "holdout_predictions": {
        "date",
        "store_id",
        "product_id",
        "units_sold",
        "model",
        "prediction",
        "error",
    },
    "permutation_importance": {
        "feature",
        "baseline_validation_wape_pct",
        "wape_increase_mean_pp",
        "wape_increase_std_pp",
        "repeats",
    },
    "inventory_recommendations": {
        "date",
        "inventory_snapshot_date",
        "store_id",
        "product_id",
        "predicted_units",
        "closing_stock",
        "reorder_point",
        "supplier_lead_time_days",
        "target_stock_units",
        "suggested_order_units",
        "coverage_days",
        "inventory_status",
        "forecast_model",
        "data_source",
        "open_purchase_orders_included",
    },
}

DATE_COLUMNS = {
    "executive_kpis": ["date_start", "date_end"],
    "monthly_performance": ["sales_month"],
    "holdout_predictions": ["date"],
    "inventory_recommendations": [
        "date",
        "inventory_snapshot_date",
    ],
}

EXPECTED_CHAMPION = "gradient_boosting_v1"


def resolve_artifact(relative_path, root=ROOT):
    """Resolve an artifact within the configured repository root."""
    base = Path(root).resolve()
    path = (base / relative_path).resolve()

    if not path.is_relative_to(base):
        raise ValueError("Artifact path escapes repository root.")

    if not path.is_file():
        raise FileNotFoundError(f"Missing dashboard artifact: {path}")

    return path


def load_csv_artifact(name, root=ROOT):
    """Load a validated CSV artifact without changing source files."""
    if name not in CSV_ARTIFACTS:
        raise ValueError(f"Unknown CSV artifact: {name}")

    path = resolve_artifact(CSV_ARTIFACTS[name], root=root)
    frame = pd.read_csv(path)

    if frame.empty:
        raise ValueError(f"Dashboard artifact is empty: {name}")

    missing = REQUIRED_COLUMNS[name] - set(frame.columns)

    if missing:
        raise ValueError(
            f"Dashboard artifact {name} missing columns: "
            f"{sorted(missing)}"
        )

    for column in DATE_COLUMNS.get(name, []):
        frame[column] = pd.to_datetime(
            frame[column],
            errors="raise",
        )

        if frame[column].isna().any():
            raise ValueError(
                f"Dashboard artifact {name} has missing dates."
            )

    return frame.copy(deep=True)


def load_json_artifact(name, root=ROOT):
    """Load a JSON object from an existing saved artifact."""
    if name not in JSON_ARTIFACTS:
        raise ValueError(f"Unknown JSON artifact: {name}")

    path = resolve_artifact(JSON_ARTIFACTS[name], root=root)

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"Dashboard JSON artifact must be an object: {name}"
        )

    return data


def validate_dashboard_provenance(datasets):
    """Verify that displayed model and planning sources agree."""
    scorecard = datasets["executive_scorecard"]
    holdout = datasets["holdout_summary"]
    planning = datasets["planning_summary"]
    selection = datasets["champion_selection"]

    for name, artifact in (
        ("executive_scorecard", scorecard),
        ("holdout_summary", holdout),
        ("planning_summary", planning),
        ("champion_selection", selection),
    ):
        if artifact.get("data_source") != "synthetic":
            raise ValueError(
                f"Dashboard source must be synthetic: {name}"
            )

    if selection.get("champion_model") != EXPECTED_CHAMPION:
        raise ValueError("Unexpected frozen champion.")

    if selection.get("selection_status") != (
        "frozen_before_test_evaluation"
    ):
        raise ValueError("Champion selection is not frozen.")

    if holdout.get("champion_model") != EXPECTED_CHAMPION:
        raise ValueError("Holdout champion does not match selection.")

    if holdout.get("test_used_for_model_selection") is not False:
        raise ValueError("Holdout selection provenance is invalid.")

    if planning.get("forecast_model") != EXPECTED_CHAMPION:
        raise ValueError("Planning model does not match champion.")

    if planning.get("forecast_horizon_days") != 1:
        raise ValueError("Planning horizon must be one day.")

    if planning.get("open_purchase_orders_included") is not False:
        raise ValueError("Planning must exclude unconfirmed orders.")

    if planning.get("store_product_count") != len(
        datasets["inventory_recommendations"]
    ):
        raise ValueError("Planning record count does not match.")

    if not datasets["inventory_recommendations"][
        "forecast_model"
    ].eq(EXPECTED_CHAMPION).all():
        raise ValueError("Planning records contain unexpected models.")

    if not datasets["inventory_recommendations"][
        "data_source"
    ].eq("synthetic").all():
        raise ValueError("Planning records contain nonsynthetic data.")

    if datasets["inventory_recommendations"][
        "open_purchase_orders_included"
    ].any():
        raise ValueError("Planning records include open orders.")

    if not datasets["holdout_metrics"]["split"].eq("test").all():
        raise ValueError("Holdout metrics contain unexpected splits.")

    expected_models = {
        EXPECTED_CHAMPION,
        "rolling_mean_7",
    }

    if set(datasets["holdout_metrics"]["model"]) != expected_models:
        raise ValueError("Unexpected holdout comparison models.")


def load_dashboard_data(root=ROOT):
    """Load and validate all dashboard datasets and metadata."""
    datasets = {
        name: load_csv_artifact(name, root=root)
        for name in CSV_ARTIFACTS
    }

    datasets.update({
        name: load_json_artifact(name, root=root)
        for name in JSON_ARTIFACTS
    })

    validate_dashboard_provenance(datasets)

    return datasets


def main():
    """Run a read-only dashboard data-loading smoke test."""
    datasets = load_dashboard_data()

    print("PASS: Dashboard artifacts loaded")

    for name in CSV_ARTIFACTS:
        print(f"{name}: {len(datasets[name])} rows")

    print("\nPROVENANCE:")
    print(
        "Champion:",
        datasets["champion_selection"]["champion_model"],
    )
    print(
        "Holdout evaluation:",
        datasets["holdout_summary"]["evaluation_protocol"],
    )
    print(
        "Planning date:",
        datasets["planning_summary"]["forecast_date"],
    )
    print(
        "Data source:",
        datasets["planning_summary"]["data_source"],
    )

    print("\nPASS: Dashboard provenance validated")


if __name__ == "__main__":
    main()