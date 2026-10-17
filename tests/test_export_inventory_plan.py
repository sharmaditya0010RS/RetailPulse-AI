"""Tests for reproducible synthetic inventory planning exports."""

import json

import numpy as np
import pandas as pd
import pytest

from src.analytics.export_inventory_plan import (
    EXPORT_COLUMNS,
    STATUS_ORDER,
    build_planning_summary,
    export_inventory_plan,
    prepare_planning_export,
)
from src.analytics.forecast_inventory_pipeline import (
    generate_inventory_plan,
    load_historical_data,
)
from src.models.forecast_inference import load_champion


@pytest.fixture(scope="module")
def plan():
    history = load_historical_data()
    champion = load_champion()
    return generate_inventory_plan(history, artifact=champion)


def test_export_schema(plan):
    prepared = prepare_planning_export(plan)

    assert prepared.columns.tolist() == EXPORT_COLUMNS
    assert len(prepared) == 100


def test_export_sorted_by_series(plan):
    prepared = prepare_planning_export(plan)

    expected = prepared.sort_values(
        ["store_id", "product_id"]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(prepared, expected)


def test_export_does_not_mutate_input(plan):
    original = plan.copy(deep=True)

    prepare_planning_export(plan)

    pd.testing.assert_frame_equal(plan, original)


def test_rejects_empty_export():
    with pytest.raises(ValueError, match="nonempty"):
        prepare_planning_export(pd.DataFrame())


def test_rejects_missing_export_column(plan):
    invalid = plan.drop(columns=["predicted_units"])

    with pytest.raises(ValueError, match="Missing planning export"):
        prepare_planning_export(invalid)


def test_rejects_duplicate_export_series(plan):
    invalid = pd.concat(
        [plan, plan.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate store-product"):
        prepare_planning_export(invalid)


def test_rejects_multiple_forecast_dates(plan):
    invalid = plan.copy()
    invalid.loc[0, "date"] = pd.Timestamp("2025-12-02")

    with pytest.raises(ValueError, match="one forecast date"):
        prepare_planning_export(invalid)


def test_rejects_multiple_snapshot_dates(plan):
    invalid = plan.copy()
    invalid.loc[0, "inventory_snapshot_date"] = pd.Timestamp(
        "2025-11-29"
    )

    with pytest.raises(ValueError, match="one snapshot date"):
        prepare_planning_export(invalid)


def test_rejects_nonconsecutive_dates(plan):
    invalid = plan.copy()
    invalid["date"] = pd.Timestamp("2025-12-03")

    with pytest.raises(ValueError, match="must follow"):
        prepare_planning_export(invalid)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_predictions(plan, value):
    invalid = plan.copy()
    invalid.loc[0, "predicted_units"] = value

    with pytest.raises(ValueError, match="must be finite"):
        prepare_planning_export(invalid)


def test_rejects_negative_predictions(plan):
    invalid = plan.copy()
    invalid.loc[0, "predicted_units"] = -1.0

    with pytest.raises(ValueError, match="cannot be negative"):
        prepare_planning_export(invalid)


def test_rejects_negative_orders(plan):
    invalid = plan.copy()
    invalid.loc[0, "suggested_order_units"] = -1

    with pytest.raises(ValueError, match="cannot be negative"):
        prepare_planning_export(invalid)


def test_rejects_fractional_orders(plan):
    invalid = plan.copy()
    invalid["suggested_order_units"] = (
        invalid["suggested_order_units"].astype(float)
    )
    invalid.loc[0, "suggested_order_units"] = 1.5

    with pytest.raises(ValueError, match="must be integers"):
        prepare_planning_export(invalid)


def test_rejects_unknown_status(plan):
    invalid = plan.copy()
    invalid.loc[0, "inventory_status"] = "UNKNOWN"

    with pytest.raises(ValueError, match="Unknown inventory"):
        prepare_planning_export(invalid)


def test_rejects_unexpected_model(plan):
    invalid = plan.copy()
    invalid.loc[0, "forecast_model"] = "other_model"

    with pytest.raises(ValueError, match="Unexpected planning forecast"):
        prepare_planning_export(invalid)


def test_rejects_nonsynthetic_source(plan):
    invalid = plan.copy()
    invalid.loc[0, "data_source"] = "production"

    with pytest.raises(ValueError, match="Unexpected planning data"):
        prepare_planning_export(invalid)


def test_rejects_included_open_orders(plan):
    invalid = plan.copy()
    invalid.loc[0, "open_purchase_orders_included"] = True

    with pytest.raises(ValueError, match="must be excluded"):
        prepare_planning_export(invalid)


def test_summary_dates_and_model(plan):
    summary = build_planning_summary(plan)

    assert summary["forecast_date"] == "2025-12-01"
    assert summary["inventory_snapshot_date"] == "2025-11-30"
    assert summary["forecast_model"] == "gradient_boosting_v1"
    assert summary["forecast_horizon_days"] == 1


def test_summary_quantities(plan):
    summary = build_planning_summary(plan)

    assert summary["store_product_count"] == 100
    assert summary["total_suggested_order_units"] == 4108
    assert summary["total_available_stock_units"] == int(
        plan["closing_stock"].sum()
    )
    assert summary["total_target_stock_units"] == int(
        plan["target_stock_units"].sum()
    )


def test_summary_status_counts(plan):
    summary = build_planning_summary(plan)

    assert summary["inventory_status_counts"] == {
        "STOCKOUT": 15,
        "BELOW_LEAD_TIME_COVERAGE": 27,
        "REPLENISHMENT_SUGGESTED": 17,
        "SUFFICIENT_FOR_POLICY": 41,
    }

    assert sum(
        summary["inventory_status_counts"].values()
    ) == 100

    assert list(summary["inventory_status_counts"]) == STATUS_ORDER


def test_summary_limitations(plan):
    summary = build_planning_summary(plan)

    assert summary["data_source"] == "synthetic"
    assert summary["open_purchase_orders_included"] is False
    assert len(summary["limitations"]) >= 5
    assert any(
        "multi-day" in limitation
        for limitation in summary["limitations"]
    )


def test_export_creates_three_files(plan, tmp_path):
    paths = export_inventory_plan(plan, output_dir=tmp_path)

    assert set(paths) == {"csv", "json", "summary"}
    assert all(path.is_file() for path in paths.values())


def test_export_filenames(plan, tmp_path):
    paths = export_inventory_plan(plan, output_dir=tmp_path)

    assert paths["csv"].name == (
        "inventory_recommendations_2025-12-01.csv"
    )
    assert paths["json"].name == (
        "inventory_recommendations_2025-12-01.json"
    )
    assert paths["summary"].name == (
        "inventory_planning_summary_2025-12-01.json"
    )


def test_csv_record_count_and_order_total(plan, tmp_path):
    paths = export_inventory_plan(plan, output_dir=tmp_path)
    csv_data = pd.read_csv(paths["csv"])

    assert len(csv_data) == 100
    assert csv_data["suggested_order_units"].sum() == 4108


def test_json_record_count_and_dates(plan, tmp_path):
    paths = export_inventory_plan(plan, output_dir=tmp_path)

    records = json.loads(
        paths["json"].read_text(encoding="utf-8")
    )

    assert len(records) == 100
    assert all(record["date"] == "2025-12-01" for record in records)
    assert all(
        record["inventory_snapshot_date"] == "2025-11-30"
        for record in records
    )


def test_json_matches_csv_order_totals(plan, tmp_path):
    paths = export_inventory_plan(plan, output_dir=tmp_path)

    csv_data = pd.read_csv(paths["csv"])
    records = json.loads(
        paths["json"].read_text(encoding="utf-8")
    )

    assert sum(
        record["suggested_order_units"]
        for record in records
    ) == csv_data["suggested_order_units"].sum()


def test_summary_json_matches_records(plan, tmp_path):
    paths = export_inventory_plan(plan, output_dir=tmp_path)

    records = json.loads(
        paths["json"].read_text(encoding="utf-8")
    )
    summary = json.loads(
        paths["summary"].read_text(encoding="utf-8")
    )

    assert summary["store_product_count"] == len(records)
    assert summary["total_suggested_order_units"] == sum(
        record["suggested_order_units"]
        for record in records
    )


def test_json_is_strictly_serializable(plan, tmp_path):
    paths = export_inventory_plan(plan, output_dir=tmp_path)

    records = json.loads(
        paths["json"].read_text(encoding="utf-8"),
        parse_constant=lambda value: pytest.fail(
            f"Nonstandard JSON constant: {value}"
        ),
    )

    assert len(records) == 100


def test_infinite_coverage_exports_as_null(plan, tmp_path):
    modified = plan.copy(deep=True)
    modified.loc[0, "coverage_days"] = np.inf

    paths = export_inventory_plan(modified, output_dir=tmp_path)

    records = json.loads(
        paths["json"].read_text(encoding="utf-8")
    )

    assert records[0]["coverage_days"] is None


def test_export_does_not_mutate_plan(plan, tmp_path):
    original = plan.copy(deep=True)

    export_inventory_plan(plan, output_dir=tmp_path)

    pd.testing.assert_frame_equal(plan, original)


def test_export_is_byte_reproducible(plan, tmp_path):
    paths = export_inventory_plan(plan, output_dir=tmp_path)

    before = {
        name: path.read_bytes()
        for name, path in paths.items()
    }

    export_inventory_plan(plan, output_dir=tmp_path)

    after = {
        name: path.read_bytes()
        for name, path in paths.items()
    }

    assert before == after