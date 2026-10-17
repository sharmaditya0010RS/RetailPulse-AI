"""Tests for read-only dashboard artifact loading."""

import json
from copy import deepcopy

import pandas as pd
import pytest

from src.dashboard.data_loader import (
    CSV_ARTIFACTS,
    EXPECTED_CHAMPION,
    JSON_ARTIFACTS,
    ROOT,
    load_csv_artifact,
    load_dashboard_data,
    load_json_artifact,
    resolve_artifact,
    validate_dashboard_provenance,
)


@pytest.fixture(scope="module")
def real_datasets():
    """Load existing saved artifacts without modifying them."""
    return load_dashboard_data()


def test_all_expected_csv_artifacts_registered():
    assert len(CSV_ARTIFACTS) == 9


def test_all_expected_json_artifacts_registered():
    assert len(JSON_ARTIFACTS) == 4


def test_all_artifacts_exist():
    for relative_path in [
        *CSV_ARTIFACTS.values(),
        *JSON_ARTIFACTS.values(),
    ]:
        assert resolve_artifact(relative_path).is_file()


def test_resolve_artifact_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="escapes repository"):
        resolve_artifact("../outside.csv", root=tmp_path)


def test_resolve_artifact_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing dashboard"):
        resolve_artifact("missing.csv", root=tmp_path)


def test_unknown_csv_artifact_rejected():
    with pytest.raises(ValueError, match="Unknown CSV"):
        load_csv_artifact("unknown")


def test_unknown_json_artifact_rejected():
    with pytest.raises(ValueError, match="Unknown JSON"):
        load_json_artifact("unknown")


def test_empty_csv_rejected(tmp_path):
    path = tmp_path / CSV_ARTIFACTS["executive_kpis"]
    path.parent.mkdir(parents=True)
    path.write_text("date_start,date_end\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_csv_artifact("executive_kpis", root=tmp_path)


def test_missing_csv_columns_rejected(tmp_path):
    path = tmp_path / CSV_ARTIFACTS["executive_kpis"]
    path.parent.mkdir(parents=True)
    path.write_text(
        "date_start,date_end\n2024-01-01,2025-11-30\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing columns"):
        load_csv_artifact("executive_kpis", root=tmp_path)


def test_invalid_csv_date_rejected(tmp_path):
    path = tmp_path / CSV_ARTIFACTS["monthly_performance"]
    path.parent.mkdir(parents=True)

    pd.DataFrame(
        {
            "sales_month": ["not-a-date"],
            "units_sold": [10],
            "lost_units": [1],
            "net_revenue": [100.0],
            "estimated_gross_profit": [40.0],
            "revenue_growth_mom_pct": [0.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_csv_artifact("monthly_performance", root=tmp_path)


def test_missing_csv_date_rejected(tmp_path):
    path = tmp_path / CSV_ARTIFACTS["monthly_performance"]
    path.parent.mkdir(parents=True)

    pd.DataFrame(
        {
            "sales_month": [None],
            "units_sold": [10],
            "lost_units": [1],
            "net_revenue": [100.0],
            "estimated_gross_profit": [40.0],
            "revenue_growth_mom_pct": [0.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing dates"):
        load_csv_artifact("monthly_performance", root=tmp_path)


def test_invalid_json_rejected(tmp_path):
    path = tmp_path / JSON_ARTIFACTS["executive_scorecard"]
    path.parent.mkdir(parents=True)
    path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_json_artifact("executive_scorecard", root=tmp_path)


def test_json_array_rejected(tmp_path):
    path = tmp_path / JSON_ARTIFACTS["executive_scorecard"]
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be an object"):
        load_json_artifact("executive_scorecard", root=tmp_path)


@pytest.mark.parametrize(
    ("name", "expected_rows"),
    [
        ("executive_kpis", 1),
        ("monthly_performance", 23),
        ("store_performance", 10),
        ("category_performance", 5),
        ("product_stockouts", 10),
        ("holdout_metrics", 2),
        ("holdout_predictions", 18200),
        ("permutation_importance", 12),
        ("inventory_recommendations", 100),
    ],
)
def test_saved_csv_row_counts(name, expected_rows, real_datasets):
    assert len(real_datasets[name]) == expected_rows


def test_dates_parsed_as_datetime(real_datasets):
    for name, column in [
        ("executive_kpis", "date_start"),
        ("executive_kpis", "date_end"),
        ("monthly_performance", "sales_month"),
        ("holdout_predictions", "date"),
        ("inventory_recommendations", "date"),
        ("inventory_recommendations", "inventory_snapshot_date"),
    ]:
        assert pd.api.types.is_datetime64_any_dtype(
            real_datasets[name][column]
        )


def test_csv_loads_are_independent():
    first = load_csv_artifact("executive_kpis")
    original_revenue = first.loc[0, "net_revenue"]

    first.loc[0, "net_revenue"] = -1

    second = load_csv_artifact("executive_kpis")
    assert second.loc[0, "net_revenue"] == original_revenue


def test_json_loads_are_independent():
    first = load_json_artifact("executive_scorecard")
    original_revenue = first["net_revenue"]

    first["net_revenue"] = -1

    second = load_json_artifact("executive_scorecard")
    assert second["net_revenue"] == original_revenue


def test_champion_matches_frozen_selection(real_datasets):
    assert (
        real_datasets["champion_selection"]["champion_model"]
        == EXPECTED_CHAMPION
    )
    assert (
        real_datasets["holdout_summary"]["champion_model"]
        == EXPECTED_CHAMPION
    )
    assert (
        real_datasets["planning_summary"]["forecast_model"]
        == EXPECTED_CHAMPION
    )


def test_holdout_was_not_used_for_selection(real_datasets):
    assert (
        real_datasets["holdout_summary"][
            "test_used_for_model_selection"
        ]
        is False
    )


def test_planning_excludes_open_purchase_orders(real_datasets):
    assert (
        real_datasets["planning_summary"][
            "open_purchase_orders_included"
        ]
        is False
    )
    assert not real_datasets["inventory_recommendations"][
        "open_purchase_orders_included"
    ].any()


def test_dashboard_provenance_passes(real_datasets):
    validate_dashboard_provenance(real_datasets)


def test_unexpected_champion_rejected(real_datasets):
    changed = deepcopy(real_datasets)
    changed["champion_selection"]["champion_model"] = "unexpected"

    with pytest.raises(ValueError, match="Unexpected frozen champion"):
        validate_dashboard_provenance(changed)


def test_holdout_selection_leakage_rejected(real_datasets):
    changed = deepcopy(real_datasets)
    changed["holdout_summary"]["test_used_for_model_selection"] = True

    with pytest.raises(ValueError, match="Holdout selection provenance"):
        validate_dashboard_provenance(changed)


def test_wrong_planning_model_rejected(real_datasets):
    changed = deepcopy(real_datasets)
    changed["planning_summary"]["forecast_model"] = "unexpected"

    with pytest.raises(ValueError, match="Planning model"):
        validate_dashboard_provenance(changed)


def test_wrong_planning_record_count_rejected(real_datasets):
    changed = deepcopy(real_datasets)
    changed["planning_summary"]["store_product_count"] = 99

    with pytest.raises(ValueError, match="record count"):
        validate_dashboard_provenance(changed)


def test_unexpected_holdout_models_rejected(real_datasets):
    changed = deepcopy(real_datasets)
    changed["holdout_metrics"].loc[0, "model"] = "unexpected"

    with pytest.raises(ValueError, match="Unexpected holdout"):
        validate_dashboard_provenance(changed)


def test_nonsynthetic_source_rejected(real_datasets):
    changed = deepcopy(real_datasets)
    changed["executive_scorecard"]["data_source"] = "production"

    with pytest.raises(ValueError, match="must be synthetic"):
        validate_dashboard_provenance(changed)


def test_root_points_to_repository():
    assert (ROOT / "src" / "dashboard" / "data_loader.py").is_file()