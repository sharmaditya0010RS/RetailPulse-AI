import json

import numpy as np
import pandas as pd
import pytest

from src.analytics.forecast_error_analysis import (
    BASELINE,
    CHAMPION,
    GROUPINGS,
    OUTPUT_DIR,
    calculate_champion_comparison,
    calculate_group_metrics,
    load_analysis_data,
    validate_global_reconciliation,
)


@pytest.fixture
def sample_predictions():
    return pd.DataFrame(
        {
            "model": [
                CHAMPION,
                CHAMPION,
                BASELINE,
                BASELINE,
            ],
            "store_id": ["S001"] * 4,
            "units_sold": [10, 20, 10, 20],
            "prediction": [12, 18, 15, 15],
            "absolute_error": [2, 2, 5, 5],
            "signed_error": [2, -2, 5, -5],
            "squared_scaled_error": [1, 1, 6.25, 6.25],
        }
    )


def test_group_metrics_known_values(sample_predictions):
    metrics = calculate_group_metrics(
        sample_predictions,
        ["store_id"],
    )

    champion = metrics.loc[
        metrics["model"] == CHAMPION
    ].iloc[0]

    assert champion["observations"] == 2
    assert champion["actual_units"] == 30
    assert champion["predicted_units"] == pytest.approx(30)
    assert champion["wape_pct"] == pytest.approx(100 * 4 / 30)
    assert champion["rmsse"] == pytest.approx(1)
    assert champion["bias_pct"] == pytest.approx(0)


def test_baseline_metrics_known_values(sample_predictions):
    metrics = calculate_group_metrics(
        sample_predictions,
        ["store_id"],
    )

    baseline = metrics.loc[
        metrics["model"] == BASELINE
    ].iloc[0]

    assert baseline["wape_pct"] == pytest.approx(100 * 10 / 30)
    assert baseline["rmsse"] == pytest.approx(2.5)
    assert baseline["bias_pct"] == pytest.approx(0)


def test_comparison_known_improvement(sample_predictions):
    metrics = calculate_group_metrics(
        sample_predictions,
        ["store_id"],
    )

    comparison = calculate_champion_comparison(
        metrics,
        ["store_id"],
    )

    assert len(comparison) == 1
    assert comparison.iloc[0]["champion_wins"]
    assert comparison.iloc[0][
        "wape_improvement_pp"
    ] == pytest.approx(20.0)


def test_group_metrics_rejects_empty_frame(sample_predictions):
    with pytest.raises(ValueError, match="empty"):
        calculate_group_metrics(
            sample_predictions.iloc[:0],
            ["store_id"],
        )


def test_group_metrics_rejects_missing_column(sample_predictions):
    with pytest.raises(ValueError, match="Missing analysis columns"):
        calculate_group_metrics(
            sample_predictions.drop(columns=["absolute_error"]),
            ["store_id"],
        )


def test_group_metrics_rejects_zero_actual_units(sample_predictions):
    modified = sample_predictions.copy()
    modified["units_sold"] = 0

    with pytest.raises(ValueError, match="positive"):
        calculate_group_metrics(
            modified,
            ["store_id"],
        )


def test_comparison_rejects_missing_baseline(sample_predictions):
    metrics = calculate_group_metrics(
        sample_predictions,
        ["store_id"],
    )

    metrics = metrics.loc[metrics["model"] == CHAMPION]

    with pytest.raises(ValueError, match="Missing baseline"):
        calculate_champion_comparison(
            metrics,
            ["store_id"],
        )


def test_saved_predictions_reconcile_with_holdout():
    predictions = load_analysis_data()

    assert len(predictions) == 18200
    assert set(predictions["model"]) == {CHAMPION, BASELINE}

    validate_global_reconciliation(predictions)


@pytest.mark.parametrize(
    ("grouping", "expected_groups"),
    [
        ("store", 10),
        ("product", 10),
        ("day_of_week", 7),
    ],
)
def test_saved_segment_reports(grouping, expected_groups):
    metrics = pd.read_csv(
        OUTPUT_DIR / f"{grouping}_metrics.csv"
    )
    comparison = pd.read_csv(
        OUTPUT_DIR / f"{grouping}_comparison.csv"
    )

    assert len(metrics) == expected_groups * 2
    assert len(comparison) == expected_groups
    assert set(metrics["model"]) == {CHAMPION, BASELINE}
    assert comparison["champion_wins"].all()

    for column in (
        "wape_pct_champion",
        "wape_pct_baseline",
        "wape_improvement_pp",
    ):
        assert np.isfinite(comparison[column]).all()

    np.testing.assert_allclose(
        comparison["wape_improvement_pp"].to_numpy(),
        (
            comparison["wape_pct_baseline"]
            - comparison["wape_pct_champion"]
        ).to_numpy(),
        rtol=0,
        atol=1e-10,
    )


def test_saved_summary_matches_reports():
    summary_path = OUTPUT_DIR / "error_analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["data_source"] == "synthetic"
    assert summary["champion_model"] == CHAMPION
    assert summary["baseline_model"] == BASELINE
    assert summary["test_rows_per_model"] == 9100
    assert summary["model_retrained"] is False
    assert summary["test_used_for_model_selection"] is False

    for grouping in GROUPINGS:
        comparison = pd.read_csv(
            OUTPUT_DIR / f"{grouping}_comparison.csv"
        )
        details = summary["groupings"][grouping]

        assert details["groups"] == len(comparison)
        assert details["champion_winning_groups"] == int(
            comparison["champion_wins"].sum()
        )
        assert details["champion_losing_or_tied_groups"] == int(
            (~comparison["champion_wins"]).sum()
        )