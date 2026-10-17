import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.forecast_baselines import (
    BASELINES,
    calculate_metrics,
    calculate_training_scales,
    evaluate_baselines,
    load_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "validation"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def metrics():
    return pd.read_csv(REPORT_DIR / "baseline_metrics.csv")


@pytest.fixture(scope="module")
def predictions():
    return pd.read_csv(
        REPORT_DIR / "baseline_predictions.csv",
        parse_dates=["date"],
    )


def test_baseline_metric_files_exist():
    for name in (
        "baseline_metrics.csv",
        "baseline_predictions.csv",
        "baseline_summary.json",
    ):
        assert (REPORT_DIR / name).is_file()


def test_metric_calculation():
    result = calculate_metrics(
        actual=[10, 20],
        predicted=[12, 18],
        scales=[4, 4],
    )

    assert result["wape_pct"] == pytest.approx(13.3333)
    assert result["rmsse"] == pytest.approx(1.0)
    assert result["bias_pct"] == pytest.approx(0.0)


def test_positive_bias_means_overforecast():
    result = calculate_metrics(
        actual=[10, 20],
        predicted=[12, 22],
        scales=[4, 4],
    )

    assert result["bias_pct"] > 0


@pytest.mark.parametrize(
    ("actual", "predicted", "scales"),
    [
        ([1, 2], [1], [1, 1]),
        ([], [], []),
        ([0, 0], [1, 1], [1, 1]),
        ([1, 2], [1, 2], [0, 1]),
        ([1, 2], [1, np.nan], [1, 1]),
    ],
)
def test_invalid_metric_inputs(actual, predicted, scales):
    with pytest.raises(ValueError):
        calculate_metrics(actual, predicted, scales)


def test_training_scales_cover_all_series(dataset):
    scales = calculate_training_scales(dataset)

    assert len(scales) == 100
    assert scales["training_scale"].notna().all()
    assert (scales["training_scale"] > 0).all()


def test_training_scales_ignore_validation_and_test(dataset):
    original = calculate_training_scales(dataset)

    modified = dataset.copy()
    future = modified["split"] != "train"
    modified.loc[future, "units_sold"] = 999999

    recalculated = calculate_training_scales(modified)

    pd.testing.assert_frame_equal(original, recalculated)


def test_validation_only_metrics(metrics):
    assert len(metrics) == 3
    assert set(metrics["model"]) == set(BASELINES)
    assert (metrics["split"] == "validation").all()
    assert (metrics["observations"] == 9200).all()


def test_prediction_row_counts(predictions):
    assert len(predictions) == 27600

    counts = predictions.groupby("model").size()

    assert set(counts.index) == set(BASELINES)
    assert (counts == 9200).all()


def test_predictions_are_validation_only(predictions):
    assert predictions["date"].min() == pd.Timestamp(
        "2025-06-01"
    )
    assert predictions["date"].max() == pd.Timestamp(
        "2025-08-31"
    )


def test_prediction_errors_reconcile(predictions):
    expected = (
        predictions["prediction"] - predictions["units_sold"]
    )

    np.testing.assert_allclose(
        predictions["error"].to_numpy(),
        expected.to_numpy(),
    )


def test_rolling_mean_is_best_baseline(metrics):
    best = metrics.sort_values("wape_pct").iloc[0]

    assert best["model"] == "rolling_mean_7"
    assert best["wape_pct"] == pytest.approx(33.5911)


def test_summary_metadata():
    path = REPORT_DIR / "baseline_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))

    assert summary["data_source"] == "synthetic"
    assert summary["evaluation_split"] == "validation"
    assert summary["forecast_horizon_days"] == 1
    assert summary["best_baseline_by_wape"] == "rolling_mean_7"


def test_recomputed_results_match_export(dataset, metrics):
    recomputed, _ = evaluate_baselines(dataset)

    actual = metrics.sort_values("model").reset_index(drop=True)
    expected = recomputed.sort_values("model").reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
    )