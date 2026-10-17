import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.forecast_baselines import (
    SERIES_KEYS,
    calculate_metrics,
    calculate_training_scales,
    load_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "test"
SELECTION_PATH = ROOT / "models" / "champion_selection.json"

CHAMPION = "gradient_boosting_v1"
BASELINE = "rolling_mean_7"


@pytest.fixture(scope="module")
def metrics():
    return pd.read_csv(
        REPORT_DIR / "holdout_metrics.csv"
    )


@pytest.fixture(scope="module")
def predictions():
    return pd.read_csv(
        REPORT_DIR / "holdout_predictions.csv",
        parse_dates=["date"],
    )


@pytest.fixture(scope="module")
def summary():
    return json.loads(
        (REPORT_DIR / "holdout_summary.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def selection():
    return json.loads(
        SELECTION_PATH.read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.mark.parametrize(
    "filename",
    [
        "holdout_metrics.csv",
        "holdout_predictions.csv",
        "holdout_summary.json",
    ],
)
def test_holdout_artifact_exists(filename):
    assert (REPORT_DIR / filename).is_file()


def test_metric_models_and_split(metrics):
    assert len(metrics) == 2
    assert set(metrics["model"]) == {CHAMPION, BASELINE}
    assert (metrics["split"] == "test").all()
    assert (metrics["observations"] == 9100).all()


def test_prediction_row_counts(predictions):
    assert len(predictions) == 18200

    counts = predictions.groupby("model").size()

    assert counts[CHAMPION] == 9100
    assert counts[BASELINE] == 9100


def test_prediction_date_boundaries(predictions):
    assert predictions["date"].min() == pd.Timestamp(
        "2025-09-01"
    )
    assert predictions["date"].max() == pd.Timestamp(
        "2025-11-30"
    )


def test_prediction_keys_unique_per_model(predictions):
    assert not predictions.duplicated(
        ["model", "date", *SERIES_KEYS]
    ).any()


def test_predictions_are_finite_and_nonnegative(predictions):
    values = predictions["prediction"].to_numpy()

    assert np.isfinite(values).all()
    assert (values >= 0).all()


def test_prediction_errors_reconcile(predictions):
    np.testing.assert_allclose(
        predictions["error"],
        predictions["prediction"] - predictions["units_sold"],
    )


def test_holdout_targets_match_dataset(dataset, predictions):
    test = dataset.loc[
        dataset["split"] == "test",
        ["date", *SERIES_KEYS, "units_sold"],
    ]

    for model_name in (CHAMPION, BASELINE):
        subset = predictions.loc[
            predictions["model"] == model_name
        ]

        joined = subset.merge(
            test,
            on=["date", *SERIES_KEYS],
            how="left",
            validate="one_to_one",
            suffixes=("_saved", "_source"),
        )

        assert len(joined) == 9100

        np.testing.assert_allclose(
            joined["units_sold_saved"],
            joined["units_sold_source"],
        )


def test_baseline_predictions_match_lagged_feature(
    dataset,
    predictions,
):
    test = dataset.loc[
        dataset["split"] == "test",
        ["date", *SERIES_KEYS, "sales_rolling_mean_7"],
    ]

    baseline = predictions.loc[
        predictions["model"] == BASELINE
    ]

    joined = baseline.merge(
        test,
        on=["date", *SERIES_KEYS],
        validate="one_to_one",
    )

    np.testing.assert_allclose(
        joined["prediction"],
        joined["sales_rolling_mean_7"],
    )


def test_saved_metrics_reconcile(dataset, predictions, metrics):
    scales = calculate_training_scales(dataset)

    for model_name in (CHAMPION, BASELINE):
        subset = predictions.loc[
            predictions["model"] == model_name
        ]

        evaluation = subset.merge(
            scales,
            on=SERIES_KEYS,
            how="left",
            validate="many_to_one",
        )

        assert evaluation["training_scale"].notna().all()

        calculated = calculate_metrics(
            evaluation["units_sold"],
            evaluation["prediction"],
            evaluation["training_scale"],
        )

        saved = metrics.loc[
            metrics["model"] == model_name
        ].iloc[0]

        for metric_name, value in calculated.items():
            assert saved[metric_name] == pytest.approx(value)


def test_champion_beats_baseline_on_test(metrics):
    champion = metrics.loc[
        metrics["model"] == CHAMPION
    ].iloc[0]

    baseline = metrics.loc[
        metrics["model"] == BASELINE
    ].iloc[0]

    assert champion["wape_pct"] < baseline["wape_pct"]
    assert champion["rmsse"] < baseline["rmsse"]


def test_summary_preserves_frozen_selection(summary, selection):
    assert summary["data_source"] == "synthetic"
    assert summary["champion_model"] == selection["champion_model"]
    assert summary["champion_frozen_before_test"] is True
    assert summary["retraining_performed"] is False
    assert summary["hyperparameter_tuning_performed"] is False
    assert summary["test_used_for_model_selection"] is False
    assert summary["test_period"] == selection["reserved_test_period"]


def test_summary_protocol(summary):
    assert summary["forecast_horizon_days"] == 1
    assert (
        summary["evaluation_protocol"]
        == "rolling_one_day_ahead"
    )
    assert summary["rmsse_scale_source"] == "training_only"
    assert summary["test_observations_per_model"] == 9100


def test_summary_metrics_match_csv(summary, metrics):
    for model_name, summary_key in (
        (CHAMPION, "champion_test_metrics"),
        (BASELINE, "baseline_test_metrics"),
    ):
        saved = metrics.loc[
            metrics["model"] == model_name
        ].iloc[0]

        recorded = summary[summary_key]

        for metric_name in (
            "wape_pct",
            "rmsse",
            "bias_pct",
        ):
            assert recorded[metric_name] == pytest.approx(
                saved[metric_name]
            )


def test_validation_to_test_change_reconciles(
    summary,
    selection,
):
    validation_wape = selection[
        "validation_metrics"
    ]["wape_pct"]

    test_wape = summary["champion_test_wape_pct"]

    assert summary["champion_validation_wape_pct"] == pytest.approx(
        validation_wape
    )

    assert summary[
        "validation_to_test_wape_change_points"
    ] == pytest.approx(
        round(test_wape - validation_wape, 4)
    )