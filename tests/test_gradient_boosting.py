import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.forecast_baselines import (
    SERIES_KEYS,
    calculate_metrics,
    calculate_training_scales,
    load_dataset,
)
from src.models.train_gradient_boosting import MODEL_NAME
from src.models.train_random_forest import prepare_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / f"{MODEL_NAME}.joblib"
REPORT_DIR = ROOT / "reports" / "validation"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def artifact():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def predictions():
    return pd.read_csv(
        REPORT_DIR / f"{MODEL_NAME}_predictions.csv",
        parse_dates=["date"],
    )


@pytest.fixture(scope="module")
def metrics():
    return pd.read_csv(
        REPORT_DIR / f"{MODEL_NAME}_metrics.csv"
    ).iloc[0]


def test_model_artifact_exists():
    assert MODEL_PATH.is_file()


@pytest.mark.parametrize(
    "filename",
    [
        "gradient_boosting_v1_metrics.csv",
        "gradient_boosting_v1_predictions.csv",
        "gradient_boosting_v1_summary.json",
        "phase5_model_comparison.csv",
    ],
)
def test_report_exists(filename):
    assert (REPORT_DIR / filename).is_file()


def test_model_metadata(artifact):
    assert artifact["model_name"] == MODEL_NAME
    assert artifact["data_source"] == "synthetic"
    assert artifact["forecast_horizon_days"] == 1
    assert artifact["training_split"] == "train"
    assert artifact["evaluation_split"] == "validation"


def test_feature_schema_excludes_target(artifact):
    columns = artifact["feature_columns"]

    assert len(columns) == len(set(columns))
    assert "units_sold" not in columns
    assert "date" not in columns
    assert "split" not in columns

    assert set(artifact["numeric_features"]).issubset(columns)

    for feature in artifact["categorical_features"]:
        assert any(
            name.startswith(f"{feature}_")
            for name in columns
        )


def test_validation_predictions_only(predictions):
    assert len(predictions) == 9200
    assert (predictions["model"] == MODEL_NAME).all()
    assert predictions["date"].min() == pd.Timestamp(
        "2025-06-01"
    )
    assert predictions["date"].max() == pd.Timestamp(
        "2025-08-31"
    )


def test_prediction_keys_unique(predictions):
    assert not predictions.duplicated(
        ["date", *SERIES_KEYS]
    ).any()


def test_predictions_nonnegative_and_finite(predictions):
    values = predictions["prediction"].to_numpy()

    assert np.isfinite(values).all()
    assert (values >= 0).all()


def test_prediction_errors_reconcile(predictions):
    np.testing.assert_allclose(
        predictions["error"].to_numpy(),
        (
            predictions["prediction"]
            - predictions["units_sold"]
        ).to_numpy(),
    )


def test_saved_model_reproduces_predictions(
    dataset,
    artifact,
    predictions,
):
    train = dataset.loc[dataset["split"] == "train"]
    validation = dataset.loc[
        dataset["split"] == "validation"
    ]

    _, x_validation = prepare_features(
        train,
        validation,
    )

    assert (
        x_validation.columns.tolist()
        == artifact["feature_columns"]
    )

    reproduced = np.maximum(
        artifact["model"].predict(x_validation),
        0.0,
    )

    np.testing.assert_allclose(
        reproduced,
        predictions["prediction"].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_metrics_reconcile(dataset, predictions, metrics):
    scales = calculate_training_scales(dataset)

    evaluation = predictions.merge(
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

    for name, value in calculated.items():
        assert float(metrics[name]) == pytest.approx(value)


def test_gradient_boosting_beats_random_forest(metrics):
    forest = pd.read_csv(
        REPORT_DIR / "random_forest_v1_metrics.csv"
    ).iloc[0]

    assert metrics["wape_pct"] < forest["wape_pct"]
    assert metrics["rmsse"] < forest["rmsse"]


def test_comparison_matches_metric_files():
    comparison = pd.read_csv(
        REPORT_DIR / "phase5_model_comparison.csv"
    )

    assert set(comparison["model"]) == {
        "rolling_mean_7",
        "random_forest_v1",
        "gradient_boosting_v1",
    }

    assert (comparison["split"] == "validation").all()
    assert (comparison["observations"] == 9200).all()

    best = comparison.sort_values("wape_pct").iloc[0]
    assert best["model"] == MODEL_NAME

    for name in ("random_forest_v1", MODEL_NAME):
        saved = pd.read_csv(
            REPORT_DIR / f"{name}_metrics.csv"
        ).iloc[0]

        row = comparison.loc[
            comparison["model"] == name
        ].iloc[0]

        for metric in ("wape_pct", "rmsse", "bias_pct"):
            assert row[metric] == pytest.approx(
                saved[metric]
            )


def test_summary_metadata():
    path = REPORT_DIR / f"{MODEL_NAME}_summary.json"
    summary = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert summary["data_source"] == "synthetic"
    assert summary["evaluation_split"] == "validation"
    assert summary["forecast_horizon_days"] == 1
    assert summary["test_split_used"] is False
    assert summary["beats_baseline"] is True
    assert summary["beats_random_forest"] is True
    assert (
        summary["best_model_by_validation_wape"]
        == MODEL_NAME
    )