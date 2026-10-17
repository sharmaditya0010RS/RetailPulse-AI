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
from src.models.train_random_forest import (
    BASELINE_WAPE,
    CATEGORICAL_FEATURES,
    MODEL_NAME,
    NUMERIC_FEATURES,
    prepare_features,
)

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


def test_reporting_artifacts_exist():
    for suffix in (
        "metrics.csv",
        "predictions.csv",
        "feature_importance.csv",
        "summary.json",
    ):
        assert (REPORT_DIR / f"{MODEL_NAME}_{suffix}").is_file()


def test_model_metadata(artifact):
    assert artifact["model_name"] == MODEL_NAME
    assert artifact["data_source"] == "synthetic"
    assert artifact["forecast_horizon_days"] == 1


def test_feature_schema(artifact):
    columns = artifact["feature_columns"]

    assert len(columns) == len(set(columns))
    assert set(NUMERIC_FEATURES).issubset(columns)
    assert all("units_sold" != name for name in columns)
    assert all("split" != name for name in columns)
    assert all("date" != name for name in columns)

    for feature in CATEGORICAL_FEATURES:
        assert any(
            name.startswith(f"{feature}_")
            for name in columns
        )


def test_training_preprocessing_schema(dataset, artifact):
    train = dataset.loc[dataset["split"] == "train"]
    validation = dataset.loc[
        dataset["split"] == "validation"
    ]

    x_train, x_validation = prepare_features(
        train,
        validation,
    )

    assert x_train.columns.tolist() == artifact["feature_columns"]
    assert x_validation.columns.tolist() == artifact["feature_columns"]


def test_predictions_are_validation_only(predictions):
    assert len(predictions) == 9200
    assert predictions["date"].min() == pd.Timestamp(
        "2025-06-01"
    )
    assert predictions["date"].max() == pd.Timestamp(
        "2025-08-31"
    )
    assert (predictions["model"] == MODEL_NAME).all()


def test_predictions_are_finite_and_nonnegative(predictions):
    values = predictions["prediction"].to_numpy()

    assert np.isfinite(values).all()
    assert (values >= 0).all()


def test_prediction_keys_are_unique(predictions):
    assert not predictions.duplicated(
        ["date", *SERIES_KEYS]
    ).any()


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
    validation = dataset.loc[
        dataset["split"] == "validation"
    ].copy()

    x_validation = pd.get_dummies(
        validation[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        dtype=int,
    ).reindex(
        columns=artifact["feature_columns"],
        fill_value=0,
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

    result = calculate_metrics(
        evaluation["units_sold"],
        evaluation["prediction"],
        evaluation["training_scale"],
    )

    for name, value in result.items():
        assert float(metrics[name]) == pytest.approx(value)


def test_model_beats_validation_baseline(metrics):
    assert metrics["model"] == MODEL_NAME
    assert metrics["split"] == "validation"
    assert metrics["observations"] == 9200
    assert metrics["wape_pct"] < BASELINE_WAPE


def test_feature_importance():
    importance = pd.read_csv(
        REPORT_DIR / f"{MODEL_NAME}_feature_importance.csv"
    )

    assert not importance["feature"].duplicated().any()
    assert (importance["importance"] >= 0).all()
    assert importance["importance"].sum() == pytest.approx(1.0)


def test_summary_metadata():
    path = REPORT_DIR / f"{MODEL_NAME}_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))

    assert summary["data_source"] == "synthetic"
    assert summary["evaluation_split"] == "validation"
    assert summary["test_split_used"] is False
    assert summary["beats_baseline"] is True
    assert summary["baseline_wape_pct"] == BASELINE_WAPE