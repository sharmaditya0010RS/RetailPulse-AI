"""Validated inference for the frozen RetailPulse forecasting champion."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SELECTION_PATH = ROOT / "models" / "champion_selection.json"
MODEL_PATH = ROOT / "models" / "gradient_boosting_v1.joblib"

EXPECTED_CHAMPION = "gradient_boosting_v1"

LAG_FEATURES = (
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
)

ROLLING_FEATURES = (
    "sales_rolling_mean_7",
    "sales_rolling_mean_14",
    "sales_rolling_mean_28",
)

CALENDAR_FEATURES = (
    "day_of_week",
    "month",
    "weekend_flag",
)

EXPECTED_NUMERIC_FEATURES = [
    *LAG_FEATURES,
    *ROLLING_FEATURES,
    *CALENDAR_FEATURES,
]

EXPECTED_CATEGORICAL_FEATURES = [
    "store_id",
    "product_id",
]


def load_champion():
    """Load the frozen selection and its saved trained model."""
    if not SELECTION_PATH.is_file():
        raise FileNotFoundError(
            f"Champion selection not found: {SELECTION_PATH}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Champion model not found: {MODEL_PATH}"
        )

    with SELECTION_PATH.open("r", encoding="utf-8") as file:
        selection = json.load(file)

    if selection.get("champion_model") != EXPECTED_CHAMPION:
        raise ValueError(
            "Frozen champion does not match the expected model."
        )

    artifact = joblib.load(MODEL_PATH)

    if artifact.get("model_name") != EXPECTED_CHAMPION:
        raise ValueError("Saved model identity mismatch.")

    if artifact.get("data_source") != "synthetic":
        raise ValueError("Unexpected model data source.")

    if artifact.get("numeric_features") != EXPECTED_NUMERIC_FEATURES:
        raise ValueError("Unexpected numeric feature schema.")

    if (
        artifact.get("categorical_features")
        != EXPECTED_CATEGORICAL_FEATURES
    ):
        raise ValueError("Unexpected categorical feature schema.")

    feature_columns = artifact.get("feature_columns")

    if not isinstance(feature_columns, list):
        raise ValueError("Encoded feature schema must be a list.")

    if len(feature_columns) != 30:
        raise ValueError("Unexpected encoded feature count.")

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Duplicate encoded feature names.")

    if not hasattr(artifact.get("model"), "predict"):
        raise ValueError("Saved artifact has no prediction method.")

    return artifact


def validate_forecast_inputs(frame, artifact):
    """Validate raw input features without changing the caller's data."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Forecast inputs must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError("Forecast input data cannot be empty.")

    numeric = artifact["numeric_features"]
    categorical = artifact["categorical_features"]

    required = [*numeric, *categorical]
    missing = [column for column in required if column not in frame]

    if missing:
        raise ValueError(f"Missing forecast features: {missing}")

    validated = frame.copy(deep=True)

    for column in numeric:
        converted = pd.to_numeric(
            validated[column],
            errors="coerce",
        )

        if not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Invalid or missing numeric values in {column}."
            )

        validated[column] = converted

    for column in (*LAG_FEATURES, *ROLLING_FEATURES):
        if (validated[column] < 0).any():
            raise ValueError(
                f"Negative demand-history values in {column}."
            )

    for column, minimum, maximum in (
        ("day_of_week", 0, 6),
        ("month", 1, 12),
        ("weekend_flag", 0, 1),
    ):
        values = validated[column]

        if (
            (values < minimum).any()
            or (values > maximum).any()
            or (values != np.floor(values)).any()
        ):
            raise ValueError(
                f"Invalid calendar values in {column}."
            )

    expected_weekend = validated["day_of_week"].isin([5, 6])

    if not (
        validated["weekend_flag"].astype(bool)
        == expected_weekend
    ).all():
        raise ValueError(
            "Weekend flag is inconsistent with day of week."
        )

    for column in categorical:
        if validated[column].isna().any():
            raise ValueError(f"Missing category values in {column}.")

        if not validated[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError(
                f"Invalid category values in {column}."
            )

    return validated


def encode_forecast_inputs(frame, artifact):
    """Encode raw features using the saved training feature order."""
    validated = validate_forecast_inputs(frame, artifact)

    numeric = artifact["numeric_features"]
    categorical = artifact["categorical_features"]
    feature_columns = artifact["feature_columns"]

    encoded = pd.get_dummies(
        validated[[*numeric, *categorical]],
        columns=categorical,
        dtype=int,
    )

    unexpected = sorted(set(encoded.columns) - set(feature_columns))

    if unexpected:
        raise ValueError(
            f"Unknown encoded categories: {unexpected}"
        )

    encoded = encoded.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    return encoded


def predict_daily_demand(frame, artifact=None):
    """Return nonnegative one-day-ahead predictions."""
    if artifact is None:
        artifact = load_champion()

    encoded = encode_forecast_inputs(frame, artifact)

    predictions = np.asarray(
        artifact["model"].predict(encoded),
        dtype=float,
    )

    if predictions.ndim != 1 or len(predictions) != len(frame):
        raise ValueError("Unexpected prediction output shape.")

    if not np.isfinite(predictions).all():
        raise ValueError("Model returned non-finite predictions.")

    predictions = np.maximum(predictions, 0.0)

    result = frame.copy(deep=True)
    result["predicted_units"] = predictions

    return result