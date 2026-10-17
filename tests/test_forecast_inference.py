"""Tests for frozen champion inference."""

import numpy as np
import pandas as pd
import pytest

from src.models.forecast_inference import (
    encode_forecast_inputs,
    load_champion,
    predict_daily_demand,
    validate_forecast_inputs,
)


@pytest.fixture(scope="module")
def champion():
    return load_champion()


@pytest.fixture
def sample_frame():
    """Use training-period rows, never held-out test rows."""
    dataset = pd.read_csv(
        "data/processed/forecasting_dataset.csv"
    )
    return (
        dataset.loc[dataset["split"] == "train"]
        .head(3)
        .copy()
    )


def test_champion_identity(champion):
    assert champion["model_name"] == "gradient_boosting_v1"
    assert champion["data_source"] == "synthetic"
    assert len(champion["feature_columns"]) == 30


def test_valid_inputs_preserve_rows(sample_frame, champion):
    result = validate_forecast_inputs(
        sample_frame,
        champion,
    )

    assert len(result) == len(sample_frame)
    assert result.index.equals(sample_frame.index)


def test_validation_does_not_mutate_input(sample_frame, champion):
    original = sample_frame.copy(deep=True)

    validate_forecast_inputs(sample_frame, champion)

    pd.testing.assert_frame_equal(sample_frame, original)


def test_rejects_non_dataframe(champion):
    with pytest.raises(TypeError, match="DataFrame"):
        validate_forecast_inputs([], champion)


def test_rejects_empty_input(sample_frame, champion):
    with pytest.raises(ValueError, match="empty"):
        validate_forecast_inputs(
            sample_frame.iloc[:0],
            champion,
        )


def test_rejects_missing_feature(sample_frame, champion):
    invalid = sample_frame.drop(columns=["sales_lag_1"])

    with pytest.raises(ValueError, match="Missing forecast"):
        validate_forecast_inputs(invalid, champion)


def test_rejects_missing_numeric_value(sample_frame, champion):
    invalid = sample_frame.copy()
    invalid.loc[invalid.index[0], "sales_lag_1"] = np.nan

    with pytest.raises(ValueError, match="Invalid or missing"):
        validate_forecast_inputs(invalid, champion)


def test_rejects_infinite_numeric_value(sample_frame, champion):
    invalid = sample_frame.copy()
    invalid.loc[invalid.index[0], "sales_lag_7"] = np.inf

    with pytest.raises(ValueError, match="Invalid or missing"):
        validate_forecast_inputs(invalid, champion)


def test_rejects_negative_demand_history(sample_frame, champion):
    invalid = sample_frame.copy()
    invalid.loc[invalid.index[0], "sales_lag_14"] = -1

    with pytest.raises(ValueError, match="Negative demand"):
        validate_forecast_inputs(invalid, champion)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("day_of_week", 7),
        ("day_of_week", 1.5),
        ("month", 13),
        ("month", 0),
        ("weekend_flag", 2),
    ],
)
def test_rejects_invalid_calendar(
    sample_frame,
    champion,
    column,
    invalid_value,
):
    invalid = sample_frame.copy()
    invalid[column] = invalid[column].astype(float)
    invalid.loc[invalid.index[0], column] = invalid_value

    with pytest.raises(ValueError, match="Invalid calendar"):
        validate_forecast_inputs(invalid, champion)


def test_rejects_inconsistent_weekend_flag(
    sample_frame,
    champion,
):
    invalid = sample_frame.copy()
    index = invalid.index[0]
    invalid.loc[index, "weekend_flag"] = (
        1 - invalid.loc[index, "weekend_flag"]
    )

    with pytest.raises(ValueError, match="Weekend flag"):
        validate_forecast_inputs(invalid, champion)


def test_rejects_missing_category(sample_frame, champion):
    invalid = sample_frame.copy()
    invalid.loc[invalid.index[0], "store_id"] = None

    with pytest.raises(ValueError, match="Missing category"):
        validate_forecast_inputs(invalid, champion)


def test_rejects_blank_category(sample_frame, champion):
    invalid = sample_frame.copy()
    invalid.loc[invalid.index[0], "product_id"] = " "

    with pytest.raises(ValueError, match="Invalid category"):
        validate_forecast_inputs(invalid, champion)


def test_rejects_unknown_category(sample_frame, champion):
    invalid = sample_frame.copy()
    invalid.loc[invalid.index[0], "store_id"] = "S999"

    with pytest.raises(ValueError, match="Unknown encoded"):
        encode_forecast_inputs(invalid, champion)


def test_encoded_feature_order(sample_frame, champion):
    encoded = encode_forecast_inputs(
        sample_frame,
        champion,
    )

    assert encoded.columns.tolist() == champion["feature_columns"]
    assert encoded.shape == (3, 30)


def test_predictions_are_nonnegative(sample_frame, champion):
    result = predict_daily_demand(
        sample_frame,
        champion,
    )

    assert len(result) == 3
    assert np.isfinite(result["predicted_units"]).all()
    assert result["predicted_units"].ge(0).all()


def test_prediction_does_not_mutate_input(
    sample_frame,
    champion,
):
    original = sample_frame.copy(deep=True)

    predict_daily_demand(sample_frame, champion)

    pd.testing.assert_frame_equal(sample_frame, original)


def test_prediction_is_reproducible(sample_frame, champion):
    first = predict_daily_demand(
        sample_frame,
        champion,
    )
    second = predict_daily_demand(
        sample_frame,
        champion,
    )

    np.testing.assert_allclose(
        first["predicted_units"].to_numpy(),
        second["predicted_units"].to_numpy(),
        rtol=0,
        atol=0,
    )


def test_prediction_preserves_identifiers(
    sample_frame,
    champion,
):
    result = predict_daily_demand(
        sample_frame,
        champion,
    )

    for column in ("date", "store_id", "product_id"):
        assert result[column].equals(sample_frame[column])