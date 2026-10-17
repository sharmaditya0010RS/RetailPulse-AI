import json

import numpy as np
import pandas as pd
import pytest

from src.analytics.forecast_explainability import (
    IMPORTANCE_PATH,
    SUMMARY_PATH,
    calculate_permutation_importance,
    calculate_wape,
    encode_features,
)


class MockModel:
    def predict(self, features):
        return (
            0.5 * features["sales_lag_1"].to_numpy()
            + 2.0 * features["store_id_S001"].to_numpy()
            + 1.0
        )


@pytest.fixture
def artifact():
    return {
        "model": MockModel(),
        "numeric_features": ["sales_lag_1"],
        "categorical_features": ["store_id"],
        "feature_columns": [
            "sales_lag_1",
            "store_id_S001",
            "store_id_S002",
        ],
    }


@pytest.fixture
def validation():
    return pd.DataFrame(
        {
            "sales_lag_1": [10, 12, 14, 16, 18, 20],
            "store_id": [
                "S001",
                "S002",
                "S001",
                "S002",
                "S001",
                "S002",
            ],
            "units_sold": [8, 7, 10, 9, 12, 11],
            "split": ["validation"] * 6,
        }
    )


def test_wape_perfect_predictions():
    assert calculate_wape(
        [10, 20],
        [10, 20],
    ) == pytest.approx(0.0)


def test_wape_known_value():
    assert calculate_wape(
        [10, 20],
        [12, 16],
    ) == pytest.approx(20.0)


def test_wape_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        calculate_wape([10, 20], [10])


def test_wape_rejects_nonfinite_actual():
    with pytest.raises(ValueError, match="Non-finite"):
        calculate_wape([np.nan, 10], [5, 10])


def test_wape_rejects_nonfinite_prediction():
    with pytest.raises(ValueError, match="Non-finite"):
        calculate_wape([10, 20], [np.inf, 20])


def test_wape_rejects_zero_denominator():
    with pytest.raises(ValueError, match="denominator"):
        calculate_wape([0, 0], [1, 2])


def test_encoding_preserves_feature_order(validation, artifact):
    encoded = encode_features(validation, artifact)

    assert encoded.columns.tolist() == artifact["feature_columns"]
    assert encoded.shape == (6, 3)


def test_encoding_one_hot_values(validation, artifact):
    encoded = encode_features(validation, artifact)

    assert encoded["store_id_S001"].tolist() == [
        1, 0, 1, 0, 1, 0
    ]


def test_encoding_rejects_unknown_category(validation, artifact):
    modified = validation.copy()
    modified.loc[0, "store_id"] = "S999"

    with pytest.raises(ValueError, match="Unexpected encoded"):
        encode_features(modified, artifact)


def test_encoding_rejects_missing_numeric(validation, artifact):
    modified = validation.copy()
    modified.loc[0, "sales_lag_1"] = np.nan

    with pytest.raises(ValueError, match="Missing feature"):
        encode_features(modified, artifact)


def test_permutation_rejects_zero_repeats(validation, artifact):
    with pytest.raises(ValueError, match="At least one repeat"):
        calculate_permutation_importance(
            validation,
            artifact,
            repeats=0,
        )


def test_permutation_rejects_empty_data(artifact):
    empty = pd.DataFrame(
        columns=[
            "sales_lag_1",
            "store_id",
            "units_sold",
            "split",
        ]
    )

    with pytest.raises(ValueError, match="empty"):
        calculate_permutation_importance(empty, artifact)


def test_permutation_rejects_test_rows(validation, artifact):
    modified = validation.copy()
    modified.loc[0, "split"] = "test"

    with pytest.raises(ValueError, match="Only validation"):
        calculate_permutation_importance(modified, artifact)


def test_permutation_feature_count(validation, artifact):
    importance, baseline = calculate_permutation_importance(
        validation,
        artifact,
        repeats=2,
    )

    assert len(importance) == 2
    assert set(importance["feature"]) == {
        "sales_lag_1",
        "store_id",
    }
    assert baseline >= 0


def test_permutation_is_reproducible(validation, artifact):
    first, first_wape = calculate_permutation_importance(
        validation,
        artifact,
        repeats=3,
        random_state=42,
    )

    second, second_wape = calculate_permutation_importance(
        validation,
        artifact,
        repeats=3,
        random_state=42,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_wape == pytest.approx(second_wape)


def test_permutation_does_not_mutate_input(validation, artifact):
    original = validation.copy(deep=True)

    calculate_permutation_importance(
        validation,
        artifact,
        repeats=2,
    )

    pd.testing.assert_frame_equal(validation, original)


def test_permutation_report_exists():
    assert IMPORTANCE_PATH.is_file()
    assert SUMMARY_PATH.is_file()


def test_saved_importance_has_expected_features():
    importance = pd.read_csv(IMPORTANCE_PATH)

    expected = {
        "sales_lag_1",
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_28",
        "sales_rolling_mean_7",
        "sales_rolling_mean_14",
        "sales_rolling_mean_28",
        "day_of_week",
        "month",
        "weekend_flag",
        "store_id",
        "product_id",
    }

    assert len(importance) == 12
    assert set(importance["feature"]) == expected
    assert importance["feature"].is_unique
    assert (importance["repeats"] == 3).all()


def test_saved_importance_is_sorted():
    importance = pd.read_csv(IMPORTANCE_PATH)

    values = importance["wape_increase_mean_pp"].to_numpy()

    assert np.isfinite(values).all()
    assert np.all(values[:-1] >= values[1:])


def test_saved_summary_metadata():
    summary = json.loads(
        SUMMARY_PATH.read_text(encoding="utf-8")
    )

    assert summary["data_source"] == "synthetic"
    assert summary["model"] == "gradient_boosting_v1"
    assert summary["analysis_split"] == "validation"
    assert summary["validation_observations"] == 9200
    assert summary["features_evaluated"] == 12
    assert summary["repeats"] == 3
    assert summary["model_retrained"] is False
    assert summary["test_data_used"] is False


def test_saved_summary_matches_importance():
    summary = json.loads(
        SUMMARY_PATH.read_text(encoding="utf-8")
    )
    importance = pd.read_csv(IMPORTANCE_PATH)

    np.testing.assert_allclose(
        importance["baseline_validation_wape_pct"].to_numpy(),
        summary["original_validation_wape_pct"],
        rtol=0,
        atol=0.0001,
    )