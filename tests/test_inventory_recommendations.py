"""Tests for illustrative inventory replenishment recommendations."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.inventory_recommendations import (
    recommend_replenishment,
    validate_inventory_inputs,
)


@pytest.fixture
def inventory_frame():
    return pd.DataFrame(
        {
            "store_id": ["S001", "S001", "S002"],
            "product_id": ["P001", "P002", "P001"],
            "predicted_units": [20.0, 10.0, 15.0],
            "closing_stock": [30, 100, 0],
            "reorder_point": [40, 50, 25],
            "supplier_lead_time_days": [3, 2, 4],
        }
    )


def test_validation_preserves_rows(inventory_frame):
    result = validate_inventory_inputs(inventory_frame)

    assert len(result) == 3
    assert result.index.equals(inventory_frame.index)


def test_validation_does_not_mutate_input(inventory_frame):
    original = inventory_frame.copy(deep=True)

    validate_inventory_inputs(inventory_frame)

    pd.testing.assert_frame_equal(inventory_frame, original)


def test_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        validate_inventory_inputs([])


def test_rejects_empty_frame(inventory_frame):
    with pytest.raises(ValueError, match="empty"):
        validate_inventory_inputs(inventory_frame.iloc[:0])


def test_rejects_missing_column(inventory_frame):
    invalid = inventory_frame.drop(columns=["closing_stock"])

    with pytest.raises(ValueError, match="Missing inventory"):
        validate_inventory_inputs(invalid)


@pytest.mark.parametrize(
    "column",
    ["store_id", "product_id"],
)
def test_rejects_missing_identifier(inventory_frame, column):
    invalid = inventory_frame.copy()
    invalid.loc[0, column] = None

    with pytest.raises(ValueError, match="Missing identifier"):
        validate_inventory_inputs(invalid)


@pytest.mark.parametrize(
    "column",
    ["store_id", "product_id"],
)
def test_rejects_blank_identifier(inventory_frame, column):
    invalid = inventory_frame.copy()
    invalid.loc[0, column] = " "

    with pytest.raises(ValueError, match="Invalid identifier"):
        validate_inventory_inputs(invalid)


def test_rejects_duplicate_store_product(inventory_frame):
    invalid = pd.concat(
        [inventory_frame, inventory_frame.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate store-product"):
        validate_inventory_inputs(invalid)


@pytest.mark.parametrize(
    "column",
    [
        "predicted_units",
        "closing_stock",
        "reorder_point",
        "supplier_lead_time_days",
    ],
)
def test_rejects_missing_numeric(inventory_frame, column):
    invalid = inventory_frame.copy()
    invalid[column] = invalid[column].astype(float)
    invalid.loc[0, column] = np.nan

    with pytest.raises(ValueError, match="Invalid numeric"):
        validate_inventory_inputs(invalid)


@pytest.mark.parametrize(
    "column",
    [
        "predicted_units",
        "closing_stock",
        "reorder_point",
        "supplier_lead_time_days",
    ],
)
def test_rejects_infinite_numeric(inventory_frame, column):
    invalid = inventory_frame.copy()
    invalid[column] = invalid[column].astype(float)
    invalid.loc[0, column] = np.inf

    with pytest.raises(ValueError, match="Invalid numeric"):
        validate_inventory_inputs(invalid)


@pytest.mark.parametrize(
    "column",
    [
        "predicted_units",
        "closing_stock",
        "reorder_point",
        "supplier_lead_time_days",
    ],
)
def test_rejects_negative_numeric(inventory_frame, column):
    invalid = inventory_frame.copy()
    invalid[column] = invalid[column].astype(float)
    invalid.loc[0, column] = -1

    with pytest.raises(ValueError, match="Negative values"):
        validate_inventory_inputs(invalid)


@pytest.mark.parametrize(
    "column",
    [
        "closing_stock",
        "reorder_point",
        "supplier_lead_time_days",
    ],
)
def test_rejects_fractional_integer_fields(inventory_frame, column):
    invalid = inventory_frame.copy()
    invalid[column] = invalid[column].astype(float)
    invalid.loc[0, column] = 1.5

    with pytest.raises(ValueError, match="Non-integer"):
        validate_inventory_inputs(invalid)


def test_rejects_zero_supplier_lead_time(inventory_frame):
    invalid = inventory_frame.copy()
    invalid.loc[0, "supplier_lead_time_days"] = 0

    with pytest.raises(ValueError, match="at least one day"):
        validate_inventory_inputs(invalid)


@pytest.mark.parametrize(
    "safety_days",
    [-1, np.nan, np.inf, True, "two"],
)
def test_rejects_invalid_safety_days(inventory_frame, safety_days):
    with pytest.raises(ValueError, match="Safety days"):
        validate_inventory_inputs(
            inventory_frame,
            safety_days=safety_days,
        )


def test_default_safety_days(inventory_frame):
    result = recommend_replenishment(inventory_frame)

    assert result["safety_days"].tolist() == [2.0, 2.0, 2.0]


def test_target_stock_calculation(inventory_frame):
    result = recommend_replenishment(
        inventory_frame,
        safety_days=2,
    )

    assert result["target_stock_units"].tolist() == [100, 40, 90]


def test_suggested_order_calculation(inventory_frame):
    result = recommend_replenishment(
        inventory_frame,
        safety_days=2,
    )

    assert result["suggested_order_units"].tolist() == [70, 0, 90]


def test_lead_time_demand_calculation(inventory_frame):
    result = recommend_replenishment(inventory_frame)

    assert result["lead_time_demand_units"].tolist() == [
        60.0,
        20.0,
        60.0,
    ]


def test_safety_stock_calculation(inventory_frame):
    result = recommend_replenishment(inventory_frame)

    assert result["safety_stock_units"].tolist() == [40, 20, 30]


def test_existing_reorder_point_flag(inventory_frame):
    result = recommend_replenishment(inventory_frame)

    assert result["below_existing_reorder_point"].tolist() == [
        True,
        False,
        True,
    ]


def test_stockout_flag(inventory_frame):
    result = recommend_replenishment(inventory_frame)

    assert result["stockout_flag"].tolist() == [
        False,
        False,
        True,
    ]


def test_inventory_status(inventory_frame):
    result = recommend_replenishment(inventory_frame)

    assert result["inventory_status"].tolist() == [
        "BELOW_LEAD_TIME_COVERAGE",
        "SUFFICIENT_FOR_POLICY",
        "STOCKOUT",
    ]


def test_coverage_days(inventory_frame):
    result = recommend_replenishment(inventory_frame)

    np.testing.assert_allclose(
        result["coverage_days"].to_numpy(),
        [1.5, 10.0, 0.0],
    )


def test_zero_demand_requires_no_order():
    frame = pd.DataFrame(
        {
            "store_id": ["S001"],
            "product_id": ["P001"],
            "predicted_units": [0.0],
            "closing_stock": [10],
            "reorder_point": [20],
            "supplier_lead_time_days": [3],
        }
    )

    result = recommend_replenishment(frame)

    assert result["target_stock_units"].iloc[0] == 0
    assert result["suggested_order_units"].iloc[0] == 0
    assert np.isinf(result["coverage_days"].iloc[0])
    assert (
        result["inventory_status"].iloc[0]
        == "SUFFICIENT_FOR_POLICY"
    )


def test_fractional_forecast_rounds_target_up():
    frame = pd.DataFrame(
        {
            "store_id": ["S001"],
            "product_id": ["P001"],
            "predicted_units": [2.25],
            "closing_stock": [3],
            "reorder_point": [5],
            "supplier_lead_time_days": [2],
        }
    )

    result = recommend_replenishment(frame, safety_days=1)

    assert result["target_stock_units"].iloc[0] == 7
    assert result["suggested_order_units"].iloc[0] == 4
    assert result["safety_stock_units"].iloc[0] == 3


def test_zero_safety_days(inventory_frame):
    result = recommend_replenishment(
        inventory_frame,
        safety_days=0,
    )

    assert result["target_stock_units"].tolist() == [60, 20, 60]
    assert result["safety_stock_units"].tolist() == [0, 0, 0]


def test_recommendations_do_not_mutate_input(inventory_frame):
    original = inventory_frame.copy(deep=True)

    recommend_replenishment(inventory_frame)

    pd.testing.assert_frame_equal(inventory_frame, original)


def test_recommendations_preserve_identifiers(inventory_frame):
    result = recommend_replenishment(inventory_frame)

    for column in ("store_id", "product_id"):
        assert result[column].equals(inventory_frame[column])


def test_recommendations_are_nonnegative_integers(inventory_frame):
    result = recommend_replenishment(inventory_frame)

    assert result["suggested_order_units"].ge(0).all()
    assert pd.api.types.is_integer_dtype(
        result["suggested_order_units"]
    )


def test_recommendations_are_reproducible(inventory_frame):
    first = recommend_replenishment(inventory_frame)
    second = recommend_replenishment(inventory_frame)

    pd.testing.assert_frame_equal(first, second)