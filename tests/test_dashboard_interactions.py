"""Tests for read-only dashboard selection and filtering."""

from copy import deepcopy

import pandas as pd
import pytest

from src.dashboard.data_loader import load_dashboard_data
from src.dashboard.forecast_performance import prepare_forecast_series
from src.dashboard.inventory_planner import (
    STATUS_ORDER,
    filter_inventory_plan,
)


@pytest.fixture(scope="module")
def dashboard_data():
    """Load existing saved dashboard artifacts once."""
    return load_dashboard_data()


def test_all_inventory_statuses_are_registered(dashboard_data):
    actual = set(
        dashboard_data["inventory_recommendations"][
            "inventory_status"
        ].unique()
    )
    assert actual == set(STATUS_ORDER)


def test_unfiltered_inventory_plan_preserves_all_rows(dashboard_data):
    plan = dashboard_data["inventory_recommendations"]
    result = filter_inventory_plan(
        plan,
        stores=plan["store_id"].unique().tolist(),
        statuses=STATUS_ORDER,
        positive_orders_only=False,
    )

    assert len(result) == 100
    assert result["suggested_order_units"].sum() == 4108


def test_positive_order_filter(dashboard_data):
    plan = dashboard_data["inventory_recommendations"]
    result = filter_inventory_plan(
        plan,
        stores=plan["store_id"].unique().tolist(),
        statuses=STATUS_ORDER,
        positive_orders_only=True,
    )

    assert len(result) == 59
    assert (result["suggested_order_units"] > 0).all()
    assert result["suggested_order_units"].sum() == 4108


@pytest.mark.parametrize("status", STATUS_ORDER)
def test_filter_by_inventory_status(dashboard_data, status):
    plan = dashboard_data["inventory_recommendations"]
    result = filter_inventory_plan(
        plan,
        stores=plan["store_id"].unique().tolist(),
        statuses=[status],
        positive_orders_only=False,
    )

    assert not result.empty
    assert result["inventory_status"].eq(status).all()


def test_filter_by_store(dashboard_data):
    plan = dashboard_data["inventory_recommendations"]
    selected_store = plan["store_id"].iloc[0]

    result = filter_inventory_plan(
        plan,
        stores=[selected_store],
        statuses=STATUS_ORDER,
        positive_orders_only=False,
    )

    assert len(result) == 10
    assert result["store_id"].eq(selected_store).all()


def test_empty_store_selection(dashboard_data):
    plan = dashboard_data["inventory_recommendations"]

    result = filter_inventory_plan(
        plan,
        stores=[],
        statuses=STATUS_ORDER,
        positive_orders_only=False,
    )

    assert result.empty


def test_empty_status_selection(dashboard_data):
    plan = dashboard_data["inventory_recommendations"]

    result = filter_inventory_plan(
        plan,
        stores=plan["store_id"].unique().tolist(),
        statuses=[],
        positive_orders_only=False,
    )

    assert result.empty


def test_inventory_filter_does_not_mutate_source(dashboard_data):
    plan = dashboard_data["inventory_recommendations"]
    original = plan.copy(deep=True)

    result = filter_inventory_plan(
        plan,
        stores=plan["store_id"].unique().tolist(),
        statuses=STATUS_ORDER,
        positive_orders_only=True,
    )
    result.loc[:, "suggested_order_units"] = -1

    pd.testing.assert_frame_equal(plan, original)


def test_forecast_series_selection(dashboard_data):
    predictions = dashboard_data["holdout_predictions"]
    first = predictions.iloc[0]

    result = prepare_forecast_series(
        predictions,
        model=first["model"],
        store_id=first["store_id"],
        product_id=first["product_id"],
    )

    assert len(result) == 91
    assert result["model"].eq(first["model"]).all()
    assert result["store_id"].eq(first["store_id"]).all()
    assert result["product_id"].eq(first["product_id"]).all()
    assert result["date"].is_monotonic_increasing


def test_forecast_selection_does_not_mutate_source(dashboard_data):
    predictions = dashboard_data["holdout_predictions"]
    original = predictions.copy(deep=True)
    first = predictions.iloc[0]

    result = prepare_forecast_series(
        predictions,
        model=first["model"],
        store_id=first["store_id"],
        product_id=first["product_id"],
    )

    result.loc[:, "prediction"] = -1
    pd.testing.assert_frame_equal(predictions, original)


def test_unknown_forecast_model_returns_empty(dashboard_data):
    predictions = dashboard_data["holdout_predictions"]
    first = predictions.iloc[0]

    result = prepare_forecast_series(
        predictions,
        model="unknown_model",
        store_id=first["store_id"],
        product_id=first["product_id"],
    )

    assert result.empty


def test_dashboard_data_is_not_changed_by_filters(dashboard_data):
    original = deepcopy(dashboard_data)

    plan = dashboard_data["inventory_recommendations"]
    filter_inventory_plan(
        plan,
        stores=plan["store_id"].unique().tolist(),
        statuses=STATUS_ORDER,
        positive_orders_only=True,
    )

    pd.testing.assert_frame_equal(
        dashboard_data["inventory_recommendations"],
        original["inventory_recommendations"],
    )