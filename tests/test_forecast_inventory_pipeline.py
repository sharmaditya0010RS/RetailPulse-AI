"""Tests for next-day forecast-to-inventory integration."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.forecast_inventory_pipeline import (
    build_inventory_snapshot,
    build_next_day_features,
    generate_inventory_plan,
    load_historical_data,
)
from src.models.forecast_inference import load_champion


@pytest.fixture(scope="module")
def history():
    return load_historical_data()


@pytest.fixture(scope="module")
def champion():
    return load_champion()


def test_historical_data_shape(history):
    assert len(history) == 70000
    assert history.groupby(["store_id", "product_id"]).ngroups == 100


def test_historical_data_has_no_duplicate_keys(history):
    assert not history.duplicated(
        ["date", "store_id", "product_id"]
    ).any()


def test_next_day_feature_count(history):
    features = build_next_day_features(history)

    assert len(features) == 100
    assert not features.duplicated(
        ["store_id", "product_id"]
    ).any()


def test_forecast_date_is_next_day(history):
    features = build_next_day_features(history)

    expected = pd.Timestamp("2025-12-01")

    assert features["date"].eq(expected).all()
    assert expected == history["date"].max() + pd.Timedelta(days=1)


def test_calendar_features(history):
    features = build_next_day_features(history)

    assert features["day_of_week"].eq(0).all()
    assert features["month"].eq(12).all()
    assert features["weekend_flag"].eq(0).all()


def test_lags_match_historical_observations(history):
    features = build_next_day_features(history)

    group = history.loc[
        (history["store_id"] == "S001")
        & (history["product_id"] == "P001")
    ].sort_values("date")

    row = features.loc[
        (features["store_id"] == "S001")
        & (features["product_id"] == "P001")
    ].iloc[0]

    values = group["units_sold"].to_numpy()

    for lag in (1, 7, 14, 28):
        assert row[f"sales_lag_{lag}"] == values[-lag]


def test_rolling_means_match_history(history):
    features = build_next_day_features(history)

    group = history.loc[
        (history["store_id"] == "S001")
        & (history["product_id"] == "P001")
    ].sort_values("date")

    row = features.loc[
        (features["store_id"] == "S001")
        & (features["product_id"] == "P001")
    ].iloc[0]

    values = group["units_sold"].to_numpy(dtype=float)

    for window in (7, 14, 28):
        assert row[f"sales_rolling_mean_{window}"] == pytest.approx(
            values[-window:].mean()
        )


def test_feature_generation_does_not_mutate_history(history):
    original = history.copy(deep=True)

    build_next_day_features(history)

    pd.testing.assert_frame_equal(history, original)


def test_rejects_empty_feature_history():
    with pytest.raises(ValueError, match="nonempty"):
        build_next_day_features(pd.DataFrame())


def test_rejects_missing_history_column(history):
    invalid = history.drop(columns=["units_sold"])

    with pytest.raises(ValueError, match="Missing demand history"):
        build_next_day_features(invalid)


def test_rejects_duplicate_history_dates(history):
    invalid = pd.concat(
        [history, history.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        build_next_day_features(invalid)


def test_rejects_insufficient_history(history):
    short = history.loc[
        (history["store_id"] == "S001")
        & (history["product_id"] == "P001")
    ].tail(27)

    with pytest.raises(ValueError, match="Insufficient history"):
        build_next_day_features(short)


def test_rejects_stale_series(history):
    invalid = history.loc[
        ~(
            (history["store_id"] == "S001")
            & (history["product_id"] == "P001")
            & (history["date"] == history["date"].max())
        )
    ].copy()

    with pytest.raises(ValueError, match="not current"):
        build_next_day_features(invalid)


def test_rejects_missing_recent_day(history):
    series = history.loc[
        (history["store_id"] == "S001")
        & (history["product_id"] == "P001")
    ].copy()

    missing_date = series["date"].max() - pd.Timedelta(days=5)

    series = series.loc[series["date"] != missing_date]

    with pytest.raises(ValueError, match="Missing recent"):
        build_next_day_features(series)


def test_inventory_snapshot_count(history):
    snapshot = build_inventory_snapshot(history)

    assert len(snapshot) == 100
    assert snapshot["inventory_snapshot_date"].eq(
        pd.Timestamp("2025-11-30")
    ).all()


def test_inventory_snapshot_matches_latest_stock(history):
    snapshot = build_inventory_snapshot(history)

    actual = history.loc[
        (history["date"] == pd.Timestamp("2025-11-30"))
        & (history["store_id"] == "S001")
        & (history["product_id"] == "P001")
    ]["closing_stock"].iloc[0]

    selected = snapshot.loc[
        (snapshot["store_id"] == "S001")
        & (snapshot["product_id"] == "P001")
    ]["closing_stock"].iloc[0]

    assert selected == actual == 145


def test_rejects_incomplete_inventory_snapshot(history):
    invalid = history.loc[
        ~(
            (history["store_id"] == "S001")
            & (history["product_id"] == "P001")
            & (history["date"] == history["date"].max())
        )
    ].copy()

    with pytest.raises(ValueError, match="does not cover all series"):
        build_inventory_snapshot(invalid)


def test_inventory_snapshot_does_not_mutate_history(history):
    original = history.copy(deep=True)

    build_inventory_snapshot(history)

    pd.testing.assert_frame_equal(history, original)


def test_integration_produces_100_rows(history, champion):
    plan = generate_inventory_plan(history, artifact=champion)

    assert len(plan) == 100
    assert plan["date"].eq(pd.Timestamp("2025-12-01")).all()


def test_integration_uses_frozen_champion(history, champion):
    plan = generate_inventory_plan(history, artifact=champion)

    assert plan["forecast_model"].eq("gradient_boosting_v1").all()
    assert plan["data_source"].eq("synthetic").all()


def test_integration_predictions_are_finite(history, champion):
    plan = generate_inventory_plan(history, artifact=champion)

    assert np.isfinite(plan["predicted_units"]).all()
    assert plan["predicted_units"].ge(0).all()


def test_integration_orders_are_nonnegative(history, champion):
    plan = generate_inventory_plan(history, artifact=champion)

    assert plan["suggested_order_units"].ge(0).all()
    assert pd.api.types.is_integer_dtype(
        plan["suggested_order_units"]
    )


def test_integration_has_no_duplicate_series(history, champion):
    plan = generate_inventory_plan(history, artifact=champion)

    assert not plan.duplicated(
        ["store_id", "product_id"]
    ).any()


def test_integration_excludes_unconfirmed_open_orders(history, champion):
    plan = generate_inventory_plan(history, artifact=champion)

    assert not plan["open_purchase_orders_included"].any()
    assert (
        plan["inventory_position_units"]
        == plan["closing_stock"]
    ).all()


def test_integration_preserves_history(history, champion):
    original = history.copy(deep=True)

    generate_inventory_plan(history, artifact=champion)

    pd.testing.assert_frame_equal(history, original)


def test_integration_reproducible(history, champion):
    first = generate_inventory_plan(history, artifact=champion)
    second = generate_inventory_plan(history, artifact=champion)

    pd.testing.assert_frame_equal(first, second)