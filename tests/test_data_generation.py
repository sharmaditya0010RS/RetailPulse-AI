from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "retailpulse_synthetic_70k.csv"


@pytest.fixture(scope="module")
def retail_data():
    assert CSV_PATH.is_file(), "Generate the dataset first."

    return pd.read_csv(
        CSV_PATH,
        parse_dates=["date"],
        keep_default_na=False,
        na_values=[""],
    )


def test_dataset_shape(retail_data):
    assert retail_data.shape == (70000, 52)


def test_dataset_dimensions(retail_data):
    assert retail_data["store_id"].nunique() == 10
    assert retail_data["product_id"].nunique() == 10
    assert retail_data["date"].nunique() == 700


def test_date_range(retail_data):
    assert retail_data["date"].min() == pd.Timestamp("2024-01-01")
    assert retail_data["date"].max() == pd.Timestamp("2025-11-30")


def test_primary_key_uniqueness(retail_data):
    assert not retail_data.duplicated(
        ["date", "store_id", "product_id"]
    ).any()


def test_no_missing_values(retail_data):
    assert not retail_data.isna().any().any()


def test_inventory_balance(retail_data):
    assert (
        retail_data["closing_stock"]
        == retail_data["opening_stock"] - retail_data["units_sold"]
    ).all()


def test_demand_balance(retail_data):
    assert (
        retail_data["latent_demand_units"]
        == retail_data["units_sold"] + retail_data["lost_sales_units"]
    ).all()


def test_stockout_logic(retail_data):
    assert (
        retail_data["stockout_flag"]
        == (retail_data["lost_sales_units"] > 0).astype(int)
    ).all()


def test_nonnegative_inventory(retail_data):
    for column in [
        "opening_stock",
        "closing_stock",
        "units_sold",
        "units_ordered",
        "units_received",
        "lost_sales_units",
    ]:
        assert (retail_data[column] >= 0).all()


def test_revenue_balance(retail_data):
    expected = (
        retail_data["gross_revenue"]
        - retail_data["refund_amount"]
    )

    assert (
        (retail_data["net_revenue"] - expected).abs() < 0.011
    ).all()


def test_discount_consistency(retail_data):
    assert (
        retail_data["promotion_flag"]
        == (retail_data["discount_pct"] > 0).astype(int)
    ).all()


def test_inventory_continuity(retail_data):
    ordered = retail_data.sort_values(
        ["store_id", "product_id", "date"]
    )

    previous_closing = ordered.groupby(
        ["store_id", "product_id"]
    )["closing_stock"].shift(1)

    has_previous = previous_closing.notna()

    expected_opening = (
        previous_closing[has_previous]
        + ordered.loc[has_previous, "units_received"]
    )

    actual_opening = ordered.loc[
        has_previous, "opening_stock"
    ]

    assert (actual_opening == expected_opening).all()


def test_promotion_type_values(retail_data):
    assert set(retail_data["promotion_type"].unique()) == {
        "None",
        "Percentage",
    }

    inactive = retail_data["promotion_flag"] == 0
    active = retail_data["promotion_flag"] == 1

    assert (retail_data.loc[inactive, "promotion_type"] == "None").all()
    assert (
        retail_data.loc[active, "promotion_type"] == "Percentage"
    ).all()