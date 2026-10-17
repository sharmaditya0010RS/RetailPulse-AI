import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from config.database import get_engine

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

EXPECTED_CSVS = {
    "monthly": 23,
    "categories": 5,
    "stores": 10,
    "stockouts": 10,
    "promotions": 2,
}

EXPECTED_FIGURES = [
    "monthly_revenue.png",
    "monthly_units_sold.png",
    "category_revenue.png",
    "store_revenue.png",
    "product_lost_sales.png",
    "promotion_stockout_rate.png",
]


@pytest.fixture(scope="module")
def kpis():
    path = REPORT_DIR / "phase3_kpi_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def engine():
    database_engine = get_engine()
    yield database_engine
    database_engine.dispose()


def test_kpi_file_exists():
    assert (REPORT_DIR / "phase3_kpi_summary.json").is_file()


def test_kpi_source_is_synthetic(kpis):
    assert kpis["data_source"] == "synthetic"


def test_kpi_date_range(kpis):
    assert kpis["date_start"] == "2024-01-01"
    assert kpis["date_end"] == "2025-11-30"


def test_kpi_expected_counts(kpis):
    assert kpis["observations"] == 70000
    assert kpis["stores"] == 10
    assert kpis["products"] == 10
    assert kpis["days"] == 700


def test_kpi_expected_sales(kpis):
    assert kpis["units_sold"] == 1483884
    assert kpis["lost_units"] == 79971


def test_kpi_expected_revenue(kpis):
    assert Decimal(str(kpis["net_revenue"])) == Decimal(
        "116500690.78"
    )


@pytest.mark.parametrize(
    ("dataset", "expected_rows"),
    list(EXPECTED_CSVS.items()),
)
def test_analytics_csv_exists_and_has_rows(dataset, expected_rows):
    path = REPORT_DIR / f"phase3_{dataset}.csv"

    assert path.is_file()

    frame = pd.read_csv(path)

    assert len(frame) == expected_rows
    assert not frame.empty
    assert not frame.isna().all(axis=1).any()


@pytest.mark.parametrize("filename", EXPECTED_FIGURES)
def test_eda_figure_exists(filename):
    path = FIGURE_DIR / filename

    assert path.is_file()
    assert path.stat().st_size > 1000

    with path.open("rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"


def test_monthly_revenue_reconciles(kpis):
    monthly = pd.read_csv(
        REPORT_DIR / "phase3_monthly.csv"
    )

    monthly_revenue = sum(
        Decimal(str(value))
        for value in monthly["net_revenue"]
    )

    assert monthly_revenue == Decimal(
        str(kpis["net_revenue"])
    )


def test_category_revenue_reconciles(kpis):
    categories = pd.read_csv(
        REPORT_DIR / "phase3_categories.csv"
    )

    category_revenue = sum(
        Decimal(str(value))
        for value in categories["net_revenue"]
    )

    assert category_revenue == Decimal(
        str(kpis["net_revenue"])
    )


def test_store_revenue_reconciles(kpis):
    stores = pd.read_csv(
        REPORT_DIR / "phase3_stores.csv"
    )

    store_revenue = sum(
        Decimal(str(value))
        for value in stores["net_revenue"]
    )

    assert store_revenue == Decimal(
        str(kpis["net_revenue"])
    )


def test_lost_units_reconcile(kpis):
    stockouts = pd.read_csv(
        REPORT_DIR / "phase3_stockouts.csv"
    )

    assert int(stockouts["lost_units"].sum()) == int(
        kpis["lost_units"]
    )


def test_exported_kpis_match_database(engine, kpis):
    query = text(
        """
        SELECT
            COUNT(*) AS observations,
            SUM(units_sold) AS units_sold,
            SUM(lost_sales_units) AS lost_units,
            SUM(net_revenue) AS net_revenue
        FROM retailpulse.fact_daily_retail
        """
    )

    with engine.connect() as connection:
        result = connection.execute(query).mappings().one()

    assert int(result["observations"]) == kpis["observations"]
    assert int(result["units_sold"]) == kpis["units_sold"]
    assert int(result["lost_units"]) == kpis["lost_units"]

    assert Decimal(str(result["net_revenue"])) == Decimal(
        str(kpis["net_revenue"])
    )