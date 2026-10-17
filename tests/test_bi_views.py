from decimal import Decimal

import pytest
from sqlalchemy import text

from config.database import get_engine

EXPECTED_VIEWS = {
    "bi_executive_kpis": 1,
    "bi_monthly_performance": 23,
    "bi_category_performance": 5,
    "bi_store_performance": 10,
    "bi_product_stockouts": 10,
}

EXPECTED_REVENUE = Decimal("116500690.78")
EXPECTED_UNITS = 1483884
EXPECTED_LOST_UNITS = 79971


@pytest.fixture(scope="module")
def engine():
    database_engine = get_engine()
    yield database_engine
    database_engine.dispose()


def fetch_one(engine, query):
    with engine.connect() as connection:
        return connection.execute(text(query)).mappings().one()


@pytest.mark.parametrize(
    ("view_name", "expected_count"),
    list(EXPECTED_VIEWS.items()),
)
def test_bi_view_row_count(engine, view_name, expected_count):
    result = fetch_one(
        engine,
        f"SELECT COUNT(*) AS row_count "
        f"FROM retailpulse.{view_name}",
    )
    assert result["row_count"] == expected_count


def test_executive_kpi_reconciliation(engine):
    row = fetch_one(
        engine,
        "SELECT * FROM retailpulse.bi_executive_kpis",
    )

    assert row["observations"] == 70000
    assert row["stores"] == 10
    assert row["products"] == 10
    assert row["days"] == 700
    assert row["units_sold"] == EXPECTED_UNITS
    assert row["lost_units"] == EXPECTED_LOST_UNITS
    assert row["net_revenue"] == EXPECTED_REVENUE


@pytest.mark.parametrize(
    "view_name",
    [
        "bi_monthly_performance",
        "bi_category_performance",
        "bi_store_performance",
    ],
)
def test_revenue_reconciles_across_views(engine, view_name):
    row = fetch_one(
        engine,
        f"""
        SELECT SUM(net_revenue) AS total_revenue
        FROM retailpulse.{view_name}
        """,
    )
    assert row["total_revenue"] == EXPECTED_REVENUE


@pytest.mark.parametrize(
    "view_name",
    [
        "bi_monthly_performance",
        "bi_category_performance",
        "bi_store_performance",
        "bi_product_stockouts",
    ],
)
def test_units_reconcile_across_views(engine, view_name):
    row = fetch_one(
        engine,
        f"""
        SELECT
            SUM(units_sold) AS units_sold,
            SUM(lost_units) AS lost_units
        FROM retailpulse.{view_name}
        """,
    )

    assert row["units_sold"] == EXPECTED_UNITS
    assert row["lost_units"] == EXPECTED_LOST_UNITS


def test_executive_kpi_percentages(engine):
    row = fetch_one(
        engine,
        """
        SELECT
            estimated_gross_margin_pct,
            stockout_observation_rate_pct,
            unmet_demand_rate_pct
        FROM retailpulse.bi_executive_kpis
        """,
    )

    assert row["estimated_gross_margin_pct"] == Decimal("38.64")
    assert row["stockout_observation_rate_pct"] == Decimal("6.93")
    assert row["unmet_demand_rate_pct"] == Decimal("5.11")


def test_monthly_date_coverage(engine):
    row = fetch_one(
        engine,
        """
        SELECT
            MIN(sales_month) AS first_month,
            MAX(sales_month) AS last_month
        FROM retailpulse.bi_monthly_performance
        """,
    )

    assert str(row["first_month"]) == "2024-01-01"
    assert str(row["last_month"]) == "2025-11-01"


def test_monthly_growth_initial_value(engine):
    row = fetch_one(
        engine,
        """
        SELECT revenue_growth_mom_pct
        FROM retailpulse.bi_monthly_performance
        WHERE sales_month = DATE '2024-01-01'
        """,
    )

    assert row["revenue_growth_mom_pct"] is None


def test_category_revenue_share(engine):
    row = fetch_one(
        engine,
        """
        SELECT SUM(revenue_share_pct) AS total_share
        FROM retailpulse.bi_category_performance
        """,
    )

    assert abs(
        row["total_share"] - Decimal("100.00")
    ) <= Decimal("0.05")


def test_store_ranking(engine):
    row = fetch_one(
        engine,
        """
        SELECT store_id, revenue_rank
        FROM retailpulse.bi_store_performance
        WHERE revenue_rank = 1
        """,
    )

    assert row["store_id"] == "S006"
    assert row["revenue_rank"] == 1


def test_top_lost_sales_product(engine):
    row = fetch_one(
        engine,
        """
        SELECT product_id, lost_units
        FROM retailpulse.bi_product_stockouts
        ORDER BY lost_units DESC
        LIMIT 1
        """,
    )

    assert row["product_id"] == "P003"
    assert row["lost_units"] == 20452


def test_bi_views_have_no_null_primary_metrics(engine):
    checks = {
        "bi_executive_kpis": "net_revenue",
        "bi_monthly_performance": "net_revenue",
        "bi_category_performance": "net_revenue",
        "bi_store_performance": "net_revenue",
        "bi_product_stockouts": "lost_units",
    }

    with engine.connect() as connection:
        for view_name, metric in checks.items():
            null_count = connection.execute(
                text(
                    f"""
                    SELECT COUNT(*)
                    FROM retailpulse.{view_name}
                    WHERE {metric} IS NULL
                    """
                )
            ).scalar_one()

            assert null_count == 0