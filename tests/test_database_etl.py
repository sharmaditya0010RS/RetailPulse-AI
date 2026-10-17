from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from config.database import get_engine

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "retailpulse_synthetic_70k.csv"


@pytest.fixture(scope="module")
def source():
    return pd.read_csv(
        CSV_PATH,
        keep_default_na=False,
        na_values=[""],
    )


@pytest.fixture(scope="module")
def engine():
    database_engine = get_engine()
    yield database_engine
    database_engine.dispose()


def test_expected_tables_exist(engine):
    query = text(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'retailpulse'
          AND table_type = 'BASE TABLE'
        """
    )

    with engine.connect() as connection:
        actual = {
            row[0]
            for row in connection.execute(query)
        }

    expected = {
        "dim_supplier",
        "dim_store",
        "dim_product",
        "fact_daily_retail",
    }

    assert expected.issubset(actual)


def test_dimension_row_counts(engine):
    expected = {
        "dim_supplier": 10,
        "dim_store": 10,
        "dim_product": 10,
    }

    with engine.connect() as connection:
        for table_name, expected_count in expected.items():
            actual = connection.execute(
                text(
                    f"SELECT COUNT(*) "
                    f"FROM retailpulse.{table_name}"
                )
            ).scalar_one()

            assert actual == expected_count


def test_fact_row_count(engine):
    with engine.connect() as connection:
        count = connection.execute(
            text(
                "SELECT COUNT(*) "
                "FROM retailpulse.fact_daily_retail"
            )
        ).scalar_one()

    assert count == 70000


def test_fact_primary_key_uniqueness(engine):
    query = text(
        """
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT (date, store_id, product_id))
                AS unique_rows
        FROM retailpulse.fact_daily_retail
        """
    )

    with engine.connect() as connection:
        row = connection.execute(query).one()

    assert row.total_rows == row.unique_rows == 70000


def test_date_coverage(engine):
    query = text(
        """
        SELECT
            MIN(date) AS start_date,
            MAX(date) AS end_date,
            COUNT(DISTINCT date) AS total_days
        FROM retailpulse.fact_daily_retail
        """
    )

    with engine.connect() as connection:
        result = connection.execute(query).mappings().one()

    assert str(result["start_date"]) == "2024-01-01"
    assert str(result["end_date"]) == "2025-11-30"
    assert result["total_days"] == 700


def test_inventory_constraints(engine):
    query = text(
        """
        SELECT COUNT(*)
        FROM retailpulse.fact_daily_retail
        WHERE closing_stock <> opening_stock - units_sold
           OR closing_stock < 0
           OR units_returned > units_sold
           OR latent_demand_units
              <> units_sold + lost_sales_units
        """
    )

    with engine.connect() as connection:
        invalid_count = connection.execute(query).scalar_one()

    assert invalid_count == 0


def test_foreign_key_integrity(engine):
    query = text(
        """
        SELECT COUNT(*)
        FROM retailpulse.fact_daily_retail f
        LEFT JOIN retailpulse.dim_store s
            ON f.store_id = s.store_id
        LEFT JOIN retailpulse.dim_product p
            ON f.product_id = p.product_id
        WHERE s.store_id IS NULL
           OR p.product_id IS NULL
        """
    )

    with engine.connect() as connection:
        invalid_count = connection.execute(query).scalar_one()

    assert invalid_count == 0


def test_promotion_type_preserved(engine):
    query = text(
        """
        SELECT COUNT(*)
        FROM retailpulse.fact_daily_retail
        WHERE (promotion_flag = 0 AND promotion_type <> 'None')
           OR (promotion_flag = 1
               AND promotion_type <> 'Percentage')
        """
    )

    with engine.connect() as connection:
        invalid_count = connection.execute(query).scalar_one()

    assert invalid_count == 0


def test_units_reconcile(engine, source):
    query = text(
        """
        SELECT
            SUM(units_sold) AS units_sold,
            SUM(lost_sales_units) AS lost_units
        FROM retailpulse.fact_daily_retail
        """
    )

    with engine.connect() as connection:
        result = connection.execute(query).mappings().one()

    assert int(result["units_sold"]) == int(
        source["units_sold"].sum()
    )

    assert int(result["lost_units"]) == int(
        source["lost_sales_units"].sum()
    )


def test_revenue_reconciles(engine, source):
    query = text(
        """
        SELECT SUM(net_revenue)
        FROM retailpulse.fact_daily_retail
        """
    )

    with engine.connect() as connection:
        database_revenue = connection.execute(
            query
        ).scalar_one()

    source_revenue = sum(
        Decimal(str(value))
        for value in source["net_revenue"]
    )

    assert abs(
        database_revenue - source_revenue
    ) <= Decimal("0.01")


def test_cost_stored_at_fact_grain(engine):
    query = text(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = 'retailpulse'
          AND table_name = 'fact_daily_retail'
          AND column_name = 'unit_cost'
        """
    )

    with engine.connect() as connection:
        column_count = connection.execute(
            query
        ).scalar_one()

    assert column_count == 1


def test_no_null_fact_values(engine):
    query = text(
        """
        SELECT COUNT(*)
        FROM retailpulse.fact_daily_retail
        WHERE date IS NULL
           OR store_id IS NULL
           OR product_id IS NULL
           OR unit_cost IS NULL
           OR promotion_type IS NULL
           OR units_sold IS NULL
           OR net_revenue IS NULL
        """
    )

    with engine.connect() as connection:
        invalid_count = connection.execute(query).scalar_one()

    assert invalid_count == 0