import argparse
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from config.database import get_engine

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "raw" / "retailpulse_synthetic_70k.csv"

STORE_COLUMNS = [
    "store_id",
    "region",
    "store_tier",
    "store_area_sqm",
]

PRODUCT_COLUMNS = [
    "product_id",
    "category",
    "brand",
    "supplier_id",
]

SUPPLIER_COLUMNS = ["supplier_id"]

DIMENSION_ATTRIBUTES = [
    "region",
    "store_tier",
    "store_area_sqm",
    "category",
    "brand",
    "supplier_id",
]

EXPECTED_ROWS = 70000
EXPECTED_COLUMNS = 52


def read_source() -> pd.DataFrame:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Source CSV does not exist: {CSV_PATH}"
        )

    frame = pd.read_csv(
        CSV_PATH,
        parse_dates=["date"],
        keep_default_na=False,
        na_values=[""],
    )

    return frame


def validate_source(frame: pd.DataFrame) -> None:
    if frame.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(
            f"Unexpected dataset shape: {frame.shape}"
        )

    if frame.isna().any().any():
        raise ValueError("Source contains missing values.")

    key = ["date", "store_id", "product_id"]

    if frame.duplicated(key).any():
        raise ValueError("Duplicate daily retail keys found.")

    if frame["store_id"].nunique() != 10:
        raise ValueError("Expected exactly 10 stores.")

    if frame["product_id"].nunique() != 10:
        raise ValueError("Expected exactly 10 products.")

    if frame["date"].nunique() != 700:
        raise ValueError("Expected exactly 700 dates.")

    if not (
        frame["closing_stock"]
        == frame["opening_stock"] - frame["units_sold"]
    ).all():
        raise ValueError("Inventory balance failed.")

    if not (
        frame["latent_demand_units"]
        == frame["units_sold"] + frame["lost_sales_units"]
    ).all():
        raise ValueError("Demand balance failed.")

    if not (
        frame["stockout_flag"]
        == (frame["lost_sales_units"] > 0).astype(int)
    ).all():
        raise ValueError("Stockout flag mismatch.")

    if not (
        frame["units_returned"] <= frame["units_sold"]
    ).all():
        raise ValueError("Returns exceed sales.")

    if not (
        frame["unit_cost"] >= 0
    ).all():
        raise ValueError("Negative unit cost found.")

    if not (
        frame["promotion_flag"]
        == (frame["discount_pct"] > 0).astype(int)
    ).all():
        raise ValueError("Promotion consistency failed.")

    if not (
        frame["promotion_type"].isin(["None", "Percentage"])
    ).all():
        raise ValueError("Unexpected promotion type.")

    if not (
        (frame["net_revenue"] -
         (frame["gross_revenue"] - frame["refund_amount"]))
        .abs() < 0.011
    ).all():
        raise ValueError("Revenue balance failed.")

    for key_column, attributes in [
        ("store_id", STORE_COLUMNS[1:]),
        ("product_id", PRODUCT_COLUMNS[1:]),
    ]:
        counts = frame.groupby(key_column)[attributes].nunique()

        if (counts > 1).any().any():
            raise ValueError(
                f"Inconsistent dimension attributes for {key_column}."
            )

    print("PASS: Source validation")


def build_tables(frame: pd.DataFrame) -> dict:
    suppliers = (
        frame[SUPPLIER_COLUMNS]
        .drop_duplicates()
        .sort_values("supplier_id")
        .reset_index(drop=True)
    )

    stores = (
        frame[STORE_COLUMNS]
        .drop_duplicates()
        .sort_values("store_id")
        .reset_index(drop=True)
    )

    products = (
        frame[PRODUCT_COLUMNS]
        .drop_duplicates()
        .sort_values("product_id")
        .reset_index(drop=True)
    )

    facts = frame.drop(
        columns=DIMENSION_ATTRIBUTES
    ).copy()

    facts["date"] = facts["date"].dt.date

    return {
        "dim_supplier": suppliers,
        "dim_store": stores,
        "dim_product": products,
        "fact_daily_retail": facts,
    }


def load_tables(tables: dict) -> dict:
    engine = get_engine()

    with engine.begin() as connection:
        fact_count = connection.execute(
            text(
                "SELECT COUNT(*) "
                "FROM retailpulse.fact_daily_retail"
            )
        ).scalar_one()

        print(f"Existing fact rows: {fact_count}")

        connection.execute(
            text("DELETE FROM retailpulse.fact_daily_retail")
        )
        connection.execute(
            text("DELETE FROM retailpulse.dim_product")
        )
        connection.execute(
            text("DELETE FROM retailpulse.dim_store")
        )
        connection.execute(
            text("DELETE FROM retailpulse.dim_supplier")
        )

        for table_name in [
            "dim_supplier",
            "dim_store",
            "dim_product",
            "fact_daily_retail",
        ]:
            table_frame = tables[table_name]

            table_frame.to_sql(
                name=table_name,
                con=connection,
                schema="retailpulse",
                if_exists="append",
                index=False,
                chunksize=500,
            )

            print(
                f"Loaded {table_name}: "
                f"{len(table_frame):,} rows"
            )

        result = connection.execute(
            text(
                """
                SELECT
                    COUNT(*) AS row_count,
                    COUNT(DISTINCT store_id) AS stores,
                    COUNT(DISTINCT product_id) AS products,
                    COUNT(DISTINCT date) AS days,
                    COALESCE(SUM(units_sold), 0) AS units_sold,
                    COALESCE(SUM(lost_sales_units), 0) AS lost_units,
                    COALESCE(SUM(net_revenue), 0) AS net_revenue
                FROM retailpulse.fact_daily_retail
                """
            )
        ).mappings().one()

        summary = {
            "row_count": int(result["row_count"]),
            "stores": int(result["stores"]),
            "products": int(result["products"]),
            "days": int(result["days"]),
            "units_sold": int(result["units_sold"]),
            "lost_units": int(result["lost_units"]),
            "net_revenue": str(result["net_revenue"]),
        }

        if summary["row_count"] != EXPECTED_ROWS:
            raise ValueError("PostgreSQL row count mismatch.")

        if summary["stores"] != 10:
            raise ValueError("PostgreSQL store count mismatch.")

        if summary["products"] != 10:
            raise ValueError("PostgreSQL product count mismatch.")

        if summary["days"] != 700:
            raise ValueError("PostgreSQL date count mismatch.")

        source = tables["fact_daily_retail"]

        if summary["units_sold"] != int(source["units_sold"].sum()):
            raise ValueError("Units sold reconciliation failed.")

        if summary["lost_units"] != int(
            source["lost_sales_units"].sum()
        ):
            raise ValueError("Lost sales reconciliation failed.")

        source_revenue = sum(
            Decimal(str(value))
            for value in source["net_revenue"]
        )

        db_revenue = Decimal(summary["net_revenue"])

        if abs(source_revenue - db_revenue) > Decimal("0.01"):
            raise ValueError("Revenue reconciliation failed.")

        print("PASS: Database reconciliation")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    print("=" * 55)
    print("RETAILPULSE AI - PHASE 2 ETL")
    print("=" * 55)

    frame = read_source()

    print(f"Read source: {len(frame):,} rows")

    validate_source(frame)

    tables = build_tables(frame)

    summary = load_tables(tables)

    print("\nETL completed successfully.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()