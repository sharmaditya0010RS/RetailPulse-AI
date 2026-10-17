import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from config.database import get_engine

ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = ROOT / "data" / "processed" / "bi"

BI_VIEWS = {
    "executive_kpis": "bi_executive_kpis",
    "monthly_performance": "bi_monthly_performance",
    "category_performance": "bi_category_performance",
    "store_performance": "bi_store_performance",
    "product_stockouts": "bi_product_stockouts",
}

EXPECTED_ROWS = {
    "executive_kpis": 1,
    "monthly_performance": 23,
    "category_performance": 5,
    "store_performance": 10,
    "product_stockouts": 10,
}


def json_safe(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def fetch_bi_data():
    engine = get_engine()

    try:
        with engine.connect() as connection:
            return {
                name: pd.read_sql_query(
                    text(
                        f"SELECT * FROM retailpulse.{view_name}"
                    ),
                    connection,
                )
                for name, view_name in BI_VIEWS.items()
            }
    finally:
        engine.dispose()


def validate_data(datasets):
    for name, expected_count in EXPECTED_ROWS.items():
        frame = datasets[name]

        if len(frame) != expected_count:
            raise ValueError(
                f"{name}: expected {expected_count} rows, "
                f"found {len(frame)}"
            )

        if frame.isna().all(axis=1).any():
            raise ValueError(
                f"{name}: contains an entirely empty row"
            )

    executive = datasets["executive_kpis"].iloc[0]

    if int(executive["observations"]) != 70000:
        raise ValueError("Executive observation count mismatch.")

    if int(executive["units_sold"]) != 1483884:
        raise ValueError("Executive units sold mismatch.")

    if int(executive["lost_units"]) != 79971:
        raise ValueError("Executive lost units mismatch.")

    expected_revenue = Decimal("116500690.78")

    if Decimal(str(executive["net_revenue"])) != expected_revenue:
        raise ValueError("Executive revenue mismatch.")

    for name in (
        "monthly_performance",
        "category_performance",
        "store_performance",
    ):
        total = sum(
            Decimal(str(value))
            for value in datasets[name]["net_revenue"]
        )

        if total != expected_revenue:
            raise ValueError(
                f"{name}: revenue does not reconcile"
            )


def export_data(datasets):
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    for name, frame in datasets.items():
        output_path = EXPORT_DIR / f"{name}.csv"
        frame.to_csv(output_path, index=False)
        print(f"Exported: {output_path.name} ({len(frame)} rows)")

    executive = datasets["executive_kpis"].iloc[0]

    scorecard = {
        key: json_safe(value)
        for key, value in executive.items()
    }

    scorecard["data_source"] = "synthetic"
    scorecard["gross_profit_note"] = (
        "Estimated gross profit; excludes some operating costs."
    )

    scorecard_path = EXPORT_DIR / "executive_scorecard.json"

    scorecard_path.write_text(
        json.dumps(scorecard, indent=2),
        encoding="utf-8",
    )

    print(f"Exported: {scorecard_path.name}")


def main():
    print("RETAILPULSE AI - PHASE 4 BI EXPORT")
    print("Source: PostgreSQL reporting views")
    print("Dataset: synthetic retail data")

    datasets = fetch_bi_data()
    validate_data(datasets)
    export_data(datasets)

    print("PASS: BI row counts")
    print("PASS: Executive KPI reconciliation")
    print("PASS: Revenue reconciliation across views")
    print("PASS: BI export completed")


if __name__ == "__main__":
    main()