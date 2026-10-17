import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import text

from config.database import get_engine

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

QUERIES = {
    "monthly": """
        SELECT
            DATE_TRUNC('month', date)::DATE AS month,
            SUM(units_sold) AS units_sold,
            SUM(net_revenue) AS net_revenue,
            SUM(lost_sales_units) AS lost_units
        FROM retailpulse.fact_daily_retail
        GROUP BY 1
        ORDER BY 1
    """,
    "categories": """
        SELECT
            p.category,
            SUM(f.net_revenue) AS net_revenue,
            SUM(f.units_sold) AS units_sold
        FROM retailpulse.fact_daily_retail f
        JOIN retailpulse.dim_product p
            ON f.product_id = p.product_id
        GROUP BY p.category
        ORDER BY net_revenue DESC
    """,
    "stores": """
        SELECT
            f.store_id,
            SUM(f.net_revenue) AS net_revenue,
            SUM(f.units_sold) AS units_sold
        FROM retailpulse.fact_daily_retail f
        GROUP BY f.store_id
        ORDER BY net_revenue DESC
    """,
    "stockouts": """
        SELECT
            product_id,
            SUM(lost_sales_units) AS lost_units,
            ROUND(100.0 * AVG(stockout_flag), 2)
                AS stockout_rate_pct
        FROM retailpulse.fact_daily_retail
        GROUP BY product_id
        ORDER BY lost_units DESC
    """,
    "promotions": """
        SELECT
            CASE
                WHEN promotion_flag = 1 THEN 'Promotion'
                ELSE 'No Promotion'
            END AS promotion_group,
            ROUND(AVG(units_sold), 2) AS avg_units_sold,
            ROUND(100.0 * AVG(stockout_flag), 2)
                AS stockout_rate_pct
        FROM retailpulse.fact_daily_retail
        GROUP BY promotion_flag
        ORDER BY promotion_flag DESC
    """,
}

KPI_QUERY = """
    SELECT
        COUNT(*) AS observations,
        COUNT(DISTINCT store_id) AS stores,
        COUNT(DISTINCT product_id) AS products,
        COUNT(DISTINCT date) AS days,
        SUM(units_sold) AS units_sold,
        SUM(lost_sales_units) AS lost_units,
        SUM(net_revenue) AS net_revenue,
        SUM(gross_profit_estimate) AS estimated_gross_profit,
        ROUND(
            100.0 * SUM(gross_profit_estimate)
            / NULLIF(SUM(net_revenue), 0), 2
        ) AS estimated_gross_margin_pct,
        ROUND(100.0 * AVG(stockout_flag), 2)
            AS stockout_observation_rate_pct,
        ROUND(
            100.0 * SUM(lost_sales_units)
            / NULLIF(SUM(latent_demand_units), 0), 2
        ) AS unmet_demand_rate_pct
    FROM retailpulse.fact_daily_retail
"""


def fetch_data():
    engine = get_engine()

    try:
        with engine.connect() as connection:
            kpis = dict(
                connection.execute(
                    text(KPI_QUERY)
                ).mappings().one()
            )

            datasets = {
                name: pd.read_sql_query(text(query), connection)
                for name, query in QUERIES.items()
            }
    finally:
        engine.dispose()

    return kpis, datasets


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / filename,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()


def generate_figures(data):
    monthly = data["monthly"].copy()
    monthly["month"] = pd.to_datetime(monthly["month"])
    monthly["net_revenue"] = monthly["net_revenue"].astype(float)

    plt.figure(figsize=(11, 5))
    plt.plot(
        monthly["month"],
        monthly["net_revenue"] / 1_000_000,
        marker="o",
        linewidth=2,
    )
    plt.title("Monthly Net Revenue — Synthetic Retail Data")
    plt.xlabel("Month")
    plt.ylabel("Net Revenue (Millions)")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.25)
    save_figure("monthly_revenue.png")

    plt.figure(figsize=(11, 5))
    plt.plot(
        monthly["month"],
        monthly["units_sold"],
        marker="o",
        linewidth=2,
    )
    plt.title("Monthly Units Sold — Synthetic Retail Data")
    plt.xlabel("Month")
    plt.ylabel("Units Sold")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.25)
    save_figure("monthly_units_sold.png")

    categories = data["categories"].copy()
    categories["net_revenue"] = categories["net_revenue"].astype(float)

    plt.figure(figsize=(9, 5))
    plt.barh(
        categories["category"],
        categories["net_revenue"] / 1_000_000,
    )
    plt.gca().invert_yaxis()
    plt.title("Revenue by Category — Synthetic Retail Data")
    plt.xlabel("Net Revenue (Millions)")
    save_figure("category_revenue.png")

    stores = data["stores"].copy()
    stores["net_revenue"] = stores["net_revenue"].astype(float)

    plt.figure(figsize=(10, 5))
    plt.bar(
        stores["store_id"],
        stores["net_revenue"] / 1_000_000,
    )
    plt.title("Store Revenue Ranking — Synthetic Retail Data")
    plt.xlabel("Store")
    plt.ylabel("Net Revenue (Millions)")
    save_figure("store_revenue.png")

    stockouts = data["stockouts"]

    plt.figure(figsize=(10, 5))
    plt.bar(
        stockouts["product_id"],
        stockouts["lost_units"],
    )
    plt.title("Lost Sales Units by Product — Synthetic Retail Data")
    plt.xlabel("Product")
    plt.ylabel("Lost Sales Units")
    save_figure("product_lost_sales.png")

    promotions = data["promotions"]

    plt.figure(figsize=(7, 5))
    plt.bar(
        promotions["promotion_group"],
        promotions["stockout_rate_pct"].astype(float),
    )
    plt.title("Stockout Rate by Promotion Group — Synthetic Data")
    plt.xlabel("Promotion Group")
    plt.ylabel("Stockout Observation Rate (%)")
    save_figure("promotion_stockout_rate.png")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("RETAILPULSE AI - PHASE 3 EDA")
    print("Reading analytics data from PostgreSQL...")

    kpis, datasets = fetch_data()

    if int(kpis["observations"]) != 70000:
        raise ValueError("Unexpected PostgreSQL observation count.")

    if int(kpis["units_sold"]) != 1483884:
        raise ValueError("Units sold do not match the source.")

    if int(kpis["lost_units"]) != 79971:
        raise ValueError("Lost sales do not match the source.")

    kpi_export = {
        key: float(value) if value is not None else None
        for key, value in kpis.items()
    }

    kpi_export["data_source"] = "synthetic"
    kpi_export["date_start"] = "2024-01-01"
    kpi_export["date_end"] = "2025-11-30"

    output = REPORT_DIR / "phase3_kpi_summary.json"

    output.write_text(
        json.dumps(kpi_export, indent=2),
        encoding="utf-8",
    )

    for name, frame in datasets.items():
        csv_path = REPORT_DIR / f"phase3_{name}.csv"
        frame.to_csv(csv_path, index=False)

    generate_figures(datasets)

    print("PASS: KPI validation")
    print("PASS: Analytics CSV exports")
    print("PASS: Six EDA figures generated")
    print(f"KPI summary: {output}")
    print(f"Figures: {FIGURE_DIR}")
    print("PHASE 3 EDA COMPLETE")


if __name__ == "__main__":
    main()