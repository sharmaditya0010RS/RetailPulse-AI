import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
START_DATE = "2024-01-01"
N_DAYS = 700
SEED = 42

CATEGORIES = [
    "Grocery", "Grocery", "Beverages", "Beverages", "Snacks",
    "Snacks", "Personal Care", "Personal Care", "Household", "Household",
]
BRANDS = [
    "FreshCo", "DailyFarm", "Hydra", "BrewHouse", "CrunchCo",
    "Snackly", "PureGlow", "CarePlus", "HomeBright", "CleanNest",
]
REGIONS = ["North", "South", "East", "West", "Central"]
BASE_PRICES = np.array([45, 80, 35, 150, 30, 65, 120, 210, 95, 175], dtype=float)
BASE_DEMAND = np.array([20, 14, 24, 10, 28, 16, 12, 8, 14, 9], dtype=float)

COLUMNS = [
    "date", "store_id", "product_id", "category", "region", "brand",
    "supplier_id", "store_tier", "store_area_sqm", "base_price",
    "unit_cost", "selling_price", "competitor_price", "promotion_flag",
    "promotion_type", "discount_pct", "advertising_spend",
    "holiday_flag", "festival_flag", "weekend_flag", "day_of_week",
    "month", "iso_week", "day_of_year", "temperature_c", "rainfall_mm",
    "humidity_pct", "store_footfall", "staff_count", "operating_hours",
    "loyalty_customer_share", "repeat_customer_share",
    "average_basket_size", "local_demand_index", "inflation_index",
    "competitor_activity_index", "opening_stock", "units_sold",
    "units_returned", "closing_stock", "lost_sales_units",
    "stockout_flag", "reorder_point", "units_ordered", "units_received",
    "supplier_lead_time_days", "supplier_reliability", "gross_revenue",
    "refund_amount", "net_revenue", "gross_profit_estimate",
    "latent_demand_units",
]


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    records = []

    store_scale = rng.uniform(0.75, 1.45, size=10)
    store_area = rng.integers(500, 2401, size=10)
    store_tiers = ["Premium" if x > 1.1 else "Standard" for x in store_scale]
    supplier_reliability = rng.uniform(0.84, 0.98, size=10)

    for store_idx in range(10):
        region = REGIONS[store_idx % len(REGIONS)]
        for product_idx in range(10):
            stock = int(rng.integers(100, 220))
            pending = {}
            product_price = float(BASE_PRICES[product_idx])
            unit_cost = round(product_price * float(rng.uniform(0.52, 0.72)), 2)
            reorder_point = int(BASE_DEMAND[product_idx] * store_scale[store_idx] * 5)
            lead_time = int(rng.integers(2, 7))
            reliability = float(supplier_reliability[product_idx])

            for day_idx, date in enumerate(dates):
                received = pending.pop(day_idx, 0)
                opening_stock = stock + received

                weekday = date.weekday()
                weekend = int(weekday >= 5)
                month = date.month
                doy = date.dayofyear
                festival = int((month == 10 and date.day >= 15) or month == 11)
                holiday = int(
                    (month == 1 and date.day in (1, 26))
                    or (month == 8 and date.day == 15)
                    or (month == 10 and date.day == 2)
                    or (month == 12 and date.day == 25)
                )

                seasonal = 1 + 0.14 * np.sin(2 * np.pi * doy / 365.25)
                weekly = 1.17 if weekend else 1.0
                festival_effect = 1.22 if festival else 1.0
                holiday_effect = 1.13 if holiday else 1.0
                inflation = 1 + 0.045 * day_idx / 365.25
                base_price = round(product_price * inflation, 2)

                promotion = int(rng.random() < (0.17 if weekend else 0.10))
                discount = float(rng.choice([0.05, 0.10, 0.15, 0.20])) if promotion else 0.0
                selling_price = round(base_price * (1 - discount), 2)
                competitor_price = round(
                    base_price * float(rng.uniform(0.94, 1.09)), 2
                )
                competitor_activity = float(rng.uniform(0.85, 1.18))
                advertising = round(float(rng.uniform(80, 550)) if promotion else 0.0, 2)

                temperature = round(
                    26 + 10 * np.sin(2 * np.pi * (doy - 90) / 365.25)
                    + float(rng.normal(0, 2.5)), 1
                )
                rainfall = round(
                    float(rng.gamma(2, 5)) if month in (6, 7, 8, 9) else
                    float(rng.gamma(1, 1.5)), 1
                )
                humidity = round(float(np.clip(58 + rainfall * 0.5 + rng.normal(0, 8), 25, 95)), 1)

                footfall = max(
                    40,
                    int(rng.normal(300 * store_scale[store_idx] * weekly, 35)),
                )
                staff = max(4, int(round(8 * store_scale[store_idx] + weekend * 2)))
                hours = 11.0 if weekend else 10.0
                loyalty = round(float(rng.uniform(0.25, 0.65)), 3)
                repeat_share = round(float(rng.uniform(0.20, 0.55)), 3)
                basket = round(float(rng.uniform(2.1, 5.2)), 2)
                local_index = float(np.clip(rng.normal(1.0, 0.08), 0.75, 1.25))

                price_ratio = selling_price / max(competitor_price, 1)
                demand_mean = (
                    BASE_DEMAND[product_idx]
                    * store_scale[store_idx]
                    * seasonal
                    * weekly
                    * festival_effect
                    * holiday_effect
                    * local_index
                    * (1 + 1.7 * discount)
                    * (1 + 0.0003 * advertising)
                    * np.clip(price_ratio ** -1.2, 0.7, 1.4)
                    * np.clip(footfall / (300 * store_scale[store_idx]), 0.65, 1.5)
                    / competitor_activity
                )

                latent_demand = int(rng.poisson(max(demand_mean, 0.1)))
                sold = min(latent_demand, opening_stock)
                lost = max(0, latent_demand - sold)
                closing = opening_stock - sold
                returns = int(rng.binomial(sold, 0.025))
                stockout = int(lost > 0)

                ordered = 0
                outstanding = sum(pending.values())
                if closing + outstanding <= reorder_point:
                    target = max(reorder_point * 3, 80)
                    ordered = max(0, target - closing - outstanding)
                    delivery_day = day_idx + lead_time
                    if delivery_day < N_DAYS:
                        delivered = int(rng.binomial(ordered, reliability))
                        pending[delivery_day] = pending.get(delivery_day, 0) + delivered

                gross_revenue = round(sold * selling_price, 2)
                refund_amount = round(returns * selling_price, 2)
                net_revenue = round(gross_revenue - refund_amount, 2)
                gross_profit = round(
                    net_revenue - (sold - returns) * unit_cost, 2
                )

                records.append({
                    "date": date,
                    "store_id": f"S{store_idx + 1:03d}",
                    "product_id": f"P{product_idx + 1:03d}",
                    "category": CATEGORIES[product_idx],
                    "region": region,
                    "brand": BRANDS[product_idx],
                    "supplier_id": f"SUP{product_idx + 1:03d}",
                    "store_tier": store_tiers[store_idx],
                    "store_area_sqm": int(store_area[store_idx]),
                    "base_price": base_price,
                    "unit_cost": unit_cost,
                    "selling_price": selling_price,
                    "competitor_price": competitor_price,
                    "promotion_flag": promotion,
                    "promotion_type": "Percentage" if promotion else "None",
                    "discount_pct": discount,
                    "advertising_spend": advertising,
                    "holiday_flag": holiday,
                    "festival_flag": festival,
                    "weekend_flag": weekend,
                    "day_of_week": weekday,
                    "month": month,
                    "iso_week": int(date.isocalendar().week),
                    "day_of_year": doy,
                    "temperature_c": temperature,
                    "rainfall_mm": rainfall,
                    "humidity_pct": humidity,
                    "store_footfall": footfall,
                    "staff_count": staff,
                    "operating_hours": hours,
                    "loyalty_customer_share": loyalty,
                    "repeat_customer_share": repeat_share,
                    "average_basket_size": basket,
                    "local_demand_index": round(local_index, 4),
                    "inflation_index": round(inflation, 5),
                    "competitor_activity_index": round(competitor_activity, 4),
                    "opening_stock": opening_stock,
                    "units_sold": sold,
                    "units_returned": returns,
                    "closing_stock": closing,
                    "lost_sales_units": lost,
                    "stockout_flag": stockout,
                    "reorder_point": reorder_point,
                    "units_ordered": ordered,
                    "units_received": received,
                    "supplier_lead_time_days": lead_time,
                    "supplier_reliability": round(reliability, 4),
                    "gross_revenue": gross_revenue,
                    "refund_amount": refund_amount,
                    "net_revenue": net_revenue,
                    "gross_profit_estimate": gross_profit,
                    "latent_demand_units": latent_demand,
                })

                stock = closing

    frame = pd.DataFrame.from_records(records, columns=COLUMNS)
    return frame.sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)


def validate(frame: pd.DataFrame) -> dict:
    assert len(frame) == 70000
    assert len(frame.columns) == 52
    assert frame["store_id"].nunique() == 10
    assert frame["product_id"].nunique() == 10
    assert frame["date"].nunique() == 700
    assert not frame.isna().any().any()
    assert not frame.duplicated(["date", "store_id", "product_id"]).any()

    assert (frame["units_sold"] <= frame["opening_stock"]).all()
    assert (
        frame["closing_stock"]
        == frame["opening_stock"] - frame["units_sold"]
    ).all()
    assert (
        frame["latent_demand_units"]
        == frame["units_sold"] + frame["lost_sales_units"]
    ).all()
    assert (frame["units_returned"] <= frame["units_sold"]).all()
    assert (frame["closing_stock"] >= 0).all()

    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "stores": int(frame["store_id"].nunique()),
        "products": int(frame["product_id"].nunique()),
        "days": int(frame["date"].nunique()),
        "start_date": str(frame["date"].min().date()),
        "end_date": str(frame["date"].max().date()),
        "total_units_sold": int(frame["units_sold"].sum()),
        "total_lost_sales_units": int(frame["lost_sales_units"].sum()),
        "stockout_rate": round(float(frame["stockout_flag"].mean()), 4),
        "net_revenue": round(float(frame["net_revenue"].sum()), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    frame = generate(args.seed)
    summary = validate(frame)

    output_dir = ROOT / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "retailpulse_synthetic_70k.csv"
    summary_path = output_dir / "retailpulse_generation_summary.json"

    frame.to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Dataset generated:", csv_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()