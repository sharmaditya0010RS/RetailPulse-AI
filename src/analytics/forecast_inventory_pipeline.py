"""Integrate frozen next-day forecasts with synthetic inventory planning."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.inventory_recommendations import recommend_replenishment
from src.models.forecast_inference import load_champion, predict_daily_demand

ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = ROOT / "data" / "raw" / "retailpulse_synthetic_70k.csv"

HISTORY_COLUMNS = (
    "date",
    "store_id",
    "product_id",
    "units_sold",
)

INVENTORY_COLUMNS = (
    "date",
    "store_id",
    "product_id",
    "closing_stock",
    "reorder_point",
    "supplier_lead_time_days",
    "supplier_reliability",
    "units_ordered",
    "units_received",
)

SERIES_KEYS = ["store_id", "product_id"]


def load_historical_data(path=RAW_DATA_PATH):
    """Load historical demand and inventory without modifying source files."""
    columns = sorted(set(HISTORY_COLUMNS) | set(INVENTORY_COLUMNS))

    frame = pd.read_csv(path, usecols=columns)

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")

    if frame.empty:
        raise ValueError("Historical retail data is empty.")

    if frame[list(HISTORY_COLUMNS)].isna().any().any():
        raise ValueError("Missing historical demand identifiers or values.")

    if frame.duplicated(["date", *SERIES_KEYS]).any():
        raise ValueError("Duplicate historical store-product dates.")

    demand = pd.to_numeric(frame["units_sold"], errors="coerce")

    if (
        not np.isfinite(demand.to_numpy(dtype=float)).all()
        or (demand < 0).any()
        or (demand != np.floor(demand)).any()
    ):
        raise ValueError("Historical units sold must be nonnegative integers.")

    frame["units_sold"] = demand.astype("int64")

    return frame.sort_values(
        [*SERIES_KEYS, "date"]
    ).reset_index(drop=True)


def build_next_day_features(history):
    """Build one-day-ahead features from observations through the cutoff."""
    if not isinstance(history, pd.DataFrame) or history.empty:
        raise ValueError("Historical demand must be a nonempty DataFrame.")

    required = set(HISTORY_COLUMNS)
    missing = sorted(required - set(history.columns))

    if missing:
        raise ValueError(f"Missing demand history columns: {missing}")

    frame = history[list(HISTORY_COLUMNS)].copy(deep=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")

    if frame.duplicated(["date", *SERIES_KEYS]).any():
        raise ValueError("Duplicate store-product history dates.")

    if frame[list(HISTORY_COLUMNS)].isna().any().any():
        raise ValueError("Missing demand history values.")

    demand = pd.to_numeric(frame["units_sold"], errors="coerce")

    if (
        not np.isfinite(demand.to_numpy(dtype=float)).all()
        or (demand < 0).any()
    ):
        raise ValueError("Invalid historical demand values.")

    frame["units_sold"] = demand

    cutoff = frame["date"].max()
    forecast_date = cutoff + pd.Timedelta(days=1)

    frame = frame.sort_values(
        [*SERIES_KEYS, "date"]
    ).reset_index(drop=True)

    rows = []

    for (store_id, product_id), group in frame.groupby(
        SERIES_KEYS,
        sort=True,
    ):
        if len(group) < 28:
            raise ValueError(
                f"Insufficient history for {store_id}/{product_id}."
            )

        if group["date"].max() != cutoff:
            raise ValueError(
                f"History is not current for {store_id}/{product_id}."
            )

        expected_dates = pd.date_range(
            end=cutoff,
            periods=28,
            freq="D",
        )

        observed_dates = pd.DatetimeIndex(group["date"].tail(28))

        if not observed_dates.equals(expected_dates):
            raise ValueError(
                f"Missing recent daily history for {store_id}/{product_id}."
            )

        values = group["units_sold"].to_numpy(dtype=float)

        row = {
            "date": forecast_date,
            "store_id": store_id,
            "product_id": product_id,
            "sales_lag_1": values[-1],
            "sales_lag_7": values[-7],
            "sales_lag_14": values[-14],
            "sales_lag_28": values[-28],
            "sales_rolling_mean_7": values[-7:].mean(),
            "sales_rolling_mean_14": values[-14:].mean(),
            "sales_rolling_mean_28": values[-28:].mean(),
            "day_of_week": forecast_date.dayofweek,
            "month": forecast_date.month,
            "weekend_flag": int(forecast_date.dayofweek >= 5),
        }

        rows.append(row)

    features = pd.DataFrame(rows)

    if features.duplicated(SERIES_KEYS).any():
        raise ValueError("Duplicate forecast feature rows.")

    return features


def build_inventory_snapshot(history):
    """Select the latest inventory state for every store-product series."""
    required = set(INVENTORY_COLUMNS)
    missing = sorted(required - set(history.columns))

    if missing:
        raise ValueError(f"Missing inventory history columns: {missing}")

    frame = history[list(INVENTORY_COLUMNS)].copy(deep=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")

    latest_date = frame["date"].max()
    snapshot = frame.loc[frame["date"] == latest_date].copy()

    if snapshot.empty:
        raise ValueError("Latest inventory snapshot is empty.")

    if snapshot.duplicated(SERIES_KEYS).any():
        raise ValueError("Duplicate store-product inventory records.")

    historical_keys = history[SERIES_KEYS].drop_duplicates()

    if len(snapshot) != len(historical_keys):
        raise ValueError(
            "Latest inventory snapshot does not cover all series."
        )

    if not snapshot[SERIES_KEYS].merge(
        historical_keys,
        on=SERIES_KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )["_merge"].eq("both").all():
        raise ValueError("Inventory snapshot contains unexpected series.")

    return snapshot.rename(
        columns={"date": "inventory_snapshot_date"}
    ).sort_values(SERIES_KEYS).reset_index(drop=True)


def generate_inventory_plan(history, artifact=None, safety_days=2):
    """Produce synthetic next-day forecasts and replenishment suggestions."""
    features = build_next_day_features(history)
    snapshot = build_inventory_snapshot(history)

    forecast_date = features["date"].iloc[0]
    snapshot_date = snapshot["inventory_snapshot_date"].iloc[0]

    if forecast_date != snapshot_date + pd.Timedelta(days=1):
        raise ValueError(
            "Forecast date must immediately follow inventory snapshot."
        )

    if artifact is None:
        artifact = load_champion()

    predictions = predict_daily_demand(features, artifact)

    plan = predictions.merge(
        snapshot,
        on=SERIES_KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not plan["_merge"].eq("both").all():
        raise ValueError("Forecast and inventory series do not align.")

    plan = plan.drop(columns="_merge")

    plan = recommend_replenishment(
        plan,
        safety_days=safety_days,
    )

    plan["forecast_model"] = artifact["model_name"]
    plan["planning_method"] = "one_day_forecast_times_coverage_days"
    plan["data_source"] = "synthetic"
    plan["open_purchase_orders_included"] = False

    return plan.sort_values(SERIES_KEYS).reset_index(drop=True)


def main():
    """Run a read-only next-day planning demonstration."""
    history = load_historical_data()
    plan = generate_inventory_plan(history)

    print("MODEL:", plan["forecast_model"].iloc[0])
    print("INVENTORY DATE:", plan["inventory_snapshot_date"].iloc[0].date())
    print("FORECAST DATE:", plan["date"].iloc[0].date())
    print("STORE-PRODUCT SERIES:", len(plan))
    print("TOTAL SUGGESTED ORDER UNITS:", int(plan["suggested_order_units"].sum()))

    print("\nINVENTORY STATUS COUNTS:")
    print(plan["inventory_status"].value_counts().to_string())

    print("\nSAMPLE RECOMMENDATIONS:")
    columns = [
        "date",
        "store_id",
        "product_id",
        "predicted_units",
        "closing_stock",
        "supplier_lead_time_days",
        "target_stock_units",
        "suggested_order_units",
        "inventory_status",
    ]

    print(plan[columns].head(10).to_string(index=False))

    if len(plan) != 100:
        raise ValueError("Expected 100 synthetic store-product plans.")

    print("\nPASS: Forecast-to-inventory integration")


if __name__ == "__main__":
    main()