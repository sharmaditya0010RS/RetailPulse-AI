from pathlib import Path

import pandas as pd
from sqlalchemy import text

from config.database import get_engine

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "forecasting_dataset.csv"

SOURCE_QUERY = """
    SELECT
        date,
        store_id,
        product_id,
        units_sold
    FROM retailpulse.fact_daily_retail
    ORDER BY store_id, product_id, date
"""

EXPECTED_ROWS = 70000
EXPECTED_SERIES = 100


def load_history():
    engine = get_engine()

    try:
        with engine.connect() as connection:
            frame = pd.read_sql_query(
                text(SOURCE_QUERY),
                connection,
            )
    finally:
        engine.dispose()

    return frame


def build_features(history):
    frame = history.copy()

    frame["date"] = pd.to_datetime(frame["date"])

    frame = frame.sort_values(
        ["store_id", "product_id", "date"]
    ).reset_index(drop=True)

    if len(frame) != EXPECTED_ROWS:
        raise ValueError("Unexpected historical row count.")

    if frame.duplicated(
        ["date", "store_id", "product_id"]
    ).any():
        raise ValueError("Duplicate store-product-date keys.")

    series_count = frame[
        ["store_id", "product_id"]
    ].drop_duplicates().shape[0]

    if series_count != EXPECTED_SERIES:
        raise ValueError("Unexpected number of time series.")

    groups = frame.groupby(
        ["store_id", "product_id"],
        sort=False,
    )["units_sold"]

    frame["sales_lag_1"] = groups.shift(1)
    frame["sales_lag_7"] = groups.shift(7)
    frame["sales_lag_14"] = groups.shift(14)
    frame["sales_lag_28"] = groups.shift(28)

    for window in (7, 14, 28):
        frame[f"sales_rolling_mean_{window}"] = (
            frame.groupby(
                ["store_id", "product_id"],
                sort=False,
            )["units_sold"]
            .transform(
                lambda values: values.shift(1)
                .rolling(window, min_periods=window)
                .mean()
            )
        )

    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month
    frame["weekend_flag"] = (
        frame["day_of_week"] >= 5
    ).astype(int)

    frame = frame.dropna().reset_index(drop=True)

    expected_feature_rows = EXPECTED_ROWS - (
        EXPECTED_SERIES * 28
    )

    if len(frame) != expected_feature_rows:
        raise ValueError(
            "Unexpected forecasting feature row count."
        )

    return frame


def assign_splits(frame):
    result = frame.copy()

    train_end = pd.Timestamp("2025-05-31")
    validation_end = pd.Timestamp("2025-08-31")

    result["split"] = "test"

    result.loc[
        result["date"] <= train_end,
        "split",
    ] = "train"

    result.loc[
        (result["date"] > train_end)
        & (result["date"] <= validation_end),
        "split",
    ] = "validation"

    if result["split"].nunique() != 3:
        raise ValueError("Expected three chronological splits.")

    return result


def main():
    print("RETAILPULSE AI - PHASE 5 FORECASTING DATASET")
    print("Target: next-day observed units sold")
    print("Source: synthetic PostgreSQL retail data")

    history = load_history()
    features = build_features(history)
    dataset = assign_splits(features)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Historical observations: {len(history):,}")
    print(f"Forecasting observations: {len(dataset):,}")

    print("Split counts:")
    print(dataset["split"].value_counts().to_string())

    print(f"Output: {OUTPUT_PATH}")
    print("PASS: Lagged and rolling features created")
    print("PASS: Chronological splits assigned")
    print("PHASE 5 STEP 1 COMPLETE")


if __name__ == "__main__":
    main()