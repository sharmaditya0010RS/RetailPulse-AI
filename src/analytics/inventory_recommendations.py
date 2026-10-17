"""Illustrative inventory recommendations using one-day demand forecasts."""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "store_id",
    "product_id",
    "predicted_units",
    "closing_stock",
    "reorder_point",
    "supplier_lead_time_days",
)

INTEGER_COLUMNS = (
    "closing_stock",
    "reorder_point",
    "supplier_lead_time_days",
)

DEFAULT_SAFETY_DAYS = 2


def validate_inventory_inputs(frame, safety_days=DEFAULT_SAFETY_DAYS):
    """Validate the inputs for a simple inventory planning heuristic."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Inventory inputs must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError("Inventory inputs cannot be empty.")

    if (
        isinstance(safety_days, bool)
        or not isinstance(safety_days, (int, float))
        or not np.isfinite(safety_days)
        or safety_days < 0
    ):
        raise ValueError("Safety days must be a finite nonnegative number.")

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(f"Missing inventory columns: {missing}")

    validated = frame.copy(deep=True)

    for column in ("store_id", "product_id"):
        if validated[column].isna().any():
            raise ValueError(f"Missing identifier values in {column}.")

        if not validated[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError(f"Invalid identifier values in {column}.")

    if validated.duplicated(["store_id", "product_id"]).any():
        raise ValueError("Duplicate store-product inventory records.")

    for column in ("predicted_units", *INTEGER_COLUMNS):
        values = pd.to_numeric(validated[column], errors="coerce")

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid numeric values in {column}.")

        if (values < 0).any():
            raise ValueError(f"Negative values in {column}.")

        if column in INTEGER_COLUMNS:
            if (values != np.floor(values)).any():
                raise ValueError(f"Non-integer values in {column}.")

            validated[column] = values.astype("int64")
        else:
            validated[column] = values.astype(float)

    if (validated["supplier_lead_time_days"] < 1).any():
        raise ValueError("Supplier lead time must be at least one day.")

    return validated


def recommend_replenishment(frame, safety_days=DEFAULT_SAFETY_DAYS):
    """Generate nonnegative order suggestions from daily demand estimates.

    This is an illustrative policy, not a validated multi-day forecast.
    Outstanding purchase orders are not deducted because their open status
    cannot be established from the available snapshot.
    """
    validated = validate_inventory_inputs(frame, safety_days=safety_days)

    result = validated.copy(deep=True)

    demand = result["predicted_units"]
    lead_time = result["supplier_lead_time_days"]
    available_stock = result["closing_stock"]

    result["safety_days"] = float(safety_days)

    result["lead_time_demand_units"] = (
        demand * lead_time
    )

    result["safety_stock_units"] = np.ceil(
        demand * safety_days
    ).astype("int64")

    result["target_stock_units"] = np.ceil(
        demand * (lead_time + safety_days)
    ).astype("int64")

    result["inventory_position_units"] = available_stock

    result["suggested_order_units"] = (
        result["target_stock_units"] - available_stock
    ).clip(lower=0).astype("int64")

    result["below_existing_reorder_point"] = (
        available_stock < result["reorder_point"]
    )

    result["stockout_flag"] = available_stock.eq(0)

    result["coverage_days"] = np.where(
        demand > 0,
        available_stock / demand.replace(0, np.nan),
        np.inf,
    )

    result["inventory_status"] = np.select(
        [
            result["stockout_flag"],
            result["coverage_days"] < lead_time,
            result["suggested_order_units"] > 0,
        ],
        [
            "STOCKOUT",
            "BELOW_LEAD_TIME_COVERAGE",
            "REPLENISHMENT_SUGGESTED",
        ],
        default="SUFFICIENT_FOR_POLICY",
    )

    return result