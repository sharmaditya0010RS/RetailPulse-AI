import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "forecasting_dataset.csv"
REPORT_DIR = ROOT / "reports" / "validation"

SERIES_KEYS = ["store_id", "product_id"]

BASELINES = {
    "naive_1": "sales_lag_1",
    "seasonal_naive_7": "sales_lag_7",
    "rolling_mean_7": "sales_rolling_mean_7",
}


def load_dataset():
    frame = pd.read_csv(DATA_PATH, parse_dates=["date"])

    required = {
        "date",
        "store_id",
        "product_id",
        "units_sold",
        "split",
        *BASELINES.values(),
    }

    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if frame.duplicated(["date", *SERIES_KEYS]).any():
        raise ValueError("Duplicate forecasting keys.")

    if frame[list(required)].isna().any().any():
        raise ValueError("Missing values in required columns.")

    if (frame["units_sold"] < 0).any():
        raise ValueError("Negative actual sales detected.")

    return frame


def calculate_training_scales(frame):
    train = frame.loc[frame["split"] == "train"].copy()

    if train.empty:
        raise ValueError("Training split is empty.")

    train = train.sort_values([*SERIES_KEYS, "date"])

    train["previous_sales"] = train.groupby(
        SERIES_KEYS
    )["units_sold"].shift(1)

    train["squared_difference"] = (
        train["units_sold"] - train["previous_sales"]
    ) ** 2

    scales = (
        train.groupby(SERIES_KEYS)["squared_difference"]
        .mean()
        .rename("training_scale")
        .reset_index()
    )

    if scales["training_scale"].isna().any():
        raise ValueError("Insufficient training history.")

    if (scales["training_scale"] <= 0).any():
        raise ValueError(
            "Zero training scale detected. "
            "RMSSE requires a positive denominator."
        )

    return scales


def calculate_metrics(actual, predicted, scales):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    scales = np.asarray(scales, dtype=float)

    if not (
        len(actual) == len(predicted) == len(scales)
    ):
        raise ValueError("Metric input lengths differ.")

    if len(actual) == 0:
        raise ValueError("No observations to evaluate.")

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
        and np.isfinite(scales).all()
    ):
        raise ValueError("Non-finite metric input.")

    if (scales <= 0).any():
        raise ValueError("Training scales must be positive.")

    actual_total = actual.sum()
    if actual_total <= 0:
        raise ValueError("WAPE denominator must be positive.")

    errors = predicted - actual

    wape = 100.0 * np.abs(errors).sum() / actual_total

    # Pooled RMSSE: average scaled squared error
    # across all validation observations, then square root.
    rmsse = np.sqrt(np.mean((errors**2) / scales))

    # Positive bias means overforecasting.
    bias_pct = 100.0 * errors.sum() / actual_total

    return {
        "wape_pct": round(float(wape), 4),
        "rmsse": round(float(rmsse), 4),
        "bias_pct": round(float(bias_pct), 4),
    }


def evaluate_baselines(frame):
    scales = calculate_training_scales(frame)

    validation = frame.loc[
        frame["split"] == "validation"
    ].copy()

    if validation.empty:
        raise ValueError("Validation split is empty.")

    validation = validation.merge(
        scales,
        on=SERIES_KEYS,
        how="left",
        validate="many_to_one",
    )

    if validation["training_scale"].isna().any():
        raise ValueError("Validation series missing training scale.")

    results = []
    predictions = []

    for model_name, feature_name in BASELINES.items():
        predicted = validation[feature_name].to_numpy(
            dtype=float
        )

        actual = validation["units_sold"].to_numpy(
            dtype=float
        )

        metrics = calculate_metrics(
            actual,
            predicted,
            validation["training_scale"],
        )

        results.append(
            {
                "model": model_name,
                "split": "validation",
                "observations": len(validation),
                **metrics,
            }
        )

        prediction_frame = validation[
            ["date", *SERIES_KEYS, "units_sold"]
        ].copy()

        prediction_frame["model"] = model_name
        prediction_frame["prediction"] = predicted
        prediction_frame["error"] = predicted - actual

        predictions.append(prediction_frame)

    results_frame = pd.DataFrame(results).sort_values(
        ["wape_pct", "rmsse"]
    ).reset_index(drop=True)

    predictions_frame = pd.concat(
        predictions,
        ignore_index=True,
    )

    return results_frame, predictions_frame


def main():
    print("RETAILPULSE AI - PHASE 5 BASELINE FORECASTS")
    print("Dataset: synthetic retail data")
    print("Evaluation: rolling one-day-ahead validation")

    frame = load_dataset()

    results, predictions = evaluate_baselines(frame)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORT_DIR / "baseline_metrics.csv"
    predictions_path = REPORT_DIR / "baseline_predictions.csv"
    summary_path = REPORT_DIR / "baseline_summary.json"

    results.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    summary = {
        "data_source": "synthetic",
        "evaluation_split": "validation",
        "forecast_horizon_days": 1,
        "evaluation_protocol": "rolling_one_day_ahead",
        "rmsse_scaling": (
            "Per-series mean squared first difference "
            "from training data only"
        ),
        "bias_convention": (
            "Positive = overforecast; negative = underforecast"
        ),
        "best_baseline_by_wape": results.iloc[0]["model"],
        "metrics": results.to_dict(orient="records"),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print(results.to_string(index=False))
    print()
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")
    print("PASS: Baseline evaluation complete")


if __name__ == "__main__":
    main()