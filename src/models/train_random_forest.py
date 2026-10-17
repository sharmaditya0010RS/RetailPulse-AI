import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.forecast_baselines import (
    REPORT_DIR,
    SERIES_KEYS,
    calculate_metrics,
    calculate_training_scales,
    load_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

MODEL_NAME = "random_forest_v1"

NUMERIC_FEATURES = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_mean_14",
    "sales_rolling_mean_28",
    "day_of_week",
    "month",
    "weekend_flag",
]

CATEGORICAL_FEATURES = [
    "store_id",
    "product_id",
]

BASELINE_WAPE = 33.5911


def prepare_features(train, validation):
    train_features = pd.get_dummies(
        train[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        dtype=int,
    )

    validation_features = pd.get_dummies(
        validation[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        dtype=int,
    )

    validation_features = validation_features.reindex(
        columns=train_features.columns,
        fill_value=0,
    )

    if train_features.isna().any().any():
        raise ValueError("Missing training features.")

    if validation_features.isna().any().any():
        raise ValueError("Missing validation features.")

    return train_features, validation_features


def train_and_evaluate(frame):
    train = frame.loc[frame["split"] == "train"].copy()
    validation = frame.loc[
        frame["split"] == "validation"
    ].copy()

    if len(train) != 48900:
        raise ValueError("Unexpected training row count.")

    if len(validation) != 9200:
        raise ValueError("Unexpected validation row count.")

    if train["date"].max() >= validation["date"].min():
        raise ValueError("Chronological split overlap.")

    x_train, x_validation = prepare_features(
        train,
        validation,
    )

    y_train = train["units_sold"].to_numpy(dtype=float)
    y_validation = validation["units_sold"].to_numpy(
        dtype=float
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_validation)
    predictions = np.maximum(predictions, 0.0)

    scales = calculate_training_scales(frame)

    evaluation = validation.merge(
        scales,
        on=SERIES_KEYS,
        how="left",
        validate="many_to_one",
    )

    if evaluation["training_scale"].isna().any():
        raise ValueError("Missing training scale.")

    metrics = calculate_metrics(
        y_validation,
        predictions,
        evaluation["training_scale"].to_numpy(),
    )

    result = {
        "model": MODEL_NAME,
        "split": "validation",
        "observations": len(validation),
        **metrics,
    }

    prediction_frame = validation[
        ["date", *SERIES_KEYS, "units_sold"]
    ].copy()

    prediction_frame["model"] = MODEL_NAME
    prediction_frame["prediction"] = predictions
    prediction_frame["error"] = predictions - y_validation

    importance = pd.DataFrame(
        {
            "feature": x_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    artifact = {
        "model": model,
        "feature_columns": x_train.columns.tolist(),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_name": MODEL_NAME,
        "data_source": "synthetic",
        "forecast_horizon_days": 1,
    }

    return result, prediction_frame, importance, artifact


def main():
    print("RETAILPULSE AI - PHASE 5 RANDOM FOREST")
    print("Dataset: synthetic retail data")
    print("Training: chronological train split only")
    print("Evaluation: rolling one-day-ahead validation")
    print("Test split: untouched")
    print()

    frame = load_dataset()

    result, predictions, importance, artifact = (
        train_and_evaluate(frame)
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{MODEL_NAME}.joblib"
    metrics_path = REPORT_DIR / f"{MODEL_NAME}_metrics.csv"
    predictions_path = (
        REPORT_DIR / f"{MODEL_NAME}_predictions.csv"
    )
    importance_path = (
        REPORT_DIR / f"{MODEL_NAME}_feature_importance.csv"
    )
    summary_path = REPORT_DIR / f"{MODEL_NAME}_summary.json"

    joblib.dump(artifact, model_path)

    pd.DataFrame([result]).to_csv(
        metrics_path,
        index=False,
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    improvement_pct = (
        100.0
        * (BASELINE_WAPE - result["wape_pct"])
        / BASELINE_WAPE
    )

    summary = {
        "model": MODEL_NAME,
        "data_source": "synthetic",
        "evaluation_split": "validation",
        "forecast_horizon_days": 1,
        "evaluation_protocol": "rolling_one_day_ahead",
        "training_only_preprocessing": True,
        "test_split_used": False,
        "baseline_model": "rolling_mean_7",
        "baseline_wape_pct": BASELINE_WAPE,
        "relative_wape_improvement_pct": round(
            improvement_pct,
            4,
        ),
        "beats_baseline": result["wape_pct"] < BASELINE_WAPE,
        "metrics": result,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("VALIDATION RESULTS")
    print(pd.DataFrame([result]).to_string(index=False))
    print()
    print(f"Baseline WAPE: {BASELINE_WAPE:.4f}%")
    print(f"ML WAPE: {result['wape_pct']:.4f}%")
    print(f"Relative WAPE improvement: {improvement_pct:.2f}%")
    print()
    print("TOP 10 FEATURES")
    print(importance.head(10).to_string(index=False))
    print()
    print(f"Model: {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Feature importance: {importance_path}")
    print(f"Summary: {summary_path}")
    print("PASS: Random Forest training completed")


if __name__ == "__main__":
    main()