import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.forecast_baselines import (
    REPORT_DIR,
    SERIES_KEYS,
    calculate_metrics,
    calculate_training_scales,
    load_dataset,
)
from src.models.train_random_forest import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_features,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

MODEL_NAME = "gradient_boosting_v1"
BASELINE_MODEL = "rolling_mean_7"


def load_comparison_metrics():
    baseline_path = REPORT_DIR / "baseline_metrics.csv"
    forest_path = REPORT_DIR / "random_forest_v1_metrics.csv"

    baseline = pd.read_csv(baseline_path)
    forest = pd.read_csv(forest_path)

    baseline_row = baseline.loc[
        baseline["model"] == BASELINE_MODEL
    ]

    if len(baseline_row) != 1 or len(forest) != 1:
        raise ValueError("Unexpected comparison metric rows.")

    if forest.iloc[0]["model"] != "random_forest_v1":
        raise ValueError("Unexpected Random Forest model.")

    return baseline_row.iloc[0], forest.iloc[0]


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

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.07,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predicted = np.maximum(
        model.predict(x_validation),
        0.0,
    )

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
        predicted,
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
    prediction_frame["prediction"] = predicted
    prediction_frame["error"] = predicted - y_validation

    artifact = {
        "model": model,
        "model_name": MODEL_NAME,
        "feature_columns": x_train.columns.tolist(),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "data_source": "synthetic",
        "forecast_horizon_days": 1,
        "training_split": "train",
        "evaluation_split": "validation",
    }

    return result, prediction_frame, artifact


def main():
    print("RETAILPULSE AI - PHASE 5 GRADIENT BOOSTING")
    print("Dataset: synthetic retail data")
    print("Training: chronological train split only")
    print("Evaluation: rolling one-day-ahead validation")
    print("Test split: untouched")
    print()

    frame = load_dataset()

    result, predictions, artifact = train_and_evaluate(frame)

    baseline, forest = load_comparison_metrics()

    comparison = pd.DataFrame(
        [
            baseline.to_dict(),
            forest.to_dict(),
            result,
        ]
    ).sort_values(
        ["wape_pct", "rmsse"]
    ).reset_index(drop=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{MODEL_NAME}.joblib"
    metrics_path = REPORT_DIR / f"{MODEL_NAME}_metrics.csv"
    predictions_path = (
        REPORT_DIR / f"{MODEL_NAME}_predictions.csv"
    )
    summary_path = REPORT_DIR / f"{MODEL_NAME}_summary.json"
    comparison_path = REPORT_DIR / "phase5_model_comparison.csv"

    joblib.dump(artifact, model_path)

    pd.DataFrame([result]).to_csv(
        metrics_path,
        index=False,
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    baseline_wape = float(baseline["wape_pct"])
    forest_wape = float(forest["wape_pct"])
    boosting_wape = result["wape_pct"]

    summary = {
        "model": MODEL_NAME,
        "data_source": "synthetic",
        "evaluation_split": "validation",
        "forecast_horizon_days": 1,
        "evaluation_protocol": "rolling_one_day_ahead",
        "test_split_used": False,
        "training_only_preprocessing": True,
        "baseline_wape_pct": baseline_wape,
        "random_forest_wape_pct": forest_wape,
        "gradient_boosting_wape_pct": boosting_wape,
        "beats_baseline": boosting_wape < baseline_wape,
        "beats_random_forest": boosting_wape < forest_wape,
        "best_model_by_validation_wape": comparison.iloc[0][
            "model"
        ],
        "metrics": result,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("VALIDATION MODEL COMPARISON")
    print(comparison.to_string(index=False))
    print()
    print(
        "Best model by validation WAPE: "
        f"{comparison.iloc[0]['model']}"
    )
    print()
    print(f"Model: {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")
    print(f"Comparison: {comparison_path}")
    print("PASS: Gradient Boosting training completed")


if __name__ == "__main__":
    main()