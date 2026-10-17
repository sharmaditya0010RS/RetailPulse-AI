import json

import joblib
import numpy as np
import pandas as pd

from src.models.forecast_baselines import (
    SERIES_KEYS,
    calculate_metrics,
    calculate_training_scales,
    load_dataset,
)
from src.models.select_champion import ROOT, SELECTION_PATH

REPORT_DIR = ROOT / "reports" / "test"

METRICS_PATH = REPORT_DIR / "holdout_metrics.csv"
PREDICTIONS_PATH = REPORT_DIR / "holdout_predictions.csv"
SUMMARY_PATH = REPORT_DIR / "holdout_summary.json"

BASELINE_NAME = "rolling_mean_7"
BASELINE_FEATURE = "sales_rolling_mean_7"


def load_frozen_champion():
    if not SELECTION_PATH.is_file():
        raise FileNotFoundError(
            "Frozen champion selection is missing."
        )

    selection = json.loads(
        SELECTION_PATH.read_text(encoding="utf-8")
    )

    if (
        selection["selection_status"]
        != "frozen_before_test_evaluation"
    ):
        raise ValueError("Champion selection is not frozen.")

    if selection["test_metrics_used_for_selection"] is not False:
        raise ValueError("Selection used test metrics.")

    if selection["champion_model"] != "gradient_boosting_v1":
        raise ValueError("Unexpected frozen champion.")

    if selection["forecast_horizon_days"] != 1:
        raise ValueError("Unexpected forecast horizon.")

    if (
        selection["evaluation_protocol"]
        != "rolling_one_day_ahead"
    ):
        raise ValueError("Unexpected evaluation protocol.")

    expected_artifact = (
        "models/gradient_boosting_v1.joblib"
    )

    if selection["champion_artifact"] != expected_artifact:
        raise ValueError("Unexpected champion artifact.")

    artifact_path = ROOT / expected_artifact

    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)

    artifact = joblib.load(artifact_path)

    if artifact["model_name"] != selection["champion_model"]:
        raise ValueError("Artifact model mismatch.")

    if artifact["data_source"] != "synthetic":
        raise ValueError("Unexpected data source.")

    if artifact["training_split"] != "train":
        raise ValueError("Unexpected artifact training split.")

    if artifact["evaluation_split"] != "validation":
        raise ValueError("Unexpected artifact evaluation split.")

    return selection, artifact


def prepare_holdout_features(test, artifact):
    numeric = artifact["numeric_features"]
    categorical = artifact["categorical_features"]
    columns = artifact["feature_columns"]

    features = pd.get_dummies(
        test[numeric + categorical],
        columns=categorical,
        dtype=int,
    )

    unknown = set(features.columns) - set(columns)

    if unknown:
        raise ValueError(
            f"Unknown holdout features: {sorted(unknown)}"
        )

    features = features.reindex(
        columns=columns,
        fill_value=0,
    )

    if features.isna().any().any():
        raise ValueError("Missing holdout features.")

    return features


def evaluate_holdout(frame, artifact):
    train = frame.loc[frame["split"] == "train"]
    validation = frame.loc[
        frame["split"] == "validation"
    ]
    test = frame.loc[frame["split"] == "test"].copy()

    if len(train) != 48900:
        raise ValueError("Unexpected training row count.")

    if len(validation) != 9200:
        raise ValueError("Unexpected validation row count.")

    if len(test) != 9100:
        raise ValueError("Unexpected test row count.")

    if not (
        train["date"].max()
        < validation["date"].min()
        <= validation["date"].max()
        < test["date"].min()
    ):
        raise ValueError("Chronological split overlap.")

    if test["date"].min() != pd.Timestamp("2025-09-01"):
        raise ValueError("Unexpected test start date.")

    if test["date"].max() != pd.Timestamp("2025-11-30"):
        raise ValueError("Unexpected test end date.")

    if test.duplicated(["date", *SERIES_KEYS]).any():
        raise ValueError("Duplicate holdout keys.")

    features = prepare_holdout_features(test, artifact)

    actual = test["units_sold"].to_numpy(dtype=float)

    champion_predictions = np.maximum(
        artifact["model"].predict(features),
        0.0,
    )

    baseline_predictions = test[
        BASELINE_FEATURE
    ].to_numpy(dtype=float)

    scales = calculate_training_scales(frame)

    evaluation = test.merge(
        scales,
        on=SERIES_KEYS,
        how="left",
        sort=False,
        validate="many_to_one",
    )

    if evaluation["training_scale"].isna().any():
        raise ValueError("Missing training scale.")

    if not evaluation[
        ["date", *SERIES_KEYS]
    ].reset_index(drop=True).equals(
        test[
            ["date", *SERIES_KEYS]
        ].reset_index(drop=True)
    ):
        raise ValueError("Holdout row order changed.")

    scale_values = evaluation[
        "training_scale"
    ].to_numpy(dtype=float)

    metric_rows = []
    prediction_frames = []

    candidates = [
        (artifact["model_name"], champion_predictions),
        (BASELINE_NAME, baseline_predictions),
    ]

    for name, predicted in candidates:
        metrics = calculate_metrics(
            actual,
            predicted,
            scale_values,
        )

        metric_rows.append(
            {
                "model": name,
                "split": "test",
                "observations": len(test),
                **metrics,
            }
        )

        predictions = test[
            ["date", *SERIES_KEYS, "units_sold"]
        ].copy()

        predictions["model"] = name
        predictions["prediction"] = predicted
        predictions["error"] = predicted - actual

        prediction_frames.append(predictions)

    return (
        pd.DataFrame(metric_rows),
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ),
    )


def main():
    existing = [
        path
        for path in (
            METRICS_PATH,
            PREDICTIONS_PATH,
            SUMMARY_PATH,
        )
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Holdout results already exist. "
            "Refusing to repeat or overwrite evaluation: "
            f"{existing}"
        )

    selection, artifact = load_frozen_champion()
    frame = load_dataset()

    metrics, predictions = evaluate_holdout(
        frame,
        artifact,
    )

    champion = metrics.loc[
        metrics["model"] == selection["champion_model"]
    ].iloc[0]

    baseline = metrics.loc[
        metrics["model"] == BASELINE_NAME
    ].iloc[0]

    validation_wape = selection[
        "validation_metrics"
    ]["wape_pct"]

    summary = {
        "project": "RetailPulse AI 2.0",
        "data_source": "synthetic",
        "selection_record": (
            "models/champion_selection.json"
        ),
        "champion_model": selection["champion_model"],
        "champion_frozen_before_test": True,
        "retraining_performed": False,
        "hyperparameter_tuning_performed": False,
        "test_used_for_model_selection": False,
        "test_period": selection["reserved_test_period"],
        "forecast_horizon_days": 1,
        "evaluation_protocol": "rolling_one_day_ahead",
        "test_observations_per_model": 9100,
        "rmsse_scale_source": "training_only",
        "champion_validation_wape_pct": validation_wape,
        "champion_test_wape_pct": float(
            champion["wape_pct"]
        ),
        "validation_to_test_wape_change_points": round(
            float(champion["wape_pct"]) - validation_wape,
            4,
        ),
        "baseline_test_wape_pct": float(
            baseline["wape_pct"]
        ),
        "champion_test_metrics": champion.to_dict(),
        "baseline_test_metrics": baseline.to_dict(),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(
        METRICS_PATH,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    with SUMMARY_PATH.open(
        "x",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print("RETAILPULSE AI - FINAL HOLDOUT EVALUATION")
    print("Data source: synthetic")
    print(f"Frozen champion: {selection['champion_model']}")
    print("Model retrained: False")
    print("Test used for selection: False")
    print()
    print("HELD-OUT TEST RESULTS")
    print(metrics.to_string(index=False))
    print()
    print(
        "Champion validation WAPE: "
        f"{validation_wape:.4f}%"
    )
    print(
        "Champion test WAPE: "
        f"{champion['wape_pct']:.4f}%"
    )
    print(
        "Validation-to-test WAPE change: "
        f"{summary['validation_to_test_wape_change_points']:+.4f} "
        "percentage points"
    )
    print()
    print(f"Metrics: {METRICS_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("PASS: Final holdout evaluation completed")


if __name__ == "__main__":
    main()