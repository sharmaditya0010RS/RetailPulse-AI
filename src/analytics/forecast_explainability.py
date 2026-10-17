import json

import joblib
import numpy as np
import pandas as pd

from src.models.forecast_baselines import load_dataset
from src.models.select_champion import ROOT, SELECTION_PATH

REPORT_DIR = ROOT / "reports" / "validation"

IMPORTANCE_PATH = (
    REPORT_DIR / "gradient_boosting_permutation_importance.csv"
)
SUMMARY_PATH = (
    REPORT_DIR / "gradient_boosting_explainability_summary.json"
)

REPEATS = 3
RANDOM_STATE = 42


def load_champion():
    selection = json.loads(
        SELECTION_PATH.read_text(encoding="utf-8")
    )

    if selection["champion_model"] != "gradient_boosting_v1":
        raise ValueError("Unexpected champion model.")

    if (
        selection["selection_status"]
        != "frozen_before_test_evaluation"
    ):
        raise ValueError("Champion selection is not frozen.")

    if selection["test_metrics_used_for_selection"] is not False:
        raise ValueError("Test data influenced champion selection.")

    artifact_path = ROOT / selection["champion_artifact"]

    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)

    artifact = joblib.load(artifact_path)

    if artifact["model_name"] != selection["champion_model"]:
        raise ValueError("Champion artifact mismatch.")

    if artifact["data_source"] != "synthetic":
        raise ValueError("Unexpected data source.")

    if artifact["training_split"] != "train":
        raise ValueError("Unexpected training split.")

    if artifact["evaluation_split"] != "validation":
        raise ValueError("Unexpected evaluation split.")

    return selection, artifact


def encode_features(frame, artifact):
    numeric = artifact["numeric_features"]
    categorical = artifact["categorical_features"]
    columns = artifact["feature_columns"]

    encoded = pd.get_dummies(
        frame[numeric + categorical],
        columns=categorical,
        dtype=int,
    )

    unknown = set(encoded.columns) - set(columns)

    if unknown:
        raise ValueError(
            f"Unexpected encoded features: {sorted(unknown)}"
        )

    encoded = encoded.reindex(
        columns=columns,
        fill_value=0,
    )

    if encoded.isna().any().any():
        raise ValueError("Missing feature values.")

    return encoded


def calculate_wape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError("Actual/prediction length mismatch.")

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
    ):
        raise ValueError("Non-finite evaluation values.")

    denominator = actual.sum()

    if denominator <= 0:
        raise ValueError("WAPE denominator must be positive.")

    return (
        np.abs(actual - predicted).sum()
        / denominator
        * 100.0
    )


def calculate_permutation_importance(
    validation,
    artifact,
    repeats=REPEATS,
    random_state=RANDOM_STATE,
):
    if repeats < 1:
        raise ValueError("At least one repeat is required.")

    raw_features = (
        artifact["numeric_features"]
        + artifact["categorical_features"]
    )

    if validation.empty:
        raise ValueError("Validation data is empty.")

    if not (validation["split"] == "validation").all():
        raise ValueError("Only validation rows are permitted.")

    working = validation.reset_index(drop=True).copy()

    actual = working["units_sold"].to_numpy(dtype=float)

    original_features = encode_features(
        working,
        artifact,
    )

    original_predictions = np.maximum(
        artifact["model"].predict(original_features),
        0.0,
    )

    original_wape = calculate_wape(
        actual,
        original_predictions,
    )

    rng = np.random.default_rng(random_state)
    rows = []

    for feature in raw_features:
        shuffled_wapes = []

        for _ in range(repeats):
            modified = working.copy()

            modified[feature] = rng.permutation(
                modified[feature].to_numpy()
            )

            encoded = encode_features(
                modified,
                artifact,
            )

            predicted = np.maximum(
                artifact["model"].predict(encoded),
                0.0,
            )

            shuffled_wapes.append(
                calculate_wape(actual, predicted)
            )

        changes = np.asarray(shuffled_wapes) - original_wape

        rows.append(
            {
                "feature": feature,
                "baseline_validation_wape_pct": round(
                    original_wape, 4
                ),
                "permuted_wape_mean_pct": round(
                    float(np.mean(shuffled_wapes)), 4
                ),
                "wape_increase_mean_pp": round(
                    float(np.mean(changes)), 4
                ),
                "wape_increase_std_pp": round(
                    float(np.std(changes)), 4
                ),
                "repeats": repeats,
            }
        )

    importance = pd.DataFrame(rows)

    importance = importance.sort_values(
        ["wape_increase_mean_pp", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return importance, original_wape


def main():
    existing = [
        path
        for path in (IMPORTANCE_PATH, SUMMARY_PATH)
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Explainability report already exists. "
            f"Refusing to overwrite: {existing}"
        )

    selection, artifact = load_champion()
    frame = load_dataset()

    validation = frame.loc[
        frame["split"] == "validation"
    ].copy()

    if len(validation) != 9200:
        raise ValueError("Unexpected validation row count.")

    if validation["date"].min() != pd.Timestamp(
        "2025-06-01"
    ):
        raise ValueError("Unexpected validation start date.")

    if validation["date"].max() != pd.Timestamp(
        "2025-08-31"
    ):
        raise ValueError("Unexpected validation end date.")

    importance, original_wape = (
        calculate_permutation_importance(
            validation,
            artifact,
        )
    )

    expected_wape = selection[
        "validation_metrics"
    ]["wape_pct"]

    if not np.isclose(
        original_wape,
        expected_wape,
        atol=0.0001,
    ):
        raise ValueError(
            "Recomputed validation WAPE does not "
            "match frozen selection."
        )

    summary = {
        "project": "RetailPulse AI 2.0",
        "data_source": "synthetic",
        "model": selection["champion_model"],
        "analysis_split": "validation",
        "validation_observations": len(validation),
        "method": "raw_feature_permutation_importance",
        "metric": "wape",
        "importance_unit": "percentage_points",
        "repeats": REPEATS,
        "random_state": RANDOM_STATE,
        "original_validation_wape_pct": round(
            original_wape, 4
        ),
        "features_evaluated": len(importance),
        "model_retrained": False,
        "test_data_used": False,
        "interpretation": (
            "Positive WAPE increase indicates that "
            "shuffling the feature worsened validation "
            "predictions. Importance is associational, "
            "not causal."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    with SUMMARY_PATH.open(
        "x",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print("RETAILPULSE AI - CHAMPION EXPLAINABILITY")
    print("Data source: synthetic")
    print(f"Model: {selection['champion_model']}")
    print("Analysis split: validation")
    print("Test data used: False")
    print("Model retrained: False")
    print(
        "Original validation WAPE: "
        f"{original_wape:.4f}%"
    )
    print()
    print("PERMUTATION IMPORTANCE")
    print(
        importance[
            [
                "feature",
                "wape_increase_mean_pp",
                "wape_increase_std_pp",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Importance report: {IMPORTANCE_PATH}")
    print(f"Summary report: {SUMMARY_PATH}")
    print("PASS: Champion explainability completed")


if __name__ == "__main__":
    main()