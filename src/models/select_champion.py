import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "validation"
MODEL_DIR = ROOT / "models"

COMPARISON_PATH = REPORT_DIR / "phase5_model_comparison.csv"
SELECTION_PATH = MODEL_DIR / "champion_selection.json"

EXPECTED_MODELS = {
    "rolling_mean_7",
    "random_forest_v1",
    "gradient_boosting_v1",
}


def select_champion(comparison):
    required = {
        "model",
        "split",
        "observations",
        "wape_pct",
        "rmsse",
        "bias_pct",
    }

    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(
            f"Missing comparison columns: {sorted(missing)}"
        )

    if len(comparison) != len(EXPECTED_MODELS):
        raise ValueError("Unexpected model comparison row count.")

    if set(comparison["model"]) != EXPECTED_MODELS:
        raise ValueError("Unexpected model comparison entries.")

    if comparison["model"].duplicated().any():
        raise ValueError("Duplicate model comparison entries.")

    if not (comparison["split"] == "validation").all():
        raise ValueError("Only validation metrics are allowed.")

    if not (comparison["observations"] == 9200).all():
        raise ValueError("Unexpected validation observations.")

    metric_columns = ["wape_pct", "rmsse", "bias_pct"]

    if comparison[metric_columns].isna().any().any():
        raise ValueError("Missing validation metrics.")

    if not (
        comparison["wape_pct"].ge(0).all()
        and comparison["rmsse"].ge(0).all()
    ):
        raise ValueError("Invalid error metrics.")

    ranked = comparison.sort_values(
        ["wape_pct", "rmsse", "model"]
    ).reset_index(drop=True)

    champion = ranked.iloc[0]

    if champion["model"] == "rolling_mean_7":
        raise ValueError(
            "Baseline won; no serialized ML champion selected."
        )

    model_path = MODEL_DIR / f"{champion['model']}.joblib"

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    selection = {
        "project": "RetailPulse AI 2.0",
        "data_source": "synthetic",
        "selection_status": "frozen_before_test_evaluation",
        "selection_metric": "validation_wape_pct",
        "tie_breaker": "validation_rmsse",
        "champion_model": str(champion["model"]),
        "champion_artifact": str(
            model_path.relative_to(ROOT)
        ).replace("\\", "/"),
        "validation_metrics": {
            "wape_pct": float(champion["wape_pct"]),
            "rmsse": float(champion["rmsse"]),
            "bias_pct": float(champion["bias_pct"]),
        },
        "training_period": {
            "start": "2024-01-29",
            "end": "2025-05-31",
        },
        "validation_period": {
            "start": "2025-06-01",
            "end": "2025-08-31",
        },
        "reserved_test_period": {
            "start": "2025-09-01",
            "end": "2025-11-30",
        },
        "forecast_horizon_days": 1,
        "evaluation_protocol": "rolling_one_day_ahead",
        "test_metrics_used_for_selection": False,
        "rmsse_scaling": (
            "Per-series mean squared first difference "
            "on feature-ready training rows only"
        ),
        "candidate_metrics": ranked.to_dict(
            orient="records"
        ),
    }

    return selection


def main():
    if SELECTION_PATH.exists():
        raise FileExistsError(
            "Champion selection already exists. "
            "Refusing to overwrite the frozen selection."
        )

    comparison = pd.read_csv(COMPARISON_PATH)
    selection = select_champion(comparison)

    selection["frozen_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with SELECTION_PATH.open(
        "x",
        encoding="utf-8",
    ) as file:
        json.dump(selection, file, indent=2)
        file.write("\n")

    print("RETAILPULSE AI - CHAMPION SELECTION")
    print(f"Champion: {selection['champion_model']}")
    print(
        "Validation WAPE: "
        f"{selection['validation_metrics']['wape_pct']:.4f}%"
    )
    print(
        "Validation RMSSE: "
        f"{selection['validation_metrics']['rmsse']:.4f}"
    )
    print("Test metrics used for selection: False")
    print(f"Selection record: {SELECTION_PATH}")
    print("PASS: Champion selection frozen")


if __name__ == "__main__":
    main()