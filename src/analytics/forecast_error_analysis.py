import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.forecast_baselines import (
    SERIES_KEYS,
    calculate_training_scales,
    load_dataset,
)

ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    ROOT / "reports" / "test" / "holdout_predictions.csv"
)
METRICS_PATH = (
    ROOT / "reports" / "test" / "holdout_metrics.csv"
)
OUTPUT_DIR = ROOT / "reports" / "test" / "error_analysis"

CHAMPION = "gradient_boosting_v1"
BASELINE = "rolling_mean_7"

GROUPINGS = {
    "store": ["store_id"],
    "product": ["product_id"],
    "day_of_week": ["day_of_week"],
}


def load_analysis_data():
    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["date"],
    )

    expected_models = {CHAMPION, BASELINE}

    if set(predictions["model"]) != expected_models:
        raise ValueError("Unexpected prediction models.")

    if len(predictions) != 18200:
        raise ValueError("Unexpected prediction row count.")

    if predictions.duplicated(
        ["model", "date", *SERIES_KEYS]
    ).any():
        raise ValueError("Duplicate prediction keys.")

    if predictions["date"].min() != pd.Timestamp(
        "2025-09-01"
    ):
        raise ValueError("Unexpected test start date.")

    if predictions["date"].max() != pd.Timestamp(
        "2025-11-30"
    ):
        raise ValueError("Unexpected test end date.")

    if not np.isfinite(
        predictions[["units_sold", "prediction"]].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError("Non-finite prediction values.")

    dataset = load_dataset()

    training_scales = calculate_training_scales(dataset)

    predictions = predictions.merge(
        training_scales,
        on=SERIES_KEYS,
        how="left",
        validate="many_to_one",
    )

    if predictions["training_scale"].isna().any():
        raise ValueError("Missing training-only RMSSE scale.")

    predictions["day_of_week"] = (
        predictions["date"].dt.day_name()
    )

    predictions["absolute_error"] = (
        predictions["prediction"] - predictions["units_sold"]
    ).abs()

    predictions["squared_scaled_error"] = (
        (
            predictions["prediction"]
            - predictions["units_sold"]
        ) ** 2
        / predictions["training_scale"]
    )

    predictions["signed_error"] = (
        predictions["prediction"] - predictions["units_sold"]
    )

    return predictions


def calculate_group_metrics(frame, group_columns):
    required = {
        "model",
        "units_sold",
        "prediction",
        "absolute_error",
        "squared_scaled_error",
        "signed_error",
        *group_columns,
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing analysis columns: {sorted(missing)}"
        )

    if frame.empty:
        raise ValueError("Cannot analyze empty data.")

    grouped = (
        frame.groupby(
            ["model", *group_columns],
            dropna=False,
        )
        .agg(
            observations=("units_sold", "size"),
            actual_units=("units_sold", "sum"),
            predicted_units=("prediction", "sum"),
            absolute_error_sum=("absolute_error", "sum"),
            mean_squared_scaled_error=(
                "squared_scaled_error",
                "mean",
            ),
            signed_error_sum=("signed_error", "sum"),
        )
        .reset_index()
    )

    if (grouped["actual_units"] <= 0).any():
        raise ValueError(
            "Group actual-unit total must be positive."
        )

    grouped["wape_pct"] = (
        100.0
        * grouped["absolute_error_sum"]
        / grouped["actual_units"]
    )

    grouped["rmsse"] = np.sqrt(
        grouped["mean_squared_scaled_error"]
    )

    grouped["bias_pct"] = (
        100.0
        * grouped["signed_error_sum"]
        / grouped["actual_units"]
    )

    return grouped[
        [
            "model",
            *group_columns,
            "observations",
            "actual_units",
            "predicted_units",
            "wape_pct",
            "rmsse",
            "bias_pct",
        ]
    ].sort_values(
        [*group_columns, "model"]
    ).reset_index(drop=True)


def calculate_champion_comparison(group_metrics, group_columns):
    champion = group_metrics.loc[
        group_metrics["model"] == CHAMPION
    ].drop(columns=["model"])

    baseline = group_metrics.loc[
        group_metrics["model"] == BASELINE
    ].drop(columns=["model"])

    comparison = champion.merge(
        baseline,
        on=group_columns,
        how="inner",
        suffixes=("_champion", "_baseline"),
        validate="one_to_one",
    )

    if len(comparison) != len(champion):
        raise ValueError("Missing baseline comparison group.")

    comparison["wape_improvement_pp"] = (
        comparison["wape_pct_baseline"]
        - comparison["wape_pct_champion"]
    )

    comparison["wape_relative_improvement_pct"] = (
        100.0
        * comparison["wape_improvement_pp"]
        / comparison["wape_pct_baseline"]
    )

    comparison["champion_wins"] = (
        comparison["wape_pct_champion"]
        < comparison["wape_pct_baseline"]
    )

    return comparison.sort_values(
        "wape_pct_champion",
        ascending=False,
    ).reset_index(drop=True)


def validate_global_reconciliation(predictions):
    saved = pd.read_csv(METRICS_PATH)

    for model_name in (CHAMPION, BASELINE):
        subset = predictions.loc[
            predictions["model"] == model_name
        ]

        actual_total = subset["units_sold"].sum()

        calculated_wape = (
            100.0
            * subset["absolute_error"].sum()
            / actual_total
        )

        calculated_rmsse = np.sqrt(
            subset["squared_scaled_error"].mean()
        )

        calculated_bias = (
            100.0
            * subset["signed_error"].sum()
            / actual_total
        )

        record = saved.loc[
            saved["model"] == model_name
        ]

        if len(record) != 1:
            raise ValueError("Missing saved holdout metric.")

        record = record.iloc[0]

        for metric, calculated in (
            ("wape_pct", calculated_wape),
            ("rmsse", calculated_rmsse),
            ("bias_pct", calculated_bias),
        ):
            if not np.isclose(
                record[metric],
                calculated,
                atol=0.0001,
            ):
                raise ValueError(
                    f"Holdout reconciliation failed: "
                    f"{model_name}, {metric}"
                )


def main():
    expected_outputs = []

    for grouping in GROUPINGS:
        expected_outputs.extend(
            [
                OUTPUT_DIR / f"{grouping}_metrics.csv",
                OUTPUT_DIR / f"{grouping}_comparison.csv",
            ]
        )

    expected_outputs.append(
        OUTPUT_DIR / "error_analysis_summary.json"
    )

    existing = [
        path
        for path in expected_outputs
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Error-analysis output already exists. "
            f"Refusing to overwrite: {existing}"
        )

    predictions = load_analysis_data()

    validate_global_reconciliation(predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "project": "RetailPulse AI 2.0",
        "data_source": "synthetic",
        "analysis_type": "descriptive_holdout_error_analysis",
        "champion_model": CHAMPION,
        "baseline_model": BASELINE,
        "test_rows_per_model": 9100,
        "model_retrained": False,
        "test_used_for_model_selection": False,
        "groupings": {},
    }

    for name, columns in GROUPINGS.items():
        metrics = calculate_group_metrics(
            predictions,
            columns,
        )

        comparison = calculate_champion_comparison(
            metrics,
            columns,
        )

        metrics.to_csv(
            OUTPUT_DIR / f"{name}_metrics.csv",
            index=False,
        )

        comparison.to_csv(
            OUTPUT_DIR / f"{name}_comparison.csv",
            index=False,
        )

        summary["groupings"][name] = {
            "groups": len(comparison),
            "champion_winning_groups": int(
                comparison["champion_wins"].sum()
            ),
            "champion_losing_or_tied_groups": int(
                (~comparison["champion_wins"]).sum()
            ),
        }

        print(f"\n{name.upper()} ERROR ANALYSIS")
        print(
            comparison[
                [
                    *columns,
                    "wape_pct_champion",
                    "wape_pct_baseline",
                    "wape_improvement_pp",
                ]
            ].to_string(index=False)
        )

    with (
        OUTPUT_DIR / "error_analysis_summary.json"
    ).open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print("\nRETAILPULSE AI - ERROR ANALYSIS")
    print("Data source: synthetic")
    print("Source: saved held-out predictions")
    print("Model retrained: False")
    print("Test used for model selection: False")
    print(f"Reports: {OUTPUT_DIR}")
    print("PASS: Descriptive holdout error analysis completed")


if __name__ == "__main__":
    main()