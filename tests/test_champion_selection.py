import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.select_champion import (
    COMPARISON_PATH,
    EXPECTED_MODELS,
    ROOT,
    SELECTION_PATH,
    select_champion,
)


@pytest.fixture(scope="module")
def comparison():
    return pd.read_csv(COMPARISON_PATH)


@pytest.fixture(scope="module")
def selection():
    return json.loads(
        SELECTION_PATH.read_text(encoding="utf-8")
    )


def test_selection_file_exists():
    assert SELECTION_PATH.is_file()


def test_selection_metadata(selection):
    assert selection["project"] == "RetailPulse AI 2.0"
    assert selection["data_source"] == "synthetic"
    assert (
        selection["selection_status"]
        == "frozen_before_test_evaluation"
    )
    assert (
        selection["selection_metric"]
        == "validation_wape_pct"
    )
    assert selection["test_metrics_used_for_selection"] is False


def test_champion_is_gradient_boosting(selection):
    assert selection["champion_model"] == "gradient_boosting_v1"
    assert selection["validation_metrics"]["wape_pct"] == pytest.approx(
        28.3538
    )
    assert selection["validation_metrics"]["rmsse"] == pytest.approx(
        0.6265
    )


def test_champion_artifact_exists(selection):
    artifact_path = ROOT / selection["champion_artifact"]

    assert artifact_path.is_file()
    assert artifact_path.suffix == ".joblib"


def test_selection_periods(selection):
    assert selection["training_period"] == {
        "start": "2024-01-29",
        "end": "2025-05-31",
    }
    assert selection["validation_period"] == {
        "start": "2025-06-01",
        "end": "2025-08-31",
    }
    assert selection["reserved_test_period"] == {
        "start": "2025-09-01",
        "end": "2025-11-30",
    }


def test_selection_uses_one_day_ahead_protocol(selection):
    assert selection["forecast_horizon_days"] == 1
    assert (
        selection["evaluation_protocol"]
        == "rolling_one_day_ahead"
    )


def test_selection_matches_comparison(comparison, selection):
    recalculated = select_champion(comparison)

    assert recalculated == {
        key: value
        for key, value in selection.items()
        if key != "frozen_at_utc"
    }


def test_candidate_models_match_expected(selection):
    candidates = selection["candidate_metrics"]

    assert len(candidates) == 3
    assert {row["model"] for row in candidates} == EXPECTED_MODELS
    assert all(row["split"] == "validation" for row in candidates)


def test_missing_column_rejected(comparison):
    modified = comparison.drop(columns=["wape_pct"])

    with pytest.raises(ValueError, match="Missing comparison"):
        select_champion(modified)


def test_duplicate_model_rejected(comparison):
    modified = comparison.copy()
    modified.loc[
        modified.index[-1],
        "model",
    ] = modified.iloc[0]["model"]

    with pytest.raises(ValueError):
        select_champion(modified)


def test_test_split_rejected(comparison):
    modified = comparison.copy()
    modified.loc[modified.index[0], "split"] = "test"

    with pytest.raises(ValueError, match="validation"):
        select_champion(modified)


def test_missing_metric_rejected(comparison):
    modified = comparison.copy()
    modified.loc[modified.index[0], "wape_pct"] = float("nan")

    with pytest.raises(ValueError, match="Missing validation"):
        select_champion(modified)


def test_negative_wape_rejected(comparison):
    modified = comparison.copy()
    modified.loc[modified.index[0], "wape_pct"] = -1.0

    with pytest.raises(ValueError, match="Invalid error"):
        select_champion(modified)


def test_unexpected_observation_count_rejected(comparison):
    modified = comparison.copy()
    modified.loc[modified.index[0], "observations"] = 1

    with pytest.raises(ValueError, match="observations"):
        select_champion(modified)


def test_baseline_champion_rejected(comparison):
    modified = comparison.copy()
    baseline_mask = modified["model"] == "rolling_mean_7"

    modified.loc[baseline_mask, "wape_pct"] = 0.0

    with pytest.raises(ValueError, match="Baseline won"):
        select_champion(modified)


def test_frozen_timestamp_is_present(selection):
    timestamp = selection["frozen_at_utc"]

    assert isinstance(timestamp, str)
    assert timestamp.endswith("+00:00")


def test_selection_file_is_not_empty():
    assert Path(SELECTION_PATH).stat().st_size > 0