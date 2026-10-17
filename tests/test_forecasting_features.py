from pathlib import Path

import pandas as pd
import pytest

from src.features.build_forecasting_dataset import (
    assign_splits,
    build_features,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "processed" / "forecasting_dataset.csv"

FEATURE_COLUMNS = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_mean_14",
    "sales_rolling_mean_28",
]


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(
        DATASET_PATH,
        parse_dates=["date"],
    )


def test_forecasting_dataset_exists():
    assert DATASET_PATH.is_file()


def test_forecasting_dataset_row_count(dataset):
    assert len(dataset) == 67200


def test_forecasting_dataset_series_count(dataset):
    assert (
        dataset[["store_id", "product_id"]]
        .drop_duplicates()
        .shape[0]
        == 100
    )


def test_forecasting_dataset_has_no_duplicate_keys(dataset):
    assert not dataset.duplicated(
        ["date", "store_id", "product_id"]
    ).any()


def test_forecasting_dataset_has_no_missing_features(dataset):
    assert not dataset[FEATURE_COLUMNS].isna().any().any()


@pytest.mark.parametrize(
    ("split", "expected_rows"),
    [
        ("train", 48900),
        ("validation", 9200),
        ("test", 9100),
    ],
)
def test_split_row_counts(dataset, split, expected_rows):
    assert (dataset["split"] == split).sum() == expected_rows


def test_splits_are_chronological(dataset):
    train = dataset.loc[dataset["split"] == "train", "date"]
    validation = dataset.loc[
        dataset["split"] == "validation",
        "date",
    ]
    test = dataset.loc[dataset["split"] == "test", "date"]

    assert train.max() < validation.min()
    assert validation.max() < test.min()

    assert train.max() == pd.Timestamp("2025-05-31")
    assert validation.min() == pd.Timestamp("2025-06-01")
    assert validation.max() == pd.Timestamp("2025-08-31")
    assert test.min() == pd.Timestamp("2025-09-01")


@pytest.mark.parametrize("lag", [1, 7, 14, 28])
def test_lag_values_match_prior_sales(dataset, lag):
    history = dataset.sort_values(
        ["store_id", "product_id", "date"]
    )

    # Check dates after the initial feature warm-up period.
    # Earlier lag values require raw history that is not
    # retained in the feature dataset.
    expected = history.groupby(
        ["store_id", "product_id"]
    )["units_sold"].shift(lag)

    valid = expected.notna()

    actual = history.loc[valid, f"sales_lag_{lag}"]
    reference = expected.loc[valid]

    assert (actual.to_numpy() == reference.to_numpy()).all()


@pytest.mark.parametrize("window", [7, 14, 28])
def test_rolling_means_use_only_prior_sales(dataset, window):
    history = dataset.sort_values(
        ["store_id", "product_id", "date"]
    )

    expected = history.groupby(
        ["store_id", "product_id"]
    )["units_sold"].transform(
        lambda values: values.shift(1)
        .rolling(window, min_periods=window)
        .mean()
    )

    valid = expected.notna()

    actual = history.loc[
        valid,
        f"sales_rolling_mean_{window}",
    ]

    assert (
        abs(actual.to_numpy() - expected.loc[valid].to_numpy())
        < 1e-9
    ).all()


def test_calendar_features_match_dates(dataset):
    assert (
        dataset["day_of_week"].to_numpy()
        == dataset["date"].dt.dayofweek.to_numpy()
    ).all()

    assert (
        dataset["month"].to_numpy()
        == dataset["date"].dt.month.to_numpy()
    ).all()


def test_feature_builder_rejects_duplicate_keys():
    dates = pd.date_range("2024-01-01", periods=700)

    rows = []
    for store_number in range(1, 11):
        for product_number in range(1, 11):
            for day in dates:
                rows.append(
                    {
                        "date": day,
                        "store_id": f"S{store_number:03d}",
                        "product_id": f"P{product_number:03d}",
                        "units_sold": 10,
                    }
                )

    history = pd.DataFrame(rows)
    history.loc[1, "date"] = history.loc[0, "date"]

    with pytest.raises(ValueError, match="Duplicate"):
        build_features(history)


def test_assign_splits_preserves_row_count(dataset):
    reassigned = assign_splits(dataset.drop(columns=["split"]))

    assert len(reassigned) == len(dataset)
    assert reassigned["split"].equals(dataset["split"])