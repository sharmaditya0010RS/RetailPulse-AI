import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.export_bi_data import (
    BI_VIEWS,
    EXPECTED_ROWS,
    json_safe,
    validate_data,
)

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "data" / "processed" / "bi"
EXPECTED_REVENUE = Decimal("116500690.78")


@pytest.fixture(scope="module")
def datasets():
    return {
        name: pd.read_csv(EXPORT_DIR / f"{name}.csv")
        for name in BI_VIEWS
    }


@pytest.fixture(scope="module")
def scorecard():
    path = EXPORT_DIR / "executive_scorecard.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("name", "expected_rows"),
    list(EXPECTED_ROWS.items()),
)
def test_exported_csv_row_counts(name, expected_rows):
    path = EXPORT_DIR / f"{name}.csv"

    assert path.is_file()

    frame = pd.read_csv(path)

    assert len(frame) == expected_rows


def test_scorecard_exists():
    assert (EXPORT_DIR / "executive_scorecard.json").is_file()


def test_scorecard_identifies_synthetic_data(scorecard):
    assert scorecard["data_source"] == "synthetic"


def test_scorecard_financial_precision(scorecard):
    assert Decimal(str(scorecard["net_revenue"])) == EXPECTED_REVENUE


def test_scorecard_operational_totals(scorecard):
    assert int(scorecard["observations"]) == 70000
    assert int(scorecard["units_sold"]) == 1483884
    assert int(scorecard["lost_units"]) == 79971


@pytest.mark.parametrize(
    "name",
    [
        "monthly_performance",
        "category_performance",
        "store_performance",
    ],
)
def test_exported_revenue_reconciliation(datasets, name):
    total = sum(
        Decimal(str(value))
        for value in datasets[name]["net_revenue"]
    )

    assert total == EXPECTED_REVENUE


@pytest.mark.parametrize(
    "name",
    [
        "monthly_performance",
        "category_performance",
        "store_performance",
        "product_stockouts",
    ],
)
def test_exported_unit_reconciliation(datasets, name):
    frame = datasets[name]

    assert int(frame["units_sold"].sum()) == 1483884
    assert int(frame["lost_units"].sum()) == 79971


def test_json_safe_decimal():
    assert json_safe(Decimal("123.45")) == "123.45"


def test_validate_data_accepts_valid_exports(datasets):
    validate_data(datasets)


def test_validate_data_rejects_wrong_row_count(datasets):
    invalid = {
        name: frame.copy()
        for name, frame in datasets.items()
    }

    invalid["monthly_performance"] = invalid[
        "monthly_performance"
    ].iloc[:-1]

    with pytest.raises(ValueError, match="expected 23 rows"):
        validate_data(invalid)