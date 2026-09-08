"""Tests for the data cleaning pipeline (05_data_cleaning.py).

These tests run offline — no API key, no network. Each test checks one
behaviour of the cleaning pipeline, so a failure tells you exactly what broke.

Run with:  pytest -v
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# The module name starts with a digit, so we load it by file path.
SPEC = importlib.util.spec_from_file_location(
    "data_cleaning", Path(__file__).parent.parent / "05_data_cleaning.py"
)
data_cleaning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data_cleaning)


@pytest.fixture
def cleaned() -> pd.DataFrame:
    """The raw sample dataset, after the full cleaning pipeline."""
    return data_cleaning.clean(data_cleaning.RAW)


# --- Date parsing ---------------------------------------------------------
def test_parse_date_handles_known_formats():
    expected = pd.Timestamp("2026-05-12")
    assert data_cleaning.parse_date("2026-05-12") == expected
    assert data_cleaning.parse_date("12/05/2026") == expected
    assert data_cleaning.parse_date("12-05-2026") == expected
    assert data_cleaning.parse_date("2026/05/12") == expected


def test_parse_date_returns_nat_on_garbage():
    """Unparseable input must be flagged as missing, never silently guessed."""
    assert pd.isna(data_cleaning.parse_date("not_recorded"))
    assert pd.isna(data_cleaning.parse_date(None))


# --- Text normalization ---------------------------------------------------
def test_text_columns_are_normalized(cleaned):
    assert (cleaned["truck_id"] == cleaned["truck_id"].str.upper()).all()
    assert (cleaned["intervention"] == cleaned["intervention"].str.lower()).all()
    assert not cleaned["intervention"].str.contains(r"^\s|\s$").any()


# --- Deduplication --------------------------------------------------------
def test_duplicates_are_removed(cleaned):
    """'FH16-001' and 'fh16-001' on the same date are the same intervention."""
    keys = cleaned[["truck_id", "intervention", "date"]]
    assert not keys.duplicated().any()
    assert len(cleaned) < len(data_cleaning.RAW)


# --- Types and required fields --------------------------------------------
def test_types_are_correct(cleaned):
    assert pd.api.types.is_numeric_dtype(cleaned["cost_eur"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])


def test_rows_without_truck_id_are_dropped(cleaned):
    assert cleaned["truck_id"].notna().all()


# --- Missing values and outliers ------------------------------------------
def test_missing_costs_are_imputed(cleaned):
    assert cleaned["cost_eur"].notna().all()


def test_outliers_are_flagged_not_deleted(cleaned):
    """Auditability: a suspicious value is kept and marked, never dropped."""
    assert cleaned["cost_outlier"].any()
    assert (cleaned.loc[cleaned["cost_outlier"], "cost_eur"] > 5000).all()


def test_imputed_values_are_not_polluted_by_outliers(cleaned):
    """The 999999 EUR outlier must not inflate the imputed brake-pad cost."""
    imputed = cleaned.loc[~cleaned["cost_outlier"], "cost_eur"]
    assert imputed.max() < 5000


# --- Non-regression on the pipeline as a whole ----------------------------
def test_clean_does_not_mutate_input():
    """The pipeline must return a new frame, not modify the caller's data."""
    before = data_cleaning.RAW.copy()
    data_cleaning.clean(data_cleaning.RAW)
    pd.testing.assert_frame_equal(data_cleaning.RAW, before)
