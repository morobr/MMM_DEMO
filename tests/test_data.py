"""Tests for data module."""

import pandas as pd
import pytest

from mmm_test.data import (
    compute_sale_days,
    preprocess,
    validate_mmm_data,
    validate_raw_data,
)


def test_validate_raw_data_passes(sample_dataframe):
    """Test validation passes for valid data."""
    validate_raw_data(sample_dataframe)


def test_validate_raw_data_empty():
    """Test validation fails for empty DataFrame."""
    with pytest.raises(ValueError, match="empty"):
        validate_raw_data(pd.DataFrame())


def test_preprocess_sorts_by_date(sample_dataframe):
    """Test that preprocessing sorts by date."""
    shuffled = sample_dataframe.sample(frac=1, random_state=42)
    result = preprocess(shuffled, "date", "sales")
    assert result["date"].is_monotonic_increasing


def test_preprocess_converts_date(sample_dataframe):
    """Test that preprocessing converts date column to datetime."""
    df = sample_dataframe.copy()
    df["date"] = df["date"].astype(str)
    result = preprocess(df, "date", "sales")
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_preprocess_missing_column(sample_dataframe):
    """Test that preprocessing raises on missing columns."""
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess(sample_dataframe, "nonexistent", "sales")


def test_preprocess_null_target(sample_dataframe):
    """Test that preprocessing raises when target has nulls."""
    df = sample_dataframe.copy()
    df.loc[0, "sales"] = None
    with pytest.raises(ValueError, match="null values"):
        preprocess(df, "date", "sales")


def test_compute_sale_days(tmp_path):
    """Test sale_days computation from SpecialSale.csv."""
    special_csv = tmp_path / "SpecialSale.csv"
    special_csv.write_text(
        "Date,Sales Name\n7/18/2015,Sale A\n7/19/2015,Sale A\n8/15/2015,Sale B\n"
    )
    result = compute_sale_days(tmp_path)
    assert "sale_days" in result.columns
    assert "Date" in result.columns
    jul = result[result["Date"] == pd.Timestamp("2015-07-01")]
    assert jul["sale_days"].values[0] == 2
    aug = result[result["Date"] == pd.Timestamp("2015-08-01")]
    assert aug["sale_days"].values[0] == 1


def test_compute_sale_days_file_not_found(tmp_path):
    """Test that compute_sale_days raises when file is missing."""
    with pytest.raises(FileNotFoundError):
        compute_sale_days(tmp_path)


def test_validate_mmm_data_passes(mmm_sample_dataframe):
    """Test validation passes for properly structured data."""
    validate_mmm_data(mmm_sample_dataframe)


def test_validate_mmm_data_missing_column(mmm_sample_dataframe):
    """Test validation fails when a required column is missing."""
    df = mmm_sample_dataframe.drop(columns=["NPS"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_mmm_data(df)


def test_validate_mmm_data_wrong_row_count(mmm_sample_dataframe):
    """Test validation fails when row count is not 12."""
    df = mmm_sample_dataframe.head(6)
    with pytest.raises(ValueError, match="Expected 12"):
        validate_mmm_data(df)


def test_validate_mmm_data_null_target(mmm_sample_dataframe):
    """Test validation fails when target has null values."""
    df = mmm_sample_dataframe.copy()
    df.loc[0, "total_gmv"] = None
    with pytest.raises(ValueError, match="total_gmv"):
        validate_mmm_data(df)


def test_validate_mmm_data_null_channel(mmm_sample_dataframe):
    """Test validation fails when a channel column has null values."""
    df = mmm_sample_dataframe.copy()
    df.loc[0, "TV"] = None
    with pytest.raises(ValueError, match="TV"):
        validate_mmm_data(df)
