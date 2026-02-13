"""Tests for data module."""

import pandas as pd
import pytest

from mmm_test.data import preprocess, validate_raw_data


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
