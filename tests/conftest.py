"""Shared fixtures and mocked model fits for tests."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mmm_demo.config import ModelConfig


@pytest.fixture
def sample_config() -> ModelConfig:
    """Create a sample model configuration for testing."""
    return ModelConfig(
        date_column="date",
        target_column="sales",
        channel_columns=["tv_spend", "radio_spend", "digital_spend"],
        control_columns=["price"],
        adstock_max_lag=4,
        chains=2,
        draws=100,
        tune=100,
    )


@pytest.fixture
def real_config() -> ModelConfig:
    """Create a ModelConfig with actual dataset defaults."""
    return ModelConfig()


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a small synthetic DataFrame for generic tests."""
    n_weeks = 52
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n_weeks, freq="W"),
            "sales": rng.normal(10000, 2000, n_weeks).clip(min=0),
            "tv_spend": rng.uniform(500, 5000, n_weeks),
            "radio_spend": rng.uniform(200, 2000, n_weeks),
            "digital_spend": rng.uniform(300, 3000, n_weeks),
            "price": rng.uniform(8, 15, n_weeks),
        }
    )


@pytest.fixture
def mmm_sample_dataframe() -> pd.DataFrame:
    """Create synthetic DataFrame mimicking the DT Mart MMM schema (grouped)."""
    n = 12
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2015-07-01", periods=n, freq="MS"),
            "total_gmv": rng.uniform(1.5e8, 5.5e8, n),
            "TV": rng.uniform(0, 9.3e7, n),
            "Sponsorship": rng.uniform(1.1e7, 8.47e8, n),
            "Digital": rng.uniform(1e7, 4.8e8, n),
            "Online": rng.uniform(2e6, 3.2e8, n),
            "NPS": rng.uniform(44, 60, n),
            "total_Discount": rng.uniform(1.2e8, 4.2e8, n),
            "sale_days": rng.integers(0, 10, n),
        }
    )


@pytest.fixture
def mmm_raw_sample_dataframe() -> pd.DataFrame:
    """Create synthetic DataFrame with raw (ungrouped) 7-channel monthly schema."""
    n = 12
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2015-07-01", periods=n, freq="MS"),
            "total_gmv": rng.uniform(1.5e8, 5.5e8, n),
            "TV": rng.uniform(0, 9.3e7, n),
            "Digital": rng.uniform(5e6, 1.26e8, n),
            "Sponsorship": rng.uniform(1.1e7, 8.47e8, n),
            "Content.Marketing": rng.uniform(0, 3.4e7, n),
            "Online.marketing": rng.uniform(1e6, 2.44e8, n),
            "Affiliates": rng.uniform(1e6, 7.4e7, n),
            "SEM": rng.uniform(5e6, 3.19e8, n),
            "NPS": rng.uniform(44, 60, n),
            "total_Discount": rng.uniform(1.2e8, 4.2e8, n),
            "sale_days": rng.integers(0, 10, n),
        }
    )


@pytest.fixture
def mmm_weekly_sample_dataframe() -> pd.DataFrame:
    """Create synthetic weekly DataFrame mimicking the weekly MMM schema (grouped)."""
    n = 52
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2015-06-29", periods=n, freq="W-MON"),
            "total_gmv": rng.uniform(2e7, 1.5e8, n),
            "TV": rng.uniform(0, 2.5e7, n),
            "Sponsorship": rng.uniform(2e6, 2e8, n),
            "Digital": rng.uniform(2e6, 1.2e8, n),
            "Online": rng.uniform(4e5, 7.8e7, n),
            "NPS": rng.uniform(44, 60, n),
            "total_Discount": rng.uniform(2e7, 1e8, n),
            "sale_days": rng.integers(0, 5, n),
        }
    )


@pytest.fixture
def mock_idata() -> MagicMock:
    """Create a mock ArviZ InferenceData with passing diagnostics."""
    idata = MagicMock()

    # Mock sample_stats with no divergences
    diverging = xr.DataArray(np.zeros((2, 100), dtype=bool), dims=["chain", "draw"])
    idata.sample_stats = {"diverging": diverging}

    return idata


@pytest.fixture
def mock_mmm_model() -> MagicMock:
    """Create a mock fitted MMM model."""
    model = MagicMock()
    model.idata = MagicMock()
    return model
