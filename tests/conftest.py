"""Shared fixtures and mocked model fits for tests."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mmm_test.config import ModelConfig


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
def sample_dataframe() -> pd.DataFrame:
    """Create a small synthetic DataFrame mimicking the DT Mart schema."""
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
