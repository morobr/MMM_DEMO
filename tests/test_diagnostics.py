"""Tests for diagnostics module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mmm_test.diagnostics import (
    ESS_THRESHOLD,
    RHAT_THRESHOLD,
    check_convergence,
    validate_before_interpretation,
)


@pytest.fixture
def _good_summary():
    """Mock az.summary returning passing diagnostics."""
    summary_df = pd.DataFrame({"r_hat": [1.0, 1.001], "ess_bulk": [500.0, 600.0]})
    with patch("mmm_test.diagnostics.az.summary", return_value=summary_df):
        yield


@pytest.fixture
def _bad_rhat_summary():
    """Mock az.summary returning failing R-hat."""
    summary_df = pd.DataFrame({"r_hat": [1.05, 1.001], "ess_bulk": [500.0, 600.0]})
    with patch("mmm_test.diagnostics.az.summary", return_value=summary_df):
        yield


@pytest.fixture
def _bad_ess_summary():
    """Mock az.summary returning failing ESS."""
    summary_df = pd.DataFrame({"r_hat": [1.0, 1.001], "ess_bulk": [100.0, 200.0]})
    with patch("mmm_test.diagnostics.az.summary", return_value=summary_df):
        yield


def test_check_convergence_passes(mock_idata, _good_summary):
    """Test diagnostics pass with good values."""
    result = check_convergence(mock_idata)
    assert result.passed is True
    assert result.rhat_ok is True
    assert result.ess_ok is True
    assert result.divergences == 0


def test_check_convergence_fails_rhat(mock_idata, _bad_rhat_summary):
    """Test diagnostics fail with high R-hat."""
    result = check_convergence(mock_idata)
    assert result.passed is False
    assert result.rhat_ok is False
    assert result.max_rhat > RHAT_THRESHOLD


def test_check_convergence_fails_ess(mock_idata, _bad_ess_summary):
    """Test diagnostics fail with low ESS."""
    result = check_convergence(mock_idata)
    assert result.passed is False
    assert result.ess_ok is False
    assert result.min_ess < ESS_THRESHOLD


def test_check_convergence_fails_divergences(_good_summary):
    """Test diagnostics fail with divergences."""
    idata = MagicMock()
    diverging = xr.DataArray(np.ones((2, 100), dtype=bool), dims=["chain", "draw"])
    idata.sample_stats = {"diverging": diverging}

    result = check_convergence(idata)
    assert result.passed is False
    assert result.divergences > 0


def test_validate_before_interpretation_passes(mock_idata, _good_summary):
    """Test validation gate passes with good diagnostics."""
    validate_before_interpretation(mock_idata)


def test_validate_before_interpretation_raises(mock_idata, _bad_rhat_summary):
    """Test validation gate raises with bad diagnostics."""
    with pytest.raises(RuntimeError, match="not converged"):
        validate_before_interpretation(mock_idata)
