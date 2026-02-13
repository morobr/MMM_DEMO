"""Tests for model module."""

import pytest

from mmm_test.config import ModelConfig
from mmm_test.model import build_model, sample_prior_predictive


def test_build_model_returns_mmm(real_config):
    """Test that build_model returns an MMM instance."""
    model = build_model(real_config)
    assert model is not None
    assert model.date_column == real_config.date_column


def test_build_model_no_channels():
    """Test that build_model raises with no channel columns."""
    config = ModelConfig(channel_columns=[])
    with pytest.raises(ValueError, match="channel column"):
        build_model(config)


def test_build_model_adstock_max_lag(real_config):
    """Test that adstock max_lag is set from config."""
    model = build_model(real_config)
    assert model.adstock.l_max == real_config.adstock_max_lag


def test_build_model_default_adstock_is_8():
    """Test that default config uses l_max=8 for weekly data."""
    config = ModelConfig()
    model = build_model(config)
    assert model.adstock.l_max == 8


def test_build_model_with_controls(real_config):
    """Test that build_model includes control columns."""
    model = build_model(real_config)
    assert model.control_columns == real_config.control_columns


def test_build_model_no_controls_passes_none():
    """Test that empty control_columns passes None to MMM."""
    config = ModelConfig(control_columns=[])
    model = build_model(config)
    assert model.control_columns is None


def test_sample_prior_predictive_calls_model(mock_mmm_model, mmm_sample_dataframe):
    """Test that sample_prior_predictive wrapper calls the model method."""
    config = ModelConfig()
    cols = [config.date_column, *config.channel_columns, *config.control_columns]
    x = mmm_sample_dataframe[cols]
    y = mmm_sample_dataframe[config.target_column]

    sample_prior_predictive(mock_mmm_model, x, y)
    mock_mmm_model.sample_prior_predictive.assert_called_once()
