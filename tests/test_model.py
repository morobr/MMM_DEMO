"""Tests for model module."""

import pytest

from mmm_test.config import ModelConfig
from mmm_test.model import build_model


def test_build_model_returns_mmm(sample_config):
    """Test that build_model returns an MMM instance."""
    model = build_model(sample_config)
    assert model is not None
    assert model.date_column == sample_config.date_column


def test_build_model_no_channels():
    """Test that build_model raises with no channel columns."""
    config = ModelConfig(channel_columns=[])
    with pytest.raises(ValueError, match="channel column"):
        build_model(config)


def test_build_model_adstock_max_lag(sample_config):
    """Test that adstock max_lag is set from config."""
    model = build_model(sample_config)
    assert model.adstock.l_max == sample_config.adstock_max_lag
