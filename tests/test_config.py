"""Tests for config module."""

from mmm_test.config import ModelConfig


def test_default_config():
    """Test that ModelConfig has sensible defaults."""
    config = ModelConfig()
    assert config.date_column == "date"
    assert config.target_column == "sales"
    assert config.channel_columns == []
    assert config.control_columns == []
    assert config.adstock_max_lag == 8
    assert config.chains == 4
    assert config.draws == 1000


def test_custom_config():
    """Test that ModelConfig accepts custom values."""
    config = ModelConfig(
        channel_columns=["tv", "radio"],
        adstock_max_lag=12,
    )
    assert config.channel_columns == ["tv", "radio"]
    assert config.adstock_max_lag == 12


def test_config_independent_instances():
    """Test that mutable defaults are independent between instances."""
    config1 = ModelConfig()
    config2 = ModelConfig()
    config1.channel_columns.append("tv")
    assert config2.channel_columns == []
