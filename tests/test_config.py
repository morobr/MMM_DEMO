"""Tests for config module."""

from mmm_test.config import ModelConfig


def test_default_config():
    """Test that ModelConfig has sensible defaults for the DT Mart dataset."""
    config = ModelConfig()
    assert config.date_column == "Date"
    assert config.target_column == "total_gmv"
    assert len(config.channel_columns) == 4
    assert "TV" in config.channel_columns
    assert "Online" in config.channel_columns
    assert config.control_columns == ["NPS", "total_Discount", "sale_days"]
    assert config.adstock_max_lag == 4
    assert config.target_accept == 0.95
    assert config.chains == 4
    assert config.draws == 1000
    assert config.tune == 2000


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
    config1.channel_columns.append("new_channel")
    assert "new_channel" not in config2.channel_columns


def test_get_model_config():
    """Test that get_model_config returns a dict with expected prior keys."""
    config = ModelConfig()
    mc = config.get_model_config()
    assert isinstance(mc, dict)
    assert "intercept" in mc
    assert "likelihood" in mc
    assert "adstock_alpha" in mc
    assert "saturation_lam" in mc
    assert "saturation_beta" in mc
    assert "gamma_control" in mc
