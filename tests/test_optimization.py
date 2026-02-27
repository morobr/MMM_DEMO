"""Tests for optimization module."""

from unittest.mock import MagicMock

import pytest

from mmm_demo.optimization import optimize_budget


def test_optimize_budget_unfitted_model():
    """Test that optimization raises for unfitted model."""
    model = MagicMock()
    model.idata = None
    with pytest.raises(ValueError, match="fitted"):
        optimize_budget(model, total_budget=10000, channels=["tv"])


def test_optimize_budget_calls_optimizer(mock_mmm_model):
    """Test that optimization calls the model's optimizer method."""
    mock_mmm_model.optimize_channel_budget_for_maximum_contribution.return_value = {
        "channel": ["tv", "radio"],
        "budget": [6000, 4000],
    }

    result = optimize_budget(
        mock_mmm_model,
        total_budget=10000,
        channels=["tv", "radio"],
    )

    mock_mmm_model.optimize_channel_budget_for_maximum_contribution.assert_called_once()
    assert len(result) == 2
