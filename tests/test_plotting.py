"""Tests for plotting module."""

from unittest.mock import patch

from mmm_demo.plotting import _save_plot


def test_save_plot_creates_file(tmp_path):
    """Test that _save_plot saves a figure to the correct path."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])

    with patch("mmm_demo.plotting.OUTPUTS_DIR", tmp_path):
        path = _save_plot(fig, "diagnostics", "test_plot")

    assert path.exists()
    assert path.suffix == ".png"
    assert "diagnostics" in str(path)
