"""Charts and visualizations for MMM analysis."""

from datetime import datetime
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
from pymc_marketing.mmm import MMM

from mmm_test.config import OUTPUTS_DIR


def _save_plot(fig: plt.Figure, subdir: str, name: str) -> Path:
    """Save a matplotlib figure to the outputs directory.

    Parameters
    ----------
    fig : plt.Figure
        The figure to save.
    subdir : str
        Subdirectory under outputs/plots/ (e.g., "diagnostics", "contributions").
    name : str
        Filename without extension.

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_path = OUTPUTS_DIR / "plots" / subdir
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f"{name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_trace(idata: az.InferenceData) -> Path:
    """Plot MCMC trace plots for convergence inspection.

    Parameters
    ----------
    idata : az.InferenceData
        The fitted model trace.

    Returns
    -------
    Path
        Path to the saved plot.
    """
    axes = az.plot_trace(idata, compact=True)
    fig = axes.ravel()[0].figure
    date_str = datetime.now().strftime("%Y-%m-%d")
    return _save_plot(fig, "diagnostics", f"trace_{date_str}")


def plot_posterior_predictive(model: MMM) -> Path:
    """Plot posterior predictive check.

    Parameters
    ----------
    model : MMM
        A fitted MMM model instance.

    Returns
    -------
    Path
        Path to the saved plot.
    """
    fig = model.plot_posterior_predictive()
    if not isinstance(fig, plt.Figure):
        fig = fig.figure
    date_str = datetime.now().strftime("%Y-%m-%d")
    return _save_plot(fig, "diagnostics", f"posterior_predictive_{date_str}")


def plot_channel_contributions(model: MMM) -> Path:
    """Plot channel contribution decomposition.

    Parameters
    ----------
    model : MMM
        A fitted MMM model instance.

    Returns
    -------
    Path
        Path to the saved plot.
    """
    fig = model.plot_channel_contributions()
    if not isinstance(fig, plt.Figure):
        fig = fig.figure
    date_str = datetime.now().strftime("%Y-%m-%d")
    return _save_plot(fig, "contributions", f"channel_contributions_{date_str}")
