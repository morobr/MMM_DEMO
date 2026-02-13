"""Budget optimization using PyMC-Marketing's BudgetOptimizer."""

import pandas as pd
from pymc_marketing.mmm import MMM


def optimize_budget(
    model: MMM,
    total_budget: float,
    channels: list[str],
    budget_bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Find the optimal budget allocation across marketing channels.

    Parameters
    ----------
    model : MMM
        A fitted MMM model instance.
    total_budget : float
        Total budget to allocate across channels.
    channels : list[str]
        List of channel names to optimize.
    budget_bounds : dict[str, tuple[float, float]] | None, optional
        Min/max spend bounds per channel as {channel: (min, max)}.
        If None, no bounds are applied.

    Returns
    -------
    pd.DataFrame
        Optimization results with channel allocations and predicted outcomes.

    Raises
    ------
    ValueError
        If the model has not been fitted.
    """
    if not hasattr(model, "idata") or model.idata is None:
        raise ValueError("Model must be fitted before optimization")

    optimizer = model.optimize_channel_budget_for_maximum_contribution(
        total_budget=total_budget,
        channels=channels,
        budget_bounds=budget_bounds,
    )

    return pd.DataFrame(optimizer)
