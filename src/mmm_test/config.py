"""Model hyperparameters, channel definitions, priors, and constants."""

from dataclasses import dataclass, field
from pathlib import Path

# Dataset
DATASET_ID = "datatattle/dt-mart-market-mix-modeling"
DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")


@dataclass
class ModelConfig:
    """Central configuration for the MMM model.

    Parameters
    ----------
    date_column : str
        Name of the date/time column in the dataset.
    target_column : str
        Name of the target variable column (e.g., sales/revenue).
    channel_columns : list[str]
        Marketing channel spend columns to model.
    control_columns : list[str]
        Non-marketing control variable columns.
    adstock_max_lag : int
        Maximum lag for adstock carryover effect in time periods.
    target_accept : float
        Target acceptance rate for MCMC sampling. Higher values (0.9-0.99)
        reduce divergences but slow sampling.
    chains : int
        Number of MCMC chains to run.
    draws : int
        Number of posterior draws per chain.
    tune : int
        Number of tuning steps per chain.
    """

    date_column: str = "date"
    target_column: str = "sales"
    channel_columns: list[str] = field(default_factory=list)
    control_columns: list[str] = field(default_factory=list)
    adstock_max_lag: int = 8
    target_accept: float = 0.9
    chains: int = 4
    draws: int = 1000
    tune: int = 1000
