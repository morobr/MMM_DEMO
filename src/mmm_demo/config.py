"""Model hyperparameters, channel definitions, priors, and constants."""

from dataclasses import dataclass, field
from pathlib import Path

from pymc_marketing.prior import Prior

# Dataset
DATASET_ID = "datatattle/dt-mart-market-mix-modeling"
# Anchor to project root: src/mmm_demo/config.py → up three levels
_PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

# Channels excluded due to excessive missing data (9/12 NaN)
DROP_CHANNELS: list[str] = ["Radio", "Other"]


@dataclass
class ModelConfig:
    """Central configuration for the MMM model.

    Parameters
    ----------
    date_column : str
        Name of the date/time column in the dataset.
    target_column : str
        Name of the target variable column.
    channel_columns : list[str]
        Marketing channel spend columns to model.
    control_columns : list[str]
        Non-marketing control variable columns.
    adstock_max_lag : int
        Maximum lag for adstock carryover effect in time periods (weeks).
    target_accept : float
        Target acceptance rate for MCMC sampling. Set to 0.95 to
        reduce divergences with small dataset.
    chains : int
        Number of MCMC chains to run.
    draws : int
        Number of posterior draws per chain.
    tune : int
        Number of tuning steps per chain.
    """

    date_column: str = "Date"
    target_column: str = "total_gmv"
    channel_columns: list[str] = field(
        default_factory=lambda: [
            "TV",
            "Sponsorship",
            "Digital",
            "Online",
        ]
    )
    control_columns: list[str] = field(
        default_factory=lambda: [
            "NPS",
            "total_Discount",
            "sale_days",
        ]
    )
    adstock_max_lag: int = 4
    target_accept: float = 0.95
    chains: int = 4
    draws: int = 1000
    tune: int = 2000

    def get_model_config(self) -> dict:
        """Return model_config dict with informative priors.

        Priors are calibrated for MaxAbsScaled data (channel and target
        values scaled to [0, 1] internally by PyMC-Marketing).

        Returns
        -------
        dict
            Prior specifications keyed by parameter name.
        """
        return {
            "intercept": Prior("Normal", mu=0, sigma=2),
            "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=0.5)),
            "adstock_alpha": Prior("Beta", alpha=1, beta=3),
            "saturation_lam": Prior("Gamma", alpha=3, beta=1),
            "saturation_beta": Prior("HalfNormal", sigma=2),
            "gamma_control": Prior("Normal", mu=0, sigma=2, dims="control"),
        }
