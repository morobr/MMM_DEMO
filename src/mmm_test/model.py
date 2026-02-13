"""MMM model definition and fitting."""

import arviz as az
import pandas as pd
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

from mmm_test.config import ModelConfig


def build_model(config: ModelConfig) -> MMM:
    """Initialize a PyMC-Marketing MMM model from configuration.

    Parameters
    ----------
    config : ModelConfig
        Model configuration with channels, controls, and hyperparameters.

    Returns
    -------
    MMM
        An initialized (unfitted) MMM model instance.

    Raises
    ------
    ValueError
        If no channel columns are specified in the config.
    """
    if not config.channel_columns:
        raise ValueError("At least one channel column must be specified in config")

    model = MMM(
        date_column=config.date_column,
        channel_columns=config.channel_columns,
        control_columns=config.control_columns,
        adstock=GeometricAdstock(l_max=config.adstock_max_lag),
        saturation=LogisticSaturation(),
    )
    return model


def fit_model(
    model: MMM,
    x: pd.DataFrame,
    y: pd.Series,
    config: ModelConfig,
) -> az.InferenceData:
    """Fit the MMM model using MCMC sampling.

    Parameters
    ----------
    model : MMM
        An initialized MMM model instance.
    x : pd.DataFrame
        Feature matrix with date, channel, and control columns.
    y : pd.Series
        Target variable (sales).
    config : ModelConfig
        Model configuration with sampling parameters.

    Returns
    -------
    az.InferenceData
        The fitted model trace containing posterior samples.
    """
    model.fit(
        X=x,
        y=y,
        chains=config.chains,
        draws=config.draws,
        tune=config.tune,
        target_accept=config.target_accept,
    )
    return model.idata
