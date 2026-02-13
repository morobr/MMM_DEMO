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
        Model configuration with channels, controls, priors, and hyperparameters.

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
        control_columns=config.control_columns if config.control_columns else None,
        adstock=GeometricAdstock(l_max=config.adstock_max_lag),
        saturation=LogisticSaturation(),
        model_config=config.get_model_config(),
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
        Target variable (total_gmv).
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


def sample_prior_predictive(
    model: MMM,
    x: pd.DataFrame,
    y: pd.Series,
    samples: int = 500,
) -> az.InferenceData:
    """Sample from the prior predictive distribution.

    Call before fitting to verify that priors produce plausible target values.

    Parameters
    ----------
    model : MMM
        An initialized (unfitted) MMM model instance.
    x : pd.DataFrame
        Feature matrix with date, channel, and control columns.
    y : pd.Series
        Target variable (for building the model graph).
    samples : int, optional
        Number of prior predictive samples, by default 500.

    Returns
    -------
    az.InferenceData
        Prior predictive samples.
    """
    model.sample_prior_predictive(X=x, y=y, samples=samples)
    return model.idata


def sample_posterior_predictive(model: MMM, x: pd.DataFrame) -> az.InferenceData:
    """Sample from the posterior predictive distribution.

    Call after fitting to check model's ability to reproduce observed data.

    Parameters
    ----------
    model : MMM
        A fitted MMM model instance.
    x : pd.DataFrame
        Feature matrix used for fitting.

    Returns
    -------
    az.InferenceData
        Updated inference data with posterior predictive samples.
    """
    model.sample_posterior_predictive(X=x)
    return model.idata
