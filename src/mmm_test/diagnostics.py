"""Model diagnostics and convergence validation."""

from dataclasses import dataclass

import arviz as az

RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400


@dataclass
class DiagnosticResult:
    """Result of convergence diagnostics.

    Parameters
    ----------
    passed : bool
        Whether all diagnostic checks passed.
    rhat_ok : bool
        Whether all R-hat values are below threshold.
    ess_ok : bool
        Whether all ESS values are above threshold.
    divergences : int
        Number of divergent transitions.
    max_rhat : float
        Maximum R-hat value across all parameters.
    min_ess : float
        Minimum ESS value across all parameters.
    summary : str
        Human-readable summary of the diagnostics.
    """

    passed: bool
    rhat_ok: bool
    ess_ok: bool
    divergences: int
    max_rhat: float
    min_ess: float
    summary: str


def check_convergence(idata: az.InferenceData) -> DiagnosticResult:
    """Run convergence diagnostics on a fitted model trace.

    Checks R-hat (< 1.01), ESS (> 400), and divergences (== 0).

    Parameters
    ----------
    idata : az.InferenceData
        The fitted model trace from MCMC sampling.

    Returns
    -------
    DiagnosticResult
        Diagnostic results with pass/fail status and details.
    """
    summary = az.summary(idata)
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())

    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(idata.sample_stats["diverging"].sum().values)

    rhat_ok = max_rhat < RHAT_THRESHOLD
    ess_ok = min_ess > ESS_THRESHOLD
    no_divergences = divergences == 0
    passed = rhat_ok and ess_ok and no_divergences

    messages = []
    if not rhat_ok:
        messages.append(f"R-hat too high: {max_rhat:.4f} (threshold: {RHAT_THRESHOLD})")
    if not ess_ok:
        messages.append(f"ESS too low: {min_ess:.0f} (threshold: {ESS_THRESHOLD})")
    if not no_divergences:
        messages.append(f"Divergences detected: {divergences}")
    if passed:
        messages.append("All convergence checks passed")

    return DiagnosticResult(
        passed=passed,
        rhat_ok=rhat_ok,
        ess_ok=ess_ok,
        divergences=divergences,
        max_rhat=max_rhat,
        min_ess=min_ess,
        summary="; ".join(messages),
    )


def validate_before_interpretation(idata: az.InferenceData) -> None:
    """Gate check: raise if convergence diagnostics fail.

    Call this before interpreting model results to ensure the model
    has converged properly.

    Parameters
    ----------
    idata : az.InferenceData
        The fitted model trace.

    Raises
    ------
    RuntimeError
        If any convergence diagnostic fails.
    """
    result = check_convergence(idata)
    if not result.passed:
        raise RuntimeError(f"Model has not converged: {result.summary}")
