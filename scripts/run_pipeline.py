"""End-to-end MMM pipeline runner.

Executes every pipeline step non-interactively:
  data loading → model fitting → diagnostics → decomposition → optimization → outputs

Usage
-----
    uv run python scripts/run_pipeline.py
    uv run python scripts/run_pipeline.py --strict          # abort if convergence fails
    uv run python scripts/run_pipeline.py --skip-optimization
    uv run python scripts/run_pipeline.py --skip-prior-predictive

Outputs
-------
    outputs/models/mmm_fit_{date}_pipeline.nc
    outputs/plots/diagnostics/trace_{date}.png
    outputs/plots/diagnostics/posterior_predictive_{date}.png
    outputs/plots/contributions/channel_contributions_{date}.png
    outputs/tables/channel_contributions_{date}.csv
    outputs/optimization/budget_optimization_{date}.csv  (unless --skip-optimization)
"""

import argparse
import logging
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must come before any pyplot import


from mmm_demo.config import OUTPUTS_DIR, ModelConfig
from mmm_demo.data import load_mmm_weekly_data
from mmm_demo.diagnostics import check_convergence
from mmm_demo.model import (
    build_model,
    fit_model,
    sample_posterior_predictive,
    sample_prior_predictive,
)
from mmm_demo.plotting import (
    plot_channel_contributions,
    plot_posterior_predictive,
    plot_trace,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _step(n: int, total: int, name: str) -> None:
    log.info("=" * 60)
    log.info("STEP %d/%d: %s", n, total, name)
    log.info("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the full MMM pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort if convergence diagnostics fail (default: warn and continue).",
    )
    parser.add_argument(
        "--skip-prior-predictive",
        action="store_true",
        help="Skip the prior predictive check step.",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip the budget optimization step.",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full MMM pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments controlling pipeline behaviour.

    Raises
    ------
    SystemExit
        If --strict is set and convergence diagnostics fail.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    total_steps = 8 if args.skip_optimization else 9

    # ------------------------------------------------------------------
    # STEP 1: Load & validate data
    # ------------------------------------------------------------------
    _step(1, total_steps, "Load & validate data")
    df = load_mmm_weekly_data()
    config = ModelConfig()
    feature_cols = [
        config.date_column,
        *config.channel_columns,
        *config.control_columns,
    ]
    x = df[feature_cols]
    y = df[config.target_column]
    log.info(
        "Loaded %d weekly observations (%s to %s)",
        len(df),
        df["Date"].min().date(),
        df["Date"].max().date(),
    )
    log.info("Channels: %s", config.channel_columns)
    log.info("Controls: %s", config.control_columns)
    log.info(
        "Target (%s): min=%.0f  mean=%.0f  max=%.0f",
        config.target_column,
        y.min(),
        y.mean(),
        y.max(),
    )

    # ------------------------------------------------------------------
    # STEP 2: Build model
    # ------------------------------------------------------------------
    _step(2, total_steps, "Build model")
    mmm = build_model(config)
    log.info(
        "MMM initialised  adstock=%s (max_lag=%d)  saturation=%s",
        type(mmm.adstock).__name__,
        config.adstock_max_lag,
        type(mmm.saturation).__name__,
    )

    # ------------------------------------------------------------------
    # STEP 3: Prior predictive check
    # ------------------------------------------------------------------
    if args.skip_prior_predictive:
        log.info("STEP 3/%d: Prior predictive check — skipped", total_steps)
    else:
        _step(3, total_steps, "Prior predictive check")
        sample_prior_predictive(mmm, x, y, samples=500)
        log.info("Prior predictive samples drawn (500 samples)")

    # ------------------------------------------------------------------
    # STEP 4: Fit model (MCMC sampling)
    # ------------------------------------------------------------------
    _step(4, total_steps, "Fit model (MCMC sampling)")
    log.info(
        "chains=%d  draws=%d  tune=%d  target_accept=%.2f",
        config.chains,
        config.draws,
        config.tune,
        config.target_accept,
    )
    idata = fit_model(mmm, x, y, config)
    log.info("Sampling complete")

    # ------------------------------------------------------------------
    # STEP 5: Save model
    # ------------------------------------------------------------------
    _step(5, total_steps, "Save model")
    model_path = OUTPUTS_DIR / "models" / f"mmm_fit_{date_str}_pipeline.nc"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    mmm.save(str(model_path))
    log.info("Model saved to %s", model_path)

    # ------------------------------------------------------------------
    # STEP 6: Convergence diagnostics gate
    # ------------------------------------------------------------------
    _step(6, total_steps, "Convergence diagnostics")
    diag = check_convergence(idata)
    log.info("R-hat  (max): %.4f  %s", diag.max_rhat, "OK" if diag.rhat_ok else "FAIL")
    log.info("ESS    (min): %.0f   %s", diag.min_ess, "OK" if diag.ess_ok else "FAIL")
    log.info(
        "Divergences:  %d     %s",
        diag.divergences,
        "OK" if diag.divergences == 0 else "FAIL",
    )
    log.info("Overall:      %s", "PASSED" if diag.passed else "FAILED")

    trace_path = plot_trace(idata)
    log.info("Trace plot saved to %s", trace_path)

    if not diag.passed:
        msg = "Convergence failed: %s"
        if args.strict:
            log.error(msg, diag.summary)
            log.error(
                "Aborting (--strict). Fix convergence before interpreting results."
            )
            sys.exit(1)
        else:
            log.warning(msg, diag.summary)
            log.warning(
                "Continuing with caution. "
                "Pass --strict to abort on convergence failure."
            )

    # ------------------------------------------------------------------
    # STEP 7: Posterior predictive check
    # ------------------------------------------------------------------
    _step(7, total_steps, "Posterior predictive check")
    sample_posterior_predictive(mmm, x)
    pp_path = plot_posterior_predictive(mmm)
    log.info("Posterior predictive plot saved to %s", pp_path)

    # ------------------------------------------------------------------
    # STEP 8: Channel decomposition
    # ------------------------------------------------------------------
    _step(8, total_steps, "Channel decomposition")
    contrib_plot_path = plot_channel_contributions(mmm)
    log.info("Channel contributions plot saved to %s", contrib_plot_path)

    contributions = mmm.compute_mean_contributions_over_time(original_scale=True)
    tables_dir = OUTPUTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    contrib_csv = tables_dir / f"channel_contributions_{date_str}.csv"
    contributions.to_csv(contrib_csv)
    log.info("Contributions table saved to %s", contrib_csv)

    # ------------------------------------------------------------------
    # STEP 9: Budget optimization (best-effort)
    # ------------------------------------------------------------------
    opt_path: str | None = None
    if not args.skip_optimization:
        _step(9, total_steps, "Budget optimization")
        try:
            from mmm_demo.optimization import optimize_budget

            total_budget = float(df[config.channel_columns].mean().sum())
            budget_bounds = {
                ch: (df[ch].mean() * 0.20, df[ch].mean() * 1.80)
                for ch in config.channel_columns
            }
            result = optimize_budget(
                model=mmm,
                total_budget=total_budget,
                channels=config.channel_columns,
                budget_bounds=budget_bounds,
            )
            opt_dir = OUTPUTS_DIR / "optimization"
            opt_dir.mkdir(parents=True, exist_ok=True)
            opt_csv = opt_dir / f"budget_optimization_{date_str}.csv"
            result.to_csv(opt_csv, index=False)
            opt_path = str(opt_csv)
            log.info("Optimization results saved to %s", opt_csv)
        except Exception as exc:
            log.warning("Budget optimization failed: %s", exc)
            log.warning("Skipping. Run notebook 05 for manual scenario analysis.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Pipeline complete. Outputs:")
    log.info("  Model:              %s", model_path)
    log.info("  Trace plot:         %s", trace_path)
    log.info("  Posterior pred.:    %s", pp_path)
    log.info("  Contributions plot: %s", contrib_plot_path)
    log.info("  Contributions CSV:  %s", contrib_csv)
    if opt_path:
        log.info("  Optimization CSV:   %s", opt_path)
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline(parse_args())
