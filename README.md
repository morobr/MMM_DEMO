# MMM Demo — Bayesian Marketing Mix Modeling

[![CI](https://github.com/morobr/MMM_DEMO/actions/workflows/ci.yml/badge.svg)](https://github.com/morobr/MMM_DEMO/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![PyMC-Marketing](https://img.shields.io/badge/PyMC--Marketing-latest-orange.svg)](https://github.com/pymc-labs/pymc-marketing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Also available in: [Português Brasileiro](README.pt-BR.md)

> **A production-style demo of Bayesian Marketing Mix Modeling (MMM) using PyMC-Marketing.**
> Built on real-world Indian retail data to demonstrate end-to-end channel attribution,
> adstock/saturation modeling, and data-driven budget optimization.

---

## What This Demo Shows

This project walks through the complete MMM workflow — from raw transaction data to
an optimized media budget — using a fully Bayesian approach:

| Step | What You Learn |
|------|---------------|
| **Data Engineering** | Aggregating daily sales and monthly spend into a weekly panel |
| **Channel Grouping** | Handling multicollinearity by collapsing correlated channels |
| **Bayesian Modeling** | Prior selection, GeometricAdstock, LogisticSaturation |
| **Diagnostics** | R-hat, ESS, divergences, prior/posterior predictive checks |
| **Decomposition** | Isolating each channel's contribution to total GMV |
| **Optimization** | Finding the budget allocation that maximizes predicted sales |

---

## Dataset

**Source:** [DT Mart Market Mix Modeling — Kaggle](https://www.kaggle.com/datasets/datatattle/dt-mart-market-mix-modeling)

Real Indian e-commerce data covering one year of operations (Jul 2015 – Jun 2016):

| File | Description |
|------|-------------|
| `Sales.csv` | Raw daily transaction records (~1M rows) |
| `firstfile.csv` | Daily aggregated GMV by product vertical |
| `Secondfile.csv` | Pre-built monthly MMM dataset (12 rows) |
| `MediaInvestment.csv` | Monthly spend by 9 marketing channels (values in Crores INR) |
| `MonthlyNPSscore.csv` | Monthly Net Promoter Score (range: 44–60) |
| `SpecialSale.csv` | 44 special sale event dates across 12 promotions |

Data is **not committed** to the repository. It downloads automatically from KaggleHub
on first run via `mmm_demo.data.load_mmm_weekly_data()`.

### Channel Setup

Seven raw spend columns are collapsed into four model channels to reduce multicollinearity
(pairwise correlations up to r = 0.99 between correlated channels):

| Model Channel | Raw Columns | Spend Share |
|---------------|-------------|-------------|
| **TV** | TV | 5.6% |
| **Sponsorship** | Sponsorship | 46.6% |
| **Digital** | Digital, SEM, Content.Marketing | 16.3% |
| **Online** | Online.marketing, Affiliates | 31.5% |

Radio and Other are excluded (9/12 months of missing data).

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A [Kaggle API token](https://www.kaggle.com/docs/api) at `~/.kaggle/kaggle.json`

### Setup

```bash
# Clone the repository
git clone https://github.com/morobr/MMM_DEMO.git
cd MMM_DEMO

# Create virtual environment and install all dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Verify installation
uv run python -c "import pymc_marketing; print('Ready:', pymc_marketing.__version__)"
```

### Run the Notebooks

```bash
uv run jupyter lab
```

Open notebooks in order from the `notebooks/` directory. Each notebook builds on the
output of the previous one.

### Run the Full Pipeline (non-interactive)

```bash
uv run python scripts/run_pipeline.py
```

This executes every pipeline step end-to-end: data loading → model fitting →
diagnostics → decomposition → optimization → saved outputs.

### Run Tests

```bash
uv run pytest -v
```

---

## Notebooks

The demo is structured as a progressive sequence of notebooks, each focused on one
stage of the MMM workflow:

| Notebook | Description |
|----------|-------------|
| [`01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) | Dataset overview, schema validation, correlation analysis, channel selection rationale |
| [`02_model_fitting.ipynb`](notebooks/02_model_fitting.ipynb) | Weekly data construction, prior predictive checks, MCMC sampling, trace inspection |
| [`03_diagnostics.ipynb`](notebooks/03_diagnostics.ipynb) | Convergence validation (R-hat, ESS, divergences), posterior predictive checks |
| [`04_contributions.ipynb`](notebooks/04_contributions.ipynb) | Channel decomposition, adstock/saturation curves, GMV attribution breakdown |
| [`05_optimization.ipynb`](notebooks/05_optimization.ipynb) | Budget optimization with BudgetOptimizer, scenario comparison, ROI analysis |

> **Tip:** Notebooks 03–05 depend on a fitted model saved in `outputs/models/`.
> Run notebook 02 first, or use the pre-fitted trace if provided.

---

## Project Structure

```
MMM_DEMO/
├── README.md
├── CLAUDE.md                      # Project spec and AI coding instructions
├── pyproject.toml                 # Dependencies and tooling config
├── uv.lock
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       └── ci.yml                 # Lint + format + test on every push
│
├── notebooks/                     # Progressive demo sequence
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_fitting.ipynb
│   ├── 03_diagnostics.ipynb
│   ├── 04_contributions.ipynb
│   └── 05_optimization.ipynb
│
├── scripts/
│   └── run_pipeline.py            # End-to-end pipeline runner
│
├── src/
│   └── mmm_demo/
│       ├── __init__.py
│       ├── config.py              # ModelConfig dataclass — all hyperparameters
│       ├── data.py                # Data loading, validation, weekly aggregation
│       ├── model.py               # MMM build + fit + predictive sampling
│       ├── diagnostics.py         # R-hat, ESS, divergence checks
│       ├── optimization.py        # BudgetOptimizer wrapper
│       └── plotting.py            # All visualization functions
│
├── tests/
│   ├── conftest.py                # Shared fixtures and mocked model fits
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_diagnostics.py
│   ├── test_optimization.py
│   └── test_plotting.py
│
├── data/                          # Auto-downloaded from KaggleHub (gitignored)
└── outputs/                       # Generated artifacts (gitignored)
    ├── models/                    # Fitted traces — mmm_fit_{date}_{desc}.nc
    ├── plots/
    │   ├── diagnostics/           # Trace, posterior predictive plots
    │   └── contributions/         # Channel decomposition plots
    ├── tables/                    # Summary CSVs
    └── optimization/              # Budget scenario results
```

---

## Architecture

The pipeline follows a **sequential flow with validation gates** — each step must pass
before the next begins:

```
KaggleHub Raw Data
        │
        ▼
  [data.py] Load & validate schema
        │
        ▼
  [data.py] Build weekly panel
  (aggregate GMV, distribute spend,
   assign NPS, count sale days,
   group correlated channels)
        │
        ▼
  [config.py] ModelConfig
  (channels, controls, priors,
   adstock_max_lag, sampling params)
        │
        ▼
  [model.py] Build MMM
  (GeometricAdstock + LogisticSaturation)
        │
        ▼
  [model.py] Prior Predictive Check
  ── verify plausible GMV ranges ──
        │
        ▼
  [model.py] MCMC Sampling
  (4 chains × 1000 draws + 2000 tune)
        │
        ▼
  [diagnostics.py] Convergence Gate ◄── MUST PASS (R-hat < 1.01, ESS > 400, divergences = 0)
        │
        ▼
  [model.py] Posterior Predictive Check
        │
        ▼
  [model.py] Channel Decomposition
        │
        ▼
  [optimization.py] Budget Optimization
        │
        ▼
  [plotting.py] Save All Outputs
        │
        ▼
  outputs/ (models, plots, tables, optimization)
```

---

## MMM Concepts

### Adstock (Carryover Effects)

Marketing spend today continues to influence sales for days or weeks afterward.
`GeometricAdstock` models this with exponential decay:

```
adstock_t = spend_t + α × adstock_{t-1}
```

where `α ∈ (0, 1)` is the decay rate and `l_max` caps the lag window (default: 4 weeks).

### Saturation (Diminishing Returns)

Doubling ad spend does not double the effect.
`LogisticSaturation` applies an S-curve transformation:

```
saturation(x) = (1 - exp(-λx)) / (1 + exp(-λx))
```

where `λ` controls the steepness of the curve. Higher λ → faster saturation.

### Priors

All priors are calibrated for MaxAbsScaled data (PyMC-Marketing scales inputs internally):

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| `intercept` | Normal(0, 2) | Uninformative baseline |
| `adstock_alpha` | Beta(1, 3) | Skewed toward low decay (short carryover) |
| `saturation_lam` | Gamma(3, 1) | Moderate saturation expected |
| `saturation_beta` | HalfNormal(2) | Non-negative channel coefficients |
| `gamma_control` | Normal(0, 2) | Uninformative control effects |

### Why Bayesian?

Bayesian MMM provides full uncertainty quantification. Instead of point estimates for
channel ROI, you get posterior distributions — enabling robust decision-making that
accounts for model uncertainty.

---

## Tech Stack

| Tool | Role |
|------|------|
| [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) | MMM framework (GeometricAdstock, LogisticSaturation, BudgetOptimizer) |
| [PyMC 5.x](https://www.pymc.io/) | Bayesian engine (MCMC sampling via NUTS) |
| [ArviZ](https://python.arviz.org/) | Posterior analysis, diagnostics, trace visualization |
| [pandas](https://pandas.pydata.org/) | Data manipulation and feature engineering |
| [numpy](https://numpy.org/) | Numerical operations |
| [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) | Visualization |
| [kagglehub](https://github.com/Kaggle/kagglehub) | Dataset download |
| [uv](https://docs.astral.sh/uv/) | Fast Python package and project manager |
| [ruff](https://docs.astral.sh/ruff/) | Linting and formatting |
| [pytest](https://pytest.org/) | Testing |

---

## Development

### Common Commands

```bash
# Lint
uv run ruff check .

# Fix lint issues automatically
uv run ruff check --fix .

# Format
uv run ruff format .

# Full verification (lint + format + tests)
uv run ruff check . && uv run ruff format --check . && uv run pytest

# Run a single test module
uv run pytest tests/test_data.py -v
```

### Testing Philosophy

- **Unit tests** for all public functions in every module
- **Synthetic fixtures** mimicking the DT Mart schema (no real data in tests)
- **Mocked MCMC** — model fits are always mocked to keep the test suite fast
- CI runs on every push via GitHub Actions

---

## Suggested Improvements (Roadmap)

This is a demo project. Known gaps and logical next steps:

- [ ] Time-varying adstock decay (non-stationary carryover)
- [ ] Holdout validation (train/test split for out-of-sample evaluation)
- [ ] Multiple budget scenarios (conservative, base, aggressive)
- [ ] Hierarchical priors across channels
- [ ] Export contribution tables to `outputs/tables/`

---

## Known Limitations

- **Small dataset:** Only ~52 weekly observations. Posteriors are wide and priors have
  significant influence. Results should be interpreted with caution.
- **Single market:** One retail market only; no regional/geographic breakdown.
- **No external regressors:** Competitor activity, pricing changes, and macroeconomic
  factors are not modeled.
- **Monthly → weekly distribution:** Media spend is pro-rata distributed across weeks,
  which is a simplification of actual scheduling.

---

## References

- [PyMC-Marketing GitHub](https://github.com/pymc-labs/pymc-marketing)
- [Marketing Mix Modeling — A Complete Guide (PyMC Labs)](https://www.pymc-labs.com/blog-posts/marketing-mix-modeling-a-complete-guide)
- [DT Mart Dataset on Kaggle](https://www.kaggle.com/datasets/datatattle/dt-mart-market-mix-modeling)
- [ArviZ Documentation](https://python.arviz.org/)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
