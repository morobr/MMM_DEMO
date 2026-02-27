# mmm_demo

Marketing Mix Modeling project using Bayesian inference to analyze marketing channel effectiveness. Built on PyMC-Marketing with the DT Mart dataset from KaggleHub. The project identifies channel contributions to sales, models adstock and saturation effects, explains channel behavior, and optimizes marketing investment allocation.

## Tech Stack

- **Language:** Python 3.12
- **Core framework:** PyMC-Marketing (MMM class, GeometricAdstock, LogisticSaturation, BudgetOptimizer)
- **Bayesian engine:** PyMC 5.x with ArviZ for diagnostics
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Data source:** kagglehub (DT Mart Market Mix Modeling dataset)
- **Notebooks:** Jupyter (exploration and prototyping)
- **Package manager:** uv with pyproject.toml
- **Linting/formatting:** ruff (linting and formatting in one tool)
- **Testing:** pytest
- **Pre-commit:** pre-commit with ruff hooks

## Project Structure

```
mmm_demo/
├── CLAUDE.md
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       └── ci.yml                  # ruff check + ruff format --check + pytest
├── notebooks/                      # Jupyter notebooks for exploration
├── src/
│   └── mmm_demo/
│       ├── __init__.py
│       ├── config.py               # Model hyperparameters, channel definitions, priors, constants
│       ├── data.py                 # Data loading, downloading, and preprocessing
│       ├── model.py                # MMM model definition and fitting
│       ├── diagnostics.py          # Model diagnostics and convergence validation
│       ├── optimization.py         # Budget optimization
│       └── plotting.py             # Charts and visualizations
├── data/                           # Raw and processed data (gitignored)
├── outputs/
│   ├── models/                     # Fitted model traces (netCDF via ArviZ)
│   ├── plots/
│   │   ├── diagnostics/            # Convergence, posterior, trace plots
│   │   └── contributions/          # Channel decomposition and attribution plots
│   ├── tables/                     # Summary CSVs
│   └── optimization/               # Budget optimization scenario results
└── tests/
    ├── conftest.py                 # Shared fixtures, mocked model fits
    ├── test_config.py
    ├── test_data.py
    ├── test_model.py
    ├── test_diagnostics.py
    ├── test_optimization.py
    └── test_plotting.py
```

### Module Responsibilities

- **`config.py`** -- Python dataclasses holding all configuration: `ModelConfig` with channel names, date column, control columns, prior distributions, adstock max_lag, saturation parameters. Provides IDE autocomplete and type validation.
- **`data.py`** -- Downloads dataset from KaggleHub on demand if not present in `data/`. Loads, validates schema/completeness, preprocesses, and feature-engineers the data. The `data/` directory is gitignored.
- **`model.py`** -- Defines the MMM model using PyMC-Marketing's MMM class, sets priors from config, fits the model, and returns the fitted trace.
- **`diagnostics.py`** -- Runs convergence checks (R-hat, ESS, divergences), prior/posterior predictive checks, and model validation. Acts as a gate before interpretation.
- **`optimization.py`** -- Uses PyMC-Marketing's BudgetOptimizer to find optimal channel allocation given constraints.
- **`plotting.py`** -- All visualization functions: diagnostic plots, channel contribution charts, adstock/saturation curves, optimization comparison plots.

## Development Setup

```bash
# Clone the repository
git clone <repo-url>
cd mmm_demo

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Verify installation
uv run python -c "import pymc_marketing; print(pymc_marketing.__version__)"
```

### Data Setup

Data is NOT committed to the repository. The `data.py` module downloads it automatically:

```python
from mmm_demo.data import load_data

df = load_data()  # Downloads from KaggleHub if data/ is empty, then loads and validates
```

The dataset source is: `kagglehub.dataset_download("datatattle/dt-mart-market-mix-modeling")`

Expected columns: Week, Sales, and various channel spend columns, plus control variables.

## Common Commands

```bash
# Run all tests
uv run pytest

# Run tests for a specific module
uv run pytest tests/test_data.py

# Run tests with verbose output
uv run pytest -v

# Lint check
uv run ruff check .

# Fix lint issues automatically
uv run ruff check --fix .

# Format check
uv run ruff format --check .

# Format files
uv run ruff format .

# Run lint + format + tests (full verification)
uv run ruff check . && uv run ruff format --check . && uv run pytest

# Launch Jupyter
uv run jupyter lab

# Add a new dependency
uv add <package-name>
```

## Architecture

### Data Flow Pipeline

The project follows a **sequential pipeline with validation gates** between each step. No step should proceed unless the previous step's output has been validated.

```
Raw CSV (KaggleHub)
  │
  ▼
[data.py] Load & validate schema/completeness
  │
  ▼
[data.py] Preprocess & feature engineer → validate transformed data
  │
  ▼
[config.py] Load model configuration (priors, channels, controls)
  │
  ▼
[model.py] Initialize MMM → set priors → fit model
  │
  ▼
[diagnostics.py] Validate convergence (R-hat, ESS, divergences) ← GATE: must pass before continuing
  │
  ▼
[model.py / diagnostics.py] Channel decomposition & contribution analysis
  │
  ▼
[diagnostics.py] Validate decomposition results
  │
  ▼
[optimization.py] Budget optimization via BudgetOptimizer
  │
  ▼
[plotting.py] Generate all visualizations
  │
  ▼
outputs/ (models, plots, tables, optimization results)
```

### Configuration Pattern

All model configuration lives in `config.py` as Python dataclasses with type hints:

```python
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Central configuration for the MMM model."""

    date_column: str = "Week"
    target_column: str = "Sales"
    channel_columns: list[str] = field(default_factory=list)
    control_columns: list[str] = field(default_factory=list)
    adstock_max_lag: int = 8
    # Prior distributions, saturation parameters, etc.
```

### Output Naming Convention

All output artifacts follow this naming pattern:

- **Models:** `outputs/models/mmm_fit_{YYYY-MM-DD}_{description}.nc`
- **Diagnostic plots:** `outputs/plots/diagnostics/{step}_{plot_name}.png`
- **Contribution plots:** `outputs/plots/contributions/{channel}_{plot_name}.png`
- **Summary tables:** `outputs/tables/{description}_{YYYY-MM-DD}.csv`
- **Optimization results:** `outputs/optimization/budget_scenario_{description}.csv`

All names are lowercase, snake_case, date-stamped for versioning, with descriptive suffixes.

## Coding Standards

### Formatting and Linting

- **Ruff** handles both linting and formatting
- Run `uv run ruff check .` and `uv run ruff format .` before committing
- Pre-commit hooks enforce this automatically

### Naming Conventions

- **Files:** lowercase snake_case -- `model_config.py`, `test_data.py`
- **Classes:** PascalCase -- `ModelConfig`, `ChannelContribution`
- **Functions/methods:** lowercase snake_case -- `load_data()`, `fit_model()`
- **Variables:** lowercase snake_case -- `channel_spend`, `adstock_decay`
- **Constants:** UPPER_SNAKE_CASE -- `DEFAULT_ADSTOCK_MAX_LAG`, `DATASET_ID`
- **Dataclass fields:** lowercase snake_case with type annotations

### Type Hints

- Required on **all public function signatures** (parameters and return types)
- Required on **all dataclass fields**
- Not required on local variables unless it improves readability
- Use modern Python 3.12 syntax: `list[str]` not `List[str]`, `str | None` not `Optional[str]`

```python
# Good
def fit_model(config: ModelConfig, data: pd.DataFrame) -> az.InferenceData:
    ...

# Bad -- missing type hints on public function
def fit_model(config, data):
    ...
```

### Docstrings

- **NumPy style** on all public functions and classes
- Include Parameters, Returns, and Raises sections where applicable

```python
def load_data(force_download: bool = False) -> pd.DataFrame:
    """Load the DT Mart dataset, downloading from KaggleHub if necessary.

    Parameters
    ----------
    force_download : bool, optional
        If True, re-download even if data exists locally. Default is False.

    Returns
    -------
    pd.DataFrame
        The raw dataset with validated schema.

    Raises
    ------
    ValueError
        If the downloaded data fails schema validation.
    """
```

### Error Handling

- Use specific exception types, not bare `except:`
- Validation functions should raise `ValueError` with descriptive messages
- Data loading failures should raise `FileNotFoundError` or `RuntimeError` with context
- Always include the problematic value in error messages when possible

### Import Ordering

Ruff handles import sorting automatically. The order is:

1. Standard library
2. Third-party packages
3. Local imports (`from mmm_demo import ...`)

## Testing

### Framework and Structure

- **pytest** with test files mirroring the module structure
- Tests live in `tests/` at the project root
- Shared fixtures and mocked model fits live in `tests/conftest.py`

### Test File Mapping

| Module | Test File |
|--------|-----------|
| `src/mmm_demo/config.py` | `tests/test_config.py` |
| `src/mmm_demo/data.py` | `tests/test_data.py` |
| `src/mmm_demo/model.py` | `tests/test_model.py` |
| `src/mmm_demo/diagnostics.py` | `tests/test_diagnostics.py` |
| `src/mmm_demo/optimization.py` | `tests/test_optimization.py` |
| `src/mmm_demo/plotting.py` | `tests/test_plotting.py` |

### Testing Approach

- **Unit tests** for all public functions in every module
- **Fixtures** for sample data (small synthetic DataFrames that mimic the DT Mart schema)
- **Mocked model fits** for testing diagnostics, optimization, and plotting without running actual MCMC sampling (which is slow)
- Use `pytest.fixture` for reusable test data and model objects
- Use `unittest.mock` or `pytest-mock` for mocking PyMC-Marketing internals
- Smoke tests for model initialization (ensure the model builds without errors)

### Running Tests

```bash
# All tests
uv run pytest

# Single module
uv run pytest tests/test_data.py

# Verbose with print output
uv run pytest -v -s

# Stop on first failure
uv run pytest -x
```

## Workflow

### Git Branching

- **`main`** branch is the stable branch
- **Feature branches** for all new work, named descriptively: `feat/channel-decomposition`, `fix/data-validation-nulls`, `refactor/config-dataclass`
- Merge feature branches into `main` when work is complete and CI passes

### Commit Messages

Follow **conventional commits**:

```
feat: add adstock transformation to model pipeline
fix: handle missing values in channel spend columns
refactor: extract validation logic into separate functions
docs: add docstrings to data module
test: add unit tests for budget optimizer
chore: update pre-commit hooks configuration
```

Format: `<type>: <short description in imperative mood>`

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`

### CI/CD

GitHub Actions runs on every push:

1. `uv run ruff check .` -- lint check
2. `uv run ruff format --check .` -- format check
3. `uv run pytest` -- full test suite

### Pre-commit Hooks

Pre-commit runs ruff linting and formatting checks before every commit. Install with:

```bash
uv run pre-commit install
```

## Do's and Don'ts

### Always Do

- Run `uv run ruff check .`, `uv run ruff format .`, and `uv run pytest` after making changes
- Add type hints to all public function signatures and dataclass fields
- Write NumPy-style docstrings for all public functions and classes
- Include validation gates between pipeline steps
- Explain the reasoning when suggesting or changing model priors
- Write or update tests when adding or modifying module code
- Follow the sequential workflow when adding a new channel: update `config.py` first, then `model.py`, then add/update tests, then update the relevant notebook
- Ask the user before proceeding when identifying opportunities for improvements or new workflows
- Use the output naming conventions defined in this document

### Never Do

- Never delete or overwrite files in `data/` or `outputs/`
- Never commit data files to the repository (`data/` is gitignored)
- Never modify `uv.lock` without being asked -- do not run `uv add` or `uv sync` unless explicitly instructed
- Never change model priors without explaining the reasoning and getting confirmation
- Never use bare `except:` -- always catch specific exceptions
- Never use `from typing import List, Dict, Optional` -- use modern Python 3.12 syntax (`list[str]`, `dict[str, int]`, `str | None`)
- Never skip validation gates in the pipeline
- Never interpret model results if convergence diagnostics have not passed

### When Uncertain

- **Always ask the user.** Do not guess on model configuration, prior choices, feature engineering decisions, or architectural changes. Present options with trade-offs and let the user decide.

## Domain Knowledge

### Marketing Mix Modeling (MMM) Concepts

- **Channel contribution:** The estimated effect each marketing channel (TV, radio, digital, etc.) has on sales, isolated from other factors
- **Adstock:** The lagged/carryover effect of advertising -- spending today continues to affect sales for days/weeks afterward. Modeled with `GeometricAdstock` which applies exponential decay with a `decay` parameter and `max_lag` (maximum number of time periods for the effect)
- **Saturation:** Diminishing returns from increased spending in a channel. Modeled with `LogisticSaturation` which applies an S-curve transformation with `lam` (steepness) parameter
- **Decomposition:** Breaking down total sales into the contribution of each channel plus baseline and controls
- **Budget optimization:** Given total budget and channel constraints, finding the allocation that maximizes predicted sales using `BudgetOptimizer`
- **Prior distributions:** Bayesian priors encode assumptions about parameters before seeing data. Choose them carefully and always validate with prior predictive checks

### Dataset

- **Source:** `kagglehub.dataset_download("datatattle/dt-mart-market-mix-modeling")`
- **Granularity:** Weekly
- **Target:** Sales
- **Features:** Marketing channel spend columns plus control variables
- **Time index:** Week column

### PyMC-Marketing Workflow

The standard workflow for this project:

1. **Configure** -- Define channels, controls, priors in `ModelConfig`
2. **Load & preprocess** -- Download data, validate, transform
3. **Initialize model** -- Create `MMM` instance with config
4. **Prior predictive check** -- `model.prior_predictive()` to verify priors produce plausible sales ranges
5. **Fit** -- `model.fit()` to run MCMC sampling
6. **Diagnose** -- Check R-hat (< 1.01), ESS (> 400), divergences (== 0), trace plots
7. **Posterior predictive check** -- Verify the model can reproduce observed data
8. **Decompose** -- Extract channel contributions to sales
9. **Optimize** -- Use `BudgetOptimizer` to find optimal allocation

## Known Issues & Gotchas

### PyMC-Marketing Specifics

- **R-hat values must be < 1.01** for all parameters before interpreting results. Values above this indicate the chains have not converged. Do not skip this check.
- **Effective Sample Size (ESS)** should be > 400 for reliable posterior estimates. Low ESS means the sampler is not exploring the posterior efficiently.
- **Divergences indicate sampling problems.** Any divergence count > 0 should be investigated. Common fixes: increase `target_accept` (e.g., 0.95 or 0.99), reparameterize, or adjust priors.
- **`adstock_max_lag` affects model speed significantly.** Higher values mean more parameters and slower sampling. Start with 4-8 and increase only if justified by the data.
- **Prior predictive checks are essential.** Always run them before fitting. If prior predictive samples produce unrealistic sales values (negative, astronomically large), the priors need adjustment.
- **Model fitting can be slow.** MCMC sampling may take minutes to hours depending on data size and model complexity. Always mock model fits in tests.
- **ArviZ InferenceData objects** are the standard format for storing traces. Save as netCDF (`.nc`) for portability and compatibility.

### General Data Science Gotchas

- **Multicollinearity between channels** is common (e.g., TV and radio campaigns often run simultaneously). This can make individual channel contributions unreliable. Check correlation matrices during EDA.
- **Seasonality and trends** should be accounted for as control variables, otherwise they get incorrectly attributed to marketing channels.
- **Scaling/normalization** of channel spend data may be needed depending on the magnitude differences between channels.

### References

- PyMC-Marketing repository: https://github.com/pymc-labs/pymc-marketing
- MMM complete guide: https://www.pymc-labs.com/blog-posts/marketing-mix-modeling-a-complete-guide
