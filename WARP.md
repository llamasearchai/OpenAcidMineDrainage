# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

- Environment setup (Python >= 3.10)
  - Using uv (recommended):
    - uv venv && source .venv/bin/activate
    - uv pip install -e ".[dev,test]"
  - Using pip:
    - python3 -m venv .venv && source .venv/bin/activate
    - python -m pip install -U pip
    - pip install -e ".[dev,test]"
  - Optional extras:
    - Hydro (FloPy): pip install -e ".[hydro]"
    - Geochem (phreeqpy): pip install -e ".[geochem]"
    - Visualization (plotly export): pip install -e ".[viz]"

- Common commands
  - Build (sdist+wheel via hatchling backend):
    - python -m build
  - Lint (ruff):
    - ruff check src tests
    - ruff check --fix src tests
    - Via tox: tox -e lint
  - Type-check (mypy):
    - mypy src
    - Via tox: tox -e type
  - Tests (pytest configured via pyproject, quiet by default):
    - All: tox -q
    - Direct: pytest -q
    - Single test (direct): pytest tests/test_analyze.py::test_regression_by_parameter_has_valid_r2 -q
    - Single test (tox, choose one interpreter): tox -e py310 -- tests/test_analyze.py::test_regression_by_parameter_has_valid_r2 -q
  - Pre-commit (optional but configured):
    - pre-commit install
    - pre-commit run -a
  - CLI (installed as "oamd"):
    - oamd --help
    - Example (ingest → analyze → forecast → report):
      - oamd ingest data/example.csv --out artifacts/clean.csv
      - oamd analyze artifacts/clean.csv --param pH
      - oamd forecast artifacts/clean.csv --param pH --site-id D1 --horizon 14 --out artifacts/forecast.csv
      - oamd report artifacts/clean.csv --out artifacts/report.html
    - If PATH issues in dev, you can also run: python -m oamd.cli ...

- High-level architecture and flow
  - CLI orchestration (src/oamd/cli.py)
    - Subcommands: ingest, analyze, forecast, report, hydro, geochem, visualize, alert, regulatory.
    - Each subcommand calls a focused module and prints concise results/paths.
  - Data schema & validation
    - Pydantic models (data_models.py) define Measurement and SiteType; canonical columns include: timestamp, site_id, site_type, pH, conductivity_uScm, Fe_mg_L, Mn_mg_L, Al_mg_L, sulfate_mg_L.
    - validation.py standardizes incoming column names (synonym mapping), enforces types, timezone-aware timestamps, and categorical site_type.
  - Ingestion (ingest.py)
    - Loads one or more CSVs, concatenates, validates, and writes a cleaned dataset (artifacts/clean.csv by default).
  - Statistical analysis (analyze.py)
    - ANOVA across site_type groups; linear regression using one-hot site_type and a time index; optional seasonal decomposition. Uses scipy and statsmodels.
  - ML forecasting (ml_forecast.py)
    - RandomForestRegressor with calendar features and parameter lags; returns MAE and a predictions DataFrame for the forecast horizon.
  - Reporting (reporting.py + templates/report.html.jinja)
    - Renders an HTML report combining ANOVA/regression summaries and an inline base64 PNG timeseries (matplotlib) via Jinja2 templates.
  - Hydro integration (hydro.py)
    - Builds a minimal MF6 model via FloPy. If mf6 is present on PATH, runs it; otherwise writes inputs and reports status.
  - Geochemical integration (geochem.py)
    - Runs a minimal PHREEQC speciation calculation via phreeqpy if available; returns a status dict and avoids raising if unavailable.
  - Alerting (alerts.py)
    - Threshold-based checks over cleaned data; optional webhook POST to $OAMD_ALERT_WEBHOOK in JSON format.
  - Regulatory export (regulatory.py)
    - Writes a fixed, canonical CSV subset suitable for regulatory portals.
  - Visualization (visualize.py)
    - Saves static timeseries PNGs per parameter and site grouping.
  - Utilities (utils.py)
    - Directory helpers, artifact path resolution, column normalization.

- Tooling and CI
  - tox.ini
    - envlist: py310, py311, py312, lint, type
    - testenv runs: pytest {posargs}
    - lint runs: ruff check src tests
    - type runs: mypy src
    - tox-uv is required; tox uses uv automatically when available
  - pyproject.toml
    - project.scripts: oamd = oamd.cli:main
    - [tool.pytest.ini_options]: addopts = "-q", testpaths = ["tests"]
    - [tool.ruff]: line-length = 100, target-version = "py310", extend-select = ["E","F","I"]
    - [tool.mypy]: python_version = 3.10, ignore_missing_imports = true
  - Pre-commit: trimming whitespace, EOF fixer, YAML check, ruff lint/format are configured.
  - GitHub Actions: CI runs tox across Python 3.10–3.12.

- Operational notes
  - Outputs are written under artifacts/ (created automatically and git-ignored).
  - Optional integrations degrade gracefully when external tools are missing (FloPy/mf6, phreeqpy/PHREEQC DB).
  - Alerting webhook: export OAMD_ALERT_WEBHOOK with your endpoint to receive JSON alerts.
  - Reporting requires Jinja2; if you see ImportError for jinja2 when running "oamd report", install it: pip install jinja2 (or add it to your environment/extras).
  - License: MIT.

This file should be saved as WARP.md at the repository root.

