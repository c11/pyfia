# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pyFIA** is a high-performance Python library for analyzing USDA Forest Inventory and Analysis (FIA) data. It provides statistically valid estimation methods following Bechtold & Patterson (2005) methodology.

## Design Philosophy

### Simplicity First
- **No over-engineering**: Avoid unnecessary patterns (Strategy, Factory, Builder)
- **Direct functions**: `volume(db)` not `VolumeEstimatorFactory.create().estimate()`
- **YAGNI**: Don't build for hypothetical future needs
- **Flat structure**: Maximum 3 levels of directory nesting

### Statistical Rigor
- Design-based estimation following Bechtold & Patterson (2005)
- Results must match EVALIDator (official USFS tool)
- Always include uncertainty estimates (SE, confidence intervals)
- Never compromise accuracy for convenience

### User Trust
- Show your work: Transparent methodology
- Validate against official sources
- Clear error messages when queries can't be answered
- Honest about limitations

## Documentation Map

| Document | Purpose |
|----------|---------|
| [README.md](./README.md) | Quick start for users |
| [docs/DEVELOPMENT.md](./docs/DEVELOPMENT.md) | Technical setup, architecture |
| [docs/fia_technical_context.md](./docs/fia_technical_context.md) | FIA methodology reference |
| [~/business/](../business/) | Business strategy and market analysis (outside repo) |

## Development Quick Reference

```bash
# Setup
uv venv && source .venv/bin/activate && uv pip install -e .[dev]
pre-commit install

# Test
uv run pytest                                          # all tests
uv run pytest tests/unit/test_mortality_estimator.py  # single file
uv run pytest --cov=pyfia --cov-report=html           # with coverage
uv run pytest tests/property/test_property_based.py -v # property-based

# Quality
uv run ruff format && uv run ruff check --fix && uv run mypy src/pyfia/
uv run pre-commit run --all-files

# Docs
uv run mkdocs serve
```

Tests are organized into `tests/unit/`, `tests/integration/`, `tests/validation/`, `tests/property/`, and `tests/e2e/`. Validation tests compare against EVALIDator results and require real FIA databases.

## Core Principles for Contributors

1. **User value first**: Every feature should reduce friction for end users
2. **Statistical validity**: Never ship estimates that could mislead
3. **Simplicity**: When in doubt, choose the simpler approach
4. **Real data testing**: Always test with actual FIA databases
5. **Documentation**: If it's not documented, it doesn't exist

## Important Notes

- **No backward compatibility debt**: Refactor freely, don't maintain old APIs
- **Performance matters**: Choose fast implementations over elegant abstractions
- **YAML schemas are source of truth**: FIA table definitions live in YAML
- **`mortality()` is the documentation gold standard**: Match its docstring quality

## Architecture

### Estimation Flow

All public estimation functions (`area`, `volume`, `tpa`, `mortality`, `growth`, `removals`, `biomass`, `area_change`, `site_index`, `tree_metrics`) follow the same pattern via `BaseEstimator` (Template Method):

1. **Load data** — `FIADataReader` fetches FIA tables (TREE, COND, PLOT, POP_* etc.)
2. **Filter** — `filtering/` applies `tree_domain` and `area_domain` SQL-like expressions
3. **Aggregate** — condition-level → plot-level → population totals
4. **Variance** — `estimation/variance.py` implements Bechtold & Patterson (2005) stratified formula
5. **Return** — Polars DataFrame with estimates + SE + confidence intervals

GRM-based estimators (`mortality`, `growth`, `removals`) extend `GRMBaseEstimator` instead and use `TREE_GRM_COMPONENT` + `TREE_GRM_MIDPT` tables.

### Key Classes and Files

| File | Role |
|------|------|
| `core/fia.py` | `FIA` class — database connection, `clip_by_evalid/state/clip_most_recent()` |
| `core/data_reader.py` | `FIADataReader` — efficient table loading with WHERE clause pushdown |
| `core/backends/` | DuckDB (default), SQLite, MotherDuck backends |
| `estimation/base.py` | `BaseEstimator` ABC + `AggregationResult` dataclass |
| `estimation/grm_base.py` | `GRMBaseEstimator` for mortality/growth/removals |
| `estimation/variance.py` | `calculate_domain_total_variance()` — matches EVALIDator within 1–3% |
| `filtering/parser.py` | Parses domain strings (e.g. `"DIA >= 10.0"`) to Polars expressions |
| `evalidator/` | HTTP client for EVALIDator API; used in `tests/validation/` |

### Dependencies

- **Polars** — primary dataframe library (use LazyFrame for memory efficiency)
- **DuckDB** — default backend (10–100x faster than SQLite)
- **Pydantic v2** — settings only, not for data objects
- **ConnectorX** — fast database connectivity

### EVALID System

EVALIDs follow `SSYYTT` format (state FIPS + 2-digit year + eval type). States with single-digit FIPS (e.g., AL=1) produce 5-digit EVALIDs. Always `clip_by_state()` or `clip_by_evalid()` before calling an estimator, and match `eval_type` to the function (`EXPVOL` → `volume()`, `EXPALL` → `area()`, `EXPMORT`/`EXPGROW` → `mortality()`/`growth()`).

```python
with FIA("data/nfi_south.duckdb") as db:
    db.clip_by_state(37)                   # North Carolina (FIPS 37)
    db.clip_most_recent(eval_type="VOL")
    results = volume(db, tree_domain="STATUSCD == 1")
```
