# pyFIA SQL Implementation

This document fully describes the SQL query builder layer added to pyFIA. Every public estimation function (`area`, `volume`, `tpa`, `biomass`, `mortality`, `growth`, `removals`, `area_change`, `site_index`, `tree_metrics`, `carbon`, `carbon_pool`, `carbon_flux`, `panel`) has a parallel `*_sql()` function that takes the same parameters and returns a complete SQL string that can be executed directly against FIADB tables to reproduce identical statistical estimates.

---

## Table of Contents

1. [Purpose and Design Goals](#1-purpose-and-design-goals)
2. [File Structure](#2-file-structure)
3. [Public API](#3-public-api)
4. [Statistical Methodology](#4-statistical-methodology)
5. [Infrastructure: `sql/base.py`](#5-infrastructure-sqlbasepy)
6. [GRM Infrastructure: `sql/_grm_base.py`](#6-grm-infrastructure-sql_grm_basepy)
7. [Estimator SQL Builders](#7-estimator-sql-builders)
   - [area_sql](#71-area_sql)
   - [volume_sql](#72-volume_sql)
   - [tpa_sql](#73-tpa_sql)
   - [biomass_sql](#74-biomass_sql)
   - [mortality_sql](#75-mortality_sql)
   - [growth_sql](#76-growth_sql)
   - [removals_sql](#77-removals_sql)
   - [area_change_sql](#78-area_change_sql)
   - [site_index_sql](#79-site_index_sql)
   - [tree_metrics_sql](#710-tree_metrics_sql)
   - [carbon_pool_sql](#711-carbon_pool_sql)
   - [carbon_sql](#712-carbon_sql)
   - [carbon_flux_sql](#713-carbon_flux_sql)
   - [panel_sql](#714-panel_sql)
8. [Variance CTE Chain](#8-variance-cte-chain)
9. [Grouping Support](#9-grouping-support)
10. [Domain Filters](#10-domain-filters)
11. [EVALID Handling](#11-evalid-handling)
12. [Adjustment Factors](#12-adjustment-factors)
13. [Usage Examples](#13-usage-examples)
14. [Limitations and Design Choices](#14-limitations-and-design-choices)

---

## 1. Purpose and Design Goals

The Python estimators in `pyfia.estimation.estimators.*` load FIA tables into Polars LazyFrames, apply filters, aggregate, and compute variance entirely in Python. The SQL layer provides an alternative that:

- **Returns a SQL string** — not computed data. The caller executes it.
- **Matches exactly** — the same two-stage aggregation, the same adjustment factors, the same B&P (2005) variance formula.
- **Is self-contained** — each query uses only standard FIA tables (no external views or stored procedures). Any DuckDB or compatible SQL engine with access to the FIADB tables can run it.
- **Preserves parameterisation** — `grp_by`, `by_species`, `land_type`, `tree_type`, `tree_domain`, `area_domain` are baked into the generated SQL at query-build time.

---

## 2. File Structure

```
src/pyfia/estimation/sql/
├── __init__.py          # Exports all *_sql functions
├── base.py              # Shared SQL-building primitives
├── _grm_base.py         # Shared GRM query builder (mortality/growth/removals)
├── area.py              # area_sql()
├── volume.py            # volume_sql()
├── tpa.py               # tpa_sql()
├── biomass.py           # biomass_sql()
├── mortality.py         # mortality_sql()  → wraps _grm_sql("mortality")
├── growth.py            # growth_sql()     → wraps _grm_sql("growth")
├── removals.py          # removals_sql()   → wraps _grm_sql("removals")
├── area_change.py       # area_change_sql()
├── site_index.py        # site_index_sql()
├── tree_metrics.py      # tree_metrics_sql()
├── carbon_pool.py       # carbon_pool_sql()
├── carbon.py            # carbon_sql()      → dispatcher to carbon_pool_sql / biomass_sql
├── carbon_flux.py       # carbon_flux_sql() → combined growth+mort+remv GRM query
└── panel.py             # panel_sql()       → condition or tree t1/t2 retrieval
```

All functions are re-exported from:
- `pyfia.estimation.sql`
- `pyfia.estimation`
- `pyfia` (top-level package)

---

## 3. Public API

```python
import pyfia

# All *_sql functions accept the same parameters as their Python counterparts.
# They return a str (SQL query), not computed data.

sql = pyfia.area_sql(db, land_type="forest")
sql = pyfia.volume_sql(db, by_species=True, vol_type="net")
sql = pyfia.tpa_sql(db, grp_by="FORTYPCD")
sql = pyfia.biomass_sql(db, component="AG", land_type="forest")
sql = pyfia.mortality_sql(db, tree_type="gs", measure="volume")
sql = pyfia.growth_sql(db, tree_type="gs", measure="volume")
sql = pyfia.removals_sql(db, land_type="timber")
sql = pyfia.area_change_sql(db, change_type="net")
sql = pyfia.site_index_sql(db, grp_by="FORTYPCD")
sql = pyfia.tree_metrics_sql(db, metrics=["qmd", "mean_dia"], grp_by="SPCD")
sql = pyfia.carbon_pool_sql(db, pool="total", land_type="forest")
sql = pyfia.carbon_sql(db, pool="live", by_species=True)
sql = pyfia.carbon_flux_sql(db, grp_by="STATECD")
sql = pyfia.panel_sql(db, level="condition", land_type="forest")
```

Each function requires a `FIA` instance with an EVALID already set:

```python
with pyfia.FIA("path/to/fiadb.duckdb") as db:
    db.clip_by_state(37)                 # North Carolina
    db.clip_most_recent(eval_type="VOL")
    sql = pyfia.volume_sql(db, tree_type="gs", land_type="timber")
    # Execute against the same database
    result = db.execute_sql(sql)
```

---

## 4. Statistical Methodology

All builders (except `tree_metrics_sql`) implement the same two-stage post-stratified design-based estimation described in Bechtold & Patterson (2005).

### Two-Stage Aggregation

**Stage 1 — condition level (`cond_level` CTE)**

For tree estimators (volume, TPA, biomass):
```
y_ic = Σ(metric × TPA_UNADJ × ADJ_FACTOR)   per (PLT_CN, CONDID, [group])
x_ic = CONDPROP_UNADJ × ADJ_FACTOR_AREA      (area denominator per condition)
```

For area:
```
y_ic = CONDPROP_UNADJ × ADJ_FACTOR_AREA × DOMAIN_IND
x_ic = CONDPROP_UNADJ × ADJ_FACTOR_AREA
```

For GRM estimators (mortality, growth, removals):
```
y_ic = Σ(metric × ADJ_FACTOR)   per (PLT_CN, [group])   -- no CONDID in GRM tables
x_ic = MAX(CONDPROP_UNADJ)                               -- plot-level area
```

**Stage 2 — population totals (`pop_totals` CTE)**
```
METRIC_TOTAL = Σ(y_ic × EXPNS)
AREA_TOTAL   = Σ(x_ic × EXPNS)
METRIC_ACRE  = METRIC_TOTAL / AREA_TOTAL
```

### Stratification Join

All queries open with the `strat` CTE:
```sql
strat AS (
    SELECT ppsa.PLT_CN, ps.STRATUM_CN, ps.EXPNS,
           ps.ADJ_FACTOR_MICR, ps.ADJ_FACTOR_SUBP, ps.ADJ_FACTOR_MACR,
           ps.STRATUM_WGT, ps.P2POINTCNT, ps.ESTN_UNIT_CN,
           ps.AREA_USED, ps.P1PNTCNT_EU
    FROM POP_PLOT_STRATUM_ASSGN ppsa
    JOIN POP_STRATUM ps ON ppsa.STRATUM_CN = ps.STRATUM_CN
    WHERE ps.EVALID IN (<evalids>)
)
```

This JOIN is the primary EVALID filter. Every plot that appears in the evaluation is enumerated here, including those with zero values — which is critical for unbiased variance estimation.

---

## 5. Infrastructure: `sql/base.py`

`base.py` provides all the shared building blocks. No estimator SQL is written directly; everything goes through these functions.

### 5.1 Domain Expression Translation

```python
_domain_to_sql(expr: str | None) -> str | None
```

Converts Python-style domain expressions to SQL by replacing `==` with `=`. All other operators (`!=`, `<`, `>`, `<=`, `>=`, `IN`, `BETWEEN`, `AND`, `OR`, `LIKE`) are valid SQL as-is.

### 5.2 Land Type Filter

```python
_land_type_sql(land_type: str, c: str = "c") -> str
```

| `land_type` | SQL fragment |
|---|---|
| `"forest"` | `c.COND_STATUS_CD = 1` |
| `"timber"` | `c.COND_STATUS_CD = 1 AND c.RESERVCD = 0 AND c.SITECLCD <= 6` |
| `"all"` | `1 = 1` (no restriction) |

### 5.3 Tree Type Filter

```python
_tree_type_sql(tree_type: str, t: str = "t") -> str
```

| `tree_type` | SQL fragment |
|---|---|
| `"live"` | `t.STATUSCD = 1` |
| `"dead"` | `t.STATUSCD = 2` |
| `"gs"` | `t.STATUSCD = 1 AND t.TREECLCD = 2` |
| `"all"` | `1 = 1` |

### 5.4 Tree Adjustment Factor

```python
_tree_adj_sql(t: str = "t", p: str = "p", s: str = "s") -> str
```

Mirrors `get_adjustment_factor_expr()` in `tree_expansion.py`. Selects the subplot design-based correction factor by tree diameter and plot macroplot breakpoint:

```sql
CASE
    WHEN t.DIA IS NULL THEN s.ADJ_FACTOR_SUBP
    WHEN t.DIA < 5.0   THEN s.ADJ_FACTOR_MICR
    WHEN t.DIA < COALESCE(CAST(p.MACRO_BREAKPOINT_DIA AS DOUBLE), 9999.0)
                       THEN s.ADJ_FACTOR_SUBP
    ELSE                    s.ADJ_FACTOR_MACR
END
```

### 5.5 Area Adjustment Factor

```python
_area_adj_sql(c: str = "c", s: str = "s") -> str
```

Mirrors `get_area_adjustment_factor_expr()` in `tree_expansion.py`. The area adjustment is based on the condition's `PROP_BASIS`:

```sql
CASE c.PROP_BASIS
    WHEN 'MACR' THEN s.ADJ_FACTOR_MACR
    ELSE             s.ADJ_FACTOR_SUBP
END
```

### 5.6 GRM Adjustment Factor

```python
_grm_adj_sql(s: str = "s") -> str
```

Mirrors `apply_grm_adjustment()` in `grm.py`. GRM tables use `SUBPTYP_GRM` (not tree DIA) to select the adjustment factor:

```sql
CASE gc.SUBPTYP_GRM
    WHEN 0 THEN 0.0
    WHEN 1 THEN s.ADJ_FACTOR_SUBP
    WHEN 2 THEN s.ADJ_FACTOR_MICR
    WHEN 3 THEN s.ADJ_FACTOR_MACR
    ELSE        0.0
END
```

### 5.7 EVALID List

```python
_evalid_list(db: FIA) -> str
```

Formats `db.evalid` (int or list of int) as a SQL `IN`-list literal, e.g. `"482101, 482119"`. Raises `ValueError` if no EVALID is set.

### 5.8 GROUP BY Helpers

```python
_gb_select(group_cols, alias="", trailing_comma=True) -> str
_gb_group(group_cols, leading_comma=True) -> str
_gb_join(left_alias, right_alias, group_cols) -> str
```

These produce the repetitive GROUP BY fragments needed throughout the CTE chain:

- `_gb_select(["SPCD", "FORTYPCD"])` → `"SPCD, FORTYPCD, "`
- `_gb_group(["SPCD", "FORTYPCD"])` → `", SPCD, FORTYPCD"`
- `_gb_join("pt", "fv", ["SPCD"])` → `" AND pt.SPCD = fv.SPCD"`

### 5.9 Stratification CTE

```python
_strat_cte(evalid_str: str) -> str
```

Produces the `strat` CTE (see Section 4). This is the first CTE in every query and the sole point where EVALID filtering is applied.

### 5.10 Standard Error Expressions

```python
_se_total_expr(var_col="fv.var_total") -> str
```
```sql
SQRT(GREATEST(fv.var_total, 0.0))
```

```python
_se_acre_expr() -> str
```

Ratio-of-means variance for the per-acre estimate — full B&P formula:
```sql
CASE WHEN fv.total_x > 0 THEN
    SQRT(GREATEST(
        (1.0 / (fv.total_x * fv.total_x)) * (
              fv.var_total
            + (fv.total_y / fv.total_x)^2 * fv.var_x
            - 2.0 * (fv.total_y / fv.total_x) * fv.cov_yx
        ), 0.0))
ELSE 0.0 END
```

### 5.11 Final JOIN

```python
_final_join(group_cols) -> str
```

Joins `pop_totals` to `final_var`:
- No groups: `CROSS JOIN final_var fv`
- With groups: `JOIN final_var fv ON pt.SPCD = fv.SPCD AND ...`

---

## 6. GRM Infrastructure: `sql/_grm_base.py`

Mortality, growth, and removals all share the same query structure. The difference between them is which `COMPONENT` rows are selected and which TPA/COMPONENT/SUBPTYP columns from `TREE_GRM_COMPONENT` are used.

### Column Naming Convention

FIA GRM columns follow the pattern:
```
SUBP_{TPA_PREFIX}_UNADJ_{TREE_CODE}_{LAND_CODE}
SUBP_COMPONENT_{TREE_CODE}_{LAND_CODE}
SUBP_SUBPTYP_GRM_{TREE_CODE}_{LAND_CODE}
```

| Parameter | `TREE_CODE` |
|---|---|
| `tree_type="gs"` | `GS` |
| `tree_type="live"` or `"al"` | `AL` |
| `tree_type="sawtimber"` or `"sl"` | `SL` |

| Parameter | `LAND_CODE` |
|---|---|
| `land_type="forest"` | `FOREST` |
| `land_type="timber"` | `TIMBER` |

| `component_type` | `TPA_PREFIX` |
|---|---|
| `"growth"` | `TPAGROW` |
| `"mortality"` | `TPAMORT` |
| `"removals"` | `TPAREMV` |

Example: `mortality(tree_type="gs", land_type="timber")` uses:
- `SUBP_TPAMORT_UNADJ_GS_TIMBER`
- `SUBP_COMPONENT_GS_TIMBER`
- `SUBP_SUBPTYP_GRM_GS_TIMBER`

### Component Filters

```python
_COMPONENT_FILTER = {
    "growth":    "gc.COMPONENT LIKE 'SURVIVOR%'",
    "mortality": "gc.COMPONENT LIKE 'MORTALITY%'",
    "removals":  "gc.COMPONENT LIKE 'CUT%' OR gc.COMPONENT LIKE 'OTHER_REMOVAL%'",
}
```

### Measure Expressions

| `measure` | `y` expression | Output label |
|---|---|---|
| `"volume"` | `TPA_UNADJ * VOLCFNET` | `VOLCFNET` |
| `"biomass"` | `TPA_UNADJ * (DRYBIO_BOLE + DRYBIO_BRANCH) * 0.0005` | `BIOMASS` |
| `"basal_area"` | `TPA_UNADJ * DIA_MIDPT² * 0.005454` | `BAA` |
| `"tpa"` / count | `TPA_UNADJ` | `TPA` |

### GRM Query Structure

```
WITH strat,

cond_plot AS (
    -- Aggregate COND to plot level (GRM tables have no CONDID)
    SELECT PLT_CN, ANY_VALUE(COND_STATUS_CD), SUM(CONDPROP_UNADJ),
           ANY_VALUE(RESERVCD), ANY_VALUE(SITECLCD),
           ANY_VALUE(FORTYPCD), ANY_VALUE(OWNGRPCD)
    FROM COND GROUP BY PLT_CN
),

grm_data AS (
    -- TREE_GRM_COMPONENT × TREE_GRM_MIDPT × strat × cond_plot
    -- Applies component filter, land filter, GRM adj factor
),

cond_level AS (
    -- y_ic = Σ(metric × ADJ_FACTOR) per (PLT_CN, [group])
    -- x_ic = MAX(CONDPROP_UNADJ)
),

pop_totals AS (
    -- METRIC_TOTAL, AREA_TOTAL, METRIC_ACRE, N_PLOTS
),

<variance CTEs>,

SELECT ...
```

The `EVALID` filter is applied exclusively through the `strat` CTE JOIN — GRM tables carry no EVALID column. All plots in the evaluation appear in `strat`; the `grm_data` JOIN naturally restricts to those plots.

**Land-type filter on GRM data** is applied at the condition level using `cond_plot.COND_STATUS_CD`:
- `forest`: `cond_agg.COND_STATUS_CD = 1`
- `timber`: `cond_agg.COND_STATUS_CD = 1 AND cond_agg.RESERVCD = 0 AND cond_agg.SITECLCD <= 6`

**Growing-stock filter**: When `tree_type = "gs"`, adds `AND gc.DIA_MIDPT >= 5.0`.

---

## 7. Estimator SQL Builders

### 7.1 `area_sql`

**File:** `sql/area.py`
**Counterpart:** `area()` / `AreaEstimator`

```python
area_sql(
    db,
    grp_by=None,
    land_type="forest",
    area_domain=None,
    plot_domain=None,
) -> str
```

**Formula:**
```
AREA_TOTAL = Σ(CONDPROP_UNADJ × ADJ_FACTOR_AREA × EXPNS × DOMAIN_IND)
```

**Key design — Domain Indicator approach:**
Unlike tree estimators that filter out non-matching rows, area estimation retains **all plots** (all conditions in the evaluation) and uses `DOMAIN_IND = CASE WHEN <filter> THEN 1.0 ELSE 0.0 END`. Non-domain conditions contribute zero to sums but are included in the variance calculation. This is required for unbiased variance when the domain is a subset of the total area.

```sql
DOMAIN_IND = CASE WHEN (land_filter) [AND (area_domain)] THEN 1.0 ELSE 0.0 END
y_ic = CONDPROP_UNADJ × ADJ_FACTOR_AREA × DOMAIN_IND
x_ic = CONDPROP_UNADJ × ADJ_FACTOR_AREA
```

**Output columns:** `AREA_TOTAL`, `AREA_PERCENT`, `AREA_SE`, `AREA_VARIANCE`, `N_PLOTS`

---

### 7.2 `volume_sql`

**File:** `sql/volume.py`
**Counterpart:** `volume()` / `VolumeEstimator`

```python
volume_sql(
    db,
    grp_by=None,
    by_species=False,
    land_type="forest",
    tree_type="live",
    vol_type="net",
    tree_domain=None,
    area_domain=None,
    plot_domain=None,
) -> str
```

**Volume column mapping:**

| `vol_type` | Column |
|---|---|
| `"net"` | `VOLCFNET` |
| `"gross"` | `VOLCFGRS` |
| `"sound"` | `VOLCFSND` |
| `"sawlog"` | `VOLBFNET` |

**Formula:**
```
y_ic = Σ(VOL_COL × TPA_UNADJ × ADJ_FACTOR)   per condition
x_ic = CONDPROP_UNADJ × ADJ_FACTOR_AREA
VOL_ACRE = Σ(y_ic × EXPNS) / Σ(x_ic × EXPNS)
```

**Output columns:** `VOL_TOTAL`, `AREA_TOTAL`, `VOL_ACRE`, `N_PLOTS`, `{VOL_COL}_TOTAL_SE`, `{VOL_COL}_ACRE_SE`

---

### 7.3 `tpa_sql`

**File:** `sql/tpa.py`
**Counterpart:** `tpa()` / `TPAEstimator`

```python
tpa_sql(
    db,
    grp_by=None,
    by_species=False,
    land_type="forest",
    tree_type="live",
    tree_domain=None,
    area_domain=None,
    plot_domain=None,
) -> str
```

Computes both TPA and BAA in a single query.

**Formulas:**
```
TPA y_ic = Σ(TPA_UNADJ × ADJ_FACTOR)
BAA y_ic = Σ(0.005454 × DIA² × TPA_UNADJ × ADJ_FACTOR)
```
where `0.005454 = π / (4 × 144)` converts `DIA` in inches to basal area in ft².

**Variance handling:** The variance CTEs operate on `y_ic = tpa_ic` (TPA). A `cond_level_tpa` view aliases `tpa_ic` as `y_ic` to feed the variance chain. The BAA SE is not independently computed; call `volume_sql` with the relevant volume column for a per-acre SE on basal area.

**Output columns:** `TPA_TOTAL`, `BAA_TOTAL`, `AREA_TOTAL`, `TPA_ACRE`, `BAA_ACRE`, `N_PLOTS`, `TPA_TOTAL_SE`, `TPA_ACRE_SE`

---

### 7.4 `biomass_sql`

**File:** `sql/biomass.py`
**Counterpart:** `biomass()` / `BiomassEstimator`

```python
biomass_sql(
    db,
    grp_by=None,
    by_species=False,
    component="AG",
    land_type="forest",
    tree_type="live",
    tree_domain=None,
    area_domain=None,
    plot_domain=None,
) -> str
```

**Component mapping:**

| `component` | Expression |
|---|---|
| `"AG"` | `t.DRYBIO_AG` |
| `"BG"` | `t.DRYBIO_BG` |
| `"TOTAL"` | `(t.DRYBIO_AG + t.DRYBIO_BG)` |
| other | `t.DRYBIO_{component}` |

**Conversion constants:**
- `LBS_TO_TONS = 0.0005` (= 1/2000)
- `CARBON_FRAC = 0.47`

**Formulas:**
```
y_ic = Σ(DRYBIO × TPA_UNADJ × ADJ_FACTOR × 0.0005)   per condition
BIOMASS_ACRE  = Σ(y_ic × EXPNS) / AREA_TOTAL
CARBON_ACRE   = BIOMASS_ACRE × 0.47
CARBON_TOTAL  = BIOMASS_TOTAL × 0.47
```

**Output columns:** `BIOMASS_TOTAL`, `CARBON_TOTAL`, `AREA_TOTAL`, `BIOMASS_ACRE`, `CARBON_ACRE`, `N_PLOTS`, `BIOMASS_TOTAL_SE`, `BIOMASS_ACRE_SE`

---

### 7.5 `mortality_sql`

**File:** `sql/mortality.py`
**Counterpart:** `mortality()` / `MortalityEstimator`

```python
mortality_sql(
    db,
    grp_by=None,
    by_species=False,
    tree_type="gs",
    land_type="timber",
    measure="volume",
    tree_domain=None,
    area_domain=None,
) -> str
```

Thin wrapper around `_grm_sql("mortality", ...)`.

**Component filter:** `COMPONENT LIKE 'MORTALITY%'`

The TPA column used is `SUBP_TPAMORT_UNADJ_{TC}_{LC}` which stores pre-annualised mortality rates (trees per acre per year). No further division by remeasurement period is needed.

**Output columns (measure="volume"):** `VOLCFNET_TOTAL`, `AREA_TOTAL`, `VOLCFNET_ACRE`, `N_PLOTS`, `VOLCFNET_TOTAL_SE`, `VOLCFNET_ACRE_SE`

Output label varies by measure: `VOLCFNET`, `BIOMASS`, `BAA`, or `TPA`.

---

### 7.6 `growth_sql`

**File:** `sql/growth.py`
**Counterpart:** `growth()` / `GrowthEstimator`

```python
growth_sql(
    db,
    grp_by=None,
    by_species=False,
    tree_type="gs",
    land_type="forest",
    measure="volume",
    tree_domain=None,
    area_domain=None,
) -> str
```

Thin wrapper around `_grm_sql("growth", ...)`.

**Component filter:** `COMPONENT LIKE 'SURVIVOR%'`

**Note on methodology difference:** The Python `GrowthEstimator` uses a BEGINEND cross-join (ONEORTWO=2 ending values minus ONEORTWO=1 beginning values, summed to get net growth). This SQL implementation uses the simpler SURVIVOR-component approach — trees that survived the measurement period contribute their midpoint metric to the growth estimate. Both approaches are used in FIA practice; the SURVIVOR approach is the standard EVALIDator pattern for cumulative survivor growth.

**Output columns:** Same label pattern as mortality_sql, e.g. `VOLCFNET_TOTAL`, `VOLCFNET_ACRE`, etc.

---

### 7.7 `removals_sql`

**File:** `sql/removals.py`
**Counterpart:** `removals()` / `RemovalsEstimator`

```python
removals_sql(
    db,
    grp_by=None,
    by_species=False,
    tree_type="gs",
    land_type="forest",
    measure="volume",
    tree_domain=None,
    area_domain=None,
) -> str
```

Thin wrapper around `_grm_sql("removals", ...)`.

**Component filter:** `COMPONENT LIKE 'CUT%' OR COMPONENT LIKE 'OTHER_REMOVAL%'`

The TPA column is `SUBP_TPAREMV_UNADJ_{TC}_{LC}` — pre-annualised removal rates.

**Output columns:** Same pattern as mortality_sql.

---

### 7.8 `area_change_sql`

**File:** `sql/area_change.py`
**Counterpart:** `area_change()` / `AreaChangeEstimator`

```python
area_change_sql(
    db,
    grp_by=None,
    land_type="forest",
    change_type="net",
    area_domain=None,
) -> str
```

Uses `SUBP_COND_CHNG_MTRX` to track subplot-level land-use transitions between measurement periods.

**Change indicators per subplot-condition:**
```
gain = CASE WHEN PREV_COND_STATUS_CD != 1 AND CURR_COND_STATUS_CD = 1
             THEN COALESCE(SUBTYP_PROP_CHNG, 1.0) ELSE 0.0 END
loss = CASE WHEN PREV_COND_STATUS_CD = 1 AND CURR_COND_STATUS_CD != 1
             THEN COALESCE(SUBTYP_PROP_CHNG, 1.0) ELSE 0.0 END
```

| `change_type` | `change_value` |
|---|---|
| `"net"` | `gain - loss` |
| `"gross_gain"` | `gain` |
| `"gross_loss"` | `loss` |

**Annualisation:**
```
y_ic = Σ(change_value × ADJ_FACTOR_AREA) / REMPER   per condition
```

Division by `REMPER` (remeasurement period in years) converts total change over the period to annual rate.

**Join sequence:**
1. `SUBP_COND_CHNG_MTRX sc` — one row per subplot-condition transition
2. `COND curr_cond` — current status, CONDPROP, grouping columns
3. `COND prev_cond` — previous status (via `PREV_PLT_CN` / `PREVCOND`)
4. `PLOT p` — `REMPER`
5. `strat s` — expansion and adjustment factors

**Note:** `prev_cond` join is a LEFT JOIN because some previous plots may not appear in the current evaluation's COND table (they belong to a prior inventory cycle). Rows where `prev_cond.COND_STATUS_CD IS NULL` are excluded in the WHERE clause.

**Output columns:** `CHANGE_TOTAL`, `AREA_TOTAL`, `N_PLOTS`, `CHANGE_TOTAL_SE`

---

### 7.9 `site_index_sql`

**File:** `sql/site_index.py`
**Counterpart:** `site_index()` / `SiteIndexEstimator`

```python
site_index_sql(
    db,
    grp_by=None,
    land_type="forest",
    area_domain=None,
    plot_domain=None,
) -> str
```

Computes area-weighted mean site index (`SICOND`) from the `COND` table.

**Always groups by `SIBASE`** (site index base age), prepended to any user-specified `grp_by`. SICOND values measured at different base ages are not comparable, so results must be separated by SIBASE.

**Formula:**
```
y_ic = SICOND × CONDPROP_UNADJ × ADJ_FACTOR_AREA   (SI numerator)
x_ic = CONDPROP_UNADJ × ADJ_FACTOR_AREA             (area denominator)
SITE_INDEX_MEAN = Σ(y_ic × EXPNS) / Σ(x_ic × EXPNS)
```

Conditions with `SICOND IS NULL` are excluded (unlike area estimation, null values cannot contribute to a mean calculation).

**Output columns:** `SIBASE`, `[grp_by]`, `SITE_INDEX_MEAN`, `AREA_TOTAL`, `N_PLOTS`, `SI_WEIGHTED_SE`

---

### 7.10 `tree_metrics_sql`

**File:** `sql/tree_metrics.py`
**Counterpart:** `tree_metrics()` / `TreeMetricsEstimator`

```python
tree_metrics_sql(
    db,
    metrics,
    grp_by=None,
    land_type="forest",
    tree_type="live",
    tree_domain=None,
    area_domain=None,
    sawtimber_threshold=9.0,
) -> str
```

**These are sample-level descriptive statistics, not population estimates.** No expansion factors (`EXPNS`) are applied in the final aggregation and no standard errors are produced.

**Supported metrics:**

| `metric` | SQL expression | Output column |
|---|---|---|
| `"qmd"` | `SQRT(SUM(DIA²×TPA_UNADJ) / SUM(TPA_UNADJ))` | `QMD` |
| `"mean_dia"` | `SUM(DIA×TPA_UNADJ) / SUM(TPA_UNADJ)` | `MEAN_DIA` |
| `"mean_height"` | `SUM(HT×TPA_UNADJ WHERE HT IS NOT NULL) / SUM(TPA_UNADJ WHERE HT IS NOT NULL)` | `MEAN_HT` |
| `"softwood_prop"` | `SUM(DRYBIO_BOLE WHERE SPCD<300) / SUM(DRYBIO_BOLE)` | `SOFTWOOD_PROP` |
| `"sawtimber_prop"` | `SUM(TPA_UNADJ WHERE DIA>=threshold) / SUM(TPA_UNADJ)` | `SAWTIMBER_PROP` |
| `"max_dia"` | `MAX(DIA)` | `MAX_DIA` |
| `"stocking"` | `SUM(TPA_UNADJ × (DIA/10)^1.6)` | `STOCKING` |

**Always included:** `N_PLOTS`, `N_TREES`

An unknown metric name raises `ValueError` at query-build time (before any SQL is executed).

The query still includes the `strat` CTE and JOINs to `strat` for EVALID filtering, but does not carry `EXPNS` through to the final aggregation.

---

### 7.11 `carbon_pool_sql`

**File:** `sql/carbon_pool.py`
**Counterpart:** `carbon_pool()` / `CarbonPoolEstimator`

```python
carbon_pool_sql(
    db,
    grp_by=None,
    by_species=False,
    pool="total",
    land_type="forest",
    tree_type="live",
    tree_domain=None,
    area_domain=None,
    plot_domain=None,
) -> str
```

Uses FIA's pre-calculated `CARBON_AG` (aboveground) and `CARBON_BG` (belowground) columns from the `TREE` table, converting from pounds to short tons via `× 0.0005`. This matches EVALIDator estimate number 55000 exactly, unlike `biomass_sql` which uses `DRYBIO × 0.47`.

**Pool options:**

| `pool` | Expression | Output column |
|---|---|---|
| `"ag"` | `CARBON_AG × 0.0005` | `CARBON_AG_TOTAL`, `CARBON_AG_ACRE` |
| `"bg"` | `CARBON_BG × 0.0005` | `CARBON_BG_TOTAL`, `CARBON_BG_ACRE` |
| `"total"` | `(CARBON_AG + CARBON_BG) × 0.0005` | `CARBON_TOTAL`, `CARBON_ACRE` |

**Output columns:** pool-specific total + per-acre, `AREA_TOTAL`, `N_PLOTS`, plus SE and CI columns.

The two-stage aggregation structure is identical to `biomass_sql`: `tree_cond → cond_level → plot_var → variance chain → final_var → SELECT`.

---

### 7.12 `carbon_sql`

**File:** `sql/carbon.py`
**Counterpart:** `carbon()` estimator

```python
carbon_sql(
    db,
    grp_by=None,
    by_species=False,
    pool="live",
    land_type="forest",
    tree_type="live",
    tree_domain=None,
    area_domain=None,
    plot_domain=None,
) -> str
```

Dispatcher function that routes to the appropriate underlying SQL builder based on the `pool` argument:

| `pool` | Routes to |
|---|---|
| `"ag"` | `carbon_pool_sql(pool="ag")` |
| `"bg"` | `carbon_pool_sql(pool="bg")` |
| `"live"` | `carbon_pool_sql(pool="total")` (AG + BG for live trees) |
| `"total"` | `carbon_pool_sql(pool="total")` |
| `"dead"` | `biomass_sql(tree_type="dead", component="TOTAL")` with `× 0.47` |
| `"litter"`, `"soil"` | Raises `ValueError` (not available in FIA tree tables) |

`carbon_sql` does not generate its own CTE chain — it delegates entirely to `carbon_pool_sql` or `biomass_sql`.

---

### 7.13 `carbon_flux_sql`

**File:** `sql/carbon_flux.py`
**Counterpart:** `carbon_flux()` estimator

```python
carbon_flux_sql(
    db,
    grp_by=None,
    by_species=False,
    land_type="forest",
    tree_type="gs",
    tree_domain=None,
    area_domain=None,
) -> str
```

Computes annual net carbon flux in a single combined query, avoiding three separate round-trips. The Python `carbon_flux()` function calls `growth(measure="biomass")`, `mortality(measure="biomass")`, and `removals(measure="biomass")` separately and combines the results in Python. This SQL version uses shared CTEs for all three:

**CTE structure:**

```
strat                      -- shared
cond_plot                  -- shared (aggregated COND)
grm_grow / level_grow / totals_grow    -- growth (SURVIVOR components)
grm_mort / level_mort / totals_mort    -- mortality (MORTALITY components)
grm_remv / level_remv / totals_remv    -- removals (CUT / OTHER_REMOVAL)
```

**Formula:**
```
NET_CARBON_FLUX_TOTAL = (GROWTH_BIO_TOTAL - MORT_BIO_TOTAL - REMV_BIO_TOTAL) × 0.47
NET_CARBON_FLUX_ACRE  = NET_CARBON_FLUX_TOTAL / AREA_TOTAL
```

**No standard errors are computed.** The Python version uses a simplified approach; the combined SQL query would require propagating covariances between three correlated GRM components, which is not practical in a single SQL query. Run `growth_sql`, `mortality_sql`, and `removals_sql` individually with `measure="biomass"` to obtain per-component SEs.

**Final SELECT** joins the three `totals_*` CTEs via `CROSS JOIN` (ungrouped) or keyed `LEFT JOIN` (grouped), with `COALESCE(..., 0.0)` for components that may have no matching trees.

---

### 7.14 `panel_sql`

**File:** `sql/panel.py`
**Counterpart:** `panel()` / `PanelBuilder`

```python
panel_sql(
    db,
    level="condition",
    land_type="forest",
    tree_type="all",
    tree_domain=None,
    area_domain=None,
    min_remper=0.0,
    max_remper=None,
    harvest_only=False,
) -> str
```

**This is a data retrieval query, not a population estimator.** No expansion factors (`EXPNS`) are applied, no variance is computed. Each output row is one remeasured condition or tree with t1 (previous) and t2 (current) attributes.

Dispatches to one of two internal generators:

**Condition-level panel** (`level="condition"`):

```sql
PLOT p2                       -- current plot
JOIN PLOT p1 ON p2.PREV_PLT_CN = p1.CN   -- previous plot
JOIN COND c2 ON p2.CN = c2.PLT_CN        -- current condition
LEFT JOIN COND c1 ON p1.CN = c1.PLT_CN AND c2.CONDID = c1.CONDID  -- previous condition
JOIN strat s ON p2.CN = s.PLT_CN         -- EVALID filter
```

Columns are prefixed `T2_` (current) and `T1_` (previous), e.g. `T2_COND_STATUS_CD`, `T1_FORTYPCD`. A `DOMAIN_IND`-style `T2_IN_DOMAIN` / `T1_IN_DOMAIN` indicator is computed from `land_type` without excluding rows.

`harvest_only=True` appends `AND (c2.TRTCD1 IN (10, 20) OR c2.TRTCD2 IN (10, 20) OR c2.TRTCD3 IN (10, 20))`.

**Tree-level panel** (`level="tree"`):

```sql
TREE_GRM_COMPONENT gc
JOIN TREE_GRM_MIDPT gm ON gc.TRE_CN = gm.TRE_CN
JOIN PLOT p             ON gc.PLT_CN = p.CN
JOIN strat s            ON gc.PLT_CN = s.PLT_CN
JOIN cond_plot cp       ON gc.PLT_CN = cp.PLT_CN
```

Provides `DIA_BEGIN` (diameter at t1), `DIA_MIDPT` (midpoint), `COMPONENT` (tree fate: SURVIVOR, MORTALITY1/2, CUT1/2, OTHER_REMOVAL1/2), plus volume and biomass at midpoint.

Both panels filter `REMPER > 0` and `INVYR >= 2000`.

---

## 8. Variance CTE Chain

All estimators except `tree_metrics_sql` produce an identical variance CTE chain after `pop_totals`, generated by `_variance_ctes(group_cols)` in `base.py`. The chain implements Bechtold & Patterson (2005) post-stratified variance.

### CTE Sequence

```
cond_level          -- defined by each estimator (y_ic, x_ic per condition)
  │
  ▼
plot_var            -- sum y_ic/x_ic to plot level; one row per (PLT_CN, [group])
  │
  ├── all_plots_base  -- DISTINCT PLT_CN from strat (every plot, incl. zeros)
  │   └── [group_vals] -- DISTINCT group combinations (when grouped)
  │
  ▼
all_plots_filled    -- CROSS JOIN group_vals × all_plots_base,
                    -- LEFT JOIN plot_var; fill NULLs with 0.0
                    -- Every group × plot combination exists, zeros included
  ▼
strat_stats         -- Per (ESTN_UNIT_CN, STRATUM_CN, [group]):
                    --   n_h_actual, ybar_h, s²_yh, xbar_h, s²_xh, cov_yxh
  ▼
eu_n                -- n_eu = total plots per (ESTN_UNIT_CN, [group])
  ▼
strat_var           -- Per-stratum variance contributions:
                    --   v_yh = s²_yh / P2POINTCNT
                    --   v_xh = s²_xh / P2POINTCNT
                    --   c_yxh = cov_yxh / P2POINTCNT
  ▼
eu_var              -- Per EU: sum_v1_y, sum_v2_y, sum_v1_x, ..., total_y, total_x
  ▼
final_var           -- Aggregated across EUs:
                    --   var_total, var_x, cov_yx, total_y, total_x
```

### B&P Variance Formula

The final `var_total` computed in `eu_var` and aggregated in `final_var` is:

```
V(Ŷ) = Σ_eu [ (A²/n_eu) × Σ_h W_h s²_yh/n_h   +   (A²/n_eu²) × Σ_h (1-W_h) s²_yh/n_h ]
             └─────────────── V1 (between-stratum) ──────────────┘   └───── V2 (within) ─────┘
```

where:
- `A` = `AREA_USED` (acres in the estimation unit)
- `n_eu` = total sample plots in the estimation unit
- `W_h` = `STRATUM_WGT` (stratum weight = proportion of area in stratum)
- `s²_yh` = sample variance of `y_i` within stratum `h`
- `n_h` = `P2POINTCNT` (number of phase-2 sample points in stratum)

The `GREATEST(..., 0.0)` guards prevent negative variance from floating-point rounding.

### Zero-Fill Requirement

The `all_plots_filled` CTE is critical. Without it, strata where no trees meet the filter would have inflated variance because `s²_yh` would be computed from only the non-zero plots. The LEFT JOIN + COALESCE(y_i, 0) pattern ensures every plot in the evaluation contributes a zero to the stratum variance calculation when it has no matching trees.

For grouped queries, `CROSS JOIN group_vals` creates one row per (plot × group) combination before the LEFT JOIN, so each group gets a complete set of zero-filled plots.

---

## 9. Grouping Support

Every SQL builder accepts `grp_by` (str or list of str) and `by_species` (bool). Grouping operates at two levels:

**1. Data selection** — in the initial `tree_cond` or `grm_data` CTE, group columns are fetched with their appropriate table prefix:

```python
tree_level_cols = {"SPCD", "SPGRPCD", "DIA", "HT", "TREECLCD", ...}
# → t.SPCD for tree-level columns, c.FORTYPCD for condition-level columns
```

For GRM queries, `SPCD` and `DIA_MIDPT` come from `TREE_GRM_MIDPT (gm)`; all other group columns come from `cond_plot (cond_agg)`.

**2. Aggregation** — the `_gb_select()` / `_gb_group()` helpers inject the group columns into every `GROUP BY` clause throughout the CTE chain.

**Variance with groups** — the zero-fill pattern in `all_plots_filled` uses a CROSS JOIN against `group_vals` (all distinct group combinations observed in `cond_level`). This ensures that every group has a complete set of plots — including zero-contribution plots — for unbiased variance.

---

## 10. Domain Filters

### Tree Domain

`tree_domain` is a SQL-like filter on TREE columns, e.g. `"DIA >= 10.0"`.

For standard tree estimators it is added to the `WHERE` clause of the tree CTE:
```sql
AND (DIA >= 10.0)
```

For GRM estimators, references to `DIA` are rewritten to `gc.DIA_MIDPT`:
```python
extra_where += f"\n    AND ({tree_domain_sql.replace('DIA', 'gc.DIA_MIDPT')})"
```

### Area Domain

`area_domain` is a SQL-like filter on COND columns, e.g. `"FORTYPCD IN (161, 162)"`.

- **Tree estimators:** Added to `WHERE` in the tree-cond CTE, filtering rows before aggregation.
- **Area estimator:** Applied via the `DOMAIN_IND` expression so non-matching conditions get `DOMAIN_IND = 0` rather than being excluded from the variance pool.

### Plot Domain

`plot_domain` is a SQL-like filter on PLOT columns (available in `area_sql`, `volume_sql`, `tpa_sql`, `biomass_sql`, `site_index_sql`).

When provided, a `JOIN PLOT p ON c.PLT_CN = p.CN` is added to the main data CTE and the filter is applied in the `WHERE` clause.

---

## 11. EVALID Handling

The EVALID system uses a 6-digit code: `SSYYTT` where `SS` = state FIPS, `YY` = last 2 digits of inventory year, `TT` = evaluation type code. States with single-digit FIPS (e.g. AL=1) produce 5-digit EVALIDs.

Evaluation type codes relevant to SQL builders:

| Function | `eval_type` to use |
|---|---|
| `area_sql` | `"EXPALL"` or `"EXPCURR"` |
| `volume_sql`, `tpa_sql`, `biomass_sql` | `"EXPVOL"` |
| `mortality_sql`, `growth_sql`, `removals_sql` | `"EXPMORT"` or `"EXPGROW"` |
| `carbon_sql`, `carbon_pool_sql` | `"EXPVOL"` |
| `carbon_flux_sql` | `"EXPMORT"` or `"EXPGROW"` |
| `area_change_sql` | `"EXPCURR"` |
| `site_index_sql` | `"EXPALL"` |
| `panel_sql` | matches the underlying data needed (e.g. `"EXPMORT"` for tree panel) |

The `strat` CTE uses `WHERE ps.EVALID IN (<evalid_list>)`. Multiple EVALIDs can be active simultaneously (e.g. combining multiple states). The `_evalid_list()` function formats all active EVALIDs from `db.evalid` into the IN-list.

---

## 12. Adjustment Factors

The three adjustment factors stored in `POP_STRATUM` correct for the FIA nested plot design:

| Factor | Design element | Used when |
|---|---|---|
| `ADJ_FACTOR_MICR` | Microplot (6.8 ft radius) | `DIA < 5.0 inches` |
| `ADJ_FACTOR_SUBP` | Subplot (24 ft radius) | `5.0 ≤ DIA < MACRO_BREAKPOINT_DIA` |
| `ADJ_FACTOR_MACR` | Macroplot (58.9 ft radius) | `DIA ≥ MACRO_BREAKPOINT_DIA` |

`MACRO_BREAKPOINT_DIA` is a plot-level attribute (from the `PLOT` table). It varies by state and inventory design. If null, the subplot factor is always used.

**Area adjustment** uses `PROP_BASIS` from the `COND` table:
- `PROP_BASIS = 'MACR'` → macroplot area factor
- otherwise → subplot area factor

**GRM adjustment** uses `SUBPTYP_GRM` from `TREE_GRM_COMPONENT`:
- `0` → 0.0 (not sampled)
- `1` → subplot
- `2` → microplot
- `3` → macroplot

---

## 13. Usage Examples

### Basic query execution

```python
import pyfia

with pyfia.FIA("data/fiadb.duckdb") as db:
    db.clip_by_state(37)          # NC = FIPS 37
    db.clip_most_recent(eval_type="VOL")

    sql = pyfia.volume_sql(db, land_type="forest", tree_type="gs")
    print(sql)                     # inspect the SQL

    result = db.execute_sql(sql)   # execute against the same database
```

### Grouped by species

```python
sql = pyfia.volume_sql(db, by_species=True, vol_type="net")
# Produces one row per SPCD, ordered by SPCD
```

### With domain filters

```python
sql = pyfia.tpa_sql(
    db,
    grp_by="FORTYPCD",
    tree_domain="DIA >= 10.0",
    area_domain="STDAGE > 50",
    land_type="timber",
)
```

### Mortality by species

```python
sql = pyfia.mortality_sql(
    db,
    by_species=True,
    tree_type="gs",
    land_type="timber",
    measure="volume",
)
```

### Area change (gross gain)

```python
sql = pyfia.area_change_sql(db, change_type="gross_gain")
# Returns annualized acres of non-forest → forest transitions
```

### Site index by forest type

```python
sql = pyfia.site_index_sql(db, grp_by="FORTYPCD", land_type="timber")
# Always includes SIBASE as the first group column
```

### Sample-level tree metrics

```python
sql = pyfia.tree_metrics_sql(
    db,
    metrics=["qmd", "mean_dia", "sawtimber_prop"],
    grp_by="FORTYPCD",
    sawtimber_threshold=9.0,
)
# No EXPNS, no SE — sample statistics only
```

### Carbon pools (aboveground + belowground)

```python
db.clip_most_recent(eval_type="VOL")
sql = pyfia.carbon_pool_sql(db, pool="ag", by_species=True, land_type="forest")
# CARBON_AG_TOTAL (short tons) by species
```

### Carbon dispatcher (live vs dead)

```python
sql = pyfia.carbon_sql(db, pool="live")   # routes to carbon_pool_sql(pool="total")
sql = pyfia.carbon_sql(db, pool="dead")   # routes to biomass_sql(tree_type="dead") × 0.47
```

### Net carbon flux

```python
db.clip_most_recent(eval_type="MORT")
sql = pyfia.carbon_flux_sql(db, grp_by="STATECD", tree_type="gs")
# NET_CARBON_FLUX_TOTAL, GROWTH_CARBON_TOTAL, MORT_CARBON_TOTAL, REMV_CARBON_TOTAL
# Note: no SE columns — run growth/mortality/removals_sql separately for SEs
```

### Condition-level remeasurement panel

```python
db.clip_most_recent(eval_type="CURR")
sql = pyfia.panel_sql(db, level="condition", harvest_only=True)
# Returns one row per remeasured condition where a harvest treatment was recorded
```

### Tree-level remeasurement panel

```python
sql = pyfia.panel_sql(
    db,
    level="tree",
    tree_type="live",
    min_remper=4.0,
    max_remper=6.0,
)
# DIA_BEGIN (t1), DIA_MIDPT, COMPONENT (fate), volume/biomass at midpoint
```

### Using raw SQL directly in DuckDB

The returned SQL strings are self-contained. Given access to the same FIADB tables, they can be run in any DuckDB session:

```python
import duckdb

conn = duckdb.connect("data/fiadb.duckdb")

# Build the SQL (db instance needed only for EVALID extraction)
with pyfia.FIA("data/fiadb.duckdb") as db:
    db.clip_by_state(37)
    db.clip_most_recent(eval_type="VOL")
    sql = pyfia.volume_sql(db)

result = conn.execute(sql).df()
```

---

## 14. Limitations and Design Choices

### growth_sql uses SURVIVOR components, not BEGINEND

The Python `GrowthEstimator` uses a BEGINEND cross-join methodology:
- `ONEORTWO = 2`: ending values (positive)
- `ONEORTWO = 1`: beginning values (negative)
- Net growth = sum across both rows

The SQL `growth_sql` uses the simpler SURVIVOR-component filter (`COMPONENT LIKE 'SURVIVOR%'`). This is the standard EVALIDator approach for survivor-based growth and matches the conceptual interpretation of `TREE_GRM_COMPONENT`. The BEGINEND approach — which requires a dynamic CROSS JOIN against a 2-row reference table — would significantly complicate the SQL and is not necessary for typical growth queries.

### GRM has no CONDID

`TREE_GRM_COMPONENT` and `TREE_GRM_MIDPT` do not carry a `CONDID` column. The GRM SQL builders aggregate `COND` to plot level (`cond_plot` CTE) using `ANY_VALUE()` for single-value attributes and `SUM(CONDPROP_UNADJ)` for area. This is the same approach used in the Python GRM aggregation (`aggregate_cond_to_plot()` in `grm.py`).

As a consequence, **area domain filters on multi-condition plots** are applied approximately: the filter is checked against the dominant condition's attributes. This matches the Python estimator behaviour.

### area_change requires remeasured plots

`area_change_sql` filters out plots where `REMPER IS NULL OR REMPER = 0`. These are first-inventory plots with no previous measurement. Area change is undefined for them.

### No plot_domain for GRM estimators

The `mortality_sql`, `growth_sql`, and `removals_sql` wrappers do not accept `plot_domain`. The GRM table join structure does not include a direct PLOT join in the main data CTE. If plot-level filtering is needed (e.g. by county), add it as an `area_domain` filter using a COND column that uniquely identifies the county, or use a subquery in `tree_domain`.

### No by_size_class support

The Python estimators support `by_size_class=True` with `size_class_type` options (standard, descriptive, market). The SQL builders do not implement this; it can be approximated by passing `grp_by` with a computed bucket expression or by post-processing the per-species or per-diameter results.

### tree_metrics_sql validates metrics at build time

If an unrecognised metric is passed, `ValueError` is raised immediately (before any SQL is produced). This is consistent with fail-fast design — no partial SQL should be silently returned.

### carbon_flux_sql produces no standard errors

The Python `carbon_flux()` function uses a simplified sum-of-squares approach across the three GRM component totals. Propagating covariances between growth, mortality, and removals within a single SQL query — following B&P (2005) rigorously — would require correlating three separate variance chains, which is not practical to express as a self-contained CTE chain. Run `growth_sql`, `mortality_sql`, and `removals_sql` individually with `measure="biomass"` to obtain per-component SEs, then combine in Python.

### carbon_sql raises ValueError for litter and soil pools

`pool="litter"` and `pool="soil"` are not available from FIA tree inventory tables (they require separate ecosystem inventory protocols). These raise `ValueError` at query-build time, matching the behaviour of the Python `carbon()` function.

### panel_sql is a data retrieval query, not a population estimator

`panel_sql` returns individual condition or tree rows with t1/t2 attributes. It does not apply expansion factors, does not compute population totals, and produces no standard errors or confidence intervals. For population-level change estimates use `area_change_sql` (conditions) or `growth_sql`/`mortality_sql`/`removals_sql` (trees).

The condition panel uses `PREV_PLT_CN` to link plots across inventories. Plots with no previous measurement (`PREV_PLT_CN IS NULL`) are excluded by the `WHERE` clause.

### DuckDB-specific SQL features

The generated SQL uses:
- `GREATEST(x, 0.0)` — for non-negative variance guard
- `VAR_SAMP(x)` and `COVAR_SAMP(x, y)` — for unbiased sample statistics
- `ANY_VALUE(x)` — for non-deterministic single-value aggregation (GRM cond_plot)

These are all supported by DuckDB (the primary pyFIA backend). For SQLite compatibility, `GREATEST` would need to be replaced with `MAX(x, 0.0)` and `VAR_SAMP`/`COVAR_SAMP` would require manual implementation.
