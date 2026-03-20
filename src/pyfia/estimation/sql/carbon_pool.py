"""
SQL query builder for carbon pool estimation.

Uses FIA's pre-calculated CARBON_AG and CARBON_BG columns from the TREE table,
which incorporate species-specific conversion factors. Matches EVALIDator
snum=55000 exactly for live tree carbon pools.

Formula:
  CARBON_ACRE = Σ(CARBON_COL × TPA_UNADJ × ADJ_FACTOR × LBS_TO_TONS × EXPNS)
              / Σ(CONDPROP_UNADJ × ADJ_FACTOR_AREA × EXPNS)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    _area_adj_sql,
    _domain_to_sql,
    _evalid_list,
    _final_join,
    _gb_group,
    _gb_select,
    _land_type_sql,
    _se_acre_expr,
    _se_total_expr,
    _strat_cte,
    _tree_adj_sql,
    _tree_type_sql,
    _variance_ctes,
)

if TYPE_CHECKING:
    from ...core.fia import FIA

_LBS_TO_TONS = "0.0005"   # 1/2000


def carbon_pool_sql(
    db: "FIA",
    grp_by: str | list[str] | None = None,
    by_species: bool = False,
    pool: str = "total",
    land_type: str = "forest",
    tree_type: str = "live",
    tree_domain: str | None = None,
    area_domain: str | None = None,
    plot_domain: str | None = None,
) -> str:
    """Return a SQL query string that replicates ``carbon_pool()`` estimation.

    Estimates carbon stocks using FIA's pre-calculated ``CARBON_AG`` and
    ``CARBON_BG`` columns from the TREE table, which incorporate
    species-specific carbon conversion factors. This matches EVALIDator
    snum=55000 exactly for ``pool='total'``.

    Parameters
    ----------
    db : FIA
        Connected FIA database with EVALID set (used to extract current
        EVALID(s) for the ``strat`` CTE).
    grp_by : str or list of str, optional
        Column name(s) from the TREE, COND, or PLOT tables to group results by.
        Common options include 'FORTYPCD', 'OWNGRPCD', 'STATECD', 'SPCD',
        'COUNTYCD'. For complete column descriptions, see the USDA FIA
        Database User Guide.
    by_species : bool, default False
        If True, group results by species code (SPCD). Equivalent to adding
        'SPCD' to grp_by.
    pool : {'ag', 'bg', 'total'}, default 'total'
        Carbon pool to estimate:

        - 'ag': Aboveground carbon only (``CARBON_AG`` — stems, branches,
          foliage, bark)
        - 'bg': Belowground carbon only (``CARBON_BG`` — coarse roots)
        - 'total': Total carbon (``CARBON_AG + CARBON_BG``) — matches
          EVALIDator snum=55000
    land_type : {'forest', 'timber', 'all'}, default 'forest'
        Land type to include in estimation:

        - 'forest': All forestland (COND_STATUS_CD = 1)
        - 'timber': Timberland only (unreserved, productive forestland with
          SITECLCD in [1-6] and RESERVCD = 0)
        - 'all': All land types including non-forest
    tree_type : {'live', 'dead', 'gs', 'all'}, default 'live'
        Tree type to include:

        - 'live': All live trees (STATUSCD = 1)
        - 'dead': Standing dead trees (STATUSCD = 2)
        - 'gs': Growing stock trees (live, TREECLCD = 2)
        - 'all': All trees regardless of status
    tree_domain : str, optional
        SQL-like filter expression for tree-level attributes. Examples:

        - ``"DIA >= 10.0"``: Trees 10 inches DBH and larger
        - ``"SPCD == 131"``: Loblolly pine only
    area_domain : str, optional
        SQL-like filter expression for COND-level attributes. Examples:

        - ``"OWNGRPCD == 40 AND FORTYPCD == 161"``: Private loblolly stands
        - ``"STDAGE > 50"``: Stands older than 50 years
    plot_domain : str, optional
        SQL-like filter expression for PLOT-level attributes. Examples:

        - ``"COUNTYCD == 183"``: Wake County, NC
        - ``"LAT >= 35.0 AND LAT <= 36.0"``: Latitude range

    Returns
    -------
    str
        Complete, self-contained SQL query string. Output columns include:

        - **CARBON_ACRE** : float — Carbon per acre (short tons/acre)
        - **CARBON_TOTAL** : float — Total carbon (population level, short tons)
        - **CARBON_ACRE_SE** : float — Standard error of per-acre estimate
        - **CARBON_TOTAL_SE** : float — Standard error of total estimate
        - **AREA_TOTAL** : float — Total forest area (acres) in the estimation
        - **N_PLOTS** : int — Number of FIA plots in the estimate
        - **[grouping columns]** : varies — Columns from grp_by / by_species

    Notes
    -----
    This estimator uses FIA's pre-calculated carbon columns (CARBON_AG,
    CARBON_BG) which incorporate species-specific carbon conversion factors.
    This is more accurate than applying a flat 47% carbon fraction to biomass,
    and matches EVALIDator snum=55000 for total live tree carbon.

    CARBON columns in the FIA database are stored in pounds. The query
    converts to short tons by multiplying by 0.0005 (= 1/2000).

    For biomass-derived carbon (``DRYBIO × 0.47``), use ``biomass_sql``
    instead. For a unified dispatcher that handles both live and dead tree
    carbon, use ``carbon_sql``.

    See Also
    --------
    carbon_sql : Unified carbon dispatcher (routes to carbon_pool_sql or biomass_sql)
    biomass_sql : Biomass estimation with flat 47% carbon conversion

    Examples
    --------
    Total live tree carbon (matches EVALIDator snum=55000):

    >>> with pyfia.FIA("path/to/fia.duckdb") as db:
    ...     db.clip_by_state(37)
    ...     db.clip_most_recent(eval_type="VOL")
    ...     sql = pyfia.carbon_pool_sql(db, pool="total")

    Aboveground carbon by species:

    >>> sql = pyfia.carbon_pool_sql(db, pool="ag", by_species=True)

    Carbon on private timberland by forest type:

    >>> sql = pyfia.carbon_pool_sql(
    ...     db, pool="total", land_type="timber",
    ...     area_domain="OWNGRPCD == 40", grp_by="FORTYPCD"
    ... )
    """
    evalid_str = _evalid_list(db)
    pool = pool.lower()
    group_cols: list[str] = (
        [grp_by] if isinstance(grp_by, str) else list(grp_by) if grp_by else []
    )
    if by_species and "SPCD" not in group_cols:
        group_cols.append("SPCD")

    # Carbon column expression based on pool
    if pool == "ag":
        carbon_expr = "COALESCE(t.CARBON_AG, 0.0)"
        carbon_cols = "t.CARBON_AG,"
    elif pool == "bg":
        carbon_expr = "COALESCE(t.CARBON_BG, 0.0)"
        carbon_cols = "t.CARBON_BG,"
    else:  # total
        carbon_expr = "(COALESCE(t.CARBON_AG, 0.0) + COALESCE(t.CARBON_BG, 0.0))"
        carbon_cols = "t.CARBON_AG, t.CARBON_BG,"

    gb = _gb_select(group_cols)
    ggroup = _gb_group(group_cols)

    land_filter = _land_type_sql(land_type, c="c")
    tree_filter = _tree_type_sql(tree_type, t="t")
    tree_domain_sql = _domain_to_sql(tree_domain)
    area_domain_sql = _domain_to_sql(area_domain)
    plot_domain_sql = _domain_to_sql(plot_domain)

    extra_where = ""
    if tree_domain_sql:
        extra_where += f"\n    AND ({tree_domain_sql})"
    if area_domain_sql:
        extra_where += f"\n    AND ({area_domain_sql})"
    if plot_domain_sql:
        extra_where += f"\n    AND ({plot_domain_sql})"

    extra_cond_cols = "c.RESERVCD, c.SITECLCD," if land_type == "timber" else ""
    plot_join_clause = "\n    JOIN PLOT p ON t.PLT_CN = p.CN" if plot_domain_sql else ""

    tree_level_cols = {"SPCD", "SPGRPCD", "DIA", "HT", "TREECLCD", "CCLCD",
                       "STATUSCD", "AGENTCD", "DECAYCD"}
    grp_cols_select = ""
    if group_cols:
        grp_cols_select = ", ".join(
            f"t.{c}" if c in tree_level_cols else f"c.{c}" for c in group_cols
        ) + ","

    # Include PLOT join for adjustment factor when no plot_domain (MACRO_BREAKPOINT_DIA needed)
    if not plot_domain_sql:
        plot_join_clause = "\n    JOIN PLOT p ON t.PLT_CN = p.CN"

    query = f"""-- FIA Carbon Pool Estimation SQL
-- Matches CarbonPoolEstimator (pyfia) exactly. EVALIDator snum=55000 (pool='total').
-- EVALIDs: {evalid_str}
-- pool: {pool}, land_type: {land_type}, tree_type: {tree_type}
--
-- CARBON_ACRE = Σ(CARBON_COL × TPA_UNADJ × ADJ_FACTOR × {_LBS_TO_TONS} × EXPNS)
--             / Σ(CONDPROP_UNADJ × ADJ_FACTOR_AREA × EXPNS)
-- CARBON columns are in lbs; × {_LBS_TO_TONS} converts to short tons.
WITH {_strat_cte(evalid_str)},

tree_cond AS (
    SELECT
        t.PLT_CN,
        t.CONDID,
        t.TPA_UNADJ,
        t.DIA,
        {carbon_cols}
        {grp_cols_select}
        c.CONDPROP_UNADJ,
        c.PROP_BASIS,
        {extra_cond_cols}
        {_tree_adj_sql("t", "p", "s")} AS ADJ_FACTOR,
        {_area_adj_sql("c", "s")}      AS ADJ_FACTOR_AREA,
        s.STRATUM_CN,
        s.EXPNS,
        s.STRATUM_WGT,
        s.P2POINTCNT,
        s.ESTN_UNIT_CN,
        s.AREA_USED,
        s.P1PNTCNT_EU
    FROM TREE t
    JOIN COND c  ON t.PLT_CN = c.PLT_CN AND t.CONDID = c.CONDID
    JOIN strat s ON t.PLT_CN = s.PLT_CN{plot_join_clause}
    WHERE {tree_filter}
      AND {land_filter}
      AND t.TPA_UNADJ > 0
      AND t.DIA IS NOT NULL{extra_where}
),

-- Stage 1: condition-level aggregation
-- y_ic = Σ(carbon × TPA × ADJ × LBS_TO_TONS) per condition
-- x_ic = CONDPROP_UNADJ × ADJ_FACTOR_AREA  (area denominator)
cond_level AS (
    SELECT
        PLT_CN,
        CONDID,
        STRATUM_CN,
        EXPNS,
        STRATUM_WGT,
        P2POINTCNT,
        ESTN_UNIT_CN,
        AREA_USED,
        P1PNTCNT_EU,
        {gb}SUM({carbon_expr} * TPA_UNADJ * ADJ_FACTOR * {_LBS_TO_TONS}) AS y_ic,
        MAX(CONDPROP_UNADJ * ADJ_FACTOR_AREA)                             AS x_ic
    FROM tree_cond
    GROUP BY
        PLT_CN, CONDID, STRATUM_CN, EXPNS, STRATUM_WGT, P2POINTCNT,
        ESTN_UNIT_CN, AREA_USED, P1PNTCNT_EU{ggroup}
),

-- Population totals
pop_totals AS (
    SELECT
        {gb}SUM(y_ic * EXPNS) AS CARBON_TOTAL,
        SUM(x_ic * EXPNS)    AS AREA_TOTAL,
        CASE WHEN SUM(x_ic * EXPNS) > 0
             THEN SUM(y_ic * EXPNS) / SUM(x_ic * EXPNS) ELSE 0.0 END AS CARBON_ACRE,
        COUNT(DISTINCT CASE WHEN y_ic > 0 THEN PLT_CN END)            AS N_PLOTS
    FROM cond_level
    GROUP BY {('1' if not group_cols else ', '.join(group_cols))}
),

{_variance_ctes(group_cols)}

SELECT
    {gb}pt.CARBON_TOTAL,
    pt.AREA_TOTAL,
    pt.CARBON_ACRE,
    pt.N_PLOTS,
    {_se_total_expr()} AS CARBON_TOTAL_SE,
    {_se_acre_expr()}  AS CARBON_ACRE_SE
FROM pop_totals pt
{_final_join(group_cols)}
ORDER BY {('pt.CARBON_ACRE DESC' if not group_cols else ', '.join(group_cols))}
"""
    return query
