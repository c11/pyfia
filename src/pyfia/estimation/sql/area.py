"""
SQL query builder for forest area estimation.

Produces a query that matches AreaEstimator exactly, implementing:
  AREA_TOTAL = Σ(CONDPROP_UNADJ × ADJ_FACTOR_AREA × EXPNS × DOMAIN_IND)
with domain indicator approach and B&P (2005) variance.
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
    _se_total_expr,
    _strat_cte,
    _variance_ctes,
)

if TYPE_CHECKING:
    from ...core.fia import FIA


def area_sql(
    db: "FIA",
    grp_by: str | list[str] | None = None,
    land_type: str = "forest",
    area_domain: str | None = None,
    plot_domain: str | None = None,
) -> str:
    """Return a SQL query string that replicates ``area()`` estimation.

    Calculates area estimates using FIA's design-based estimation methods
    with proper expansion factors and stratification, implementing the
    domain-indicator approach so that all plots contribute to variance
    even when they are outside the domain.

    The returned string is directly executable against the FIADB tables
    referenced by *db*. It does not fetch data itself — pass it to
    ``db.execute_sql()`` or run it in any DuckDB session with access to
    the same tables.

    Parameters
    ----------
    db : FIA
        Connected FIA database with EVALID set (used only to extract the
        current EVALID(s) for the ``strat`` CTE).
    grp_by : str or list of str, optional
        Column name(s) from the COND or PLOT tables to group results by.
        Common options:

        **Ownership and Management:**
        - 'OWNGRPCD': Ownership group (10=National Forest, 20=Other Federal,
          30=State/Local, 40=Private)
        - 'RESERVCD': Reserved status (0=Not reserved, 1=Reserved)
        - 'ADFORCD': Administrative forest code

        **Forest Characteristics:**
        - 'FORTYPCD': Forest type code (see REF_FOREST_TYPE)
        - 'STDSZCD': Stand size class (1=Large, 2=Medium, 3=Small,
          4=Seedling/sapling, 5=Nonstocked)
        - 'STDORGCD': Stand origin (0=Natural, 1=Planted)
        - 'STDAGE': Stand age in years

        **Site Characteristics:**
        - 'SITECLCD': Site productivity class (1=225+ cu ft/ac/yr, ..., 7=0-19)
        - 'PHYSCLCD': Physiographic class code

        **Location:**
        - 'STATECD': State FIPS code
        - 'UNITCD': FIA survey unit code
        - 'COUNTYCD': County code
        - 'INVYR': Inventory year

        **Disturbance and Treatment:**
        - 'DSTRBCD1', 'DSTRBCD2', 'DSTRBCD3': Disturbance codes
        - 'TRTCD1', 'TRTCD2', 'TRTCD3': Treatment codes
    land_type : {'forest', 'timber', 'all'}, default 'forest'
        Land type to include in estimation:

        - 'forest': All forestland (COND_STATUS_CD = 1)
        - 'timber': Timberland only (unreserved, productive forestland with
          SITECLCD in [1-6] and RESERVCD = 0)
        - 'all': All land types including non-forest
    area_domain : str, optional
        SQL-like filter expression for COND-level attributes. Examples:

        - ``"STDAGE > 50"``: Stands older than 50 years
        - ``"FORTYPCD IN (161, 162)"``: Specific forest types
        - ``"OWNGRPCD == 10"``: National Forest lands only
        - ``"PHYSCLCD == 31 AND STDSZCD == 1"``: Xeric sites with large trees
    plot_domain : str, optional
        SQL-like filter expression for PLOT-level attributes. Examples:

        - ``"COUNTYCD == 183"``: Wake County, NC
        - ``"COUNTYCD IN (183, 185, 187)"``: Multiple counties
        - ``"LAT >= 35.0 AND LAT <= 36.0"``: Latitude range
        - ``"ELEV > 2000"``: Elevation above 2000 feet

    Returns
    -------
    str
        Complete, self-contained SQL query string. Output columns include:

        - **AREA_TOTAL** : float — Total forest area in acres
        - **AREA_PERCENT** : float — Area as percentage of total expansion area
        - **AREA_SE** : float — Standard error of AREA_TOTAL
        - **AREA_VARIANCE** : float — Sampling variance of the area estimate
        - **N_PLOTS** : int — Number of FIA plots in the estimate
        - **[grouping columns]** : varies — Columns from grp_by

    Notes
    -----
    The query implements the domain-indicator approach: all plots are retained
    and ``DOMAIN_IND = CASE WHEN <land+area filter> THEN 1.0 ELSE 0.0 END``
    is applied before the condition-level sum. This ensures proper variance
    estimation for rare domain subsets where many plots have zero contribution.

    The EVALID filter is applied through the ``strat`` CTE join on
    ``POP_PLOT_STRATUM_ASSGN`` × ``POP_STRATUM``. Use
    ``db.clip_by_evalid()`` or ``db.clip_by_state()`` before calling
    this function.

    Examples
    --------
    Basic forest area:

    >>> with pyfia.FIA("path/to/fia.duckdb") as db:
    ...     db.clip_by_state(37)
    ...     db.clip_most_recent(eval_type="ALL")
    ...     sql = pyfia.area_sql(db, land_type="forest")
    ...     result = db.execute_sql(sql)

    Area by ownership group on timberland:

    >>> sql = pyfia.area_sql(db, grp_by="OWNGRPCD", land_type="timber")

    Old stands by forest type:

    >>> sql = pyfia.area_sql(
    ...     db,
    ...     grp_by="FORTYPCD",
    ...     land_type="forest",
    ...     area_domain="STDAGE > 50",
    ... )

    Single-county filter:

    >>> sql = pyfia.area_sql(db, plot_domain="COUNTYCD == 183")
    """
    evalid_str = _evalid_list(db)
    group_cols: list[str] = (
        [grp_by] if isinstance(grp_by, str) else list(grp_by) if grp_by else []
    )

    gb = _gb_select(group_cols)
    ggroup = _gb_group(group_cols)

    land_filter = _land_type_sql(land_type, c="c")
    area_domain_sql = _domain_to_sql(area_domain)
    plot_domain_sql = _domain_to_sql(plot_domain)

    # Build the domain indicator (DOMAIN_IND) logic:
    # - land_type restriction AND optional area_domain
    if area_domain_sql:
        domain_cond = f"({land_filter}) AND ({area_domain_sql})"
    else:
        domain_cond = land_filter

    # Optional PLOT join for plot_domain or MACRO_BREAKPOINT_DIA (area doesn't need it
    # unless plot_domain is specified)
    if plot_domain_sql:
        plot_join = "JOIN PLOT p ON c.PLT_CN = p.CN"
        plot_where = f"AND ({plot_domain_sql.replace('PLOT.', 'p.').replace('p.CN', 'p.CN')})"
        # Replace bare column refs with p. prefix for unambiguous SQL
        plot_where_clean = (
            plot_domain_sql
            .replace("==", "=")
        )
        plot_join_clause = f"\n    JOIN PLOT p ON c.PLT_CN = p.CN"
        plot_filter_clause = f"    AND ({plot_where_clean})"
    else:
        plot_join_clause = ""
        plot_filter_clause = ""

    # Extra COND columns needed for timber land type
    extra_cond_cols = ""
    if land_type == "timber":
        extra_cond_cols = "c.RESERVCD, c.SITECLCD,"

    # grp_by columns from COND
    grp_cond_select = ""
    if group_cols:
        grp_cond_select = ", ".join(f"c.{col}" for col in group_cols) + ","

    query = f"""-- FIA Area Estimation SQL
-- Matches AreaEstimator (pyfia) exactly.
-- EVALIDs: {evalid_str}
-- land_type: {land_type}, grp_by: {grp_by}
--
-- Formula: AREA_TOTAL = Σ(CONDPROP_UNADJ × ADJ_FACTOR_AREA × EXPNS × DOMAIN_IND)
-- Variance: Bechtold & Patterson (2005) post-stratified domain total variance
WITH {_strat_cte(evalid_str)},

cond_domain AS (
    -- Condition data with domain indicator and area adjustment factor
    SELECT
        c.PLT_CN,
        c.CONDID,
        c.CONDPROP_UNADJ,
        c.PROP_BASIS,
        {grp_cond_select}
        {extra_cond_cols}
        {_area_adj_sql("c", "s")} AS ADJ_FACTOR_AREA,
        -- Domain indicator: 1 if this condition belongs to the target domain
        CASE WHEN {domain_cond} THEN 1.0 ELSE 0.0 END AS DOMAIN_IND,
        s.STRATUM_CN,
        s.EXPNS,
        s.STRATUM_WGT,
        s.P2POINTCNT,
        s.ESTN_UNIT_CN,
        s.AREA_USED,
        s.P1PNTCNT_EU
    FROM COND c
    JOIN strat s ON c.PLT_CN = s.PLT_CN{plot_join_clause}
    WHERE c.CONDPROP_UNADJ IS NOT NULL
{plot_filter_clause}),

-- Stage 1: condition-level area values
-- y_ic = CONDPROP_UNADJ × ADJ_FACTOR_AREA × DOMAIN_IND  (numerator per condition)
-- x_ic = CONDPROP_UNADJ × ADJ_FACTOR_AREA               (denominator: total forest area)
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
        {gb}CONDPROP_UNADJ * ADJ_FACTOR_AREA * DOMAIN_IND AS y_ic,
        CONDPROP_UNADJ * ADJ_FACTOR_AREA                  AS x_ic
    FROM cond_domain
),

-- Stage 2: population totals
pop_totals AS (
    SELECT
        {gb}SUM(y_ic * EXPNS) AS AREA_TOTAL,
        SUM(x_ic * EXPNS)  AS TOTAL_EXPNS,
        COUNT(DISTINCT CASE WHEN y_ic > 0 THEN PLT_CN END) AS N_PLOTS
    FROM cond_level
    GROUP BY {('1' if not group_cols else ', '.join(group_cols))}
),

{_variance_ctes(group_cols)}

-- Final results
SELECT
    {gb}pt.AREA_TOTAL,
    100.0 * pt.AREA_TOTAL / NULLIF(pt.TOTAL_EXPNS, 0) AS AREA_PERCENT,
    {_se_total_expr()}                                  AS AREA_SE,
    fv.var_total                                        AS AREA_VARIANCE,
    pt.N_PLOTS
FROM pop_totals pt
{_final_join(group_cols)}
ORDER BY {('pt.AREA_TOTAL DESC' if not group_cols else ', '.join(group_cols))}
"""
    return query
