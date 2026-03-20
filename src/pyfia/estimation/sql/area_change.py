"""
SQL query builder for area change estimation.

Uses SUBP_COND_CHNG_MTRX to track subplot-level condition transitions
between measurement periods. Annualizes by dividing by REMPER.

Formula (net change):
  CHANGE_ACRE = Σ((gain - loss) × SUBPTYP_PROP_CHNG × ADJ_FACTOR_AREA × EXPNS)
              / (REMPER × AREA_TOTAL)

where:
  gain = 1 if PREV_COND_STATUS_CD != 1 AND CURR_COND_STATUS_CD == 1
  loss = 1 if PREV_COND_STATUS_CD == 1 AND CURR_COND_STATUS_CD != 1
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
    _se_total_expr,
    _strat_cte,
    _variance_ctes,
)

if TYPE_CHECKING:
    from ...core.fia import FIA


def area_change_sql(
    db: "FIA",
    grp_by: str | list[str] | None = None,
    land_type: str = "forest",
    change_type: str = "net",
    area_domain: str | None = None,
) -> str:
    """Return a SQL query string that replicates ``area_change()`` estimation.

    Calculates net or gross change in forest or timberland area using
    remeasured plots from the ``SUBP_COND_CHNG_MTRX`` table. Only plots
    measured at two time points (current and previous) contribute to the
    estimate. The result is annualized by dividing by REMPER.

    Parameters
    ----------
    db : FIA
        Connected FIA database with EVALID set (used to extract current
        EVALID(s) for the ``strat`` CTE).
    grp_by : str or list of str, optional
        Column name(s) from the COND table to group results by. All group
        columns are read from ``curr_cond`` (the t2 condition). Common
        options include:

        - 'OWNGRPCD': Ownership group (10=NF, 20=Other Federal, 30=State, 40=Private)
        - 'FORTYPCD': Current forest type code
        - 'STATECD': State FIPS code
        - 'COUNTYCD': County code
    land_type : {'forest', 'timber'}, default 'forest'
        Land classification to track changes for:

        - 'forest': All forest land (COND_STATUS_CD = 1)
        - 'timber': Timberland only (productive, unreserved forest with
          SITECLCD in [1-6] and RESERVCD = 0)
    change_type : {'net', 'gross_gain', 'gross_loss'}, default 'net'
        Type of change to calculate:

        - 'net': Net change (gains minus losses, positive = forest gain)
        - 'gross_gain': Only area gained (non-forest → forest transitions)
        - 'gross_loss': Only area lost (forest → non-forest transitions)
    area_domain : str, optional
        SQL-like filter expression on COND columns (applied to current
        condition, ``curr_cond``). Examples:

        - ``"OWNGRPCD == 40"``: Private lands only
        - ``"FORTYPCD IN (161, 162)"``: Specific forest types

    Returns
    -------
    str
        Complete, self-contained SQL query string. Output columns include:

        - **CHANGE_TOTAL** : float — Annual area change (acres/year)
        - **CHANGE_TOTAL_SE** : float — Standard error of area change estimate
        - **AREA_TOTAL** : float — Reference forest area (acres)
        - **N_PLOTS** : int — Number of remeasured plots with non-zero change
        - **[grouping columns]** : varies — Columns from grp_by

    Notes
    -----
    Area change estimation requires remeasured plots (plots with both current
    and previous measurements). States with newer FIA programs may have fewer
    remeasured plots, resulting in higher sampling errors.

    The ``REMPER`` (remeasurement period) varies by plot but averages
    approximately 5–7 years in most states. The SQL divides by
    ``MAX(REMPER)`` at the condition level to annualize change values.

    Unlike tree estimators, area change uses the ``SUBP_COND_CHNG_MTRX``
    table with subplot-level condition change indicators (``SUBPTYP_PROP_CHNG``),
    not the TREE table. The strat CTE filters plots to the active EVALID.

    See Also
    --------
    area_sql : Static forest area estimation (single time point)
    panel_sql : Raw t1/t2 remeasurement panel dataset

    Examples
    --------
    Net annual forest area change:

    >>> with pyfia.FIA("path/to/fia.duckdb") as db:
    ...     db.clip_by_state(37)
    ...     db.clip_most_recent(eval_type="CHNG")
    ...     sql = pyfia.area_change_sql(db, land_type="forest")

    Gross forest loss by ownership group:

    >>> sql = pyfia.area_change_sql(
    ...     db, change_type="gross_loss", grp_by="OWNGRPCD"
    ... )

    Net timberland area change:

    >>> sql = pyfia.area_change_sql(db, land_type="timber", change_type="net")
    """
    evalid_str = _evalid_list(db)
    group_cols: list[str] = (
        [grp_by] if isinstance(grp_by, str) else list(grp_by) if grp_by else []
    )

    gb = _gb_select(group_cols)
    ggroup = _gb_group(group_cols)

    area_domain_sql = _domain_to_sql(area_domain)
    extra_where = f"\n    AND ({area_domain_sql})" if area_domain_sql else ""

    # Land-type filter on current condition status
    if land_type == "timber":
        curr_forest_cond = (
            "curr_cond.COND_STATUS_CD = 1 "
            "AND curr_cond.RESERVCD = 0 AND curr_cond.SITECLCD <= 6"
        )
        prev_forest_cond = "prev_cond.COND_STATUS_CD = 1"
        extra_cond_cols = "curr_cond.RESERVCD, curr_cond.SITECLCD,"
    else:
        curr_forest_cond = "curr_cond.COND_STATUS_CD = 1"
        prev_forest_cond = "prev_cond.COND_STATUS_CD = 1"
        extra_cond_cols = ""

    # Change value expression
    if change_type == "gross_gain":
        change_expr = (
            f"CASE WHEN NOT ({prev_forest_cond}) AND ({curr_forest_cond}) "
            "THEN COALESCE(sc.SUBPTYP_PROP_CHNG, 1.0) ELSE 0.0 END"
        )
    elif change_type == "gross_loss":
        change_expr = (
            f"CASE WHEN ({prev_forest_cond}) AND NOT ({curr_forest_cond}) "
            "THEN COALESCE(sc.SUBPTYP_PROP_CHNG, 1.0) ELSE 0.0 END"
        )
    else:  # net
        change_expr = (
            f"(CASE WHEN NOT ({prev_forest_cond}) AND ({curr_forest_cond}) "
            "THEN COALESCE(sc.SUBPTYP_PROP_CHNG, 1.0) ELSE 0.0 END "
            f"- CASE WHEN ({prev_forest_cond}) AND NOT ({curr_forest_cond}) "
            "THEN COALESCE(sc.SUBPTYP_PROP_CHNG, 1.0) ELSE 0.0 END)"
        )

    grp_cols_select = ""
    if group_cols:
        grp_cols_select = ", ".join(f"curr_cond.{c}" for c in group_cols) + ","

    query = f"""-- FIA Area Change Estimation SQL
-- Matches AreaChangeEstimator (pyfia) exactly.
-- EVALIDs: {evalid_str}
-- land_type: {land_type}, change_type: {change_type}
--
-- Formula: CHANGE_TOTAL = Σ(change_value × ADJ_FACTOR_AREA × EXPNS) / REMPER
-- change_value = (gain_ind - loss_ind) × SUBPTYP_PROP_CHNG per subplot-condition
-- Variance: Bechtold & Patterson (2005) post-stratified variance
WITH {_strat_cte(evalid_str)},

-- Subplot-condition change data joined with current/previous conditions
chng_data AS (
    SELECT
        sc.PLT_CN,
        sc.CONDID,
        sc.SUBP,
        {grp_cols_select}
        curr_cond.CONDPROP_UNADJ,
        curr_cond.PROP_BASIS,
        {extra_cond_cols}
        {change_expr} AS change_value,
        p.REMPER,
        {_area_adj_sql("curr_cond", "s")} AS ADJ_FACTOR_AREA,
        s.STRATUM_CN,
        s.EXPNS,
        s.STRATUM_WGT,
        s.P2POINTCNT,
        s.ESTN_UNIT_CN,
        s.AREA_USED,
        s.P1PNTCNT_EU
    FROM SUBP_COND_CHNG_MTRX sc
    JOIN COND curr_cond
        ON sc.PLT_CN = curr_cond.PLT_CN AND sc.CONDID = curr_cond.CONDID
    LEFT JOIN COND prev_cond
        ON sc.PREV_PLT_CN = prev_cond.PLT_CN AND sc.PREVCOND = prev_cond.CONDID
    JOIN PLOT p ON sc.PLT_CN = p.CN
    JOIN strat s ON sc.PLT_CN = s.PLT_CN
    WHERE curr_cond.CONDPROP_UNADJ IS NOT NULL
      AND prev_cond.COND_STATUS_CD IS NOT NULL
      AND p.REMPER IS NOT NULL
      AND p.REMPER > 0{extra_where}
),

-- Condition-level aggregation
-- y_ic = Σ(change_value × ADJ_FACTOR_AREA) / REMPER  (annualized change)
-- x_ic = CONDPROP_UNADJ × ADJ_FACTOR_AREA             (area denominator)
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
        {gb}SUM(change_value * ADJ_FACTOR_AREA) / MAX(REMPER) AS y_ic,
        MAX(CONDPROP_UNADJ * ADJ_FACTOR_AREA)                  AS x_ic
    FROM chng_data
    GROUP BY
        PLT_CN, CONDID, STRATUM_CN, EXPNS, STRATUM_WGT, P2POINTCNT,
        ESTN_UNIT_CN, AREA_USED, P1PNTCNT_EU{ggroup}
),

-- Population totals
pop_totals AS (
    SELECT
        {gb}SUM(y_ic * EXPNS) AS CHANGE_TOTAL,
        SUM(x_ic * EXPNS)  AS AREA_TOTAL,
        COUNT(DISTINCT CASE WHEN y_ic <> 0 THEN PLT_CN END) AS N_PLOTS
    FROM cond_level
    GROUP BY {('1' if not group_cols else ', '.join(group_cols))}
),

{_variance_ctes(group_cols)}

SELECT
    {gb}pt.CHANGE_TOTAL,
    pt.AREA_TOTAL,
    pt.N_PLOTS,
    {_se_total_expr()} AS CHANGE_TOTAL_SE
FROM pop_totals pt
{_final_join(group_cols)}
ORDER BY {('pt.CHANGE_TOTAL DESC' if not group_cols else ', '.join(group_cols))}
"""
    return query
