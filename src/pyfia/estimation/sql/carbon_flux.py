"""
SQL query builder for carbon flux estimation.

Calculates net annual carbon sequestration as:
  NET_CARBON_FLUX = (GROWTH_BIOMASS - MORT_BIOMASS - REMV_BIOMASS) × 0.47

All three GRM components share the same strat and cond_plot CTEs and are
computed in a single query, avoiding three separate round-trips.

Per-acre value is derived as NET_FLUX_TOTAL / AREA_TOTAL so the denominator
is consistent across all three components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    _domain_to_sql,
    _evalid_list,
    _gb_group,
    _gb_select,
    _grm_adj_sql,
    _strat_cte,
)
from ._grm_base import (
    _COMPONENT_FILTER,
    _LAND_TYPE_MAP,
    _TREE_TYPE_MAP,
    _TPA_PREFIX_MAP,
    _grm_measure_columns,
    _measure_expr,
)

if TYPE_CHECKING:
    from ...core.fia import FIA

_CARBON_FRAC = "0.47"


def carbon_flux_sql(
    db: "FIA",
    grp_by: str | list[str] | None = None,
    by_species: bool = False,
    land_type: str = "forest",
    tree_type: str = "gs",
    tree_domain: str | None = None,
    area_domain: str | None = None,
) -> str:
    """Return a SQL query string that replicates ``carbon_flux()`` estimation.

    Calculates net annual carbon sequestration by combining growth, mortality,
    and removals biomass in a single query:

        Net Carbon Flux = Growth_carbon - Mortality_carbon - Removals_carbon

    Positive values indicate net carbon sequestration (carbon sink). Negative
    values indicate net carbon emission (carbon source). All three GRM
    components share the same ``strat`` and ``cond_plot`` CTEs.

    Parameters
    ----------
    db : FIA
        Connected FIA database with EVALID set. The EVALID filter is applied
        through the ``strat`` CTE join — GRM tables carry no EVALID column.
        Use ``db.clip_by_evalid()`` or ``db.clip_most_recent(eval_type='GROW')``
        before calling this function.
    grp_by : str or list of str, optional
        Column name(s) to group results by. Columns that come from
        ``TREE_GRM_MIDPT`` (e.g. 'SPCD') are prefixed ``gm.``; all others
        are assumed to come from the ``cond_plot`` aggregate CTE. Common
        options include 'SPCD', 'FORTYPCD', 'OWNGRPCD', 'STATECD', 'COUNTYCD'.
    by_species : bool, default False
        If True, group results by species code (SPCD). Equivalent to adding
        'SPCD' to grp_by.
    land_type : {'forest', 'timber'}, default 'forest'
        Land type, mapped to GRM column suffix:

        - 'forest': All forestland (``FOREST`` columns, COND_STATUS_CD = 1)
        - 'timber': Timberland only (``TIMBER`` columns, unreserved productive)
    tree_type : {'gs', 'al', 'live', 'sawtimber', 'sl'}, default 'gs'
        Tree type, mapped to GRM column suffix:

        - 'gs': Growing stock (``GS`` columns, DIA_MIDPT ≥ 5.0" filter applied)
        - 'al' / 'live': All live trees (``AL`` columns)
        - 'sawtimber' / 'sl': Sawtimber-sized trees (``SL`` columns)
    tree_domain : str, optional
        SQL-like filter expression on tree/GRM columns. References to ``DIA``
        are rewritten to ``gc.DIA_MIDPT`` (midpoint diameter). Examples:

        - ``"DIA >= 10.0"``: Trees ≥ 10" midpoint diameter
        - ``"SPCD == 131"``: Loblolly pine only
    area_domain : str, optional
        SQL-like filter expression on COND columns, applied to the ``cond_plot``
        aggregate CTE. Examples:

        - ``"FORTYPCD IN (161, 162)"``: Specific forest types
        - ``"OWNGRPCD == 40"``: Private lands only

    Returns
    -------
    str
        Complete, self-contained SQL query string. Output columns include:

        - **NET_CARBON_FLUX_TOTAL** : float — Net carbon flux (short tons C/year)
        - **NET_CARBON_FLUX_ACRE** : float — Net carbon flux per acre (tons C/acre/year)
        - **GROWTH_CARBON_TOTAL** : float — Carbon from growth (short tons/year)
        - **MORT_CARBON_TOTAL** : float — Carbon from mortality (short tons/year)
        - **REMV_CARBON_TOTAL** : float — Carbon from removals (short tons/year)
        - **AREA_TOTAL** : float — Forest area (acres) in the estimation
        - **N_PLOTS** : int — Number of plots with non-zero growth
        - **[grouping columns]** : varies — Columns from grp_by / by_species

    Notes
    -----
    The Python ``carbon_flux()`` function runs three separate estimator calls
    and combines the results in Python. This SQL version combines all three
    into a single query for efficiency. Both produce equivalent results.

    Standard errors are **not** computed for the combined flux estimate.
    To get per-component standard errors, run ``growth_sql``,
    ``mortality_sql``, and ``removals_sql`` individually with
    ``measure='biomass'``.

    Carbon is estimated as 47% of dry biomass (DRYBIO_BOLE + DRYBIO_BRANCH),
    using the flat conversion fraction. For species-specific live tree carbon
    stocks, use ``carbon_pool_sql`` instead.

    See Also
    --------
    growth_sql : Annual tree growth with variance (use measure='biomass')
    mortality_sql : Annual tree mortality with variance (use measure='biomass')
    removals_sql : Annual removals with variance (use measure='biomass')
    carbon_pool_sql : Species-specific live tree carbon stocks

    Examples
    --------
    Net carbon flux on forestland:

    >>> with pyfia.FIA("path/to/fia.duckdb") as db:
    ...     db.clip_by_state(37)
    ...     db.clip_most_recent(eval_type="GROW")
    ...     sql = pyfia.carbon_flux_sql(db, land_type="forest")

    Carbon flux by forest type:

    >>> sql = pyfia.carbon_flux_sql(db, grp_by="FORTYPCD")

    Carbon flux by species on timberland:

    >>> sql = pyfia.carbon_flux_sql(db, by_species=True, land_type="timber")
    """
    evalid_str = _evalid_list(db)
    group_cols: list[str] = (
        [grp_by] if isinstance(grp_by, str) else list(grp_by) if grp_by else []
    )
    if by_species and "SPCD" not in group_cols:
        group_cols.append("SPCD")

    tc = _TREE_TYPE_MAP.get(tree_type.lower(), "GS")
    lc = _LAND_TYPE_MAP.get(land_type.lower(), "FOREST")

    # All three share the same measure (biomass)
    measure = "biomass"
    metric_expr, _ = _measure_expr(measure, tpa_alias="gc")
    midpt_extra_cols = _grm_measure_columns(measure)   # gm.DRYBIO_BOLE, gm.DRYBIO_BRANCH,

    tree_domain_sql = _domain_to_sql(tree_domain)
    area_domain_sql = _domain_to_sql(area_domain)

    extra_where = ""
    if tree_domain_sql:
        extra_where += f"\n    AND ({tree_domain_sql.replace('DIA', 'gc.DIA_MIDPT')})"
    if area_domain_sql:
        extra_where += f"\n    AND ({area_domain_sql})"

    # Land-type filter on cond_plot
    if land_type == "forest":
        land_filter_cond = "cp.COND_STATUS_CD = 1"
    elif land_type == "timber":
        land_filter_cond = (
            "cp.COND_STATUS_CD = 1 AND cp.RESERVCD = 0 AND cp.SITECLCD <= 6"
        )
    else:
        land_filter_cond = ""

    land_where = f"\n    AND ({land_filter_cond})" if land_filter_cond else ""

    gs_filter = "\n    AND gc.DIA_MIDPT >= 5.0" if tc == "GS" else ""

    gb = _gb_select(group_cols)
    ggroup = _gb_group(group_cols)

    # grp_cols for SELECT in grm_data CTEs
    midpt_cols = {"SPCD", "DIA_MIDPT"}
    grp_cols_select = ""
    if group_cols:
        grp_cols_select = ", ".join(
            f"gm.{c}" if c in midpt_cols else f"cp.{c}" for c in group_cols
        ) + ","

    def _grm_data_cte(component: str, suffix: str) -> str:
        """Build a grm_data CTE for one component type."""
        tpa_col = f"SUBP_{_TPA_PREFIX_MAP[component]}_UNADJ_{tc}_{lc}"
        component_col = f"SUBP_COMPONENT_{tc}_{lc}"
        subptyp_col = f"SUBP_SUBPTYP_GRM_{tc}_{lc}"
        comp_filter = _COMPONENT_FILTER[component]
        return f"""grm_{suffix} AS (
    SELECT
        gc.PLT_CN,
        {grp_cols_select}
        {midpt_extra_cols}
        gc.{tpa_col}     AS TPA_UNADJ,
        gc.{subptyp_col} AS SUBPTYP_GRM,
        cp.CONDPROP_UNADJ,
        s.EXPNS,
        s.STRATUM_CN,
        s.STRATUM_WGT,
        s.P2POINTCNT,
        s.ESTN_UNIT_CN,
        s.AREA_USED,
        s.P1PNTCNT_EU,
        {_grm_adj_sql("s")} AS ADJ_FACTOR
    FROM TREE_GRM_COMPONENT gc
    JOIN TREE_GRM_MIDPT gm ON gc.TRE_CN = gm.TRE_CN
    JOIN strat s            ON gc.PLT_CN = s.PLT_CN
    JOIN cond_plot cp       ON gc.PLT_CN = cp.PLT_CN
    WHERE gc.{tpa_col} > 0
      AND ({comp_filter}){land_where}{gs_filter}{extra_where}
)"""

    def _level_cte(suffix: str) -> str:
        return f"""level_{suffix} AS (
    SELECT
        PLT_CN, STRATUM_CN, EXPNS, STRATUM_WGT, P2POINTCNT,
        ESTN_UNIT_CN, AREA_USED, P1PNTCNT_EU,
        {gb}SUM({metric_expr} * ADJ_FACTOR) AS y_ic,
        MAX(CONDPROP_UNADJ)                 AS x_ic
    FROM grm_{suffix}
    GROUP BY
        PLT_CN, STRATUM_CN, EXPNS, STRATUM_WGT, P2POINTCNT,
        ESTN_UNIT_CN, AREA_USED, P1PNTCNT_EU{ggroup}
)"""

    def _totals_cte(suffix: str, label: str) -> str:
        return f"""totals_{suffix} AS (
    SELECT
        {gb}SUM(y_ic * EXPNS) AS {label}_TOTAL,
        SUM(x_ic * EXPNS)    AS AREA_TOTAL,
        COUNT(DISTINCT CASE WHEN y_ic > 0 THEN PLT_CN END) AS N_PLOTS
    FROM level_{suffix}
    GROUP BY {('1' if not group_cols else ', '.join(group_cols))}
)"""

    # Join logic for the final SELECT
    if not group_cols:
        join_clause = "CROSS JOIN totals_mort tm\nCROSS JOIN totals_remv tr"
        group_by_clause = ""
        select_grp = ""
        order_clause = "ORDER BY NET_CARBON_FLUX_TOTAL DESC"
    else:
        on_parts_m = " AND ".join(f"tg.{c} = tm.{c}" for c in group_cols)
        on_parts_r = " AND ".join(f"tg.{c} = tr.{c}" for c in group_cols)
        join_clause = (
            f"LEFT JOIN totals_mort tm ON {on_parts_m}\n"
            f"LEFT JOIN totals_remv tr ON {on_parts_r}"
        )
        group_by_clause = ""
        select_grp = _gb_select(group_cols, alias="tg")
        order_clause = f"ORDER BY {', '.join(group_cols)}"

    query = f"""-- FIA Carbon Flux Estimation SQL
-- Matches carbon_flux() (pyfia) exactly.
-- EVALIDs: {evalid_str}
-- land_type: {land_type}, tree_type: {tree_type}
--
-- NET_CARBON_FLUX = (GROWTH_BIO - MORT_BIO - REMV_BIO) × {_CARBON_FRAC}
-- All three GRM components computed in a single query under shared CTEs.
-- Note: SE not computed here; run each component separately with biomass measure.
WITH {_strat_cte(evalid_str)},

cond_plot AS (
    SELECT PLT_CN,
           ANY_VALUE(COND_STATUS_CD) AS COND_STATUS_CD,
           SUM(CONDPROP_UNADJ)       AS CONDPROP_UNADJ,
           ANY_VALUE(RESERVCD)       AS RESERVCD,
           ANY_VALUE(SITECLCD)       AS SITECLCD
    FROM COND
    GROUP BY PLT_CN
),

-- Growth (SURVIVOR components)
{_grm_data_cte("growth", "grow")},
{_level_cte("grow")},
{_totals_cte("grow", "GROWTH_BIO")},

-- Mortality (MORTALITY components)
{_grm_data_cte("mortality", "mort")},
{_level_cte("mort")},
{_totals_cte("mort", "MORT_BIO")},

-- Removals (CUT / OTHER_REMOVAL components)
{_grm_data_cte("removals", "remv")},
{_level_cte("remv")},
{_totals_cte("remv", "REMV_BIO")}

SELECT
    {select_grp}tg.AREA_TOTAL,
    tg.N_PLOTS,
    (tg.GROWTH_BIO_TOTAL
     - COALESCE(tm.MORT_BIO_TOTAL, 0.0)
     - COALESCE(tr.REMV_BIO_TOTAL, 0.0)) * {_CARBON_FRAC}  AS NET_CARBON_FLUX_TOTAL,
    tg.GROWTH_BIO_TOTAL * {_CARBON_FRAC}                   AS GROWTH_CARBON_TOTAL,
    COALESCE(tm.MORT_BIO_TOTAL, 0.0) * {_CARBON_FRAC}      AS MORT_CARBON_TOTAL,
    COALESCE(tr.REMV_BIO_TOTAL, 0.0) * {_CARBON_FRAC}      AS REMV_CARBON_TOTAL,
    CASE WHEN tg.AREA_TOTAL > 0
         THEN (tg.GROWTH_BIO_TOTAL
               - COALESCE(tm.MORT_BIO_TOTAL, 0.0)
               - COALESCE(tr.REMV_BIO_TOTAL, 0.0)) * {_CARBON_FRAC}
              / tg.AREA_TOTAL
         ELSE 0.0 END AS NET_CARBON_FLUX_ACRE
FROM totals_grow tg
{join_clause}
{order_clause}
"""
    return query
