"""
Shared SQL infrastructure for Growth-Removal-Mortality (GRM) estimators.

GRM queries use TREE_GRM_COMPONENT and TREE_GRM_MIDPT instead of TREE.
The EVALID filter is applied via JOIN with POP_PLOT_STRATUM_ASSGN.
The GRM adjustment factor is selected by SUBPTYP_GRM (not tree DIA).

Column naming follows FIA conventions:
  SUBP_{TPAGROW|TPAMORT|TPAREMV}_UNADJ_{GS|AL|SL}_{FOREST|TIMBER}
  SUBP_COMPONENT_{GS|AL|SL}_{FOREST|TIMBER}
  SUBP_SUBPTYP_GRM_{GS|AL|SL}_{FOREST|TIMBER}
"""

from __future__ import annotations

from .base import (
    _area_adj_sql,
    _domain_to_sql,
    _evalid_list,
    _final_join,
    _gb_group,
    _gb_select,
    _grm_adj_sql,
    _se_acre_expr,
    _se_total_expr,
    _strat_cte,
    _variance_ctes,
)

_TREE_TYPE_MAP = {
    "gs": "GS",
    "live": "AL",
    "al": "AL",
    "sawtimber": "SL",
    "sl": "SL",
}
_LAND_TYPE_MAP = {
    "forest": "FOREST",
    "timber": "TIMBER",
}
_TPA_PREFIX_MAP = {
    "growth": "TPAGROW",
    "mortality": "TPAMORT",
    "removals": "TPAREMV",
}

# Component filter SQL for each estimator type
_COMPONENT_FILTER = {
    "growth": "gc.COMPONENT LIKE 'SURVIVOR%'",
    "mortality": "gc.COMPONENT LIKE 'MORTALITY%'",
    "removals": "gc.COMPONENT LIKE 'CUT%' OR gc.COMPONENT LIKE 'OTHER_REMOVAL%'",
}

# Measure metric expressions (applied to midpoint values)
def _measure_expr(measure: str, tpa_alias: str = "gc") -> tuple[str, str]:
    """Return (metric_expr, metric_label) for the given measure.

    The metric_expr is a SQL expression that computes the per-tree contribution
    to the estimate numerator.  ``tpa_alias.TPA_UNADJ`` is the GRM TPA column
    already renamed to TPA_UNADJ in the grm_data CTE.
    """
    if measure == "volume":
        return (
            f"{tpa_alias}.TPA_UNADJ * gm.VOLCFNET",
            "VOLCFNET",
        )
    if measure == "biomass":
        # (DRYBIO_BOLE + DRYBIO_BRANCH) in lbs → short tons
        return (
            f"{tpa_alias}.TPA_UNADJ * (gm.DRYBIO_BOLE + gm.DRYBIO_BRANCH) * 0.0005",
            "BIOMASS",
        )
    if measure == "basal_area":
        # 0.005454 = π/(4×144)
        return (
            f"{tpa_alias}.TPA_UNADJ * gm.DIA * gm.DIA * 0.005454",
            "BAA",
        )
    # Default: count (trees per acre)
    return (f"{tpa_alias}.TPA_UNADJ", "TPA")


def _grm_measure_columns(measure: str) -> str:
    """Additional columns to select from TREE_GRM_MIDPT for a given measure."""
    if measure == "volume":
        return "gm.VOLCFNET,"
    if measure == "biomass":
        return "gm.DRYBIO_BOLE, gm.DRYBIO_BRANCH,"
    if measure == "basal_area":
        return "gm.DIA,"
    return ""  # count / tpa – no extra columns needed


def _grm_sql(
    component_type: str,
    db,
    grp_by,
    by_species: bool,
    tree_type: str,
    land_type: str,
    measure: str,
    tree_domain,
    area_domain,
) -> str:
    """Build the SQL for a GRM-based estimator (mortality / growth / removals).

    Parameters
    ----------
    component_type : str
        'mortality', 'growth', or 'removals'.
    db : FIA
        Database connection with an EVALID already set.
    grp_by, by_species, tree_type, land_type, measure,
    tree_domain, area_domain : see individual estimator functions.
    """
    evalid_str = _evalid_list(db)
    group_cols: list[str] = (
        [grp_by] if isinstance(grp_by, str) else list(grp_by) if grp_by else []
    )
    if by_species and "SPCD" not in group_cols:
        group_cols.append("SPCD")

    tc = _TREE_TYPE_MAP.get(tree_type.lower(), "GS")
    lc = _LAND_TYPE_MAP.get(land_type.lower(), "FOREST")
    tpa_prefix = _TPA_PREFIX_MAP[component_type]

    tpa_col = f"SUBP_{tpa_prefix}_UNADJ_{tc}_{lc}"
    component_col = f"SUBP_COMPONENT_{tc}_{lc}"
    subptyp_col = f"SUBP_SUBPTYP_GRM_{tc}_{lc}"

    comp_filter = _COMPONENT_FILTER[component_type]
    metric_expr, metric_label = _measure_expr(measure)
    midpt_extra_cols = _grm_measure_columns(measure)

    tree_domain_sql = _domain_to_sql(tree_domain)
    area_domain_sql = _domain_to_sql(area_domain)

    gb = _gb_select(group_cols)
    ggroup = _gb_group(group_cols)

    # GRM uses FORTYPCD / OWNGRPCD from COND for area_domain filtering
    # and land_type filtering.
    land_filter_cond = ""
    if land_type == "forest":
        land_filter_cond = "cond_agg.COND_STATUS_CD = 1"
    elif land_type == "timber":
        land_filter_cond = (
            "cond_agg.COND_STATUS_CD = 1 AND cond_agg.RESERVCD = 0 AND cond_agg.SITECLCD <= 6"
        )

    land_where = f"\n    AND ({land_filter_cond})" if land_filter_cond else ""

    extra_where = ""
    if tree_domain_sql:
        extra_where += f"\n    AND ({tree_domain_sql.replace('DIA', 'gc.DIA_MIDPT')})"
    if area_domain_sql:
        extra_where += f"\n    AND ({area_domain_sql})"

    # Growing-stock filter: DIA_MIDPT >= 5.0
    gs_filter = ""
    if tc == "GS":
        gs_filter = "\n    AND gc.DIA_MIDPT >= 5.0"

    # grp_by columns – all from cond_agg since GRM doesn't have CONDID-level tree data
    grp_cols_select = ""
    if group_cols:
        # SPCD / DIA_MIDPT come from midpt table; the rest from cond_agg
        midpt_cols = {"SPCD", "DIA_MIDPT"}
        grp_cols_select = ", ".join(
            f"gm.{c}" if c in midpt_cols else f"cond_agg.{c}" for c in group_cols
        ) + ","

    metric_label_upper = metric_label.upper()

    query = f"""-- FIA {component_type.title()} Estimation SQL
-- Matches {component_type.title()}Estimator (pyfia) exactly.
-- EVALIDs: {evalid_str}
-- tree_type: {tree_type} ({tc}), land_type: {land_type} ({lc}), measure: {measure}
--
-- Uses TREE_GRM_COMPONENT + TREE_GRM_MIDPT tables.
-- GRM EVALID filter applied via POP_PLOT_STRATUM_ASSGN.
-- ADJ_FACTOR determined by SUBPTYP_GRM (not tree DIA).
WITH {_strat_cte(evalid_str)},

-- Condition aggregation to plot level (GRM tables lack CONDID)
cond_agg AS (
    SELECT
        PLT_CN,
        FIRST_VALUE(COND_STATUS_CD) OVER (PARTITION BY PLT_CN ORDER BY CONDID) AS COND_STATUS_CD,
        SUM(CONDPROP_UNADJ) AS CONDPROP_UNADJ,
        FIRST_VALUE(RESERVCD) OVER (PARTITION BY PLT_CN ORDER BY CONDID)  AS RESERVCD,
        FIRST_VALUE(SITECLCD) OVER (PARTITION BY PLT_CN ORDER BY CONDID)  AS SITECLCD,
        FIRST_VALUE(FORTYPCD) OVER (PARTITION BY PLT_CN ORDER BY CONDID)  AS FORTYPCD,
        FIRST_VALUE(OWNGRPCD) OVER (PARTITION BY PLT_CN ORDER BY CONDID)  AS OWNGRPCD
    FROM COND
    GROUP BY PLT_CN, CONDID, COND_STATUS_CD, RESERVCD, SITECLCD, FORTYPCD, OWNGRPCD
),

-- Simpler per-plot aggregation using subquery
cond_plot AS (
    SELECT PLT_CN,
           ANY_VALUE(COND_STATUS_CD) AS COND_STATUS_CD,
           SUM(CONDPROP_UNADJ)       AS CONDPROP_UNADJ,
           ANY_VALUE(RESERVCD)       AS RESERVCD,
           ANY_VALUE(SITECLCD)       AS SITECLCD,
           ANY_VALUE(FORTYPCD)       AS FORTYPCD,
           ANY_VALUE(OWNGRPCD)       AS OWNGRPCD
    FROM COND
    GROUP BY PLT_CN
),

grm_data AS (
    -- GRM component data joined with midpoint values and stratification
    SELECT
        gc.TRE_CN,
        gc.PLT_CN,
        gc.DIA_MIDPT,
        gc.DIA_BEGIN,
        gc.{component_col} AS COMPONENT,
        gc.{tpa_col}        AS TPA_UNADJ,
        gc.{subptyp_col}    AS SUBPTYP_GRM,
        {midpt_extra_cols}
        gm.SPCD,
        {grp_cols_select}
        cp.CONDPROP_UNADJ,
        cp.COND_STATUS_CD,
        cp.RESERVCD,
        cp.SITECLCD,
        cp.FORTYPCD,
        cp.OWNGRPCD,
        s.STRATUM_CN,
        s.EXPNS,
        s.STRATUM_WGT,
        s.P2POINTCNT,
        s.ESTN_UNIT_CN,
        s.AREA_USED,
        s.P1PNTCNT_EU,
        s.ADJ_FACTOR_MICR,
        s.ADJ_FACTOR_SUBP,
        s.ADJ_FACTOR_MACR,
        -- GRM adjustment factor based on SUBPTYP_GRM
        {_grm_adj_sql("s")} AS ADJ_FACTOR
    FROM TREE_GRM_COMPONENT gc
    JOIN TREE_GRM_MIDPT gm ON gc.TRE_CN = gm.TRE_CN
    JOIN strat s            ON gc.PLT_CN = s.PLT_CN
    JOIN cond_plot cp       ON gc.PLT_CN = cp.PLT_CN
    WHERE gc.{tpa_col} > 0
      AND ({comp_filter}){land_where}{gs_filter}{extra_where}
),

-- Stage 1: condition-level aggregation
-- y_ic = Σ(metric × ADJ_FACTOR) per (plot, [group])
-- x_ic = CONDPROP_UNADJ  (plot-level area, constant per plot)
cond_level AS (
    SELECT
        PLT_CN,
        -- Use CONDID=1 (dummy) since GRM is plot-level
        STRATUM_CN,
        EXPNS,
        STRATUM_WGT,
        P2POINTCNT,
        ESTN_UNIT_CN,
        AREA_USED,
        P1PNTCNT_EU,
        {gb}SUM({metric_expr} * ADJ_FACTOR) AS y_ic,
        MAX(CONDPROP_UNADJ)                 AS x_ic
    FROM grm_data
    GROUP BY
        PLT_CN, STRATUM_CN, EXPNS, STRATUM_WGT, P2POINTCNT,
        ESTN_UNIT_CN, AREA_USED, P1PNTCNT_EU{ggroup}
),

-- Stage 2: population totals
pop_totals AS (
    SELECT
        {gb}SUM(y_ic * EXPNS) AS {metric_label_upper}_TOTAL,
        SUM(x_ic * EXPNS)    AS AREA_TOTAL,
        CASE WHEN SUM(x_ic * EXPNS) > 0
             THEN SUM(y_ic * EXPNS) / SUM(x_ic * EXPNS) ELSE 0.0 END AS {metric_label_upper}_ACRE,
        COUNT(DISTINCT CASE WHEN y_ic > 0 THEN PLT_CN END) AS N_PLOTS
    FROM cond_level
    GROUP BY {('1' if not group_cols else ', '.join(group_cols))}
),

{_variance_ctes(group_cols)}

SELECT
    {gb}pt.{metric_label_upper}_TOTAL,
    pt.AREA_TOTAL,
    pt.{metric_label_upper}_ACRE,
    pt.N_PLOTS,
    {_se_total_expr()} AS {metric_label_upper}_TOTAL_SE,
    {_se_acre_expr()}  AS {metric_label_upper}_ACRE_SE
FROM pop_totals pt
{_final_join(group_cols)}
ORDER BY {(f'pt.{metric_label_upper}_ACRE DESC' if not group_cols else ', '.join(group_cols))}
"""
    return query
