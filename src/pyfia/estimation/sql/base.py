"""
Shared SQL building utilities for FIA estimation queries.

All SQL builders in this package produce queries that match the exact statistical
computation performed by the corresponding Python estimators, following
Bechtold & Patterson (2005) methodology.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.fia import FIA


# ---------------------------------------------------------------------------
# Domain expression translation
# ---------------------------------------------------------------------------

def _domain_to_sql(expr: str | None) -> str | None:
    """Convert a Python domain expression to SQL.

    Replaces == with = (the only syntax difference from SQL).
    All other operators (!=, <, >, <=, >=, IN, BETWEEN, AND, OR) are valid SQL.
    """
    if expr is None:
        return None
    return expr.replace("==", "=")


# ---------------------------------------------------------------------------
# Land / tree type filters
# ---------------------------------------------------------------------------

def _land_type_sql(land_type: str, c: str = "c") -> str:
    """Return a SQL WHERE condition fragment for the given land type.

    Parameters
    ----------
    land_type : str
        'forest', 'timber', or 'all'
    c : str
        Alias used for the COND table in the query.
    """
    if land_type == "forest":
        return f"{c}.COND_STATUS_CD = 1"
    if land_type == "timber":
        return (
            f"{c}.COND_STATUS_CD = 1 AND {c}.RESERVCD = 0 AND {c}.SITECLCD <= 6"
        )
    return "1 = 1"  # 'all' – no restriction


def _tree_type_sql(tree_type: str, t: str = "t") -> str:
    """Return a SQL WHERE condition fragment for the given tree type.

    Parameters
    ----------
    tree_type : str
        'live', 'dead', 'gs', or 'all'
    t : str
        Alias used for the TREE table in the query.
    """
    if tree_type == "live":
        return f"{t}.STATUSCD = 1"
    if tree_type == "dead":
        return f"{t}.STATUSCD = 2"
    if tree_type == "gs":
        # Growing stock: live trees classified as growing stock
        return f"{t}.STATUSCD = 1 AND {t}.TREECLCD = 2"
    return "1 = 1"  # 'all'


# ---------------------------------------------------------------------------
# Adjustment factor CASE expressions
# ---------------------------------------------------------------------------

def _tree_adj_sql(t: str = "t", p: str = "p", s: str = "s") -> str:
    """SQL CASE expression that selects the correct FIA adjustment factor
    for a tree based on its diameter and the plot's macroplot breakpoint.

    This mirrors get_adjustment_factor_expr() in tree_expansion.py exactly.
    """
    return f"""CASE
        WHEN {t}.DIA IS NULL THEN {s}.ADJ_FACTOR_SUBP
        WHEN {t}.DIA < 5.0 THEN {s}.ADJ_FACTOR_MICR
        WHEN {t}.DIA < COALESCE(CAST({p}.MACRO_BREAKPOINT_DIA AS DOUBLE), 9999.0)
            THEN {s}.ADJ_FACTOR_SUBP
        ELSE {s}.ADJ_FACTOR_MACR
    END"""


def _area_adj_sql(c: str = "c", s: str = "s") -> str:
    """SQL CASE expression for area adjustment factor.

    Mirrors get_area_adjustment_factor_expr() in tree_expansion.py.
    PROP_BASIS = 'MACR' → ADJ_FACTOR_MACR, otherwise ADJ_FACTOR_SUBP.
    """
    return f"""CASE {c}.PROP_BASIS
        WHEN 'MACR' THEN {s}.ADJ_FACTOR_MACR
        ELSE {s}.ADJ_FACTOR_SUBP
    END"""


def _grm_adj_sql(s: str = "s") -> str:
    """SQL CASE expression for GRM adjustment factor based on SUBPTYP_GRM.

    Mirrors apply_grm_adjustment() in grm.py.
    SUBPTYP_GRM: 0→0.0, 1→SUBP, 2→MICR, 3→MACR.
    """
    return f"""CASE gc.SUBPTYP_GRM
        WHEN 0 THEN 0.0
        WHEN 1 THEN {s}.ADJ_FACTOR_SUBP
        WHEN 2 THEN {s}.ADJ_FACTOR_MICR
        WHEN 3 THEN {s}.ADJ_FACTOR_MACR
        ELSE 0.0
    END"""


# ---------------------------------------------------------------------------
# EVALID helpers
# ---------------------------------------------------------------------------

def _evalid_list(db: "FIA") -> str:
    """Format the db.evalid values as a SQL IN-list literal, e.g. '372019, 372024'."""
    if db.evalid is None:
        raise ValueError(
            "No EVALID set on the FIA database connection. "
            "Call db.clip_by_state(), db.clip_by_evalid(), or db.clip_most_recent() first."
        )
    evals = db.evalid if isinstance(db.evalid, (list, tuple)) else [db.evalid]
    return ", ".join(str(e) for e in evals)


# ---------------------------------------------------------------------------
# GROUP-BY helper strings
# ---------------------------------------------------------------------------

def _gb_select(group_cols: list[str], alias: str = "", trailing_comma: bool = True) -> str:
    """Column list for SELECT, optionally prefixed with a table alias.

    Returns empty string when group_cols is empty.
    """
    if not group_cols:
        return ""
    cols = ", ".join(f"{alias}.{c}" if alias else c for c in group_cols)
    return cols + (", " if trailing_comma else "")


def _gb_group(group_cols: list[str], leading_comma: bool = True) -> str:
    """Fragment for GROUP BY clause.

    Returns empty string when group_cols is empty.
    """
    if not group_cols:
        return ""
    return (", " if leading_comma else "") + ", ".join(group_cols)


def _gb_join(left_alias: str, right_alias: str, group_cols: list[str]) -> str:
    """Additional JOIN conditions for grouping columns, e.g. 'AND l.SPCD = r.SPCD'."""
    if not group_cols:
        return ""
    return " AND " + " AND ".join(f"{left_alias}.{c} = {right_alias}.{c}" for c in group_cols)


# ---------------------------------------------------------------------------
# Shared CTE blocks
# ---------------------------------------------------------------------------

def _strat_cte(evalid_str: str) -> str:
    """CTE that joins POP_PLOT_STRATUM_ASSGN ↔ POP_STRATUM for the given EVALIDs."""
    return f"""strat AS (
    SELECT
        ppsa.PLT_CN,
        ps.STRATUM_CN,
        ps.EXPNS,
        ps.ADJ_FACTOR_MICR,
        ps.ADJ_FACTOR_SUBP,
        ps.ADJ_FACTOR_MACR,
        ps.STRATUM_WGT,
        ps.P2POINTCNT,
        ps.ESTN_UNIT_CN,
        ps.AREA_USED,
        ps.P1PNTCNT_EU
    FROM POP_PLOT_STRATUM_ASSGN ppsa
    JOIN POP_STRATUM ps ON ppsa.STRATUM_CN = ps.STRATUM_CN
    WHERE ps.EVALID IN ({evalid_str})
)"""


def _variance_ctes(group_cols: list[str]) -> str:
    """Generate the variance calculation CTEs.

    These CTEs implement the full Bechtold & Patterson (2005) post-stratified
    variance formula (V1 + V2) and ratio-of-means per-acre variance.

    Assumes the caller has already defined a CTE named ``cond_level`` with:
        PLT_CN, STRATUM_CN, EXPNS, STRATUM_WGT, P2POINTCNT, ESTN_UNIT_CN,
        AREA_USED, P1PNTCNT_EU, [group_cols], y_ic, x_ic

    Outputs a CTE named ``final_var`` with columns:
        [group_cols], var_total, var_x, cov_yx, total_y, total_x
    """
    gb = _gb_select(group_cols)
    ggroup = _gb_group(group_cols)
    ggroup_nolead = _gb_group(group_cols, leading_comma=False)

    # ---- plot_var: plot-level sums from condition-level aggregation ----
    plot_var = f"""plot_var AS (
    SELECT
        PLT_CN,
        STRATUM_CN,
        EXPNS,
        STRATUM_WGT,
        P2POINTCNT,
        ESTN_UNIT_CN,
        AREA_USED,
        P1PNTCNT_EU,
        {gb}SUM(y_ic) AS y_i,
        SUM(x_ic) AS x_i
    FROM cond_level
    GROUP BY
        PLT_CN, STRATUM_CN, EXPNS, STRATUM_WGT, P2POINTCNT,
        ESTN_UNIT_CN, AREA_USED, P1PNTCNT_EU{ggroup}
)"""

    # ---- all_plots_base: every plot in the evaluation ----
    all_plots_base = """all_plots_base AS (
    SELECT DISTINCT
        PLT_CN, STRATUM_CN, EXPNS, STRATUM_WGT, P2POINTCNT,
        ESTN_UNIT_CN, AREA_USED, P1PNTCNT_EU
    FROM strat
)"""

    # ---- group_vals + CROSS JOIN for grouped case ----
    if group_cols:
        group_vals_cte = f"""group_vals AS (
    SELECT DISTINCT {', '.join(group_cols)} FROM cond_level
)"""
        cross_join_clause = "CROSS JOIN group_vals gv"
        grp_from_cross = _gb_select(group_cols, alias="gv")
        and_grp_join = _gb_join("pv", "gv", group_cols)
        cross_section = f""",
{group_vals_cte},"""
    else:
        cross_join_clause = ""
        grp_from_cross = ""
        and_grp_join = ""
        cross_section = ""

    # ---- all_plots_filled: LEFT JOIN with zero fill ----
    all_plots_filled = f"""all_plots_filled AS (
    SELECT
        apb.PLT_CN,
        apb.STRATUM_CN,
        apb.EXPNS,
        apb.STRATUM_WGT,
        apb.P2POINTCNT,
        apb.ESTN_UNIT_CN,
        apb.AREA_USED,
        apb.P1PNTCNT_EU,
        {grp_from_cross}COALESCE(pv.y_i, 0.0) AS y_i,
        COALESCE(pv.x_i, 0.0) AS x_i
    FROM all_plots_base apb
    {cross_join_clause}
    LEFT JOIN plot_var pv
        ON apb.PLT_CN = pv.PLT_CN{and_grp_join}
)"""

    # ---- strat_stats: per-stratum (× group) statistics ----
    strat_stats = f"""strat_stats AS (
    SELECT
        ESTN_UNIT_CN,
        STRATUM_CN,
        P2POINTCNT,
        STRATUM_WGT,
        {gb}MAX(AREA_USED) AS A,
        MAX(EXPNS)    AS EXPNS,
        COUNT(*)      AS n_h_actual,
        AVG(y_i)      AS ybar_h,
        COALESCE(VAR_SAMP(y_i), 0.0)        AS s2_yh,
        AVG(x_i)      AS xbar_h,
        COALESCE(VAR_SAMP(x_i), 0.0)        AS s2_xh,
        COALESCE(COVAR_SAMP(y_i, x_i), 0.0) AS cov_yxh
    FROM all_plots_filled
    GROUP BY ESTN_UNIT_CN, STRATUM_CN, P2POINTCNT, STRATUM_WGT{ggroup}
)"""

    # ---- eu_n: total actual plots per EU (× group) ----
    eu_n = f"""eu_n AS (
    SELECT ESTN_UNIT_CN, {gb}SUM(n_h_actual) AS n_eu
    FROM strat_stats
    GROUP BY ESTN_UNIT_CN{ggroup}
)"""

    # ---- strat_var: per-stratum variance components ----
    gb_ss = _gb_select(group_cols, alias="ss")
    gb_join_en = _gb_join("ss", "en", group_cols)
    strat_var = f"""strat_var AS (
    SELECT
        ss.ESTN_UNIT_CN,
        ss.STRATUM_CN,
        ss.P2POINTCNT,
        ss.STRATUM_WGT,
        ss.A,
        ss.EXPNS,
        ss.n_h_actual,
        ss.ybar_h,
        ss.xbar_h,
        {gb_ss}en.n_eu,
        CASE WHEN ss.n_h_actual > 1
             THEN ss.s2_yh  / NULLIF(ss.P2POINTCNT, 0) ELSE 0.0 END AS v_yh,
        CASE WHEN ss.n_h_actual > 1
             THEN ss.s2_xh  / NULLIF(ss.P2POINTCNT, 0) ELSE 0.0 END AS v_xh,
        CASE WHEN ss.n_h_actual > 1
             THEN ss.cov_yxh / NULLIF(ss.P2POINTCNT, 0) ELSE 0.0 END AS c_yxh
    FROM strat_stats ss
    JOIN eu_n en
        ON ss.ESTN_UNIT_CN = en.ESTN_UNIT_CN{gb_join_en}
)"""

    # ---- eu_var: per-EU (× group) aggregated variance components ----
    gb_sv = _gb_select(group_cols, alias="sv")
    eu_var = f"""eu_var AS (
    SELECT
        sv.ESTN_UNIT_CN,
        sv.n_eu,
        sv.A,
        {gb_sv}SUM(sv.STRATUM_WGT * sv.v_yh)           AS sum_v1_y,
        SUM((1.0 - sv.STRATUM_WGT) * sv.v_yh)  AS sum_v2_y,
        SUM(sv.STRATUM_WGT * sv.v_xh)           AS sum_v1_x,
        SUM((1.0 - sv.STRATUM_WGT) * sv.v_xh)  AS sum_v2_x,
        SUM(sv.STRATUM_WGT * sv.c_yxh)          AS sum_v1_cov,
        SUM((1.0 - sv.STRATUM_WGT) * sv.c_yxh) AS sum_v2_cov,
        SUM(sv.ybar_h * sv.EXPNS * sv.n_h_actual) AS total_y,
        SUM(sv.xbar_h * sv.EXPNS * sv.n_h_actual) AS total_x
    FROM strat_var sv
    GROUP BY sv.ESTN_UNIT_CN, sv.n_eu, sv.A{ggroup}
)"""

    # ---- final_var: aggregate across EUs (× group) ----
    if group_cols:
        final_group_by = f"GROUP BY {ggroup_nolead}"
        final_select_gb = gb
    else:
        final_group_by = ""
        final_select_gb = ""

    final_var = f"""final_var AS (
    SELECT
        {final_select_gb}SUM(GREATEST((A * A / n_eu)            * sum_v1_y
                              + (A * A / (n_eu * n_eu)) * sum_v2_y, 0.0)) AS var_total,
        SUM(GREATEST((A * A / n_eu)            * sum_v1_x
                              + (A * A / (n_eu * n_eu)) * sum_v2_x, 0.0)) AS var_x,
        SUM(          (A * A / n_eu)            * sum_v1_cov
                    + (A * A / (n_eu * n_eu)) * sum_v2_cov)               AS cov_yx,
        SUM(total_y) AS total_y,
        SUM(total_x) AS total_x
    FROM eu_var
    {final_group_by}
)"""

    return f"""{plot_var},
{all_plots_base},{cross_section}
{all_plots_filled},
{strat_stats},
{eu_n},
{strat_var},
{eu_var},
{final_var}"""


# ---------------------------------------------------------------------------
# Final SELECT helpers
# ---------------------------------------------------------------------------

def _se_total_expr(var_col: str = "fv.var_total") -> str:
    """SQL expression for total standard error."""
    return f"SQRT(GREATEST({var_col}, 0.0))"


def _se_acre_expr() -> str:
    """SQL expression for per-acre SE using ratio-of-means variance formula.

    V(R) = (1/X²) × [V(Y) + R² × V(X) - 2R × Cov(Y,X)]
    """
    return """CASE WHEN fv.total_x > 0 THEN
        SQRT(GREATEST(
            (1.0 / (fv.total_x * fv.total_x)) * (
                fv.var_total
                + (fv.total_y / fv.total_x) * (fv.total_y / fv.total_x) * fv.var_x
                - 2.0 * (fv.total_y / fv.total_x) * fv.cov_yx
            ), 0.0))
    ELSE 0.0 END"""


def _final_join(group_cols: list[str], pt_alias: str = "pt", fv_alias: str = "fv") -> str:
    """JOIN clause between pop_totals and final_var."""
    if not group_cols:
        return "CROSS JOIN final_var fv"
    on_parts = " AND ".join(
        f"{pt_alias}.{c} = {fv_alias}.{c}" for c in group_cols
    )
    return f"JOIN final_var fv ON {on_parts}"
