"""
SQL query builder for remeasurement panel datasets.

Creates t1/t2 (time 1 / time 2) linked condition or tree panels from FIA
remeasured plots, mirroring the panel() function.

These are data retrieval queries, NOT population estimators. No expansion
factors (EXPNS), no variance estimation. Use the output as input to further
analysis (e.g., harvest detection, growth tracking, change detection).

Condition-level panel:
  Links PLOT/COND at t2 (current) back to PLOT/COND at t1 (previous)
  via PREV_PLT_CN and matching CONDID.

Tree-level panel:
  Links TREE_GRM_COMPONENT (fate classification) with TREE_GRM_MIDPT
  (midpoint measurements) and provides t1/t2 diameter values via
  DIA_BEGIN (t1) and DIA_MIDPT/DIA_END (t2 proxies).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    _domain_to_sql,
    _evalid_list,
    _land_type_sql,
    _strat_cte,
    _tree_type_sql,
)

if TYPE_CHECKING:
    from ...core.fia import FIA


def panel_sql(
    db: "FIA",
    level: str = "condition",
    land_type: str = "forest",
    tree_type: str = "all",
    tree_domain: str | None = None,
    area_domain: str | None = None,
    min_remper: float = 0.0,
    max_remper: float | None = None,
    harvest_only: bool = False,
) -> str:
    """Return a SQL query string that replicates ``panel()`` data retrieval.

    Creates a linked t1/t2 remeasurement panel dataset from FIA's remeasured
    plots. Unlike the other SQL builders, this is **not** a population
    estimator — it returns individual rows (one per condition or tree) with
    both time-1 and time-2 attributes joined together.

    This panel data is useful for:

    - Harvest probability modeling
    - Forest change detection (area transitions)
    - Growth and mortality analysis at the individual tree level
    - Land use transition studies

    Parameters
    ----------
    db : FIA
        Connected FIA database with EVALID set (used to restrict plots
        through the ``strat`` CTE join).
    level : {'condition', 'tree'}, default 'condition'
        Level of panel to create:

        - 'condition': Condition-level panel for area/harvest analysis.
          Each row is one condition measured at two time points. Links
          PLOT/COND at t2 back to PLOT/COND at t1 via ``PREV_PLT_CN``
          and matching CONDID.
        - 'tree': Tree-level panel for individual tree tracking. Uses
          ``TREE_GRM_COMPONENT`` for fate classification and
          ``TREE_GRM_MIDPT`` for midpoint measurements. Provides t1
          diameter (``DIA_BEGIN``) and midpoint diameter (``DIA_MIDPT``).
    land_type : {'forest', 'timber', 'all'}, default 'forest'
        Land classification filter:

        - 'forest': All forest land (COND_STATUS_CD = 1)
        - 'timber': Timberland (productive, unreserved forest)
        - 'all': No land type filtering
    tree_type : {'all', 'live', 'gs'}, default 'all'
        Tree type filter (tree-level panel only). Maps to GRM column suffixes:

        - 'gs': Growing stock (merchantable trees) — uses GS columns
        - 'all': All trees — uses GS columns (default GRM behavior)
        - 'live': All live trees — uses AL columns
    tree_domain : str, optional
        SQL-like filter expression for tree/GRM columns (tree-level panel
        only). References to ``DIA`` are rewritten to ``gc.DIA_MIDPT``.
        Examples:

        - ``"SPCD == 131"``: Loblolly pine only
        - ``"DIA >= 5.0"``: Trees ≥ 5" midpoint diameter
    area_domain : str, optional
        SQL-like filter expression for COND columns. For condition-level
        panels, references to ``c.`` are rewritten to ``c2.`` (current
        condition). Examples:

        - ``"OWNGRPCD == 40"``: Private land only
        - ``"FORTYPCD IN (161, 162)"``: Specific forest types
    min_remper : float, default 0.0
        Minimum remeasurement period in years. Filters out very short
        measurement intervals (e.g., re-visits within the same cycle).
    max_remper : float, optional
        Maximum remeasurement period in years. Use to exclude unusually
        long intervals that span multiple inventory cycles.
    harvest_only : bool, default False
        If True, condition-level panel returns only conditions where a
        harvest treatment (TRTCD1/2/3 IN (10, 20)) was recorded at t2.

    Returns
    -------
    str
        Complete, self-contained SQL query string.

        For **condition-level** (level='condition'), each row has:

        - **PLT_CN** : Current plot control number (t2)
        - **STATECD**, **COUNTYCD**, **LAT**, **LON**, **ELEV** : Location
        - **INVYR_T2**, **INVYR_T1** : Inventory years at t2 and t1
        - **REMPER** : Remeasurement period (years)
        - **CONDID** : Condition identifier
        - **T2_*** : Current condition attributes (COND_STATUS_CD, FORTYPCD,
          OWNGRPCD, STDAGE, BALIVE, TRTCD1-3, DSTRBCD1-2, SICOND, SITECLCD)
        - **T1_*** : Previous condition attributes
        - **T2_IN_DOMAIN**, **T1_IN_DOMAIN** : Land-type domain indicators

        For **tree-level** (level='tree'), each row has:

        - **TRE_CN**, **PLT_CN** : Tree and plot identifiers
        - **STATECD**, **COUNTYCD**, **INVYR**, **REMPER** : Location/timing
        - **COMPONENT** : Raw GRM component (SURVIVOR, MORTALITY1, CUT1, etc.)
        - **DIA_T1** : Diameter at t1 (from ``DIA_BEGIN``)
        - **DIA_MIDPT** : Midpoint diameter
        - **SPCD**, **SPGRPCD**, **STATUSCD** : Species and status
        - **VOLCFNET**, **DRYBIO_AG**, **DRYBIO_BOLE**, **DRYBIO_BRANCH** :
          Volume/biomass at midpoint
        - **TPA_GROW_GS**, **TPA_MORT_GS**, **TPA_REMV_GS** : TPA values
        - **COND_STATUS_CD**, **FORTYPCD**, **OWNGRPCD** : Condition attributes

    Notes
    -----
    This is a data retrieval query — output rows are individual conditions
    or trees. There are no expansion factors (EXPNS) and no variance
    estimation. The strat CTE is used only to restrict plots to those in
    the active EVALID.

    Tree fate is determined from ``TREE_GRM_COMPONENT``, which provides
    authoritative classification pre-computed by FIA:

    - **SURVIVOR**: Tree alive at beginning and end of period
    - **MORTALITY1/2**: Tree died during measurement period
    - **CUT1/2**: Tree removed by harvest
    - **OTHER_REMOVAL**: Tree removed due to land use change

    For population-level change estimates, use ``area_change_sql()`` (area)
    or the GRM estimator SQL functions (``growth_sql``, ``mortality_sql``,
    ``removals_sql``) for trees.

    See Also
    --------
    area_change_sql : Population-level annual forest area change
    growth_sql : Population-level annual growth (SURVIVOR components)
    mortality_sql : Population-level annual mortality
    removals_sql : Population-level annual harvest removals

    Examples
    --------
    Condition-level panel for harvest analysis:

    >>> with pyfia.FIA("path/to/fia.duckdb") as db:
    ...     db.clip_by_state(37)
    ...     db.clip_most_recent(eval_type="GROW")
    ...     sql = pyfia.panel_sql(db, level="condition", land_type="timber")

    Tree-level panel with GRM fate classification:

    >>> sql = pyfia.panel_sql(db, level="tree", tree_type="gs")

    Harvest plots only (condition-level):

    >>> sql = pyfia.panel_sql(db, level="condition", harvest_only=True)

    Filter to 4–8 year remeasurement intervals:

    >>> sql = pyfia.panel_sql(
    ...     db, level="condition", min_remper=4.0, max_remper=8.0
    ... )
    """
    if level == "tree":
        return _tree_panel_sql(
            db=db,
            land_type=land_type,
            tree_type=tree_type,
            tree_domain=tree_domain,
            area_domain=area_domain,
            min_remper=min_remper,
            max_remper=max_remper,
        )
    return _condition_panel_sql(
        db=db,
        land_type=land_type,
        area_domain=area_domain,
        min_remper=min_remper,
        max_remper=max_remper,
        harvest_only=harvest_only,
    )


def _condition_panel_sql(
    db: "FIA",
    land_type: str,
    area_domain: str | None,
    min_remper: float,
    max_remper: float | None,
    harvest_only: bool,
) -> str:
    """Condition-level t1/t2 panel."""
    evalid_str = _evalid_list(db)
    area_domain_sql = _domain_to_sql(area_domain)

    remper_filters = ""
    if min_remper > 0:
        remper_filters += f"\n  AND p2.REMPER >= {min_remper}"
    if max_remper is not None:
        remper_filters += f"\n  AND p2.REMPER <= {max_remper}"

    land_filter_t2 = _land_type_sql(land_type, c="c2")
    land_filter_t1 = _land_type_sql(land_type, c="c1")

    extra_where = ""
    if area_domain_sql:
        extra_where += f"\n  AND ({area_domain_sql.replace('c.', 'c2.')})"
    if harvest_only:
        extra_where += (
            "\n  AND ("
            "c2.TRTCD1 IN (10, 20) OR c2.TRTCD2 IN (10, 20) OR c2.TRTCD3 IN (10, 20)"
            ")"
        )

    # Timber needs extra columns
    timber_cols_t2 = "c2.RESERVCD AS T2_RESERVCD, c2.SITECLCD AS T2_SITECLCD," if land_type == "timber" else ""
    timber_cols_t1 = "c1.RESERVCD AS T1_RESERVCD, c1.SITECLCD AS T1_SITECLCD," if land_type == "timber" else ""

    return f"""-- FIA Condition-Level Remeasurement Panel SQL
-- Matches panel(level='condition') (pyfia).
-- EVALIDs: {evalid_str}
-- land_type: {land_type}
--
-- NOTE: This is a data retrieval query, NOT a population estimator.
-- Each row is one condition with t1 (previous) and t2 (current) attributes.
-- For population estimates use area_change_sql() instead.
WITH {_strat_cte(evalid_str)}

SELECT
    -- Plot identification
    p2.CN                AS PLT_CN,
    p2.STATECD,
    p2.COUNTYCD,
    p2.INVYR             AS INVYR_T2,
    p1.INVYR             AS INVYR_T1,
    p2.REMPER,
    p2.CYCLE             AS CYCLE_T2,
    p1.CYCLE             AS CYCLE_T1,
    -- Location
    p2.LAT,
    p2.LON,
    p2.ELEV,
    -- Condition identification
    c2.CONDID,
    -- Current condition (t2)
    c2.COND_STATUS_CD    AS T2_COND_STATUS_CD,
    c2.CONDPROP_UNADJ    AS T2_CONDPROP_UNADJ,
    c2.OWNGRPCD          AS T2_OWNGRPCD,
    c2.FORTYPCD          AS T2_FORTYPCD,
    c2.STDAGE            AS T2_STDAGE,
    c2.BALIVE            AS T2_BALIVE,
    c2.SICOND            AS T2_SICOND,
    c2.SITECLCD          AS T2_SITECLCD,
    c2.SLOPE             AS T2_SLOPE,
    c2.ASPECT            AS T2_ASPECT,
    c2.TRTCD1            AS T2_TRTCD1,
    c2.TRTCD2            AS T2_TRTCD2,
    c2.TRTCD3            AS T2_TRTCD3,
    c2.TRTYR1            AS T2_TRTYR1,
    c2.DSTRBCD1          AS T2_DSTRBCD1,
    c2.DSTRBCD2          AS T2_DSTRBCD2,
    c2.DSTRBYR1          AS T2_DSTRBYR1,
    {timber_cols_t2}
    -- Previous condition (t1)
    c1.COND_STATUS_CD    AS T1_COND_STATUS_CD,
    c1.CONDPROP_UNADJ    AS T1_CONDPROP_UNADJ,
    c1.OWNGRPCD          AS T1_OWNGRPCD,
    c1.FORTYPCD          AS T1_FORTYPCD,
    c1.STDAGE            AS T1_STDAGE,
    c1.BALIVE            AS T1_BALIVE,
    c1.SICOND            AS T1_SICOND,
    c1.SITECLCD          AS T1_SITECLCD,
    {timber_cols_t1}
    -- Land-type domain indicators
    CASE WHEN {land_filter_t2} THEN 1 ELSE 0 END AS T2_IN_DOMAIN,
    CASE WHEN {land_filter_t1} THEN 1 ELSE 0 END AS T1_IN_DOMAIN
FROM PLOT p2
-- Previous plot (t1)
JOIN PLOT p1 ON p2.PREV_PLT_CN = p1.CN
-- Current condition (t2)
JOIN COND c2 ON p2.CN = c2.PLT_CN
-- Previous condition (t1), matched by CONDID
LEFT JOIN COND c1 ON p1.CN = c1.PLT_CN AND c2.CONDID = c1.CONDID
-- Restrict to plots in the current evaluation
JOIN strat s ON p2.CN = s.PLT_CN
WHERE p2.PREV_PLT_CN IS NOT NULL
  AND p2.REMPER > 0
  AND p2.INVYR >= 2000{remper_filters}{extra_where}
ORDER BY p2.STATECD, p2.COUNTYCD, p2.CN, c2.CONDID
"""


def _tree_panel_sql(
    db: "FIA",
    land_type: str,
    tree_type: str,
    tree_domain: str | None,
    area_domain: str | None,
    min_remper: float,
    max_remper: float | None,
) -> str:
    """Tree-level t1/t2 panel using GRM tables."""
    evalid_str = _evalid_list(db)
    tree_domain_sql = _domain_to_sql(tree_domain)
    area_domain_sql = _domain_to_sql(area_domain)

    remper_filters = ""
    if min_remper > 0:
        remper_filters += f"\n  AND p.REMPER >= {min_remper}"
    if max_remper is not None:
        remper_filters += f"\n  AND p.REMPER <= {max_remper}"

    land_filter = _land_type_sql(land_type, c="cp")

    extra_where = ""
    if tree_domain_sql:
        extra_where += f"\n  AND ({tree_domain_sql.replace('DIA', 'gc.DIA_MIDPT')})"
    if area_domain_sql:
        extra_where += f"\n  AND ({area_domain_sql.replace('c.', 'cp.')})"

    return f"""-- FIA Tree-Level Remeasurement Panel SQL
-- Matches panel(level='tree') (pyfia).
-- EVALIDs: {evalid_str}
-- land_type: {land_type}, tree_type: {tree_type}
--
-- NOTE: This is a data retrieval query, NOT a population estimator.
-- Each row is one tree with GRM component fate and midpoint measurements.
-- DIA_BEGIN = diameter at t1; DIA_MIDPT = midpoint diameter.
WITH {_strat_cte(evalid_str)},

cond_plot AS (
    SELECT PLT_CN,
           ANY_VALUE(COND_STATUS_CD) AS COND_STATUS_CD,
           ANY_VALUE(RESERVCD)       AS RESERVCD,
           ANY_VALUE(SITECLCD)       AS SITECLCD,
           ANY_VALUE(FORTYPCD)       AS FORTYPCD,
           ANY_VALUE(OWNGRPCD)       AS OWNGRPCD,
           SUM(CONDPROP_UNADJ)       AS CONDPROP_UNADJ
    FROM COND
    GROUP BY PLT_CN
)

SELECT
    gc.TRE_CN,
    gc.PLT_CN,
    p.STATECD,
    p.COUNTYCD,
    p.INVYR,
    p.REMPER,
    -- Tree fate
    gc.COMPONENT,
    -- Size
    gc.DIA_BEGIN    AS DIA_T1,
    gc.DIA_MIDPT    AS DIA_MIDPT,
    -- Species and status from midpoint
    gm.SPCD,
    gm.SPGRPCD,
    gm.STATUSCD,
    -- Volume / biomass at midpoint
    gm.VOLCFNET,
    gm.DRYBIO_AG,
    gm.DRYBIO_BOLE,
    gm.DRYBIO_BRANCH,
    -- TPA columns (growing stock / all live)
    gc.SUBP_TPAGROW_UNADJ_GS_FOREST AS TPA_GROW_GS,
    gc.SUBP_TPAMORT_UNADJ_GS_FOREST AS TPA_MORT_GS,
    gc.SUBP_TPAREMV_UNADJ_GS_FOREST AS TPA_REMV_GS,
    -- Condition attributes
    cp.COND_STATUS_CD,
    cp.FORTYPCD,
    cp.OWNGRPCD,
    cp.CONDPROP_UNADJ
FROM TREE_GRM_COMPONENT gc
JOIN TREE_GRM_MIDPT gm  ON gc.TRE_CN = gm.TRE_CN
JOIN PLOT p             ON gc.PLT_CN = p.CN
JOIN strat s            ON gc.PLT_CN = s.PLT_CN
JOIN cond_plot cp       ON gc.PLT_CN = cp.PLT_CN
WHERE ({land_filter})
  AND p.REMPER > 0
  AND p.INVYR >= 2000{remper_filters}{extra_where}
ORDER BY p.STATECD, p.COUNTYCD, gc.PLT_CN, gc.TRE_CN
"""
