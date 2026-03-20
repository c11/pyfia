"""
SQL query builder for tree volume estimation.

Produces a query that matches VolumeEstimator exactly, implementing:
  VOL_ACRE = Σ(VOL_COL × TPA_UNADJ × ADJ_FACTOR × EXPNS)
           / Σ(CONDPROP_UNADJ × ADJ_FACTOR_AREA × EXPNS)
with B&P (2005) ratio-of-means variance.
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

_VOL_COL_MAP = {
    "net": "VOLCFNET",
    "gross": "VOLCFGRS",
    "sound": "VOLCFSND",
    "sawlog": "VOLBFNET",
}


def volume_sql(
    db: "FIA",
    grp_by: str | list[str] | None = None,
    by_species: bool = False,
    land_type: str = "forest",
    tree_type: str = "live",
    vol_type: str = "net",
    tree_domain: str | None = None,
    area_domain: str | None = None,
    plot_domain: str | None = None,
) -> str:
    """Return a SQL query string that replicates ``volume()`` estimation.

    Calculates volume estimates using FIA's design-based estimation methods
    with proper expansion factors and stratification. The ratio-of-means
    estimator divides total volume by total area, both expanded to population
    level using ``EXPNS`` from ``POP_STRATUM``.

    Parameters
    ----------
    db : FIA
        Connected FIA database with EVALID set (used to extract current
        EVALID(s) for the ``strat`` CTE).
    grp_by : str or list of str, optional
        Column name(s) from the TREE, COND, or PLOT tables to group results by.
        Common options:

        **Tree Characteristics:**
        - 'SPCD': Species code (see REF_SPECIES table)
        - 'SPGRPCD': Species group code (hardwood/softwood groups)
        - 'TREECLCD': Tree class code (2=Growing stock, 3=Rough cull, 4=Rotten cull)
        - 'CCLCD': Crown class code (1=Open grown, 2=Dominant, 3=Codominant,
          4=Intermediate, 5=Overtopped)

        **Ownership and Management:**
        - 'OWNGRPCD': Ownership group (10=National Forest, 20=Other Federal,
          30=State/Local, 40=Private)
        - 'RESERVCD': Reserved status (0=Not reserved, 1=Reserved)

        **Forest Characteristics:**
        - 'FORTYPCD': Forest type code (see REF_FOREST_TYPE)
        - 'STDSZCD': Stand size class
        - 'STDAGE': Stand age in years
        - 'SITECLCD': Site productivity class

        **Location:**
        - 'STATECD': State FIPS code
        - 'UNITCD': FIA survey unit code
        - 'COUNTYCD': County code
        - 'INVYR': Inventory year
    by_species : bool, default False
        If True, group results by species code (SPCD). Equivalent to adding
        'SPCD' to grp_by.
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
        - 'gs': Growing stock trees (live, TREECLCD = 2, no defects)
        - 'all': All trees regardless of status
    vol_type : {'net', 'gross', 'sound', 'sawlog'}, default 'net'
        Volume type to estimate:

        - 'net': Net cubic foot volume (VOLCFNET) — gross minus defects
        - 'gross': Gross cubic foot volume (VOLCFGRS) — total stem volume
        - 'sound': Sound cubic foot volume (VOLCFSND) — gross minus rot
        - 'sawlog': Sawlog board foot volume (VOLBFNET) — net board feet
    tree_domain : str, optional
        SQL-like filter expression for tree-level attributes. Examples:

        - ``"DIA >= 10.0"``: Trees 10 inches DBH and larger
        - ``"SPCD IN (131, 110)"``: Loblolly and Virginia pine only
        - ``"HT > 50 AND CR > 30"``: Tall trees with good crowns
    area_domain : str, optional
        SQL-like filter expression for COND-level attributes. Examples:

        - ``"STDAGE > 50"``: Stands older than 50 years
        - ``"FORTYPCD IN (161, 162)"``: Specific forest types
        - ``"OWNGRPCD == 40"``: Private lands only
    plot_domain : str, optional
        SQL-like filter expression for PLOT-level attributes. Examples:

        - ``"COUNTYCD == 183"``: Wake County, NC
        - ``"LAT >= 35.0 AND LAT <= 36.0"``: Latitude range

    Returns
    -------
    str
        Complete, self-contained SQL query string. Output columns include:

        - **VOLUME_ACRE** : float — Volume per acre (units depend on vol_type)
        - **VOLUME_TOTAL** : float — Total volume expanded to population level
        - **VOLUME_ACRE_SE** : float — Standard error of per-acre estimate
        - **VOLUME_TOTAL_SE** : float — Standard error of total estimate
        - **AREA_TOTAL** : float — Total area (acres) in the estimation
        - **N_PLOTS** : int — Number of FIA plots with non-zero volume
        - **[grouping columns]** : varies — Columns from grp_by / by_species

    Notes
    -----
    Trees are adjusted for FIA's nested plot design based on diameter:

    - Trees with DIA < 5.0 inches: microplot factor (``ADJ_FACTOR_MICR``)
    - Trees 5.0 ≤ DIA < MACRO_BREAKPOINT_DIA: subplot factor (``ADJ_FACTOR_SUBP``)
    - Trees ≥ MACRO_BREAKPOINT_DIA: macroplot factor (``ADJ_FACTOR_MACR``)

    Sawlog volume (``vol_type='sawlog'``) applies only to sawtimber-sized
    trees (softwoods ≥ 9" DBH, hardwoods ≥ 11" DBH). Smaller trees will
    have NULL or 0 board foot volume.

    Examples
    --------
    Net cubic foot volume on forestland:

    >>> with pyfia.FIA("path/to/fia.duckdb") as db:
    ...     db.clip_by_state(37)
    ...     db.clip_most_recent(eval_type="VOL")
    ...     sql = pyfia.volume_sql(db, land_type="forest", vol_type="net")

    Volume by species on timberland:

    >>> sql = pyfia.volume_sql(
    ...     db, by_species=True, land_type="timber", tree_type="gs"
    ... )

    Large-tree sawlog volume by forest type:

    >>> sql = pyfia.volume_sql(
    ...     db,
    ...     grp_by="FORTYPCD",
    ...     vol_type="sawlog",
    ...     tree_type="gs",
    ...     tree_domain="DIA >= 11.0",
    ... )
    """
    evalid_str = _evalid_list(db)
    group_cols: list[str] = (
        [grp_by] if isinstance(grp_by, str) else list(grp_by) if grp_by else []
    )
    if by_species and "SPCD" not in group_cols:
        group_cols.append("SPCD")

    vol_col = _VOL_COL_MAP.get(vol_type, "VOLCFNET")

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

    # Extra COND columns for timber
    extra_cond_cols = "c.RESERVCD, c.SITECLCD," if land_type == "timber" else ""

    # grp_by columns select (from TREE and COND)
    grp_cols_select = ""
    if group_cols:
        grp_cols_select = ", ".join(
            f"t.{c}" if c in ("SPCD", "SPGRPCD", "DIA", "HT", "TREECLCD", "CCLCD",
                              "STATUSCD", "AGENTCD", "DECAYCD")
            else f"c.{c}"
            for c in group_cols
        ) + ","

    query = f"""-- FIA Volume Estimation SQL
-- Matches VolumeEstimator (pyfia) exactly.
-- EVALIDs: {evalid_str}
-- vol_type: {vol_type}, land_type: {land_type}, tree_type: {tree_type}
--
-- Formula: VOL_ACRE = Σ({vol_col} × TPA_UNADJ × ADJ_FACTOR × EXPNS)
--                   / Σ(CONDPROP_UNADJ × ADJ_FACTOR_AREA × EXPNS)
-- ADJ_FACTOR: size-based (microplot/subplot/macroplot) — see FIA nested plot design
-- Variance: Bechtold & Patterson (2005) ratio-of-means formula
WITH {_strat_cte(evalid_str)},

tree_cond AS (
    -- Join TREE × COND × PLOT × stratification, compute adjustment factors
    SELECT
        t.PLT_CN,
        t.CONDID,
        t.{vol_col},
        t.TPA_UNADJ,
        t.DIA,
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
    JOIN COND c   ON t.PLT_CN = c.PLT_CN AND t.CONDID = c.CONDID
    JOIN strat s  ON t.PLT_CN = s.PLT_CN
    JOIN PLOT p   ON t.PLT_CN = p.CN
    WHERE {tree_filter}
      AND {land_filter}
      AND t.TPA_UNADJ > 0
      AND t.{vol_col} IS NOT NULL{extra_where}
),

-- Stage 1: condition-level aggregation
-- y_ic = Σ(vol × TPA × ADJ_FACTOR) per condition  (numerator contribution)
-- x_ic = CONDPROP_UNADJ × ADJ_FACTOR_AREA          (denominator: area)
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
        {gb}SUM({vol_col} * TPA_UNADJ * ADJ_FACTOR) AS y_ic,
        MAX(CONDPROP_UNADJ * ADJ_FACTOR_AREA)        AS x_ic
    FROM tree_cond
    GROUP BY
        PLT_CN, CONDID, STRATUM_CN, EXPNS, STRATUM_WGT, P2POINTCNT,
        ESTN_UNIT_CN, AREA_USED, P1PNTCNT_EU{ggroup}
),

-- Stage 2: population totals
pop_totals AS (
    SELECT
        {gb}SUM(y_ic * EXPNS) AS VOL_TOTAL,
        SUM(x_ic * EXPNS)  AS AREA_TOTAL,
        CASE WHEN SUM(x_ic * EXPNS) > 0
             THEN SUM(y_ic * EXPNS) / SUM(x_ic * EXPNS) ELSE 0.0 END AS VOL_ACRE,
        COUNT(DISTINCT CASE WHEN y_ic > 0 THEN PLT_CN END) AS N_PLOTS
    FROM cond_level
    GROUP BY {('1' if not group_cols else ', '.join(group_cols))}
),

{_variance_ctes(group_cols)}

SELECT
    {gb}pt.VOL_TOTAL,
    pt.AREA_TOTAL,
    pt.VOL_ACRE,
    pt.N_PLOTS,
    {_se_total_expr()} AS {vol_col}_TOTAL_SE,
    {_se_acre_expr()}  AS {vol_col}_ACRE_SE
FROM pop_totals pt
{_final_join(group_cols)}
ORDER BY {('pt.VOL_ACRE DESC' if not group_cols else ', '.join(group_cols))}
"""
    return query
