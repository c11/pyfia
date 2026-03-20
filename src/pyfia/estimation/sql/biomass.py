"""
SQL query builder for biomass and carbon estimation.

Matches BiomassEstimator exactly:
  BIOMASS_ACRE = Σ(DRYBIO × TPA_UNADJ × ADJ_FACTOR × LBS_TO_TONS × EXPNS) / AREA
  CARBON_ACRE  = BIOMASS_ACRE × 0.47
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
_CARBON_FRAC = "0.47"


def biomass_sql(
    db: "FIA",
    grp_by: str | list[str] | None = None,
    by_species: bool = False,
    component: str = "AG",
    land_type: str = "forest",
    tree_type: str = "live",
    tree_domain: str | None = None,
    area_domain: str | None = None,
    plot_domain: str | None = None,
) -> str:
    """Return a SQL query string that replicates ``biomass()`` estimation.

    Calculates dry weight biomass (in short tons) and carbon content using
    FIA's standard DRYBIO_* columns and expansion factors. Implements the
    two-stage aggregation following FIA methodology.

    Carbon is estimated as 47% of dry biomass (``DRYBIO × 0.47``), following
    IPCC guidelines. For species-specific carbon, use ``carbon_pool_sql``
    instead, which uses FIA's pre-calculated ``CARBON_AG``/``CARBON_BG``
    columns and matches EVALIDator snum=55000 exactly.

    Parameters
    ----------
    db : FIA
        Connected FIA database with EVALID set (used to extract current
        EVALID(s) for the ``strat`` CTE).
    grp_by : str or list of str, optional
        Column name(s) from the TREE, COND, or PLOT tables to group results by.
        Common options include 'FORTYPCD', 'OWNGRPCD', 'STATECD', 'SPCD',
        'COUNTYCD', 'INVYR', 'STDAGE', 'SITECLCD'. For complete column
        descriptions, see the USDA FIA Database User Guide.
    by_species : bool, default False
        If True, group results by species code (SPCD). Equivalent to adding
        'SPCD' to grp_by.
    component : str, default 'AG'
        Biomass component to estimate. Valid options:

        - 'AG': Aboveground biomass (DRYBIO_AG — stem, bark, branches, foliage)
        - 'BG': Belowground biomass (DRYBIO_BG — coarse roots)
        - 'TOTAL': Total biomass (DRYBIO_AG + DRYBIO_BG)
        - 'BOLE': Main stem wood and bark (DRYBIO_BOLE)
        - 'BRANCH': Live and dead branches (DRYBIO_BRANCH)
        - 'FOLIAGE': Leaves/needles (DRYBIO_FOLIAGE)
        - 'STUMP': Stump biomass (DRYBIO_STUMP)

        Not all components are available for all species or FIA regions.
    land_type : {'forest', 'timber', 'all'}, default 'forest'
        Land type to include in estimation:

        - 'forest': All forestland (COND_STATUS_CD = 1)
        - 'timber': Timberland only (unreserved, productive forestland)
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
        - ``"DIA >= 20.0 AND TREECLCD == 2"``: Large growing-stock trees
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

        - **BIO_ACRE** : float — Biomass per acre (short tons dry weight)
        - **BIO_TOTAL** : float — Total biomass expanded to population level (short tons)
        - **CARB_ACRE** : float — Carbon per acre (47% of biomass, short tons)
        - **CARB_TOTAL** : float — Total carbon expanded to population level (short tons)
        - **BIO_ACRE_SE** : float — Standard error of per-acre biomass
        - **BIO_TOTAL_SE** : float — Standard error of total biomass
        - **CARB_ACRE_SE** : float — Standard error of per-acre carbon
        - **CARB_TOTAL_SE** : float — Standard error of total carbon
        - **AREA_TOTAL** : float — Total area (acres) in the estimation
        - **N_PLOTS** : int — Number of FIA plots in the estimate
        - **[grouping columns]** : varies — Columns from grp_by / by_species

    Notes
    -----
    Biomass is calculated from FIA's DRYBIO_* columns (in pounds) and
    converted to short tons by multiplying by 0.0005 (= 1/2000).

    For live tree carbon that matches EVALIDator snum=55000 exactly, use
    ``carbon_pool_sql(pool='total')`` instead. It uses FIA's pre-calculated
    ``CARBON_AG``/``CARBON_BG`` columns with species-specific conversion
    factors rather than the flat 47% fraction.

    Examples
    --------
    Aboveground biomass on forestland:

    >>> with pyfia.FIA("path/to/fia.duckdb") as db:
    ...     db.clip_by_state(37)
    ...     db.clip_most_recent(eval_type="VOL")
    ...     sql = pyfia.biomass_sql(db, component="AG", land_type="forest")

    Total biomass (AG + BG) by species:

    >>> sql = pyfia.biomass_sql(db, by_species=True, component="TOTAL")

    Standing dead tree biomass:

    >>> sql = pyfia.biomass_sql(
    ...     db, tree_type="dead", component="AG", grp_by="FORTYPCD"
    ... )
    """
    evalid_str = _evalid_list(db)
    group_cols: list[str] = (
        [grp_by] if isinstance(grp_by, str) else list(grp_by) if grp_by else []
    )
    if by_species and "SPCD" not in group_cols:
        group_cols.append("SPCD")

    component = component.upper()

    # Biomass column expression based on component
    if component == "AG":
        biomass_expr = "t.DRYBIO_AG"
    elif component == "BG":
        biomass_expr = "t.DRYBIO_BG"
    elif component == "TOTAL":
        biomass_expr = "(t.DRYBIO_AG + t.DRYBIO_BG)"
    else:
        biomass_expr = f"t.DRYBIO_{component}"

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

    grp_cols_select = ""
    if group_cols:
        tree_level_cols = {"SPCD", "SPGRPCD", "DIA", "HT", "TREECLCD", "CCLCD",
                           "STATUSCD", "AGENTCD"}
        grp_cols_select = ", ".join(
            f"t.{c}" if c in tree_level_cols else f"c.{c}" for c in group_cols
        ) + ","

    query = f"""-- FIA Biomass / Carbon Estimation SQL
-- Matches BiomassEstimator (pyfia) exactly.
-- EVALIDs: {evalid_str}
-- component: {component}, land_type: {land_type}, tree_type: {tree_type}
--
-- BIOMASS_ACRE = Σ({biomass_expr} × TPA_UNADJ × ADJ_FACTOR × {_LBS_TO_TONS} × EXPNS)
--             / Σ(CONDPROP_UNADJ × ADJ_FACTOR_AREA × EXPNS)
-- CARBON_ACRE  = BIOMASS_ACRE × {_CARBON_FRAC}
WITH {_strat_cte(evalid_str)},

tree_cond AS (
    SELECT
        t.PLT_CN,
        t.CONDID,
        t.TPA_UNADJ,
        t.DIA,
        t.DRYBIO_AG,
        t.DRYBIO_BG,
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
    JOIN strat s ON t.PLT_CN = s.PLT_CN
    JOIN PLOT p  ON t.PLT_CN = p.CN
    WHERE {tree_filter}
      AND {land_filter}
      AND t.TPA_UNADJ > 0
      AND t.DIA IS NOT NULL{extra_where}
),

-- Stage 1: condition-level aggregation
-- y_ic = Σ(biomass_component × TPA × ADJ × LBS_TO_TONS) per condition
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
        {gb}SUM({biomass_expr} * TPA_UNADJ * ADJ_FACTOR * {_LBS_TO_TONS}) AS y_ic,
        MAX(CONDPROP_UNADJ * ADJ_FACTOR_AREA)                              AS x_ic
    FROM tree_cond
    GROUP BY
        PLT_CN, CONDID, STRATUM_CN, EXPNS, STRATUM_WGT, P2POINTCNT,
        ESTN_UNIT_CN, AREA_USED, P1PNTCNT_EU{ggroup}
),

-- Population totals
pop_totals AS (
    SELECT
        {gb}SUM(y_ic * EXPNS) AS BIOMASS_TOTAL,
        SUM(x_ic * EXPNS)    AS AREA_TOTAL,
        CASE WHEN SUM(x_ic * EXPNS) > 0
             THEN SUM(y_ic * EXPNS) / SUM(x_ic * EXPNS) ELSE 0.0 END AS BIOMASS_ACRE,
        CASE WHEN SUM(x_ic * EXPNS) > 0
             THEN SUM(y_ic * EXPNS) / SUM(x_ic * EXPNS) * {_CARBON_FRAC} ELSE 0.0 END AS CARBON_ACRE,
        SUM(y_ic * EXPNS) * {_CARBON_FRAC}                            AS CARBON_TOTAL,
        COUNT(DISTINCT CASE WHEN y_ic > 0 THEN PLT_CN END)            AS N_PLOTS
    FROM cond_level
    GROUP BY {('1' if not group_cols else ', '.join(group_cols))}
),

{_variance_ctes(group_cols)}

SELECT
    {gb}pt.BIOMASS_TOTAL,
    pt.CARBON_TOTAL,
    pt.AREA_TOTAL,
    pt.BIOMASS_ACRE,
    pt.CARBON_ACRE,
    pt.N_PLOTS,
    {_se_total_expr()} AS BIOMASS_TOTAL_SE,
    {_se_acre_expr()}  AS BIOMASS_ACRE_SE
FROM pop_totals pt
{_final_join(group_cols)}
ORDER BY {('pt.BIOMASS_ACRE DESC' if not group_cols else ', '.join(group_cols))}
"""
    return query
