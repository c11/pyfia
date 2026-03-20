"""
SQL query builder for trees-per-acre (TPA) and basal area (BAA) estimation.

Produces a query that matches TPAEstimator exactly:
  TPA_ACRE = Σ(TPA_UNADJ × ADJ_FACTOR × EXPNS) / Σ(CONDPROP_UNADJ × ADJ_FACTOR_AREA × EXPNS)
  BAA_ACRE = Σ(π×(DIA/24)² × TPA_UNADJ × ADJ_FACTOR × EXPNS) / AREA_DENOM
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

# π / (4 × 144) expressed as a decimal – basal area in sq ft from DIA in inches
_BAA_FACTOR = "0.005454"  # = π/(4×144)


def tpa_sql(
    db: "FIA",
    grp_by: str | list[str] | None = None,
    by_species: bool = False,
    land_type: str = "forest",
    tree_type: str = "live",
    tree_domain: str | None = None,
    area_domain: str | None = None,
    plot_domain: str | None = None,
) -> str:
    """Return a SQL query string that replicates ``tpa()`` estimation.

    Calculates tree density (trees per acre) and basal area per acre using
    FIA's design-based estimation methods. TPA uses ``TPA_UNADJ`` directly;
    BAA is computed as ``π × (DIA/24)² × TPA_UNADJ`` per tree, then
    expanded to population level via the ratio-of-means estimator.

    Parameters
    ----------
    db : FIA
        Connected FIA database with EVALID set (used to extract current
        EVALID(s) for the ``strat`` CTE).
    grp_by : str or list of str, optional
        Column name(s) from the TREE, COND, or PLOT tables to group results by.
        Common options:

        **Tree Attributes:**
        - 'SPCD': Species code (see REF_SPECIES)
        - 'SPGRPCD': Species group code
        - 'TREECLCD': Tree class code (2=Growing stock, 3=Rough cull, 4=Rotten cull)
        - 'STATUSCD': Tree status (1=Live, 2=Dead, 3=Removed)
        - 'CCLCD': Crown class code (1=Open grown, 2=Dominant, 3=Codominant,
          4=Intermediate, 5=Overtopped)

        **Forest Characteristics:**
        - 'FORTYPCD': Forest type code (see REF_FOREST_TYPE)
        - 'STDSZCD': Stand size class (1=Large, 2=Medium, 3=Small,
          4=Seedling/sapling, 5=Nonstocked)
        - 'STDAGE': Stand age in years
        - 'SITECLCD': Site productivity class

        **Ownership and Location:**
        - 'OWNGRPCD': Ownership group (10=National Forest, 20=Other Federal,
          30=State/Local, 40=Private)
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
        - 'gs': Growing stock trees (live trees meeting merchantability
          standards, typically TREECLCD = 2)
        - 'all': All trees regardless of status
    tree_domain : str, optional
        SQL-like filter expression for tree-level attributes. Examples:

        - ``"DIA >= 10.0"``: Trees 10 inches DBH and larger
        - ``"SPCD IN (131, 110)"``: Specific species
        - ``"TREECLCD == 2"``: Growing stock trees only
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

        - **TPA_ACRE** : float — Trees per acre
        - **BAA_ACRE** : float — Basal area per acre (square feet)
        - **TPA_ACRE_SE** : float — Standard error of TPA estimate
        - **BAA_ACRE_SE** : float — Standard error of BAA estimate
        - **TPA_TOTAL** : float — Total trees expanded to population level
        - **BAA_TOTAL** : float — Total basal area expanded to population level
        - **AREA_TOTAL** : float — Total area (acres) in the estimation
        - **N_PLOTS** : int — Number of FIA plots in the estimate
        - **[grouping columns]** : varies — Columns from grp_by / by_species

    Notes
    -----
    The two-stage aggregation is mathematically required for statistically
    valid FIA estimates. Stage 1 sums tree values to plot-condition level;
    Stage 2 applies expansion factors. Skipping stage 1 produces results
    that can be orders of magnitude incorrect.

    FIA uses different plot sizes for different tree sizes — adjustment
    factors correct for this:

    - Microplot (6.8 ft radius): Trees 1.0–4.9" DBH → ``ADJ_FACTOR_MICR``
    - Subplot (24.0 ft radius): Trees 5.0"+ → ``ADJ_FACTOR_SUBP``
    - Macroplot (58.9 ft radius): Trees above breakpoint → ``ADJ_FACTOR_MACR``

    Examples
    --------
    Trees per acre on forestland:

    >>> with pyfia.FIA("path/to/fia.duckdb") as db:
    ...     db.clip_by_state(37)
    ...     db.clip_most_recent(eval_type="VOL")
    ...     sql = pyfia.tpa_sql(db, land_type="forest")

    TPA by species:

    >>> sql = pyfia.tpa_sql(db, by_species=True)

    Large trees only (≥10 inches DBH):

    >>> sql = pyfia.tpa_sql(db, tree_domain="DIA >= 10.0")

    TPA and BAA by ownership on timberland:

    >>> sql = pyfia.tpa_sql(
    ...     db,
    ...     grp_by="OWNGRPCD",
    ...     land_type="timber",
    ...     tree_type="gs",
    ... )
    """
    evalid_str = _evalid_list(db)
    group_cols: list[str] = (
        [grp_by] if isinstance(grp_by, str) else list(grp_by) if grp_by else []
    )
    if by_species and "SPCD" not in group_cols:
        group_cols.append("SPCD")

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
                           "STATUSCD", "AGENTCD", "DECAYCD"}
        grp_cols_select = ", ".join(
            f"t.{c}" if c in tree_level_cols else f"c.{c}" for c in group_cols
        ) + ","

    query = f"""-- FIA Trees-Per-Acre (TPA) and Basal Area (BAA) Estimation SQL
-- Matches TPAEstimator (pyfia) exactly.
-- EVALIDs: {evalid_str}
-- land_type: {land_type}, tree_type: {tree_type}
--
-- TPA_ACRE = Σ(TPA_UNADJ × ADJ_FACTOR × EXPNS) / Σ(CONDPROP × ADJ_AREA × EXPNS)
-- BAA_ACRE = Σ(π×(DIA/24)² × TPA_UNADJ × ADJ_FACTOR × EXPNS) / AREA_DENOM
--   where BAA factor {_BAA_FACTOR} = π/(4×144)  (DIA in inches → sq ft/acre)
WITH {_strat_cte(evalid_str)},

tree_cond AS (
    SELECT
        t.PLT_CN,
        t.CONDID,
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
    JOIN COND c  ON t.PLT_CN = c.PLT_CN AND t.CONDID = c.CONDID
    JOIN strat s ON t.PLT_CN = s.PLT_CN
    JOIN PLOT p  ON t.PLT_CN = p.CN
    WHERE {tree_filter}
      AND {land_filter}
      AND t.TPA_UNADJ > 0
      AND t.DIA IS NOT NULL{extra_where}
),

-- Stage 1: condition-level aggregation
-- TPA y_ic = Σ(TPA_UNADJ × ADJ_FACTOR) per condition
-- BAA y_ic = Σ(π×(DIA/24)² × TPA_UNADJ × ADJ_FACTOR) per condition
-- x_ic     = CONDPROP_UNADJ × ADJ_FACTOR_AREA  (area denominator)
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
        {gb}SUM(TPA_UNADJ * ADJ_FACTOR)                                AS tpa_ic,
        SUM({_BAA_FACTOR} * DIA * DIA * TPA_UNADJ * ADJ_FACTOR)       AS baa_ic,
        MAX(CONDPROP_UNADJ * ADJ_FACTOR_AREA)                         AS x_ic
    FROM tree_cond
    GROUP BY
        PLT_CN, CONDID, STRATUM_CN, EXPNS, STRATUM_WGT, P2POINTCNT,
        ESTN_UNIT_CN, AREA_USED, P1PNTCNT_EU{ggroup}
),

-- Rename y_ic to tpa for variance CTEs (which expect y_ic / x_ic)
cond_level_tpa AS (
    SELECT *, tpa_ic AS y_ic FROM cond_level
),

-- Population totals
pop_totals AS (
    SELECT
        {gb}SUM(tpa_ic * EXPNS) AS TPA_TOTAL,
        SUM(baa_ic * EXPNS)    AS BAA_TOTAL,
        SUM(x_ic * EXPNS)      AS AREA_TOTAL,
        CASE WHEN SUM(x_ic * EXPNS) > 0
             THEN SUM(tpa_ic * EXPNS) / SUM(x_ic * EXPNS) ELSE 0.0 END AS TPA_ACRE,
        CASE WHEN SUM(x_ic * EXPNS) > 0
             THEN SUM(baa_ic * EXPNS) / SUM(x_ic * EXPNS) ELSE 0.0 END AS BAA_ACRE,
        COUNT(DISTINCT CASE WHEN tpa_ic > 0 THEN PLT_CN END) AS N_PLOTS
    FROM cond_level
    GROUP BY {('1' if not group_cols else ', '.join(group_cols))}
),

-- Variance uses TPA metric (y_ic = tpa_ic)
{_variance_ctes(group_cols).replace('FROM cond_level', 'FROM cond_level_tpa')}

SELECT
    {gb}pt.TPA_TOTAL,
    pt.BAA_TOTAL,
    pt.AREA_TOTAL,
    pt.TPA_ACRE,
    pt.BAA_ACRE,
    pt.N_PLOTS,
    {_se_total_expr()} AS TPA_TOTAL_SE,
    {_se_acre_expr()}  AS TPA_ACRE_SE
FROM pop_totals pt
{_final_join(group_cols)}
ORDER BY {('pt.TPA_ACRE DESC' if not group_cols else ', '.join(group_cols))}
"""
    return query
