"""
SQL query builder for site index estimation.

Computes area-weighted mean site index (SICOND) following
Bechtold & Patterson (2005) methodology.

Formula:
  SITE_INDEX_MEAN = Σ(SICOND × CONDPROP_UNADJ × ADJ_FACTOR_AREA × EXPNS)
                  / Σ(CONDPROP_UNADJ × ADJ_FACTOR_AREA × EXPNS)

Always grouped by SIBASE (site index base age) since SICOND values
are only comparable within the same base age.
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


def site_index_sql(
    db: "FIA",
    grp_by: str | list[str] | None = None,
    land_type: str = "forest",
    area_domain: str | None = None,
    plot_domain: str | None = None,
) -> str:
    """Return a SQL query string that replicates ``site_index()`` estimation.

    Calculates area-weighted mean site index (SICOND) using FIA's
    design-based estimation methods. Site index represents the expected
    dominant tree height (in feet) at a specified base age, indicating
    site productivity.

    Results are always grouped by SIBASE (site index base age) to ensure
    that only comparable SICOND values are averaged together. Site index
    values are not meaningful across different base ages.

    Parameters
    ----------
    db : FIA
        Connected FIA database with EVALID set (used to extract current
        EVALID(s) for the ``strat`` CTE).
    grp_by : str or list of str, optional
        Additional column name(s) from the COND table to group results by
        (beyond SIBASE, which is always included). Common options:

        **Site Index Species:**

        - 'SISP': Species code used for site index determination

        **Forest Characteristics:**

        - 'FORTYPCD': Forest type code
        - 'STDSZCD': Stand size class
        - 'OWNGRPCD': Ownership group

        **Location:**

        - 'STATECD': State FIPS code
        - 'COUNTYCD': County code
        - 'UNITCD': FIA survey unit
    land_type : {'forest', 'timber', 'all'}, default 'forest'
        Land type to include in estimation:

        - 'forest': All forestland (COND_STATUS_CD = 1)
        - 'timber': Timberland only (unreserved, productive with SITECLCD
          in [1-6] and RESERVCD = 0)
        - 'all': All land types including non-forest
    area_domain : str, optional
        SQL-like filter expression on COND columns. Examples:

        - ``"OWNGRPCD == 40"``: Private land only
        - ``"STDAGE > 20"``: Stands over 20 years old
        - ``"FORTYPCD IN (161, 162)"``: Specific forest types
    plot_domain : str, optional
        SQL-like filter expression on PLOT columns. Examples:

        - ``"COUNTYCD == 183"``: Single county
        - ``"LAT >= 35.0 AND LAT <= 36.0"``: Latitude range

    Returns
    -------
    str
        Complete, self-contained SQL query string. Output columns include:

        - **SIBASE** : int — Site index base age (always present)
        - **SITE_INDEX_MEAN** : float — Area-weighted mean site index (feet)
        - **SI_WEIGHTED_SE** : float — Standard error of the weighted mean
        - **AREA_TOTAL** : float — Forest area (acres) in the estimation
        - **N_PLOTS** : int — Number of plots with non-null SICOND
        - **[grouping columns]** : varies — Columns from grp_by

    Notes
    -----
    Site index estimation uses the area-weighted mean formula:

        SITE_INDEX_MEAN = Σ(SICOND × CONDPROP_UNADJ × ADJ_FACTOR × EXPNS)
                        / Σ(CONDPROP_UNADJ × ADJ_FACTOR × EXPNS)

    This ratio-of-means estimator weights each condition's site index by
    its proportional area contribution.

    SIBASE is always included as a grouping column because site index values
    are only comparable within the same base age. Common base ages are 25
    years (southern pines) and 50 years (northern species). Conditions
    without site index (non-productive land, recently disturbed) are excluded.

    See Also
    --------
    area_sql : Estimate forest area (same COND-level stratification)
    tpa_sql : Tree density and basal area estimation

    Examples
    --------
    Basic site index estimation:

    >>> with pyfia.FIA("path/to/fia.duckdb") as db:
    ...     db.clip_by_state(37)
    ...     db.clip_most_recent(eval_type="VOL")
    ...     sql = pyfia.site_index_sql(db)

    Site index by ownership group:

    >>> sql = pyfia.site_index_sql(db, grp_by="OWNGRPCD")

    Site index by site index species:

    >>> sql = pyfia.site_index_sql(db, grp_by="SISP")

    Site index for private timberland:

    >>> sql = pyfia.site_index_sql(
    ...     db,
    ...     land_type="timber",
    ...     area_domain="OWNGRPCD == 40",
    ... )
    """
    evalid_str = _evalid_list(db)

    # SIBASE is always a group column for site index
    group_cols: list[str] = ["SIBASE"]
    if grp_by:
        extra = [grp_by] if isinstance(grp_by, str) else list(grp_by)
        for c in extra:
            if c not in group_cols:
                group_cols.append(c)

    gb = _gb_select(group_cols)
    ggroup = _gb_group(group_cols)

    land_filter = _land_type_sql(land_type, c="c")
    area_domain_sql = _domain_to_sql(area_domain)
    plot_domain_sql = _domain_to_sql(plot_domain)

    extra_where = ""
    if area_domain_sql:
        extra_where += f"\n    AND ({area_domain_sql})"
    if plot_domain_sql:
        extra_where += f"\n    AND ({plot_domain_sql})"

    extra_cond_cols = "c.RESERVCD, c.SITECLCD," if land_type == "timber" else ""

    plot_join_clause = "\n    JOIN PLOT p ON c.PLT_CN = p.CN" if plot_domain_sql else ""

    grp_cond_select = ""
    if group_cols:
        grp_cond_select = ", ".join(f"c.{col}" for col in group_cols) + ","

    query = f"""-- FIA Site Index Estimation SQL
-- Matches SiteIndexEstimator (pyfia) exactly.
-- EVALIDs: {evalid_str}
-- land_type: {land_type}
--
-- SITE_INDEX_MEAN = Σ(SICOND × area_weight × EXPNS) / Σ(area_weight × EXPNS)
-- area_weight = CONDPROP_UNADJ × ADJ_FACTOR_AREA
-- Always grouped by SIBASE (site index base age).
WITH {_strat_cte(evalid_str)},

cond_si AS (
    SELECT
        c.PLT_CN,
        c.CONDID,
        c.CONDPROP_UNADJ,
        c.PROP_BASIS,
        c.SICOND,
        {grp_cond_select}
        {extra_cond_cols}
        {_area_adj_sql("c", "s")} AS ADJ_FACTOR_AREA,
        s.STRATUM_CN,
        s.EXPNS,
        s.STRATUM_WGT,
        s.P2POINTCNT,
        s.ESTN_UNIT_CN,
        s.AREA_USED,
        s.P1PNTCNT_EU
    FROM COND c
    JOIN strat s ON c.PLT_CN = s.PLT_CN{plot_join_clause}
    WHERE {land_filter}
      AND c.CONDPROP_UNADJ IS NOT NULL
      AND c.SICOND IS NOT NULL{extra_where}
),

-- Condition-level weighted sums
-- y_ic = SICOND × CONDPROP_UNADJ × ADJ_FACTOR_AREA  (SI numerator)
-- x_ic = CONDPROP_UNADJ × ADJ_FACTOR_AREA            (area denominator)
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
        {gb}SICOND * CONDPROP_UNADJ * ADJ_FACTOR_AREA AS y_ic,
        CONDPROP_UNADJ * ADJ_FACTOR_AREA              AS x_ic
    FROM cond_si
),

-- Population totals
pop_totals AS (
    SELECT
        {gb}SUM(y_ic * EXPNS)    AS SI_WEIGHTED_TOTAL,
        SUM(x_ic * EXPNS)       AS AREA_TOTAL,
        CASE WHEN SUM(x_ic * EXPNS) > 0
             THEN SUM(y_ic * EXPNS) / SUM(x_ic * EXPNS) ELSE NULL END AS SITE_INDEX_MEAN,
        COUNT(DISTINCT CASE WHEN y_ic > 0 THEN PLT_CN END) AS N_PLOTS
    FROM cond_level
    GROUP BY {', '.join(group_cols)}
),

{_variance_ctes(group_cols)}

SELECT
    {gb}pt.SITE_INDEX_MEAN,
    pt.AREA_TOTAL,
    pt.N_PLOTS,
    {_se_total_expr()} AS SI_WEIGHTED_SE
FROM pop_totals pt
{_final_join(group_cols)}
ORDER BY {', '.join(group_cols)}
"""
    return query
