"""
SQL query builder for derived tree metrics.

Computes TPA-weighted sample-level descriptive statistics.
These are NOT population estimates - no expansion factors or variance.

Metrics:
  QMD         = SQRT(Σ(DIA² × TPA_UNADJ) / Σ(TPA_UNADJ))
  MEAN_DIA    = Σ(DIA × TPA_UNADJ) / Σ(TPA_UNADJ)
  MEAN_HT     = Σ(HT × TPA_UNADJ) / Σ(TPA_UNADJ)   [where HT IS NOT NULL]
  SOFTWOOD_PROP = Σ(DRYBIO_BOLE where SPCD < 300) / Σ(DRYBIO_BOLE)
  SAWTIMBER_PROP = Σ(TPA_UNADJ where DIA >= threshold) / Σ(TPA_UNADJ)
  MAX_DIA     = MAX(DIA)
  STOCKING    = Σ(TPA_UNADJ × (DIA / 10)^1.6)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    _domain_to_sql,
    _evalid_list,
    _gb_group,
    _gb_select,
    _land_type_sql,
    _strat_cte,
    _tree_type_sql,
)

if TYPE_CHECKING:
    from ...core.fia import FIA

_VALID_METRICS = frozenset(
    ["qmd", "mean_dia", "mean_height", "softwood_prop",
     "sawtimber_prop", "max_dia", "stocking"]
)


def _metric_select_exprs(metrics: list[str], sawtimber_threshold: float) -> str:
    """Build SELECT expressions for each requested metric."""
    parts: list[str] = []

    if "qmd" in metrics:
        parts.append(
            "SQRT(SUM(DIA * DIA * TPA_UNADJ) / NULLIF(SUM(TPA_UNADJ), 0)) AS QMD"
        )
    if "mean_dia" in metrics:
        parts.append(
            "SUM(DIA * TPA_UNADJ) / NULLIF(SUM(TPA_UNADJ), 0) AS MEAN_DIA"
        )
    if "mean_height" in metrics:
        parts.append(
            "SUM(CASE WHEN HT IS NOT NULL THEN HT * TPA_UNADJ ELSE 0.0 END) "
            "/ NULLIF(SUM(CASE WHEN HT IS NOT NULL THEN TPA_UNADJ ELSE 0.0 END), 0) AS MEAN_HT"
        )
    if "softwood_prop" in metrics:
        parts.append(
            "SUM(CASE WHEN SPCD < 300 THEN DRYBIO_BOLE ELSE 0.0 END) "
            "/ NULLIF(SUM(DRYBIO_BOLE), 0) AS SOFTWOOD_PROP"
        )
    if "sawtimber_prop" in metrics:
        parts.append(
            f"SUM(CASE WHEN DIA >= {sawtimber_threshold} THEN TPA_UNADJ ELSE 0.0 END) "
            "/ NULLIF(SUM(TPA_UNADJ), 0) AS SAWTIMBER_PROP"
        )
    if "max_dia" in metrics:
        parts.append("MAX(DIA) AS MAX_DIA")
    if "stocking" in metrics:
        parts.append(
            "SUM(TPA_UNADJ * POW(DIA / 10.0, 1.6)) AS STOCKING"
        )

    return ("\n        " + ",\n        ".join(parts) + ",") if parts else ""


def tree_metrics_sql(
    db: "FIA",
    metrics: list[str],
    grp_by: str | list[str] | None = None,
    land_type: str = "forest",
    tree_type: str = "live",
    tree_domain: str | None = None,
    area_domain: str | None = None,
    sawtimber_threshold: float = 9.0,
) -> str:
    """Return a SQL query string that replicates ``tree_metrics()`` estimation.

    Calculates TPA-weighted sample-level descriptive tree metrics such as
    quadratic mean diameter (QMD), mean height, and species composition.
    These are sample-level descriptive statistics — they do not use
    expansion factors or produce variance estimates.

    Parameters
    ----------
    db : FIA
        Connected FIA database with EVALID set (used to filter plots
        through the ``strat`` CTE).
    metrics : list of str
        Metrics to compute. Valid options:

        - ``"qmd"``: Quadratic mean diameter —
          ``SQRT(Σ(DIA² × TPA_UNADJ) / Σ(TPA_UNADJ))``
        - ``"mean_dia"``: Arithmetic mean diameter (TPA-weighted) —
          ``Σ(DIA × TPA_UNADJ) / Σ(TPA_UNADJ)``
        - ``"mean_height"``: Mean tree height (TPA-weighted) —
          ``Σ(HT × TPA_UNADJ) / Σ(TPA_UNADJ)`` (excludes NULL heights)
        - ``"softwood_prop"``: Softwood proportion of bole biomass —
          ``Σ(DRYBIO_BOLE where SPCD < 300) / Σ(DRYBIO_BOLE)``
        - ``"sawtimber_prop"``: Proportion of TPA above sawtimber threshold —
          ``Σ(TPA_UNADJ where DIA >= threshold) / Σ(TPA_UNADJ)``
        - ``"max_dia"``: Maximum tree diameter — ``MAX(DIA)``
        - ``"stocking"``: Rough stocking index —
          ``Σ(TPA_UNADJ × (DIA / 10)^1.6)``
    grp_by : str or list of str, optional
        Column name(s) from the TREE or COND tables to group results by.
        Supports standard FIA columns (FORTYPCD, STDAGE, OWNGRPCD, etc.)
        and plot-condition level grouping (PLT_CN, CONDID).
    land_type : {'forest', 'timber', 'all'}, default 'forest'
        Land type to include in estimation:

        - 'forest': All forestland (COND_STATUS_CD = 1)
        - 'timber': Timberland only (unreserved, productive)
        - 'all': All land types including non-forest
    tree_type : {'live', 'dead', 'gs', 'all'}, default 'live'
        Tree type to include:

        - 'live': All live trees (STATUSCD = 1)
        - 'dead': Standing dead trees (STATUSCD = 2)
        - 'gs': Growing stock trees (live, TREECLCD = 2)
        - 'all': All trees regardless of status
    tree_domain : str, optional
        SQL-like filter expression for tree-level attributes. Examples:

        - ``"DIA >= 5.0"``: Trees 5 inches DBH and larger
        - ``"SPCD == 131"``: Loblolly pine only
    area_domain : str, optional
        SQL-like filter expression for COND-level attributes. Examples:

        - ``"FORTYPCD IN (161, 162)"``: Specific forest types
        - ``"OWNGRPCD == 40"``: Private lands only
    sawtimber_threshold : float, default 9.0
        DIA threshold (inches) for the ``sawtimber_prop`` metric. FIA
        convention is 9.0" for softwoods and 11.0" for hardwoods.

    Returns
    -------
    str
        Complete, self-contained SQL query string. Output columns include:

        - **QMD** : float — Quadratic mean diameter (if requested)
        - **MEAN_DIA** : float — Arithmetic mean diameter (if requested)
        - **MEAN_HT** : float — Mean height in feet (if requested)
        - **SOFTWOOD_PROP** : float — Softwood biomass proportion (if requested)
        - **SAWTIMBER_PROP** : float — Sawtimber TPA proportion (if requested)
        - **MAX_DIA** : float — Maximum diameter (if requested)
        - **STOCKING** : float — Stocking index (if requested)
        - **N_PLOTS** : int — Number of plots included
        - **N_TREES** : int — Number of tree records included
        - **[grouping columns]** : varies — Columns from grp_by

    Raises
    ------
    ValueError
        If any metric name in ``metrics`` is not in the valid set.

    Notes
    -----
    Unlike all other SQL estimators, these are TPA-weighted sample
    statistics — they do **not** use expansion factors (EXPNS) and do
    **not** produce standard errors. The strat CTE is still used to
    restrict the plot set to the active EVALID, but no EXPNS weighting
    is applied.

    For population-level estimates (trees per acre, volume per acre),
    use ``tpa_sql`` or ``volume_sql`` instead.

    See Also
    --------
    tpa_sql : Population-level tree density (TPA) and basal area estimates
    volume_sql : Population-level volume estimates

    Examples
    --------
    QMD and mean height by forest type:

    >>> with pyfia.FIA("path/to/fia.duckdb") as db:
    ...     db.clip_by_state(37)
    ...     db.clip_most_recent(eval_type="VOL")
    ...     sql = pyfia.tree_metrics_sql(
    ...         db, metrics=["qmd", "mean_height"], grp_by="FORTYPCD"
    ...     )

    Condition-level metrics for timber valuation:

    >>> sql = pyfia.tree_metrics_sql(
    ...     db,
    ...     metrics=["qmd", "mean_height", "softwood_prop", "sawtimber_prop"],
    ...     grp_by=["FORTYPCD", "OWNGRPCD"],
    ...     land_type="timber",
    ...     tree_domain="DIA >= 1.0",
    ... )
    """
    unknown = [m for m in metrics if m not in _VALID_METRICS]
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {unknown}. Valid: {sorted(_VALID_METRICS)}"
        )

    evalid_str = _evalid_list(db)
    group_cols: list[str] = (
        [grp_by] if isinstance(grp_by, str) else list(grp_by) if grp_by else []
    )

    gb = _gb_select(group_cols)
    ggroup = _gb_group(group_cols)

    land_filter = _land_type_sql(land_type, c="c")
    tree_filter = _tree_type_sql(tree_type, t="t")
    tree_domain_sql = _domain_to_sql(tree_domain)
    area_domain_sql = _domain_to_sql(area_domain)

    extra_where = ""
    if tree_domain_sql:
        extra_where += f"\n    AND ({tree_domain_sql})"
    if area_domain_sql:
        extra_where += f"\n    AND ({area_domain_sql})"

    extra_cond_cols = "c.RESERVCD, c.SITECLCD," if land_type == "timber" else ""

    tree_level_cols = {"SPCD", "SPGRPCD", "DIA", "HT", "TREECLCD", "CCLCD",
                       "STATUSCD", "AGENTCD", "DECAYCD"}
    grp_cols_select = ""
    if group_cols:
        grp_cols_select = ", ".join(
            f"t.{c}" if c in tree_level_cols else f"c.{c}" for c in group_cols
        ) + ","

    # Determine which extra tree columns are needed
    needs_ht = "mean_height" in metrics
    needs_drybio = "softwood_prop" in metrics
    ht_col = "t.HT," if needs_ht else ""
    drybio_col = "t.DRYBIO_BOLE," if needs_drybio else ""

    metric_exprs = _metric_select_exprs(metrics, sawtimber_threshold)

    query = f"""-- FIA Tree Metrics SQL (sample-level, no expansion)
-- Matches TreeMetricsEstimator (pyfia) exactly.
-- EVALIDs: {evalid_str}
-- land_type: {land_type}, tree_type: {tree_type}
-- metrics: {metrics}
--
-- NOTE: These are TPA-weighted sample statistics, NOT population estimates.
-- No EXPNS expansion. No standard errors. Use tpa_sql/volume_sql for estimates.
WITH {_strat_cte(evalid_str)},

tree_cond AS (
    SELECT
        t.PLT_CN,
        t.CONDID,
        t.TPA_UNADJ,
        t.DIA,
        t.SPCD,
        {ht_col}
        {drybio_col}
        {grp_cols_select}
        {extra_cond_cols}
        c.CONDPROP_UNADJ
    FROM TREE t
    JOIN COND c  ON t.PLT_CN = c.PLT_CN AND t.CONDID = c.CONDID
    JOIN strat s ON t.PLT_CN = s.PLT_CN
    WHERE {tree_filter}
      AND {land_filter}
      AND t.TPA_UNADJ > 0
      AND t.DIA IS NOT NULL{extra_where}
)

SELECT
    {gb}{metric_exprs}
    COUNT(DISTINCT PLT_CN) AS N_PLOTS,
    COUNT(*)               AS N_TREES
FROM tree_cond
{('GROUP BY ' + ', '.join(group_cols)) if group_cols else ''}
{('ORDER BY ' + ', '.join(group_cols)) if group_cols else ''}
"""
    return query
