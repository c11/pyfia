"""
SQL query builder for carbon estimation (dispatcher).

Routes to the appropriate SQL builder based on carbon pool:
  - 'ag', 'bg', 'total', 'live' → carbon_pool_sql (uses CARBON_AG / CARBON_BG)
  - 'dead'                       → biomass_sql (47% of standing-dead DRYBIO_TOTAL)

Mirrors the carbon() function dispatcher in estimators/carbon.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .biomass import biomass_sql
from .carbon_pool import carbon_pool_sql

if TYPE_CHECKING:
    from ...core.fia import FIA


def carbon_sql(
    db: "FIA",
    grp_by: str | list[str] | None = None,
    by_species: bool = False,
    pool: str = "live",
    land_type: str = "forest",
    tree_type: str = "live",
    tree_domain: str | None = None,
    area_domain: str | None = None,
    plot_domain: str | None = None,
) -> str:
    """Return a SQL query string that replicates ``carbon()`` estimation.

    Provides unified access to forest carbon estimation across different
    carbon pools. Routes to the appropriate SQL builder based on the pool
    parameter:

    - ``'ag'``, ``'bg'``, ``'total'``, ``'live'``: Uses FIA's pre-calculated
      ``CARBON_AG`` / ``CARBON_BG`` columns (species-specific conversion
      factors). Matches EVALIDator snum=55000 for pool='total'/'live'.
    - ``'dead'``: Uses biomass-derived carbon (``DRYBIO_TOTAL × 0.47``) for
      standing dead trees. FIA does not provide pre-calculated carbon columns
      for dead trees.

    Parameters
    ----------
    db : FIA
        Connected FIA database with EVALID set (used to extract current
        EVALID(s) for the ``strat`` CTE).
    grp_by : str or list of str, optional
        Column name(s) from the TREE, COND, or PLOT tables to group results by.
        Common options include 'FORTYPCD', 'OWNGRPCD', 'STATECD', 'SPCD',
        'COUNTYCD'. For complete column descriptions, see the USDA FIA
        Database User Guide.
    by_species : bool, default False
        If True, group results by species code (SPCD). Only applicable for
        live tree and standing dead pools.
    pool : {'ag', 'bg', 'live', 'dead', 'total'}, default 'live'
        Carbon pool to estimate:

        - 'ag': Aboveground live tree carbon (stems, branches, foliage)
        - 'bg': Belowground live tree carbon (coarse roots)
        - 'live' / 'total': Total live tree carbon (CARBON_AG + CARBON_BG) —
          matches EVALIDator snum=55000
        - 'dead': Standing dead tree carbon (DRYBIO_TOTAL × 0.47)

        Note: 'litter' and 'soil' are not yet implemented and will raise
        a ``ValueError``.
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

        - ``"DIA >= 10.0 AND SPCD == 131"``: Large loblolly pine only
        - ``"DIA >= 5.0"``: Trees 5 inches DBH and larger
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
        Complete, self-contained SQL query string. For live tree pools
        ('ag', 'bg', 'live', 'total'), output columns include:

        - **CARBON_ACRE** : float — Carbon per acre (short tons/acre)
        - **CARBON_TOTAL** : float — Total carbon (population level, short tons)
        - **CARBON_ACRE_SE** : float — Standard error of per-acre estimate
        - **CARBON_TOTAL_SE** : float — Standard error of total estimate
        - **AREA_TOTAL** : float — Total area (acres) in the estimation
        - **N_PLOTS** : int — Number of FIA plots in the estimate
        - **[grouping columns]** : varies — Columns from grp_by / by_species

        For dead tree carbon ('dead'), the output columns match ``biomass_sql``
        (BIO_ACRE, CARB_ACRE, BIO_TOTAL, CARB_TOTAL, and their SEs).

    Raises
    ------
    ValueError
        If pool is 'litter' or 'soil' (not yet implemented in pyFIA).

    Notes
    -----
    Live tree carbon pools ('ag', 'bg', 'live', 'total') use FIA's
    pre-calculated CARBON_AG and CARBON_BG columns which incorporate
    species-specific carbon conversion factors. This is more accurate than
    applying a flat 47% carbon fraction and matches EVALIDator snum=55000.

    Dead tree carbon currently uses biomass-derived estimation (DRYBIO_TOTAL
    × 0.47) as FIA does not provide pre-calculated carbon for dead trees.

    See Also
    --------
    carbon_pool_sql : Direct access to the CARBON_AG/CARBON_BG-based estimator
    biomass_sql : Biomass estimation (also outputs CARB_ACRE = BIO_ACRE × 0.47)
    carbon_flux_sql : Net carbon flux (growth - mortality - removals)

    Examples
    --------
    Basic live tree carbon estimation (matches EVALIDator):

    >>> with pyfia.FIA("path/to/fia.duckdb") as db:
    ...     db.clip_by_state(37)
    ...     db.clip_most_recent(eval_type="VOL")
    ...     sql = pyfia.carbon_sql(db, pool="live")

    Aboveground carbon by ownership group:

    >>> sql = pyfia.carbon_sql(db, pool="ag", grp_by="OWNGRPCD")

    Standing dead tree carbon:

    >>> sql = pyfia.carbon_sql(db, pool="dead")
    """
    pool_norm = pool.lower()
    if pool_norm in ("litter", "soil"):
        raise ValueError(
            f"Carbon pool '{pool}' is not yet implemented in pyFIA. "
            "Supported pools: 'ag', 'bg', 'live', 'total', 'dead'."
        )

    if pool_norm == "dead":
        # Standing dead tree carbon: biomass × 0.47
        # biomass_sql already outputs CARBON_ACRE = BIOMASS_ACRE × 0.47
        return biomass_sql(
            db=db,
            grp_by=grp_by,
            by_species=by_species,
            component="TOTAL",
            land_type=land_type,
            tree_type="dead",
            tree_domain=tree_domain,
            area_domain=area_domain,
            plot_domain=plot_domain,
        )

    # Live tree pools: 'ag', 'bg', 'live' (alias for 'total'), 'total'
    pool_mapped = "total" if pool_norm == "live" else pool_norm
    return carbon_pool_sql(
        db=db,
        grp_by=grp_by,
        by_species=by_species,
        pool=pool_mapped,
        land_type=land_type,
        tree_type=tree_type,
        tree_domain=tree_domain,
        area_domain=area_domain,
        plot_domain=plot_domain,
    )
