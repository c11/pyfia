"""
SQL query builder for removals estimation.

Wraps the shared GRM SQL builder for the removals component:
  COMPONENT LIKE 'CUT%' OR COMPONENT LIKE 'OTHER_REMOVAL%'
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._grm_base import _grm_sql

if TYPE_CHECKING:
    from ...core.fia import FIA


def removals_sql(
    db: "FIA",
    grp_by: str | list[str] | None = None,
    by_species: bool = False,
    tree_type: str = "gs",
    land_type: str = "forest",
    measure: str = "volume",
    tree_domain: str | None = None,
    area_domain: str | None = None,
) -> str:
    """Return a SQL query string that replicates ``removals()`` estimation.

    Calculates average annual removals of trees using FIA's
    Growth-Removal-Mortality (GRM) tables following EVALIDator methodology.
    Only rows where ``COMPONENT LIKE 'CUT%' OR COMPONENT LIKE
    'OTHER_REMOVAL%'`` contribute — these are trees harvested or otherwise
    removed from the forest during the measurement period.

    Parameters
    ----------
    db : FIA
        Connected FIA database with EVALID set. The EVALID filter is applied
        through the ``strat`` CTE join — GRM tables carry no EVALID column.
        Use ``db.clip_by_evalid()`` or ``db.clip_most_recent(eval_type='GROW')``
        before calling this function.
    grp_by : str or list of str, optional
        Column name(s) to group results by. Columns that come from
        ``TREE_GRM_MIDPT`` (e.g. 'SPCD', 'DIA_MIDPT') are prefixed ``gm.``
        in the CTE; all other columns are assumed to come from the ``cond_plot``
        aggregate CTE. Common options include 'SPCD', 'FORTYPCD', 'OWNGRPCD',
        'STATECD', 'COUNTYCD'.
    by_species : bool, default False
        If True, group results by species code (SPCD). Equivalent to adding
        'SPCD' to grp_by.
    tree_type : {'gs', 'al', 'live', 'sawtimber', 'sl'}, default 'gs'
        Tree type, mapped to GRM column suffix:

        - 'gs': Growing stock (``GS`` columns, DIA_MIDPT ≥ 5.0" filter applied)
        - 'al' / 'live': All live trees (``AL`` columns)
        - 'sawtimber' / 'sl': Sawtimber-sized trees (``SL`` columns)
    land_type : {'forest', 'timber'}, default 'forest'
        Land type, mapped to GRM column suffix:

        - 'forest': All forestland (``FOREST`` columns, COND_STATUS_CD = 1)
        - 'timber': Timberland only (``TIMBER`` columns, unreserved productive)
    measure : {'volume', 'biomass', 'basal_area', 'tpa', 'count'}, default 'volume'
        What to measure in the removals estimation:

        - 'volume': Net cubic foot volume (VOLCFNET from TREE_GRM_MIDPT)
        - 'biomass': Dry weight biomass (DRYBIO_BOLE + DRYBIO_BRANCH, short tons)
        - 'basal_area': Basal area (square feet per acre)
        - 'tpa' / 'count': Trees per acre
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

        - **REMOVALS_ACRE** : float — Annual removals per acre
        - **REMOVALS_TOTAL** : float — Total annual removals (population level)
        - **REMOVALS_ACRE_SE** : float — Standard error of per-acre estimate
        - **REMOVALS_TOTAL_SE** : float — Standard error of total estimate
        - **AREA_TOTAL** : float — Forest area (acres) in the estimation
        - **N_PLOTS** : int — Number of plots with non-zero removals
        - **[grouping columns]** : varies — Columns from grp_by / by_species

    Notes
    -----
    Removals include trees cut (CUT components) or otherwise removed from
    the forest inventory, including those diverted to non-forest land use
    (OTHER_REMOVAL components). The GRM tables contain pre-calculated annual
    removals values that are already annualized by FIA.

    The EVALID filter is applied exclusively through the ``strat`` CTE, which
    joins ``POP_PLOT_STRATUM_ASSGN`` × ``POP_STRATUM WHERE EVALID IN (...)``.
    Use ``eval_type='GROW'`` when clipping, as growth/removals/mortality
    evaluations share the same GROW evaluation type.

    See Also
    --------
    growth_sql : Annual tree growth using SURVIVOR components
    mortality_sql : Annual tree mortality using MORTALITY components
    carbon_flux_sql : Combined net carbon flux (growth - mortality - removals)

    Examples
    --------
    Volume removals on forestland:

    >>> with pyfia.FIA("path/to/fia.duckdb") as db:
    ...     db.clip_by_state(37)
    ...     db.clip_most_recent(eval_type="GROW")
    ...     sql = pyfia.removals_sql(db, measure="volume", land_type="forest")

    Removals by species (tree count):

    >>> sql = pyfia.removals_sql(db, by_species=True, measure="tpa")

    Biomass removals by forest type:

    >>> sql = pyfia.removals_sql(
    ...     db, grp_by="FORTYPCD", measure="biomass", land_type="forest"
    ... )
    """
    return _grm_sql(
        component_type="removals",
        db=db,
        grp_by=grp_by,
        by_species=by_species,
        tree_type=tree_type,
        land_type=land_type,
        measure=measure,
        tree_domain=tree_domain,
        area_domain=area_domain,
    )
