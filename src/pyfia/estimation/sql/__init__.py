"""
SQL query builders for all FIA estimators.

Each function takes the same parameters as its Python counterpart and
returns a SQL string that can be executed directly against FIADB tables
(DuckDB or compatible) to reproduce the identical statistical estimates.

All queries implement Bechtold & Patterson (2005) two-stage post-stratified
estimation with ratio-of-means variance.
"""

from .area import area_sql
from .area_change import area_change_sql
from .biomass import biomass_sql
from .carbon import carbon_sql
from .carbon_flux import carbon_flux_sql
from .carbon_pool import carbon_pool_sql
from .growth import growth_sql
from .mortality import mortality_sql
from .panel import panel_sql
from .removals import removals_sql
from .site_index import site_index_sql
from .tpa import tpa_sql
from .tree_metrics import tree_metrics_sql
from .volume import volume_sql

__all__ = [
    "area_sql",
    "area_change_sql",
    "biomass_sql",
    "carbon_sql",
    "carbon_flux_sql",
    "carbon_pool_sql",
    "growth_sql",
    "mortality_sql",
    "panel_sql",
    "removals_sql",
    "site_index_sql",
    "tpa_sql",
    "tree_metrics_sql",
    "volume_sql",
]
