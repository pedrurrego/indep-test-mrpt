"""MRPT: multi-resolution Pearson test of independence on a ladder of square grids.

Public entry points: :func:`indep_test` (the test), :func:`resolve_ladder` and
:data:`DEFAULT_LADDER` (the resolution ladder), :func:`calculate_copula_values`
and :func:`bin_counts` (pseudo-observations and grid counts).  The harness
wrapper is ``indep_tests.pug_grid.PUGGridTest``.
"""

from src_pug_grid.pug_independence_test import (
    DEFAULT_LADDER,
    Impl,
    Norm,
    indep_test,
    resolve_ladder,
)
from src_pug_grid.utils import bin_counts, calculate_copula_values

__all__ = [
    "DEFAULT_LADDER",
    "Impl",
    "Norm",
    "bin_counts",
    "calculate_copula_values",
    "indep_test",
    "resolve_ladder",
]
