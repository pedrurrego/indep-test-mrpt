"""Salahub-Oldford recursive random binning independence test.

Provenance: port of the R package ``AssocBin`` (GPL >= 3); see ``binning.py``
and ``pvalues.py``. Distributed under GPL-3 (LICENSE).
"""

from indep_tests.salahub_binning.binning import (
    Bin,
    bin_chi,
    binner,
    rank_random_ties,
)
from indep_tests.salahub_binning.pvalues import (
    fitted_df,
    gamma_params,
    p_value,
    simple_df,
)
from indep_tests.salahub_binning.test import SalahubBinningTest

__all__ = [
    "Bin",
    "SalahubBinningTest",
    "bin_chi",
    "binner",
    "fitted_df",
    "gamma_params",
    "p_value",
    "rank_random_ties",
    "simple_df",
]
