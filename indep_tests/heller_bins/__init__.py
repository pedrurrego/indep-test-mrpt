"""Heller-Heller-Kaufman-Brill-Gorfine (2016) ADP binning independence test.

A faithful port of the ADP ("all derived partitions") sum statistic from the
authors' R package ``HHG`` (``barakbri/HHG``), with the ``ADP-EQP`` atom
acceleration and the ``MinP`` resolution-combining rule. See ``statistic.py`` for
the mathematics and ``test.py`` for the harness wrapper.

Provenance: port of the ADP statistic of the R package ``HHG`` (GPL-3); see
``statistic.py``. Distributed under GPL-3 (LICENSE).
"""

from indep_tests.heller_bins.statistic import (
    ADPGrid,
    nr_bins_equipartition,
    rank_ordinal,
    ranks_to_atoms,
)
from indep_tests.heller_bins.test import HellerBinsTest

__all__ = [
    "ADPGrid",
    "HellerBinsTest",
    "nr_bins_equipartition",
    "rank_ordinal",
    "ranks_to_atoms",
]
