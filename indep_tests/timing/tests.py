"""The Section-4 benchmark tests, instantiated exactly as in the manuscript's runs.

Labels match the legend labels of the paper figures (``paper_figures.config``).
``paper_tests()`` builds the *cold* roster: tests that cache a distribution-free
null (QAD, Heller-ADP, Cmiel-Ledwina) are built with ``cache_null=False`` so a
timed call pays its own null.  ``warm_tests()`` builds the *reused-null*
variants of the four tests whose null law depends on ``n`` only (MRPT, QAD,
Heller-ADP, Cmiel-Ledwina): after one untimed call fills the cache, a timed
call is the statistic plus a lookup in the stored null table.

Single-threaded BLAS is enforced by ``__main__.py`` (which re-executes itself
with the thread environment variables set *before* numpy loads); nothing here
touches the environment.
"""

from __future__ import annotations

import numpy as np

from indep_tests.base import IndependenceTest, TestResult
from indep_tests.pug_grid import PUGGridTest


class WarmPUGGridTest(PUGGridTest):
    """``density-multi`` with a cached distribution-free permutation null.

    Timing-only counterpart of the ``cache_null=True`` mode of QAD, Heller-ADP
    and Cmiel-Ledwina: the ``B`` permuted integer statistics per rung are drawn
    once per ``(n, ladder, B)`` from a deterministic seed and reused by every
    later call, which then costs the observed statistic plus ``|K(n)|`` vector
    comparisons.  Under independence the permutation null of the rank statistic
    depends on ``n`` only, so the table is valid for any continuous sample of
    the same size.  The p-value arithmetic mirrors
    ``src_pug_grid.pug_independence_test._indep_test_multi_density`` exactly
    (integer comparison, ``>=`` ties, Monte-Carlo ``+1``, Bonferroni).
    """

    _NULL_CACHE: dict[tuple[int, str, int], dict[int, np.ndarray]] = {}

    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        B: int = 500,
        seed: int | None = None,
    ) -> TestResult:
        from src_pug_grid.pug_independence_test import (
            _density_from_int,
            _density_int,
            _permutation_null,
            resolve_ladder,
        )
        from src_pug_grid.utils import calculate_copula_values

        del seed  # the cached null is seeded from its configuration
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        n = len(x)
        k_values = resolve_ladder(self.ladder, n)
        u, v = calculate_copula_values(np.column_stack((x, y)))
        S_obs = {k: _density_int(u, v, k) for k in k_values}

        key = (n, self.ladder, B)
        null = self._NULL_CACHE.get(key)
        if null is None:
            rng = np.random.default_rng(np.random.SeedSequence([n, B]))
            null = _permutation_null(u, v, k_values, n, B, rng, self.impl)
            self._NULL_CACHE[key] = null

        p_per_k = {
            k: float((np.sum(null[k] >= S_obs[k]) + 1) / (B + 1)) for k in k_values
        }
        p_value = min(1.0, len(k_values) * min(p_per_k.values()))
        return TestResult(
            statistic=max(float(_density_from_int(S_obs[k], k, n)) for k in k_values),
            p_value=p_value,
            n=n,
            B=B,
            test_name=self.name,
        )


def _fast_dcov():
    from indep_tests.dcov import DCovTest

    class FastDCovTest(DCovTest):
        """``statistic()`` through the same O(n log n) fast path ``test()`` uses.

        ``DCovTest.statistic`` calls hyppo's dense O(n^2) evaluation, whereas
        ``test(auto=True)`` takes the univariate fast path of Huo & Szekely
        (2016) with the chi-squared p-value of Shen, Panda & Vogelstein (2022).
        Timing the statistic that the calibrated call actually evaluates keeps
        panel (a) comparable.
        """

        def statistic(self, x, y):  # noqa: D401
            x = np.asarray(x).reshape(-1, 1)
            y = np.asarray(y).reshape(-1, 1)
            return float(self._dcorr().test(x, y, reps=0, auto=True).stat)

    return FastDCovTest(auto=True)


def paper_tests() -> dict[str, IndependenceTest]:
    """The eight Section-4 tests, cold (every call pays its own null)."""
    from indep_tests.cmiel_ledwina_test import CmielLedwinaTest
    from indep_tests.heller_bins import HellerBinsTest
    from indep_tests.ma_mao_fes import MaMaoFESTest
    from indep_tests.pug_grid_pearson_chi_squared.test import (
        PUGGridPearsonChiSquaredTest,
    )
    from indep_tests.qad import QADTest
    from indep_tests.salahub_binning import SalahubBinningTest
    from indep_tests.zhang_bet import ZhangBETTest

    return {
        "MRP": PUGGridTest(norm="density-multi"),
        # Same statistic with the closed-form chi^2 calibration (E = 1); its
        # call is the statistic alone. Added 2026-09-27 for the timing figure.
        "MRP-chi2": PUGGridPearsonChiSquaredTest(norm="density-multi"),
        "Zhang-BET": ZhangBETTest(),
        "GJT-QAD": QADTest(cache_null=False),
        "Ma-Mao-FES": MaMaoFESTest(),
        "SRB-dCov": _fast_dcov(),
        "Salahub-Oldford-p_simple": SalahubBinningTest(max_depth=8, p_method="simple"),
        "Heller-ADP": HellerBinsTest(cache_null=False),
        "Cmiel-Ledwina-Tn": CmielLedwinaTest(cache_null=False),
    }


def warm_tests() -> dict[str, IndependenceTest]:
    """Reused-null variants of the tests whose null law depends on ``n`` only."""
    from indep_tests.cmiel_ledwina_test import CmielLedwinaTest
    from indep_tests.heller_bins import HellerBinsTest
    from indep_tests.qad import QADTest

    return {
        "MRP": WarmPUGGridTest(norm="density-multi"),
        "GJT-QAD": QADTest(cache_null=True),
        "Heller-ADP": HellerBinsTest(cache_null=True),
        "Cmiel-Ledwina-Tn": CmielLedwinaTest(cache_null=True),
    }
