"""Distance covariance (dCov) test via hyppo library.

Implements the distance correlation test of Szekely, Rizzo, & Bakirov (2007),
which yields the same p-value as the dCov-based independence test.

Provenance
----------
Reference: Székely, G. J., Rizzo, M. L. & Bakirov, N. K. (2007). Measuring and
testing dependence by correlation of distances. Ann. Statist. 35(6):2769-2794;
fast path of Huo & Székely (2016), Technometrics 58(4):435-447, with the
chi-squared p-value of Shen, Panda & Vogelstein (2022), JCGS 31(1):254-262.
Upstream code: Python package ``hyppo`` (MIT), used as a dependency.
This module: thin wrapper. Distributed under GPL-3 (LICENSE).
"""

import numpy as np

from indep_tests.base import IndependenceTest, TestResult


class DCovTest(IndependenceTest):
    """Wrapper around ``hyppo.independence.Dcorr``.

    Parameters
    ----------
    compute_distance : str or None, default "euclidean"
        Distance metric passed to the underlying Dcorr constructor.
        Set to ``None`` when *x* and *y* are already distance matrices.
    bias : bool, default False
        Whether to use the biased (V-statistic) form of dCov.
    auto : bool, default False
        If True, hyppo uses its chi-squared approximation when n > 20 and
        ignores ``reps``/``workers``. Defaults to False so the test is
        permutation-calibrated and apples-to-apples with the rest of the
        benchmark.
    workers : int, default 1
        Number of cores for the permutation p-value computation.
        Use ``-1`` to use all available cores.
    """

    def __init__(
        self,
        compute_distance: str | None = "euclidean",
        bias: bool = False,
        auto: bool = False,
        workers: int = 1,
    ) -> None:
        self.compute_distance = compute_distance
        self.bias = bias
        self.auto = auto
        self.workers = workers

    @property
    def name(self) -> str:
        return f"dCov_auto-{self.auto}"

    def _dcorr(self):
        from hyppo.independence import Dcorr

        return Dcorr(compute_distance=self.compute_distance, bias=self.bias)

    def statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)
        return float(self._dcorr().statistic(x, y))

    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        B: int = 1000,
        seed: int | None = None,
    ) -> TestResult:
        x = np.asarray(x).reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)

        result = self._dcorr().test(
            x,
            y,
            reps=B,
            workers=self.workers,
            auto=self.auto,
            random_state=seed,
        )

        return TestResult(
            statistic=float(result.stat),
            p_value=float(result.pvalue),
            n=len(x),
            B=B,
            test_name=self.name,
            extras={"auto": self.auto, "bias": self.bias},
        )
