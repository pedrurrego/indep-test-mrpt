"""MRPT harness wrapper (``pug-grid``): the test of the paper."""

import numpy as np

from indep_tests.base import IndependenceTest, TestResult
from src_pug_grid.pug_independence_test import DEFAULT_LADDER, Impl, Norm
from src_pug_grid.pug_independence_test import indep_test as _pug_grid_indep_test


class PUGGridTest(IndependenceTest):
    """MRPT: Pearson-type statistic on the pseudo-observations binned on a
    ladder of square grids, one permutation p-value per rung, Bonferroni.

    Parameters
    ----------
    norm : str
        ``"density-multi"`` (default): the multi-resolution test of the paper
        (result label ``PUG-Grid-DENSITY-MULTI``).  ``"density"``: the
        statistic on a single ``k x k`` grid.
    k : int | None
        Grid resolution for ``norm="density"``; ``floor(sqrt(n))`` if *None*.
        Ignored by ``"density-multi"``.
    impl : str
        Permutation-null implementation: ``"vectorized"`` (default; all
        permutations in one ``np.bincount``) or ``"loop"`` (the readable
        one-at-a-time reference).  Both give bit-identical statistics and
        p-values for the same seed; only run time differs (~5-10x).  Does not
        affect ``name``, so result files are unchanged unless it is passed
        explicitly via ``--test-config``.
    ladder : str
        Resolution ladder for ``"density-multi"``, rungs joined by ``-``:
        integers are fixed resolutions (kept while below ``floor(sqrt(n))``),
        ``sqrt`` is the adaptive top rung.  The default ``"3-5-8-12-sqrt"`` is
        the production ladder and gives byte-identical results to omitting it.
        Ignored by ``"density"``.  Pass it via ``--test-config`` (e.g.
        ``{"norm": "density-multi", "ladder": "3-sqrt"}``) so the result
        filename records it.
    """

    def __init__(
        self,
        k: int | None = None,
        norm: Norm = "density-multi",
        impl: Impl = "vectorized",
        ladder: str = DEFAULT_LADDER,
    ) -> None:
        self.k = k
        self.norm = norm
        self.impl = impl
        self.ladder = ladder

    @property
    def name(self) -> str:
        suffix = self.norm.upper()
        return f"PUG-Grid-{suffix}"

    def statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the grid statistic (no permutation, raw value)."""
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        X = np.column_stack((x, y))
        n = len(X)

        if self.norm == "density-multi":
            from src_pug_grid.pug_independence_test import (
                _compute_multi_density,
                resolve_ladder,
            )
            from src_pug_grid.utils import calculate_copula_values

            k_values = resolve_ladder(self.ladder, n)
            u, v = calculate_copula_values(X)
            stats = _compute_multi_density(u, v, k_values, n)
            return max(stats.values())

        from src_pug_grid.pug_independence_test import _compute_density_statistic
        from src_pug_grid.utils import calculate_copula_values

        k = self.k if self.k is not None else int(np.floor(np.sqrt(n)))
        u, v = calculate_copula_values(X)
        return _compute_density_statistic(u, v, k, n)

    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        B: int = 500,
        seed: int | None = None,
    ) -> TestResult:
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        n = len(x)
        X = np.column_stack((x, y))

        result = _pug_grid_indep_test(
            X=X,
            n_reps=B,
            k=self.k,
            norm=self.norm,
            seed=seed if seed is not None else 1234,
            impl=self.impl,
            ladder=self.ladder,
        )

        return TestResult(
            statistic=result["statistic"],
            p_value=result["p-value"],
            n=n,
            B=B,
            test_name=self.name,
        )
