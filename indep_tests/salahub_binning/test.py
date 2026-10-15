"""IndependenceTest wrapper for the Salahub-Oldford binning test."""

from __future__ import annotations

from typing import Literal

import numpy as np

from indep_tests.base import IndependenceTest, TestResult
from indep_tests.salahub_binning.binning import (
    bin_chi,
    binner,
    rank_random_ties,
)
from indep_tests.salahub_binning.pvalues import PMethod, p_value

PermMethod = Literal["simple", "fitted", "gamma", "permutation"]


class SalahubBinningTest(IndependenceTest):
    """Recursive random binning chi-squared test (Salahub & Oldford, 2025).

    Parameters
    ----------
    max_depth:
        Maximum recursion depth. AssocBin default is 6; the paper's power
        experiments use 8.
    min_exp:
        Minimum allowable expected count for a bin. Splits that would create a
        child with expected count below this are rejected; bins whose expected
        count is already at most ``2 * min_exp`` are left unsplit.
    squarify:
        If True, at every step the longer edge of the bin is split, producing
        more nearly-square bins. Matches the ``AssocBin`` default.
    p_method:
        How to compute the p-value. ``"simple"`` uses the mnemonic
        ``chi^2_{(sqrt(K)-1)^2}``; ``"fitted"`` uses the least-squares fitted
        degrees of freedom; ``"gamma"`` uses the Gamma(alpha, theta) approximation;
        ``"permutation"`` draws ``B`` independent permutations to estimate the
        null distribution of the statistic empirically.
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_exp: float = 5.0,
        squarify: bool = True,
        p_method: PermMethod = "simple",
    ) -> None:
        self.max_depth = max_depth
        self.min_exp = min_exp
        self.squarify = squarify
        self.p_method = p_method

    @property
    def name(self) -> str:
        return f"Salahub-Binning-depth{self.max_depth}-p_{self.p_method}"

    def _compute(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[float, int]:
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        x_ranks = rank_random_ties(x, rng)
        y_ranks = rank_random_ties(y, rng)
        bins = binner(
            x_ranks,
            y_ranks,
            max_depth=self.max_depth,
            min_exp=self.min_exp,
            squarify=self.squarify,
            rng=rng,
        )
        return bin_chi(bins), len(bins)

    def statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        rng = np.random.default_rng()
        stat, _ = self._compute(x, y, rng)
        return stat

    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        B: int = 1000,
        seed: int | None = None,
    ) -> TestResult:
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        n = x.size
        rng = np.random.default_rng(seed)

        stat, K = self._compute(x, y, rng)

        if self.p_method == "permutation":
            null_stats = np.empty(B, dtype=float)
            for b in range(B):
                y_perm = rng.permutation(y)
                s, _ = self._compute(x, y_perm, rng)
                null_stats[b] = s
            pv = float(np.mean(null_stats >= stat))
            b_used = B
        else:
            method: PMethod = self.p_method  # type: ignore[assignment]
            pv = p_value(stat, K, method=method)
            b_used = None

        return TestResult(
            statistic=stat,
            p_value=pv,
            n=n,
            B=b_used,
            test_name=self.name,
            extras={
                "n_bins": K,
                "max_depth": self.max_depth,
                "min_exp": self.min_exp,
                "p_method": self.p_method,
            },
        )
