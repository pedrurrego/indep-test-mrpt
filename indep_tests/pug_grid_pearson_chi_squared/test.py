"""MRPT with closed-form chi-squared p-values (``pug-grid-chi2``).

Replaces the permutation bootstrap of ``PUGGridTest(norm="density"/"density-multi")``
with an asymptotic chi-squared approximation.

Derivation
----------
The single-resolution density statistic is

    T(k) = sum_{i,j} (c_ij/n - 1/k^2)^2,

where c_ij is the count of observations in cell (i,j) of a k x k grid of the
empirical copula.  Pearson's chi-squared statistic against the same uniform
expectation E_ij = n/k^2 is

    chi^2(k) = sum_{i,j} (c_ij - E_ij)^2 / E_ij = n * k^2 * T(k).

Under independence with continuous marginals the rank-based copula values give
near-uniform row and column sums in the k x k contingency table, so chi^2(k)
is asymptotically chi-squared with (k-1)^2 degrees of freedom.  This matches
the ``simple_df`` formula (sqrt(K) - 1)^2 from Salahub & Oldford (2025) when
K = k^2.

For the multi-resolution variant we compute one chi-squared p-value per
resolution and combine them with the same Bonferroni correction the bootstrap
version uses.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import chi2

from indep_tests.base import IndependenceTest, TestResult
from src_pug_grid.pug_independence_test import (
    DEFAULT_LADDER,
    _compute_density_statistic,
    resolve_ladder,
)
from src_pug_grid.utils import calculate_copula_values

DensityNorm = Literal["density", "density-multi"]


def _chi2_p_value(stat_T: float, k: int, n: int) -> tuple[float, float, int]:
    """Return (chi2_stat, p_value, df) for the density statistic at resolution k."""
    df = (k - 1) ** 2
    chi2_stat = n * (k**2) * stat_T
    if df <= 0:
        return chi2_stat, 1.0, df
    p = float(chi2.sf(chi2_stat, df))
    return chi2_stat, p, df


class PUGGridPearsonChiSquaredTest(IndependenceTest):
    """Density-based PUG-Grid test with chi-squared asymptotic p-values.

    Parameters
    ----------
    k : int | None
        Grid resolution.  If *None*, uses floor(sqrt(n)).  Ignored when
        ``norm="density-multi"``.
    norm : str
        ``"density"`` (single resolution) or ``"density-multi"`` (multi
        resolution with Bonferroni).
    """

    def __init__(
        self,
        k: int | None = None,
        norm: DensityNorm = "density",
    ) -> None:
        if norm not in ("density", "density-multi"):
            raise ValueError(
                f"PUGGridPearsonChiSquaredTest only supports density norms, got {norm!r}"
            )
        self.k = k
        self.norm = norm

    @property
    def name(self) -> str:
        suffix = self.norm.upper()
        return f"PUG-Grid-ChiSq-{suffix}"

    def _k_values(self, n: int) -> list[int]:
        return resolve_ladder(DEFAULT_LADDER, n)

    def statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        X = np.column_stack((x, y))
        n = len(X)
        u, v = calculate_copula_values(X)

        if self.norm == "density-multi":
            stats = [_compute_density_statistic(u, v, k, n) for k in self._k_values(n)]
            return float(max(stats))

        k = self.k if self.k is not None else int(np.floor(np.sqrt(n)))
        return float(_compute_density_statistic(u, v, k, n))

    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        B: int = 0,
        seed: int | None = None,
    ) -> TestResult:
        """Run the chi-squared test.  ``B`` and ``seed`` are accepted for
        interface compatibility but ignored (no resampling)."""
        del B, seed
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        n = len(x)
        X = np.column_stack((x, y))
        u, v = calculate_copula_values(X)

        if self.norm == "density-multi":
            k_values = self._k_values(n)
            T_per_k: dict[int, float] = {}
            chi2_per_k: dict[int, float] = {}
            p_per_k: dict[int, float] = {}
            df_per_k: dict[int, int] = {}
            for k in k_values:
                T = _compute_density_statistic(u, v, k, n)
                chi2_stat, p, df = _chi2_p_value(T, k, n)
                T_per_k[k] = float(T)
                chi2_per_k[k] = float(chi2_stat)
                p_per_k[k] = p
                df_per_k[k] = df

            m = len(k_values)
            p_min = min(p_per_k.values())
            p_corrected = min(1.0, m * p_min)

            return TestResult(
                statistic=float(max(T_per_k.values())),
                p_value=float(p_corrected),
                n=n,
                B=None,
                test_name=self.name,
                extras={
                    "k_values": k_values,
                    "T_per_k": T_per_k,
                    "chi2_per_k": chi2_per_k,
                    "df_per_k": df_per_k,
                    "p_per_k": p_per_k,
                    "p_value_raw": float(p_min),
                    "norm": self.norm,
                },
            )

        k = self.k if self.k is not None else int(np.floor(np.sqrt(n)))
        T = _compute_density_statistic(u, v, k, n)
        chi2_stat, p, df = _chi2_p_value(T, k, n)
        return TestResult(
            statistic=float(T),
            p_value=float(p),
            n=n,
            B=None,
            test_name=self.name,
            extras={
                "k": k,
                "chi2": float(chi2_stat),
                "df": df,
                "expected_per_cell": n / (k**2),
                "norm": self.norm,
            },
        )
