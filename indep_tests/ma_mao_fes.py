"""Ma-Mao Fisher Exact Scanning (FES) independence test.

Multi-scale scanning approach that partitions the empirical copula into
dyadic grids and applies Fisher's exact test to 2x2 sub-tables at every
location and scale, with three-level Sidak correction.

Reference
---------
Ma L. and Mao J. (2017). Fisher exact scanning for dependency.
http://arxiv.org/abs/1608.07885

Original R implementation: https://github.com/MaStatLab/FES

Provenance
----------
Reference: Ma, L. & Mao, J. (2019). Fisher exact scanning for dependency.
J. Amer. Statist. Assoc. 114(525):245-258.
Upstream code: https://github.com/MaStatLab/FES (R), GPL-3.
This module: implemented from the paper; the upstream code served as reference.
Distributed under GPL-3 (LICENSE).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import fisher_exact, hypergeom, norm, rankdata

from indep_tests.base import IndependenceTest, TestResult


def _ecdf_transform(x: np.ndarray) -> np.ndarray:
    """Marginal ECDF transform matching R's ``ecdf(x)(x) - 1/n``.

    Maps observations to [0, 1) via their empirical CDF values,
    shifted down by 1/n so no observation lands exactly on 1.0.
    """
    n = len(x)
    return rankdata(x, method="max") / n - 1 / n


def _build_contingency_tables(
    x_tran: np.ndarray, y_tran: np.ndarray, K1: int, K2: int
) -> dict[tuple[int, int], np.ndarray]:
    """Build contingency tables at all dyadic resolutions up to (K1, K2)."""
    tables: dict[tuple[int, int], np.ndarray] = {}
    for k1 in range(K1 + 1):
        for k2 in range(K2 + 1):
            nrows = 2**k1
            ncols = 2**k2
            row_idx = np.clip(np.floor(x_tran * nrows).astype(int), 0, nrows - 1)
            col_idx = np.clip(np.floor(y_tran * ncols).astype(int), 0, ncols - 1)
            table = np.zeros((nrows, ncols), dtype=int)
            np.add.at(table, (row_idx, col_idx), 1)
            tables[(k1, k2)] = table
    return tables


def _fisher_scan(
    tables: dict[tuple[int, int], np.ndarray],
    K1: int,
    K2: int,
    mid_p: bool,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    """Run Fisher's exact test on 2x2 sub-tables at each stratum.

    At stratum (k1, k2) the coarser 2^k1 x 2^k2 grid is used to locate
    windows, and each window is subdivided into a 2x2 table from the
    finer (k1+1, k2+1) grid.

    Returns per-stratum minimum p-values and window counts.
    """
    p_min_stratum: dict[tuple[int, int], float] = {}
    windows_stratum: dict[tuple[int, int], int] = {}

    for k1 in range(K1):
        for k2 in range(K2):
            fine_table = tables[(k1 + 1, k2 + 1)]
            nrows = 2**k1
            ncols = 2**k2
            best_p = np.inf
            n_windows = 0

            for i in range(nrows):
                for j in range(ncols):
                    t22 = fine_table[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
                    total = int(t22.sum())
                    col_sums = t22.sum(axis=0)
                    row_sums = t22.sum(axis=1)

                    if (
                        total > 25
                        and col_sums[0] > 10
                        and col_sums[1] > 10
                        and row_sums[0] > 10
                        and row_sums[1] > 10
                    ):
                        _, p = fisher_exact(t22)
                        if mid_p:
                            p -= (
                                hypergeom.pmf(
                                    t22[0, 0], total, int(col_sums[0]), int(row_sums[0])
                                )
                                / 2
                            )
                            p = max(p, 0.0)
                        n_windows += 1
                        best_p = min(best_p, p)

            p_min_stratum[(k1, k2)] = best_p
            windows_stratum[(k1, k2)] = n_windows

    return p_min_stratum, windows_stratum


def _three_level_sidak(
    p_min_stratum: dict[tuple[int, int], float],
    windows_stratum: dict[tuple[int, int], int],
    K1: int,
    K2: int,
) -> dict[str, float]:
    """Three-level Sidak correction: stratum -> resolution -> global.

    Level 1 — per stratum (k1, k2): adjust the minimum p-value for the
    number of windows tested at that stratum.

    Level 2 — per resolution r = k1 + k2: combine strata at the same
    total resolution via Sidak.

    Level 3 — global: combine across all resolutions.

    Returns dict with ``sidek``, ``bonferroni``, ``meta``, ``min`` p-values.
    """
    # --- Level 1: per-stratum Sidak ---
    p_sidek_stratum: dict[tuple[int, int], float] = {}
    p_bonf_stratum: dict[tuple[int, int], float] = {}
    for key, p_min in p_min_stratum.items():
        L = windows_stratum[key]
        if L > 0 and np.isfinite(p_min):
            p_sidek_stratum[key] = 1 - (1 - p_min) ** L
            p_bonf_stratum[key] = p_min * L
        else:
            p_sidek_stratum[key] = np.inf
            p_bonf_stratum[key] = np.inf

    # --- Level 2: per-resolution Sidak ---
    R = K1 + K2 - 1
    p_sidek_resol = np.ones(R)
    p_bonf_resol = np.ones(R)
    T = np.zeros(R, dtype=int)

    for r in range(R):
        best_sidek = np.inf
        best_bonf = np.inf
        for k1 in range(max(0, r - (K2 - 1)), min(K1 - 1, r) + 1):
            k2 = r - k1
            if np.isfinite(p_sidek_stratum[(k1, k2)]):
                best_sidek = min(best_sidek, p_sidek_stratum[(k1, k2)])
                best_bonf = min(best_bonf, p_bonf_stratum[(k1, k2)])
                T[r] += 1

        if T[r] > 0:
            p_sidek_resol[r] = 1 - (1 - best_sidek) ** T[r]
            p_bonf_resol[r] = best_bonf * T[r]

    # --- Level 3: global ---
    R1 = int(np.sum(T > 0))

    if R1 > 0:
        p_sidek_global = float(1 - (1 - np.min(p_sidek_resol)) ** R1)
        p_bonf_global = float(min(1.0, np.min(p_bonf_resol) * R1))

        valid = p_sidek_resol[T > 0]
        valid = np.clip(valid, 1e-300, 1 - 1e-15)
        p_meta_global = float(norm.cdf(np.sum(norm.ppf(valid)) / np.sqrt(R1)))
    else:
        p_sidek_global = 1.0
        p_bonf_global = 1.0
        p_meta_global = 1.0

    finite_mins = [p for p in p_min_stratum.values() if np.isfinite(p)]
    p_min_global = float(min(finite_mins)) if finite_mins else 1.0

    return {
        "sidek": float(np.clip(p_sidek_global, 0, 1)),
        "bonferroni": float(np.clip(p_bonf_global, 0, 1)),
        "meta": float(np.clip(p_meta_global, 0, 1)),
        "min": p_min_global,
    }


def fisher_exact_scanning(
    x: np.ndarray,
    y: np.ndarray,
    K1: int = 7,
    K2: int = 7,
    mid_p: bool = True,
) -> dict[str, float]:
    """Run the full FES procedure.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Bivariate observations.
    K1, K2 : int
        Maximum dyadic resolution levels (grids go up to 2^K on each axis).
    mid_p : bool
        Apply mid-p correction to Fisher's exact test p-values.

    Returns
    -------
    dict with keys ``sidek``, ``bonferroni``, ``meta``, ``min``.
    """
    x_tran = _ecdf_transform(np.asarray(x).ravel())
    y_tran = _ecdf_transform(np.asarray(y).ravel())

    tables = _build_contingency_tables(x_tran, y_tran, K1, K2)
    p_min_stratum, windows_stratum = _fisher_scan(tables, K1, K2, mid_p)
    return _three_level_sidak(p_min_stratum, windows_stratum, K1, K2)


class MaMaoFESTest(IndependenceTest):
    """Fisher Exact Scanning test for independence (Ma & Mao, 2017).

    Multi-scale scanning that partitions the empirical copula into dyadic
    grids, applies Fisher's exact test to 2x2 sub-tables at every location
    and scale, and combines via three-level Sidak correction.

    Parameters
    ----------
    K1, K2 : int
        Maximum dyadic resolution levels (default 7 each, giving grids up
        to 128 x 128).
    mid_p : bool
        Apply mid-p correction to individual Fisher tests (default True).
    correction : str
        Which global p-value to report: ``"sidek"`` (default),
        ``"bonferroni"``, or ``"meta"``.
    """

    def __init__(
        self,
        K1: int = 7,
        K2: int = 7,
        mid_p: bool = True,
        correction: str = "sidek",
    ) -> None:
        self.K1 = K1
        self.K2 = K2
        self.mid_p = mid_p
        self.correction = correction

    @property
    def name(self) -> str:
        return "Ma-Mao-FES"

    def statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the global minimum uncorrected p-value from the scan."""
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        result = fisher_exact_scanning(x, y, self.K1, self.K2, self.mid_p)
        return result["min"]

    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        B: int = 1000,
        seed: int | None = None,
    ) -> TestResult:
        """Run FES. *B* and *seed* are ignored (no resampling needed)."""
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        n = len(x)

        result = fisher_exact_scanning(x, y, self.K1, self.K2, self.mid_p)

        return TestResult(
            statistic=result["min"],
            p_value=result[self.correction],
            n=n,
            B=None,
            test_name=self.name,
            extras={
                "p_sidek": result["sidek"],
                "p_bonferroni": result["bonferroni"],
                "p_meta": result["meta"],
                "p_min": result["min"],
            },
        )
