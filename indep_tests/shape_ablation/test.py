"""``pug-partition``: the PUG-Grid-DENSITY statistic on arbitrary partitions of the unit square.

Same statistic, ladder rule, permutation scheme and Bonferroni combination as
``pug-grid norm=density-multi``; only the cell geometry changes. With
``geometry="square"`` the p-values are bit-for-bit those of density-multi
(verified 2026-09-12, 40/40 calls, after ``src_pug_grid`` switched to comparing
the integer ``sum(count^2)`` so that null/observed ties are exact; results
produced by the earlier float comparison can differ in the last decimal).
"""

from __future__ import annotations

import numpy as np

from indep_tests.base import IndependenceTest, TestResult
from src_pug_grid.utils import calculate_copula_values

from .partitions import Partition, make_partition_K, partitions_for


def _pearson(counts: np.ndarray, n: int, P: Partition) -> np.ndarray:
    """Pearson statistic per row of counts (B, K).

    For equal-area partitions the monotone-equivalent integer form sum (K c - n)^2 is
    used so that ties between null and observed values are exact (Pearson = that / (n K)).
    """
    if P.equal_area:
        return ((P.K * counts - n) ** 2).sum(axis=-1)
    expected = n * P.areas
    return (((counts - expected) ** 2) / expected).sum(axis=-1)


# Lattice tables: the pseudo-observations are ranks/n, so every point of every
# permutation lies on the n x n lattice {(i/n, j/n)}. Each rung's cell label is
# tabulated once per (partition, n) and a permutation becomes a pure gather
# (identical labels to calling ``P.codes`` on the permuted points; this only
# matters for the geometries whose ``codes`` is expensive -- golden's tree walk
# and the power diagram's dense argmin -- and makes every geometry O(n) per
# permutation like the square grid). Keyed by label + n; labels are unique per
# configuration within a geometry family.
_TABLE_CACHE: dict[tuple[str, str, int], np.ndarray] = {}


def _lattice_codes(P: Partition, n: int) -> np.ndarray:
    key = (type(P).__name__, P.label, n)
    T = _TABLE_CACHE.get(key)
    if T is None:
        g = np.arange(1, n + 1) / n
        T = np.ascontiguousarray(P.codes(g[:, None], g[None, :]))  # (n, n)
        _TABLE_CACHE[key] = T
    return T


def _rank_index(u: np.ndarray, n: int) -> np.ndarray:
    """Lattice index of rank/n values: (i/n) -> i-1."""
    return np.rint(u * n).astype(np.int64) - 1


def _counts(codes: np.ndarray, K: int) -> np.ndarray:
    """codes (B, n) -> counts (B, K) with one bincount call."""
    B = codes.shape[0]
    flat = (codes + (np.arange(B) * K)[:, None]).ravel()
    return np.bincount(flat, minlength=B * K).reshape(B, K)


class PUGPartitionTest(IndependenceTest):
    """Multi-resolution (or single) Pearson cell-count test on a chosen cell geometry.

    Parameters
    ----------
    geometry : "square" | "rect" | "rotated" | "golden" | "power"
    mode : "ladder" (density-multi rungs, cell counts matched per family) | "single"
    single_k : rung used when mode="single" (cells matched to single_k**2)
    single_K : target cell count used when mode="single" (takes precedence over
        single_k; see ``partitions.make_partition_K`` / ``profile_Ks``) -- the
        per-family resolution profile of the paper's shape section
    """

    def __init__(
        self,
        geometry: str = "square",
        mode: str = "ladder",
        single_k: int = 12,
        single_K: int | None = None,
    ):
        self.geometry, self.mode, self.single_k = geometry, mode, single_k
        self.single_K = single_K

    @property
    def name(self) -> str:
        if self.mode == "ladder":
            tag = "MULTI"
        elif self.single_K is not None:
            tag = f"K{self.single_K}"
        else:
            tag = f"k{self.single_k}"
        return f"PUG-Partition-{self.geometry}-{tag}"

    def _parts(self, n: int) -> list[Partition]:
        if self.mode == "single" and self.single_K is not None:
            return [make_partition_K(self.geometry, self.single_K)]
        return partitions_for(self.geometry, n, self.mode, self.single_k)

    def statistic(self, x, y) -> float:
        u, v = calculate_copula_values(np.column_stack((np.ravel(x), np.ravel(y))))
        n = len(u)
        return max(
            float(_pearson(_counts(P.codes(u, v)[None, :], P.K), n, P)[0])
            for P in self._parts(n)
        )

    def test(self, x, y, B: int = 500, seed: int | None = None) -> TestResult:
        u, v = calculate_copula_values(np.column_stack((np.ravel(x), np.ravel(y))))
        n = len(u)
        rng = np.random.default_rng(seed if seed is not None else 1234)
        perms = rng.permuted(
            np.tile(np.arange(n), (B, 1)), axis=1
        )  # as in src_pug_grid
        parts = self._parts(n)
        iu, iv = _rank_index(u, n), _rank_index(v, n)
        p_per, T_per = {}, {}
        for P in parts:
            T = _lattice_codes(P, n)
            T_obs = _pearson(_counts(T[iu, iv][None, :], P.K), n, P)[0]
            T_null = _pearson(_counts(T[iu[None, :], iv[perms]], P.K), n, P)
            p_per[P.label] = float((np.sum(T_null >= T_obs) + 1) / (B + 1))
            T_per[P.label] = float(T_obs)
        p_min = min(p_per.values())
        return TestResult(
            statistic=max(T_per.values()),
            p_value=min(1.0, len(parts) * p_min),
            n=n,
            B=B,
            test_name=self.name,
            extras={"p_per": p_per, "T_per": T_per, "K": [P.K for P in parts]},
        )
