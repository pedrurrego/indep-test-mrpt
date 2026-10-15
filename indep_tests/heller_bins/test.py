"""IndependenceTest wrapper for the Heller et al. (2016) ADP binning test.

Wraps the ADP sum statistic (:mod:`indep_tests.heller_bins.statistic`) behind the
common :class:`IndependenceTest` interface, combining resolutions ``m`` with the
``MinP`` rule (the default ``combining.type`` of
``hhg.univariate.ind.combined.test``) calibrated by permutation.

Distribution-free null caching
------------------------------
The statistic is rank-based, so under independence its null distribution depends
only on ``(n, nr_atoms, m_values, score)`` -- not on the data. This is the whole
point of the paper (the authors ship precomputed null tables). We exploit it: the
permutation null for a given configuration is generated once and cached, so the
Monte-Carlo replications the harness runs for one sample size reuse it instead of
re-permuting per replication. The observed statistic is still computed from the
real data every call. Set ``cache_null=False`` to regenerate the null per call
from the call's own seed.

Reference
---------
Heller, R., Heller, Y., Kaufman, S., Brill, B. & Gorfine, M. (2016).
Consistent distribution-free K-sample and independence tests for univariate
random variables. JMLR 17(29):1-54.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from indep_tests.base import IndependenceTest, TestResult
from indep_tests.heller_bins.statistic import (
    ADPGrid,
    nr_bins_equipartition,
    rank_ordinal,
    ranks_to_atoms,
)

ScoreType = Literal["LikelihoodRatio", "Pearson"]
Combining = Literal["MinP", "Fisher"]

# Distribution-free caches, keyed by configuration (see module docstring).
_GRID_CACHE: dict[tuple, ADPGrid] = {}
_NULL_CACHE: dict[tuple, np.ndarray] = {}


def _default_mmax(n: int) -> int:
    """``max(floor(sqrt(n)/2), 2)`` -- the package default for independence."""
    return max(int(np.floor(np.sqrt(n) / 2)), 2)


def _get_grid(n: int, M: int, m_values: tuple[int, ...]) -> ADPGrid:
    key = (n, M, m_values)
    grid = _GRID_CACHE.get(key)
    if grid is None:
        grid = ADPGrid(n=n, M=M, m_values=list(m_values))
        _GRID_CACHE[key] = grid
    return grid


def _generate_null(
    grid: ADPGrid,
    score_idx: int,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``B x n_m`` matrix of per-resolution null statistics.

    Under independence the joint ranks are a uniform random pairing, so we fix the
    x-atoms to the canonical order and permute the y-atoms.
    """
    n, M = grid.n, grid.M
    ax = ranks_to_atoms(np.arange(1, n + 1), n, M)
    ay0 = ranks_to_atoms(np.arange(1, n + 1), n, M)
    null = np.empty((B, len(grid.m_values)))
    for b in range(B):
        ay = ay0[rng.permutation(n)]
        chi, lr = grid.statistics(ax, ay)
        null[b] = lr if score_idx == 1 else chi
    return null


def _get_null(
    grid: ADPGrid,
    score_idx: int,
    B: int,
    cache: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    if not cache:
        return _generate_null(grid, score_idx, B, rng)
    key = (grid.n, grid.M, tuple(grid.m_values), score_idx, B)
    null = _NULL_CACHE.get(key)
    if null is None:
        # Deterministic, call-order-independent seed so every replication that
        # hits this configuration shares one null table.
        seed_seq = np.random.SeedSequence(
            [grid.n, grid.M, score_idx, B, *grid.m_values]
        )
        null = _generate_null(grid, score_idx, B, np.random.default_rng(seed_seq))
        _NULL_CACHE[key] = null
    return null


def _inner_pvalues(pool: np.ndarray) -> np.ndarray:
    """Right-tail permutation p-values per resolution for every pooled member.

    ``pool`` is ``(B+1, n_m)`` with the observed statistic as the last row and the
    ``B`` null statistics above it. For member ``k`` and resolution ``j`` the inner
    p-value is ``(#{pool[:, j] >= pool[k, j]}) / (B+1)``. Pooling the observed with
    the null keeps them exchangeable under H0, so the MinP / Fisher combinations
    are exactly calibrated (every inner p-value lies in ``[1/(B+1), 1]``).
    """
    Bp1 = pool.shape[0]
    inner = np.empty_like(pool, dtype=float)
    for j in range(pool.shape[1]):
        col = pool[:, j]
        inner[:, j] = (Bp1 - np.searchsorted(np.sort(col), col, side="left")) / Bp1
    return inner


class HellerBinsTest(IndependenceTest):
    """Heller-Heller-Kaufman-Brill-Gorfine (2016) ADP independence test.

    The ``ADP`` ("all derived partitions") sum statistic at a ladder of partition
    sizes ``m = mmin..mmax``, combined across resolutions by ``MinP`` (minimum of
    the per-``m`` permutation p-values) or ``Fisher``. For large ``n`` the ranks
    are equipartitioned into ``nr_atoms`` atoms (``ADP-EQP``); for small ``n`` the
    statistic is exact.

    Parameters
    ----------
    score_type : {"LikelihoodRatio", "Pearson"}
        Per-table score. ``LikelihoodRatio`` is the package default.
    combining : {"MinP", "Fisher"}
        How per-resolution p-values are combined. ``MinP`` is the default.
    mmin, mmax : int | None
        Partition-size ladder. ``mmax=None`` uses ``max(floor(sqrt(n)/2), 2)``.
    nr_atoms : int | None
        Number of equipartition atoms (``2 <= nr_atoms <= n``). ``None`` uses
        ``round(min(60 + 0.5*floor(sqrt(n)), n))``; equals ``n`` (exact ADP) for
        small samples.
    cache_null : bool
        Reuse a distribution-free permutation null across calls of the same
        configuration (default ``True``). See the module docstring.
    """

    def __init__(
        self,
        score_type: ScoreType = "LikelihoodRatio",
        combining: Combining = "MinP",
        mmin: int = 2,
        mmax: int | None = None,
        nr_atoms: int | None = None,
        cache_null: bool = True,
    ) -> None:
        if score_type not in ("LikelihoodRatio", "Pearson"):
            raise ValueError(f"unknown score_type {score_type!r}")
        if combining not in ("MinP", "Fisher"):
            raise ValueError(f"unknown combining {combining!r}")
        self.score_type = score_type
        self.score_idx = 1 if score_type == "LikelihoodRatio" else 0
        self.combining = combining
        self.mmin = mmin
        self.mmax = mmax
        self.nr_atoms = nr_atoms
        self.cache_null = cache_null

    @property
    def name(self) -> str:
        score = "LR" if self.score_type == "LikelihoodRatio" else "Pearson"
        return f"Heller-ADP-{score}-{self.combining}"

    def _resolve(self, n: int) -> tuple[int, tuple[int, ...]]:
        M = self.nr_atoms if self.nr_atoms is not None else nr_bins_equipartition(n)
        M = int(min(M, n))
        mmax = self.mmax if self.mmax is not None else _default_mmax(n)
        mmax = min(mmax, M)
        if mmax < self.mmin:
            mmax = self.mmin
        m_values = tuple(range(self.mmin, mmax + 1))
        return M, m_values

    def _observed(
        self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> tuple[ADPGrid, np.ndarray]:
        n = x.size
        M, m_values = self._resolve(n)
        grid = _get_grid(n, M, m_values)
        ax = ranks_to_atoms(rank_ordinal(x, rng), n, M)
        ay = ranks_to_atoms(rank_ordinal(y, rng), n, M)
        chi, lr = grid.statistics(ax, ay)
        obs = lr if self.score_idx == 1 else chi
        return grid, obs

    def statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        """Largest per-resolution ADP score (raw, no p-value)."""
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        _, obs = self._observed(x, y, np.random.default_rng())
        return float(np.max(obs))

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

        grid, obs = self._observed(x, y, rng)
        null = _get_null(grid, self.score_idx, B, self.cache_null, rng)

        # Pool the observed (last row) with the B null draws and form per-member,
        # per-resolution inner p-values, then combine across resolutions.
        pool = np.vstack([null, obs[None, :]])
        inner_p = _inner_pvalues(pool)
        p_obs = inner_p[-1]

        if self.combining == "MinP":
            combiner = inner_p.min(axis=1)  # smaller = stronger
            combined_p = float(np.mean(combiner <= combiner[-1]))
        else:  # Fisher: -sum log(p), larger = stronger
            combiner = -np.sum(np.log(inner_p), axis=1)
            combined_p = float(np.mean(combiner >= combiner[-1]))
        chosen_m = int(grid.m_values[int(np.argmin(p_obs))])

        return TestResult(
            statistic=float(np.max(obs)),
            p_value=combined_p,
            n=n,
            B=B,
            test_name=self.name,
            extras={
                "score_type": self.score_type,
                "combining": self.combining,
                "nr_atoms": grid.M,
                "m_values": list(grid.m_values),
                "per_m_statistic": obs.tolist(),
                "per_m_pvalue": p_obs.tolist(),
                "combining_statistic": float(combiner[-1]),
                "m_chosen": chosen_m,
            },
        )
