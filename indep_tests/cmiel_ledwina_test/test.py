"""IndependenceTest wrapper for the Ćmiel & Ledwina (2026) ``T_n`` test.

Wraps the trimmed-mean rank statistic ``T_n``
(:mod:`indep_tests.cmiel_ledwina_test.statistic`) behind the common
:class:`IndependenceTest` interface.

Distribution-free null caching
------------------------------
``T_n`` is a deterministic function of the two rank vectors, so under
independence -- where the ranks form a uniformly random pairing -- its law
depends only on ``(n, d, t)`` and not on the data.  The paper exploits exactly
this ("the rank process is distribution free under the independence hypothesis,
therefore ordinary Monte Carlo is sufficient to calculate the barriers").  We
generate the Monte-Carlo null once per ``(n, d, t, B)`` configuration and reuse
it across the harness's Monte-Carlo replications; the observed statistic is
recomputed from the real data on every call.  Set ``cache_null=False`` to draw a
fresh null per call from the call's own seed.

Reference
---------
Ćmiel, B. & Ledwina, T. (2026). *Detecting dependence structure: visualization
and inference.* arXiv:2410.05858.
"""

from __future__ import annotations

import numpy as np

from indep_tests.base import IndependenceTest, TestResult
from indep_tests.cmiel_ledwina_test.statistic import (
    CLGrid,
    _ranks,
    compute_tn,
    d_from_level,
    default_d,
    default_t,
    kappa_of,
)

# Distribution-free caches, keyed by configuration (see module docstring).
_GRID_CACHE: dict[tuple[int, int], CLGrid] = {}
_NULL_CACHE: dict[tuple, np.ndarray] = {}


def _get_grid(n: int, d: int) -> CLGrid:
    key = (n, d)
    grid = _GRID_CACHE.get(key)
    if grid is None:
        grid = CLGrid.build(n, d)
        _GRID_CACHE[key] = grid
    return grid


def _generate_null(
    grid: CLGrid, kappa: int, B: int, rng: np.random.Generator
) -> np.ndarray:
    """``B`` null draws of ``T_n``.

    Under independence the paired ranks are a uniform random matching, so we fix
    the x-ranks to the identity and permute the y-ranks.
    """
    n = grid.n
    identity = np.arange(1, n + 1)
    null = np.empty(B, dtype=np.float64)
    for b in range(B):
        ry = rng.permutation(n) + 1
        null[b] = compute_tn(identity, ry, grid, kappa)
    return null


def _get_null(
    grid: CLGrid,
    kappa: int,
    t: float,
    B: int,
    cache: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    if not cache:
        return _generate_null(grid, kappa, B, rng)
    key = (grid.n, grid.d, kappa, B)
    null = _NULL_CACHE.get(key)
    if null is None:
        # Deterministic, call-order-independent seed so every replication that
        # hits this configuration shares one null table.
        seed_seq = np.random.SeedSequence([grid.n, grid.d, kappa, B])
        null = _generate_null(grid, kappa, B, np.random.default_rng(seed_seq))
        _NULL_CACHE[key] = null
    return null


class CmielLedwinaTest(IndependenceTest):
    """Ćmiel & Ledwina (2026) ``T_n`` test of global independence.

    An upper-trimmed mean of the sorted ``|bar Q_n|`` values of the standardised
    weighted rank process on a regular dyadic ``d(n) x d(n)`` grid.  Rejects for
    large ``T_n``; calibrated by a distribution-free Monte-Carlo null.

    Parameters
    ----------
    d : int | None
        Grid dimension ``d(n)``.  ``None`` (default) uses the paper's explicit
        recommendation (Appendix A.2, v3): ``2**floor(log2(n/2)) - 1``.
        Overridden by ``s`` if that is given.
    s : int | None
        Dyadic level; sets ``d = 2**(s+1) - 1`` (so ``s=5`` gives ``d=63``).
        Takes precedence over ``d`` when supplied.
    t : float | None
        Smoothing / trimming parameter ``t(n) in (0, 1)``; ``T_n`` averages the
        top ``1 - t`` fraction of the ``d^2`` sorted values.  ``None`` (default)
        uses the paper's explicit recommendation (Appendix A.2, v3),
        ``t(n) = 1 - 1/(2 sqrt(n))`` — e.g. ``t(100) = 0.95``.  Pass ``t=0.95``
        for the fixed value used in the paper's ``n = 128`` power study.
        ``t -> 0`` approaches an ``l1`` norm, ``t -> 1`` the raw maximum ``V_n``
        (``l_inf``).
    randomize_ties : bool
        Break rank ties at random (the paper's *randomised ranks*).  Default
        ``True``; irrelevant for continuous data.
    cache_null : bool
        Reuse a distribution-free Monte-Carlo null across calls of the same
        ``(n, d, t, B)`` configuration (default ``True``).
    """

    def __init__(
        self,
        d: int | None = None,
        s: int | None = None,
        t: float | None = None,
        randomize_ties: bool = True,
        cache_null: bool = True,
    ) -> None:
        if t is not None and not 0.0 < t < 1.0:
            raise ValueError(f"t must lie in (0, 1); got {t!r}")
        if d is not None and d < 1:
            raise ValueError(f"d must be a positive integer; got {d!r}")
        self.d = d
        self.s = s
        self.t = t
        self.randomize_ties = randomize_ties
        self.cache_null = cache_null

    @property
    def name(self) -> str:
        return "Cmiel-Ledwina-Tn"

    def _resolve_d(self, n: int) -> int:
        if self.s is not None:
            return d_from_level(self.s)
        if self.d is not None:
            return self.d
        return default_d(n)

    def _resolve_t(self, n: int) -> float:
        return self.t if self.t is not None else default_t(n)

    def statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        """The raw ``T_n`` statistic (no p-value)."""
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        n = x.size
        d = self._resolve_d(n)
        rng = np.random.default_rng()
        rx = _ranks(x, rng, self.randomize_ties)
        ry = _ranks(y, rng, self.randomize_ties)
        grid = _get_grid(n, d)
        return compute_tn(rx, ry, grid, kappa_of(d, self._resolve_t(n)))

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
        d = self._resolve_d(n)
        t = self._resolve_t(n)
        kappa = kappa_of(d, t)
        rng = np.random.default_rng(seed)

        rx = _ranks(x, rng, self.randomize_ties)
        ry = _ranks(y, rng, self.randomize_ties)
        grid = _get_grid(n, d)
        obs = compute_tn(rx, ry, grid, kappa)

        null = _get_null(grid, kappa, t, B, self.cache_null, rng)
        # Pool observed with the B null draws (keeps exchangeability under H0);
        # reject for large T_n.
        p_value = float((1 + np.sum(null >= obs)) / (1 + B))

        return TestResult(
            statistic=obs,
            p_value=p_value,
            n=n,
            B=B,
            test_name=self.name,
            extras={
                "d": d,
                "t": t,
                "kappa": kappa,
                "K": d * d,
            },
        )
