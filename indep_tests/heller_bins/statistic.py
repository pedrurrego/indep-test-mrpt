"""Core ADP statistic for the Heller et al. (2016) univariate independence test.

This is a faithful port of the **ADP** ("all derived partitions") *sum*
statistic from the authors' R package ``HHG`` (``barakbri/HHG``). The relevant
C++ kernel is ``StatsComputer::uvs_ind_adp`` (``src/StatsComputer.cpp``) together
with the combinatorial cell-weights filled in ``TestIO::compute_adp_independence``
(``src/HHG.cpp``); ``nr_bins_equipartition`` (``R/HHG_univariate.R``) supplies the
default number of atoms used by the ``ADP-EQP`` acceleration.

The construction
----------------
Work in rank space: each margin is mapped to integer ranks ``1..n``. For a
partition size ``m`` the plane is cut into an ``m x m`` grid by placing ``m-1``
horizontal and ``m-1`` vertical cuts between ranks. ADP considers *every* such
partition. For each partition one forms the ``m x m`` contingency table of cell
counts ``O`` with independence-expected counts ``E = (row width)*(col width)/n``
and scores it by either Pearson ``sum (O-E)^2/E`` or the likelihood ratio
``sum O*log(O/E)``. The *sum*-aggregated ADP statistic at resolution ``m`` is the
sum of that score over all ``C(n-1, m-1)^2`` partitions.

Summing a cell-additive score over all partitions equals summing, over every
candidate rectangular cell, ``score(cell) * (number of partitions in which that
cell appears)``. The partition-count factorises across the two axes, and the
per-axis counts are simple binomials (see :func:`_interval_weights`). This is the
``O(grid^4)`` cell-weighting computed here -- exact, no enumeration.

ADP-EQP acceleration
--------------------
For large ``n`` the ranks are first equipartitioned into ``M = nr_atoms`` atoms
(``M <= n``); the same machinery then runs on the ``M x M`` atom grid, with the
combinatorial weights using ``M`` in place of ``n`` while the expected counts use
the *actual* rank widths spanned by each atom block. With ``M == n`` (small
samples) this reduces to exact ADP. This mirrors ``Fast.independence.test`` /
``variant='ADP-EQP'`` and keeps the cost bounded as ``n`` grows.

Reference
---------
Heller, R., Heller, Y., Kaufman, S., Brill, B. & Gorfine, M. (2016).
Consistent distribution-free K-sample and independence tests for univariate
random variables. JMLR 17(29):1-54. https://jmlr.org/papers/v17/14-441.html

Original R/C++ implementation: https://github.com/barakbri/HHG

Provenance
----------
Reference: Heller, R., Heller, Y., Kaufman, S., Brill, B. & Gorfine, M. (2016).
Consistent distribution-free K-sample and independence tests for univariate
random variables. J. Mach. Learn. Res. 17(29):1-54.
Upstream code: R package ``HHG`` (https://github.com/barakbri/HHG), GPL-3.
This module: port of the ADP sum statistic from the upstream R/C++ code.
Distributed under GPL-3 (LICENSE).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def nr_bins_equipartition(n: int) -> int:
    """Default number of atoms for ADP-EQP (``nr_bins_equipartition`` in R).

    ``round(min(60 + 0.5*floor(sqrt(n)), n))`` -- equals ``n`` for small samples
    (exact ADP) and saturates near ``60..71`` for large ``n``.
    """
    wanted = 60 + 0.5 * np.floor(np.sqrt(n))
    return int(round(min(wanted, n)))


def _choose(n: np.ndarray, k: int) -> np.ndarray:
    """Vectorised binomial ``C(n, k)`` as float, 0 where invalid.

    Mirrors ``TestIO::my_choose``: returns 0 for ``n < 0``; R's ``choose`` itself
    returns 0 for ``k < 0`` or ``k > n``.
    """
    n = np.asarray(n)
    if k < 0:
        return np.zeros(n.shape, dtype=float)
    valid = (n >= 0) & (k <= n)
    nn = np.where(valid, n, 0).astype(float)
    out = np.exp(gammaln(nn + 1.0) - gammaln(k + 1.0) - gammaln(nn - k + 1.0))
    return np.where(valid, out, 0.0)


def ranks_to_atoms(ranks: np.ndarray, n: int, M: int) -> np.ndarray:
    """Map integer ranks ``1..n`` to atom indices ``1..M`` (``ceil(r*M/n)``)."""
    atoms = np.ceil(ranks * M / n).astype(np.int64)
    return np.clip(atoms, 1, M)


def rank_ordinal(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Integer ranks ``1..n`` with ties broken at random."""
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.lexsort((rng.random(n), a))
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    return ranks


class ADPGrid:
    """Precomputed ADP geometry and cell-weights for a fixed ``(n, M, m_values)``.

    All data-independent quantities -- the candidate rectangle geometry and the
    per-resolution combinatorial weights -- are built once. :meth:`statistics`
    then turns a paired atom assignment into the per-``m`` ADP sum statistics in
    ``O(grid^4 / block)`` time, which is the only thing that varies under
    permutation.
    """

    def __init__(
        self,
        n: int,
        M: int,
        m_values: list[int],
        block: int = 256,
    ) -> None:
        if not (2 <= M <= n):
            raise ValueError(f"need 2 <= nr_atoms (={M}) <= n (={n})")
        if any(m < 2 or m > M for m in m_values):
            raise ValueError(f"partition sizes must satisfy 2 <= m <= nr_atoms (={M})")
        self.n = n
        self.M = M
        self.m_values = list(m_values)
        self.block = block

        # Atom rank boundaries: P[a] = floor(a*n/M) is the top rank of atom a, so
        # the number of ranks (= points) in atom block [lo..hi] is P[hi]-P[lo-1].
        self.P = np.floor(np.arange(M + 1) * n / M).astype(np.int64)

        # All candidate intervals 1 <= lo <= hi <= M (1-indexed atom boundaries).
        lo, hi = np.triu_indices(M, k=0)
        lo = lo + 1
        hi = hi + 1
        WX = np.stack([_interval_weights(lo, hi, M, m) for m in self.m_values], axis=1)

        # Drop intervals that carry zero weight at every resolution (e.g. the
        # whole-axis interval): they contribute nothing as either axis.
        keep = WX.max(axis=1) > 0
        self.lo = lo[keep]
        self.hi = hi[keep]
        self.WX = WX[keep]  # (T, n_m)
        self.rw = (self.P[self.hi] - self.P[self.lo - 1]).astype(float)  # rank widths

    def statistics(
        self, ax: np.ndarray, ay: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """ADP sum statistics per resolution for a paired atom assignment.

        Parameters
        ----------
        ax, ay : atom indices ``1..M`` for each point (paired by position).

        Returns
        -------
        (sum_chi, sum_lr) : each an array over ``m_values`` -- the Pearson and
        likelihood-ratio ADP sum statistics. Larger = more dependence.
        """
        M, n = self.M, self.n
        N = np.zeros((M, M), dtype=np.int64)
        np.add.at(N, (ax - 1, ay - 1), 1)
        # Padded 2-D cumulative count: C[a, b] = #points in atoms x<=a, y<=b.
        C = np.zeros((M + 1, M + 1), dtype=np.int64)
        C[1:, 1:] = N.cumsum(axis=0).cumsum(axis=1)

        lo, hi, rw, WX = self.lo, self.hi, self.rw, self.WX
        T = lo.size
        n_m = WX.shape[1]
        sum_chi = np.zeros(n_m)
        sum_lr = np.zeros(n_m)
        inv_n = 1.0 / n

        # The statistic is the quadratic form  w_m^T S w_m  with S[i, j] the score
        # of the rectangle (x-interval i) x (y-interval j). S is built block-wise
        # over x-intervals to bound memory; each block contributes
        # (WX[block] * (S_block @ WX)).sum over the block to every resolution.
        for r0 in range(0, T, self.block):
            r1 = min(r0 + self.block, T)
            hi_b = hi[r0:r1, None]
            lo_b = lo[r0:r1, None]
            Obs = (
                C[hi_b, hi[None, :]]
                - C[lo_b - 1, hi[None, :]]
                - C[hi_b, lo[None, :] - 1]
                + C[lo_b - 1, lo[None, :] - 1]
            ).astype(float)
            E = (rw[r0:r1, None] * rw[None, :]) * inv_n
            score_chi = (Obs - E) ** 2 / E
            with np.errstate(divide="ignore", invalid="ignore"):
                score_lr = np.where(Obs > 0, Obs * np.log(Obs / E), 0.0)

            sum_chi += (WX[r0:r1] * (score_chi @ WX)).sum(axis=0)
            sum_lr += (WX[r0:r1] * (score_lr @ WX)).sum(axis=0)

        return sum_chi, sum_lr


def _interval_weights(lo: np.ndarray, hi: np.ndarray, M: int, m: int) -> np.ndarray:
    """Number of ``m``-cell partitions of one axis in which interval ``[lo, hi]``
    is a cell (``count_adp_with_given_cell`` factor / ``compute_adp_independence``).

    With ``m-1`` cuts chosen among the ``M-1`` gaps:
      * left-anchored  ``lo == 1`` : choose(M - hi - 1, m - 2)
      * right-anchored ``hi == M`` : choose(lo - 2, m - 2)
      * interior                   : choose(M - (hi-lo+1) - 2, m - 3)
    The left/right branches take priority (matching the C++), so the whole-axis
    interval gets weight 0.
    """
    w_atoms = hi - lo + 1
    w_left = _choose(M - hi - 1, m - 2)
    w_right = _choose(lo - 2, m - 2)
    w_inner = _choose(M - w_atoms - 2, m - 3)
    is_left = lo == 1
    is_right = hi == M
    return np.where(is_left, w_left, np.where(is_right, w_right, w_inner))
