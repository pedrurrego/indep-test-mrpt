"""Core numerics for the Ćmiel & Ledwina (2024) ``T_n`` independence test.

Reference
---------
Ćmiel, B. & Ledwina, T. (2024). *Detecting dependence structure: visualization
and inference.* arXiv:2410.05858.

The test targets the global maximum of the standardised weighted rank process

.. math::

    \\bar Q_n(u, v) = \\sqrt n\\,\\frac{\\bar C_n(u, v) - uv}
                                       {\\sqrt{uv(1-u)(1-v)}},

evaluated on a regular dyadic grid, where :math:`\\bar C_n` is the *bilinear*
(empirical checkerboard) copula of the paper's Section 3.1.  Rather than the raw
maximum :math:`V_n` (Eq. 12) -- which is unstable because it is typically
attained at grid vertices -- the paper uses an upper-trimmed mean of the sorted
absolute process values (Section 5.1):

.. math::

    T_n = \\frac{1}{K - \\kappa + 1}\\sum_{k=\\kappa}^{K} \\bar{|Q|}_{(k)},
    \\qquad K = d(n)^2,\\quad \\kappa = \\lceil t(n)\\,K \\rceil .

Everything here is a deterministic function of the two rank vectors, so under
independence the law of ``T_n`` depends only on ``(n, d, t)`` -- see
:mod:`indep_tests.cmiel_ledwina_test.test` for the distribution-free null cache.

Provenance
----------
Reference: Ćmiel, B. & Ledwina, T. (2026). Detecting dependence structure:
visualization and inference. Statistical Papers 67(3):57 (arXiv:2410.05858).
Upstream code: none public.
This module: implemented from the paper; validated against its published
critical values. Distributed under GPL-3 (LICENSE).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def default_d(n: int) -> int:
    """The paper's recommended grid dimension ``d(n) = 2**floor(log2(n/2)) - 1``.

    This is the explicit formula of Appendix A.2 (v3), balancing sensitivity to
    local and global departures from independence.  It reproduces every grid
    used in the paper: ``d(128) = 63``, ``d(73) = 31``, ``d(517) = 255``.
    Clamped to at least 1 (degenerate for ``n < 4``).
    """
    # floor(log2(n/2)) == floor(n/2).bit_length() - 1, in exact integer maths.
    return max(2 ** ((n // 2).bit_length() - 1) - 1, 1)


def default_t(n: int) -> float:
    """The paper's recommended smoothing parameter ``t(n) = 1 - 1/(2 sqrt(n))``.

    Explicit formula of Appendix A.2 (v3); gives ``t(100) = 0.95`` and grows
    slowly towards 1, so the trimmed mean keeps a comparable number of top
    order statistics as the grid refines with ``n``.

    We return the exact formula value.  Note the paper's Table A.2 critical
    values were computed with ``t`` rounded to two decimals (as printed): with
    rounded ``t`` we reproduce them to MC precision; with the exact value the
    ``n = 200, 500`` rows differ by ~0.05 in the direction of the rounding.
    """
    return 1.0 - 1.0 / (2.0 * np.sqrt(n))


def d_from_level(s: int) -> int:
    """Grid dimension for dyadic level ``s``: ``d(s) = 2**(s + 1) - 1``."""
    return 2 ** (s + 1) - 1


def grid_points(d: int) -> np.ndarray:
    """The ``d`` interior dyadic grid coordinates ``p_{s,j} = j / (d + 1)``."""
    return np.arange(1, d + 1) / (d + 1)


def _ranks(
    a: np.ndarray, rng: np.random.Generator, randomize: bool = True
) -> np.ndarray:
    """Ordinal ranks ``1..n`` of ``a`` (a permutation of ``1..n``).

    With ``randomize=True`` ties are broken at random, matching the paper's use
    of *randomised ranks* for rounded / tied measurements (Section 4.2).  For the
    continuous data assumed throughout there are no ties and the result is the
    ordinary rank.
    """
    a = np.asarray(a, dtype=float).ravel()
    n = a.size
    if randomize:
        order = np.lexsort((rng.random(n), a))
    else:
        order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    return ranks


@dataclass
class CLGrid:
    """Precomputed, data-independent pieces for a fixed ``(n, d)`` configuration.

    Attributes
    ----------
    n, d : int
        Sample size and grid dimension.
    G : np.ndarray, shape ``(d, n)``
        Grid response indexed by *rank value*: ``G[j, s - 1]`` is the univariate
        bilinear weight of an observation whose rank equals ``s`` at grid
        coordinate ``u_j`` (Eq. 6),
        ``G[j, s-1] = 1(s <= m_j) + f_j * 1(s == m_j + 1)`` with
        ``m_j = floor(n u_j)`` and ``f_j = n u_j - m_j``.
    mean, weight : np.ndarray, shape ``(d, d)``
        The null-mean matrix ``E[j, j'] = u_j u_{j'}`` and the standardising
        denominator ``D[j, j'] = sqrt(u_j(1-u_j)) sqrt(u_{j'}(1-u_{j'}))``.
    """

    n: int
    d: int
    G: np.ndarray
    mean: np.ndarray
    weight: np.ndarray

    @classmethod
    def build(cls, n: int, d: int) -> "CLGrid":
        u = grid_points(d)  # (d,)
        nu = n * u
        m = np.floor(nu).astype(np.int64)  # (d,)
        frac = nu - m  # (d,)
        s = np.arange(1, n + 1)  # rank values (n,)
        G = (s[None, :] <= m[:, None]).astype(np.float64)
        G += frac[:, None] * (s[None, :] == (m[:, None] + 1))
        w = np.sqrt(u * (1.0 - u))
        return cls(n=n, d=d, G=G, mean=np.outer(u, u), weight=np.outer(w, w))

    def qbar(self, rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
        """The standardised process ``bar Q_n`` on the grid, shape ``(d, d)``.

        ``rx``, ``ry`` are rank vectors (values in ``1..n``).  The bilinear
        copula is ``Cbar = (1/n) A B^T`` with ``A = G[:, rx - 1]`` and
        ``B = G[:, ry - 1]`` -- i.e. select each observation's grid response by
        its rank, then accumulate the outer products (Eq. 6, matrix form).
        """
        a = self.G[:, rx - 1]  # (d, n)
        b = self.G[:, ry - 1]  # (d, n)
        cbar = (a @ b.T) / self.n  # (d, d)
        return np.sqrt(self.n) * (cbar - self.mean) / self.weight


def kappa_of(d: int, t: float) -> int:
    """Trimming index ``kappa = ceil(t K)`` clamped to ``[1, K]``, ``K = d^2``.

    The small tolerance keeps ``ceil`` exact when ``t K`` is mathematically an
    integer but floating-point rounds it just above (e.g. ``0.95 * 400``).
    """
    k = d * d
    return int(min(max(int(np.ceil(t * k - 1e-9)), 1), k))


def t_statistic(qbar: np.ndarray, kappa: int) -> float:
    """Upper-trimmed mean ``T_n`` of the sorted ``|bar Q_n|`` grid values."""
    absq = np.sort(np.abs(qbar).ravel())  # ascending
    return float(absq[kappa - 1 :].mean())


def compute_tn(
    rx: np.ndarray,
    ry: np.ndarray,
    grid: CLGrid,
    kappa: int,
) -> float:
    """``T_n`` from rank vectors and a precomputed :class:`CLGrid`."""
    return t_statistic(grid.qbar(rx, ry), kappa)
