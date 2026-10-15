"""Zhang Binary Expansion Testing (BET) for independence.

Implements ``MaxBET`` (single depth) and ``MaxBETs`` (sequential up to a
maximum depth) on bivariate data. Each marginal is ECDF-transformed and
discretised into ``2**d`` cells; for every pair of non-marginal
Walsh-Rademacher binary interactions (BIDs), the symmetry statistic
``S = 2 * n_++ + 2 * n_-- - n`` is computed. The reported test statistic
is the most extreme |S| across all ``(2**d - 1)**2`` cross-interactions,
and the p-value is Bonferroni-corrected. ``MaxBETs`` aggregates depths
``1..d_max`` by rescaling each depth's Bonferroni count to its newly
introduced interactions.

Note: this is the original BET test (Zhang 2019), not BEAST (Zhang et al.
2023). BEAUTY is the unifying theoretical framework that contains both.

Reference
---------
Zhang K. (2019). BET on independence. *Journal of the American Statistical
Association*, 114(528), 1620-1637.

Original R implementation: https://github.com/cran/BET

Provenance
----------
Reference: Zhang, K. (2019). BET on independence. J. Amer. Statist. Assoc.
114(528):1620-1637.
Upstream code: R package ``BET`` (CRAN; https://github.com/cran/BET), GPL-2 | GPL-3.
This module: implemented from the paper; the upstream code served as reference.
Distributed under GPL-3 (LICENSE).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import fisher_exact, norm, rankdata

from indep_tests.base import IndependenceTest, TestResult


def _ecdf_cells(x: np.ndarray, d: int) -> np.ndarray:
    """Map data to cell indices in ``{1, ..., 2**d}`` via ECDF ranks."""
    n = len(x)
    rank = rankdata(x, method="ordinal")  # 1..n
    L = 1 << d
    cells = np.ceil(rank / n * L).astype(np.int64)
    return np.clip(cells, 1, L)


def _unif_cells(u: np.ndarray, d: int) -> np.ndarray:
    """Map already-uniform [0, 1] values to cell indices in ``{1, ..., 2**d}``."""
    L = 1 << d
    cells = np.ceil(np.asarray(u) * L).astype(np.int64)
    return np.clip(cells, 1, L)


def _all_signs(d: int) -> np.ndarray:
    """All ``2**d`` BID sign vectors of length ``2**d``.

    Row 0 is the all-``+1`` marginal (empty subset of basic Rademachers).
    Rows ``1..2**d - 1`` enumerate every non-empty subset of the ``d``
    basic Walsh-Rademacher patterns; row ``s`` is the elementwise product
    of the basic patterns whose indices appear in the binary expansion of
    ``s``. Basic pattern ``k`` (``0 <= k < d``) splits the ``2**d`` cells
    into ``2**(k+1)`` alternating-sign blocks of size ``2**(d-k-1)``.
    """
    L = 1 << d
    cells = np.arange(L)
    basic = np.empty((d, L), dtype=np.int64)
    for k in range(d):
        block = 1 << (d - k - 1)
        basic[k] = np.where((cells // block) & 1, 1, -1)

    S = np.ones((L, L), dtype=np.int64)
    for sub in range(1, L):
        prod = np.ones(L, dtype=np.int64)
        for k in range(d):
            if (sub >> k) & 1:
                prod *= basic[k]
        S[sub] = prod
    return S


def _maxbet_bivariate(
    x: np.ndarray,
    y: np.ndarray,
    d: int,
    asymptotic: bool,
    unif_margin: bool,
) -> dict:
    """Single-depth BET on bivariate data."""
    n = len(x)
    cx = _unif_cells(x, d) if unif_margin else _ecdf_cells(x, d)
    cy = _unif_cells(y, d) if unif_margin else _ecdf_cells(y, d)

    S = _all_signs(d)
    sx = S[:, cx - 1]  # (2**d, n)
    sy = S[:, cy - 1]  # (2**d, n)
    A = sx @ sy.T  # (2**d, 2**d), entries in [-n, +n]
    A_cross = A[1:, 1:]  # drop the row/col corresponding to marginal BID 0

    flat = int(np.argmax(np.abs(A_cross)))
    i, j = np.unravel_index(flat, A_cross.shape)
    stat = int(A_cross[i, j])

    bonf = ((1 << d) - 1) ** 2  # number of cross-interactions tested

    if asymptotic:
        z = abs(stat) / np.sqrt(n)
        p_per_test = 2.0 * (1.0 - norm.cdf(z))
        p_value = float(min(1.0, p_per_test * bonf))
    else:
        sx_star = S[i + 1, cx - 1]
        sy_star = S[j + 1, cy - 1]
        n_pp = int(np.sum((sx_star > 0) & (sy_star > 0)))
        n_pm = int(np.sum((sx_star > 0) & (sy_star < 0)))
        n_mp = int(np.sum((sx_star < 0) & (sy_star > 0)))
        n_mm = int(np.sum((sx_star < 0) & (sy_star < 0)))
        _, p_per_test = fisher_exact([[n_pp, n_pm], [n_mp, n_mm]])
        p_value = float(min(1.0, p_per_test * bonf))

    return {
        "stat": stat,
        "p_value": p_value,
        "interaction": (int(i + 1), int(j + 1)),
        "n": n,
        "bonf": bonf,
    }


def _maxbets_bivariate(
    x: np.ndarray,
    y: np.ndarray,
    d_max: int,
    asymptotic: bool,
    unif_margin: bool,
) -> dict:
    """Sequential BET combining depths ``1..d_max`` with Bonferroni-of-Bonferronis."""
    if d_max < 1:
        raise ValueError("d_max must be >= 1")

    res1 = _maxbet_bivariate(x, y, 1, asymptotic, unif_margin)
    p_adj = [res1["p_value"]]
    extremes = [res1["stat"]]
    interactions = [res1["interaction"]]
    best_p = res1["p_value"]
    best_stat = res1["stat"]
    best_interaction = res1["interaction"]
    best_d = 1

    for d in range(2, d_max + 1):
        res = _maxbet_bivariate(x, y, d, asymptotic, unif_margin)
        # Total cross-interactions at depth d and at depth d-1 (p = 2):
        total_d = ((1 << d) - 1) ** 2
        prev_d = ((1 << (d - 1)) - 1) ** 2
        new_d = total_d - prev_d
        # Rescale Bonferroni: count only the interactions newly available at depth d.
        rescaled = float(min(1.0, res["p_value"] / total_d * new_d))
        p_adj.append(rescaled)
        extremes.append(res["stat"])
        interactions.append(res["interaction"])
        if rescaled < best_p:
            best_p = rescaled
            best_stat = res["stat"]
            best_interaction = res["interaction"]
            best_d = d

    final_p = float(min(min(p_adj) * d_max, 1.0))
    return {
        "p_value": final_p,
        "stat": best_stat,
        "interaction": best_interaction,
        "best_depth": best_d,
        "p_values_per_depth": p_adj,
        "extremes_per_depth": extremes,
        "interactions_per_depth": interactions,
        "n": res1["n"],
    }


class ZhangBETTest(IndependenceTest):
    """Zhang Binary Expansion Testing (BET) for bivariate independence.

    Parameters
    ----------
    mode : {"max", "maxs"}
        ``"max"``: single-depth ``MaxBET`` at depth ``d``.
        ``"maxs"``: sequential ``MaxBETs`` up to ``d_max`` (default).
    d : int
        Depth for ``mode="max"`` (default 4).
    d_max : int
        Maximum depth for ``mode="maxs"`` (default 4).
    asymptotic : bool
        If True (default), use the Gaussian asymptotic p-value with
        Bonferroni correction. If False, use Fisher's exact test on the
        2x2 contingency table at the extreme cross-interaction.
    unif_margin : bool
        Set to True when ``x`` and ``y`` are already known to be
        Uniform[0, 1]; otherwise (default) the ECDF transform is applied.
    """

    def __init__(
        self,
        mode: Literal["max", "maxs"] = "maxs",
        d: int = 4,
        d_max: int = 4,
        asymptotic: bool = True,
        unif_margin: bool = False,
    ) -> None:
        if mode not in ("max", "maxs"):
            raise ValueError(f"unknown mode {mode!r}; expected 'max' or 'maxs'")
        self.mode = mode
        self.d = d
        self.d_max = d_max
        self.asymptotic = asymptotic
        self.unif_margin = unif_margin

    @property
    def name(self) -> str:
        if self.mode == "max":
            return f"ZhangBET-d{self.d}"
        return f"ZhangBETs-dmax{self.d_max}"

    def _run(self, x: np.ndarray, y: np.ndarray) -> dict:
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        if self.mode == "max":
            return _maxbet_bivariate(x, y, self.d, self.asymptotic, self.unif_margin)
        return _maxbets_bivariate(x, y, self.d_max, self.asymptotic, self.unif_margin)

    def statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(self._run(x, y)["stat"])

    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        B: int = 1000,
        seed: int | None = None,
    ) -> TestResult:
        # B and seed are ignored: closed-form Bonferroni p-value.
        result = self._run(x, y)

        extras: dict = {"interaction": result["interaction"]}
        if self.mode == "maxs":
            extras["best_depth"] = result["best_depth"]
            extras["p_values_per_depth"] = result["p_values_per_depth"]
            extras["extremes_per_depth"] = result["extremes_per_depth"]

        return TestResult(
            statistic=float(result["stat"]),
            p_value=float(result["p_value"]),
            n=int(result["n"]),
            B=None,
            test_name=self.name,
            extras=extras,
        )
