"""Recursive random binning of paired ranks (Salahub & Oldford, 2025).

Port of the core algorithm in the R package `AssocBin`. Bins are rectangular
regions of the rank lattice [0, n] x [0, n]; each recursive step picks the
longer edge (``squarified``) and splits it at a uniformly random integer
position subject to a minimum-expected-count margin.

Provenance
----------
Reference: Salahub, C. & Oldford, R. W. (2025). Recursive random binning to
detect and display pairwise dependence. Statistical Analysis and Data Mining
18(5):e70042.
Upstream code: R package ``AssocBin`` (CRAN), GPL (>= 3).
This module: port of the binning algorithm from the upstream R code.
Distributed under GPL-3 (LICENSE).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Bin:
    x_lo: int
    x_hi: int
    y_lo: int
    y_hi: int
    x_ranks: np.ndarray
    y_ranks: np.ndarray
    expn: float
    depth: int
    stopped: bool = False

    @property
    def n_obs(self) -> int:
        return int(self.x_ranks.size)

    @property
    def dx(self) -> int:
        return self.x_hi - self.x_lo

    @property
    def dy(self) -> int:
        return self.y_hi - self.y_lo


def rank_random_ties(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Integer ranks in 1..n with ties broken at random."""
    n = x.size
    tiebreak = rng.permutation(n)
    order = np.lexsort((tiebreak, x))
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    return ranks


def _split_axis(
    b: Bin,
    axis: str,
    min_exp: float,
    rng: np.random.Generator,
) -> list[Bin] | None:
    """Split bin ``b`` on the given axis. Returns None if not splittable."""
    if axis == "x":
        delta = b.dx
        lo, hi = b.x_lo, b.x_hi
        coords = b.x_ranks
    else:
        delta = b.dy
        lo, hi = b.y_lo, b.y_hi
        coords = b.y_ranks

    if delta <= 1 or b.expn <= 0:
        return None

    margin = int(np.ceil(min_exp * delta / b.expn))
    lower = lo + max(1, margin)
    upper = hi - max(1, margin)
    if upper < lower:
        return None

    cut = int(rng.integers(lower, upper + 1))

    below_mask = coords <= cut
    above_mask = ~below_mask

    frac_below = (cut - lo) / delta
    frac_above = (hi - cut) / delta
    expn_below = b.expn * frac_below
    expn_above = b.expn * frac_above

    if axis == "x":
        b_below = Bin(
            x_lo=lo,
            x_hi=cut,
            y_lo=b.y_lo,
            y_hi=b.y_hi,
            x_ranks=b.x_ranks[below_mask],
            y_ranks=b.y_ranks[below_mask],
            expn=expn_below,
            depth=b.depth + 1,
        )
        b_above = Bin(
            x_lo=cut,
            x_hi=hi,
            y_lo=b.y_lo,
            y_hi=b.y_hi,
            x_ranks=b.x_ranks[above_mask],
            y_ranks=b.y_ranks[above_mask],
            expn=expn_above,
            depth=b.depth + 1,
        )
    else:
        b_below = Bin(
            x_lo=b.x_lo,
            x_hi=b.x_hi,
            y_lo=lo,
            y_hi=cut,
            x_ranks=b.x_ranks[below_mask],
            y_ranks=b.y_ranks[below_mask],
            expn=expn_below,
            depth=b.depth + 1,
        )
        b_above = Bin(
            x_lo=b.x_lo,
            x_hi=b.x_hi,
            y_lo=cut,
            y_hi=hi,
            x_ranks=b.x_ranks[above_mask],
            y_ranks=b.y_ranks[above_mask],
            expn=expn_above,
            depth=b.depth + 1,
        )

    return [b_below, b_above]


def _r_int_split(
    b: Bin,
    min_exp: float,
    squarify: bool,
    rng: np.random.Generator,
) -> list[Bin]:
    """Random integer split; squarified = split longer edge. Matches rIntSplit."""
    if squarify:
        if b.dx > b.dy:
            primary, secondary = "x", "y"
        elif b.dy > b.dx:
            primary, secondary = "y", "x"
        else:
            primary = "x" if rng.random() < 0.5 else "y"
            secondary = "y" if primary == "x" else "x"
    else:
        primary = "x" if rng.random() < 0.5 else "y"
        secondary = "y" if primary == "x" else "x"

    children = _split_axis(b, primary, min_exp, rng)
    if children is None:
        children = _split_axis(b, secondary, min_exp, rng)
    if children is None:
        b.stopped = True
        return [b]
    return children


def _half_split(b: Bin, rng: np.random.Generator) -> list[Bin]:
    """Initial split at the median rank on a random axis (AssocBin halfSplit)."""
    axis = "x" if rng.random() < 0.5 else "y"
    coords = b.x_ranks if axis == "x" else b.y_ranks
    if coords.size < 2:
        b.stopped = True
        return [b]
    sorted_coords = np.sort(coords)
    mid_idx = coords.size // 2
    cut = int(sorted_coords[mid_idx - 1])
    lo = b.x_lo if axis == "x" else b.y_lo
    hi = b.x_hi if axis == "x" else b.y_hi
    if cut <= lo or cut >= hi:
        cut = (lo + hi) // 2
        if cut <= lo:
            cut = lo + 1
        if cut >= hi:
            cut = hi - 1

    below_mask = coords <= cut
    above_mask = ~below_mask

    delta = hi - lo
    expn_below = b.expn * (cut - lo) / delta
    expn_above = b.expn * (hi - cut) / delta

    if axis == "x":
        return [
            Bin(
                x_lo=lo,
                x_hi=cut,
                y_lo=b.y_lo,
                y_hi=b.y_hi,
                x_ranks=b.x_ranks[below_mask],
                y_ranks=b.y_ranks[below_mask],
                expn=expn_below,
                depth=1,
            ),
            Bin(
                x_lo=cut,
                x_hi=hi,
                y_lo=b.y_lo,
                y_hi=b.y_hi,
                x_ranks=b.x_ranks[above_mask],
                y_ranks=b.y_ranks[above_mask],
                expn=expn_above,
                depth=1,
            ),
        ]
    return [
        Bin(
            x_lo=b.x_lo,
            x_hi=b.x_hi,
            y_lo=lo,
            y_hi=cut,
            x_ranks=b.x_ranks[below_mask],
            y_ranks=b.y_ranks[below_mask],
            expn=expn_below,
            depth=1,
        ),
        Bin(
            x_lo=b.x_lo,
            x_hi=b.x_hi,
            y_lo=cut,
            y_hi=hi,
            x_ranks=b.x_ranks[above_mask],
            y_ranks=b.y_ranks[above_mask],
            expn=expn_above,
            depth=1,
        ),
    ]


def _bin_stopped(b: Bin, max_depth: int, min_exp: float) -> bool:
    if b.stopped:
        return True
    if b.depth >= max_depth:
        return True
    if b.n_obs < 1:
        return True
    if b.expn <= 2 * min_exp:
        return True
    return False


def binner(
    x_ranks: np.ndarray,
    y_ranks: np.ndarray,
    max_depth: int,
    min_exp: float,
    squarify: bool,
    rng: np.random.Generator,
) -> list[Bin]:
    """Iterative recursive random binner (``Algorithm 1`` of the paper)."""
    n = x_ranks.size
    root = Bin(
        x_lo=0,
        x_hi=n,
        y_lo=0,
        y_hi=n,
        x_ranks=x_ranks,
        y_ranks=y_ranks,
        expn=float(n),
        depth=0,
    )
    bins: list[Bin] = _half_split(root, rng)

    while True:
        new_bins: list[Bin] = []
        any_split = False
        for b in bins:
            if _bin_stopped(b, max_depth, min_exp):
                new_bins.append(b)
                continue
            children = _r_int_split(b, min_exp, squarify, rng)
            if len(children) == 1 and children[0] is b:
                new_bins.append(b)
            else:
                any_split = True
                new_bins.extend(children)
        bins = new_bins
        if not any_split:
            break
    return bins


def bin_chi(bins: list[Bin]) -> float:
    """Pearson chi-squared statistic from a list of final bins."""
    stat = 0.0
    for b in bins:
        if b.expn <= 0:
            continue
        diff = b.n_obs - b.expn
        stat += diff * diff / b.expn
    return float(stat)
