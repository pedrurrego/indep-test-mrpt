"""The rBEX (random bisection expanding cross) pattern.

Zhang (2019) motivates the BET with the *bisection expanding cross* (BEX), a
sequence of one-dimensional manifolds ``BEX_d`` in ``[0, 1]^2`` (his §1.1).
For every integer ``d > 0``,

    BEX_d = { (x, y) in [0,1]^2 : gamma_d(x, y) = 0 },

    gamma_d(x, y) =
        sum_{i=1}^{2^{d-1}} sum_{j=1}^{2^{d-1}}
            ( |x - i/2^{d-1} + 1/2^d| - |y - j/2^{d-1} + 1/2^d| )
            * I(|x - i/2^{d-1} + 1/2^d| <= 1/2^d)
            * I(|y - j/2^{d-1} + 1/2^d| <= 1/2^d).

Geometrically this is a ``2^{d-1} x 2^{d-1}`` grid of square cells, and inside
each cell the manifold is the pair of diagonals ``|x - c_x| = |y - c_y|`` — an
"X". So ``BEX_1`` is one big X spanning the square; ``BEX_2`` is a 2x2 lattice
of X's (a 45-degree diamond grid); and so on, doubling each level. Zhang proves
that ``(X_d, Y_d)`` drawn uniformly on ``BEX_d`` has Uniform[0,1] margins yet a
joint law that, although degenerate (TV distance 1 from independence), grows
arbitrarily close to the independence copula as ``d`` increases — the paradox
that drives the paper.

Salahub & Oldford (2025, §5.3, Figs. rBEX_pattern / quasi_power_rBEX) sample
``N`` points uniformly on ``BEX_d`` and call the resulting scatter the **rBEX
pattern** at *resolution* ``d`` ("to distinguish the random sample from its
support"). No noise is added: the dependence signal is the manifold itself.
Their quasi-power x-axis is ``null, 1, 2, ..., 10`` where "null" is independent
uniform data (the size anchor) and ``1..10`` are BEX resolutions.

We follow that convention exactly:

- ``resolution >= 1`` -> a uniform sample of ``BEX_resolution``.
- ``resolution == 0`` -> independent Uniform[0,1] data. This is the "null"
  anchor (used to estimate size); it is *not* a Zhang BEX level, just the
  ``d -> infinity`` limit made concrete as a left-hand baseline.

Sampling is exact and O(n): a point is placed by choosing a cell uniformly,
choosing one of its two diagonals uniformly, and a uniform position along it.
Because every diagonal segment has equal length, this is uniform on the
manifold, and the margins are Uniform[0,1] by construction.
"""

from __future__ import annotations

import numpy as np


def resolution_to_grid(resolution: int) -> int:
    """Cells per side for a BEX resolution: ``2^{d-1}`` (``d >= 1``).

    resolution 1 -> 1 cell (single X), 2 -> 2, 3 -> 4, ..., i.e. the manifold
    is a ``m x m`` lattice of X-crosses with ``m = 2^{resolution-1}``.
    """
    if resolution < 1:
        raise ValueError("grid only defined for resolution >= 1")
    return 2 ** (resolution - 1)


def rbex_sample(
    n: int, resolution: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` points from the rBEX pattern at the given resolution.

    Parameters
    ----------
    n : int
        Number of points.
    resolution : int
        BEX level ``d >= 1``; ``0`` returns independent Uniform[0,1] data
        (the "null" size anchor, see module docstring).
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    (x, y) : tuple of np.ndarray
        Two length-``n`` arrays, each marginally Uniform[0, 1].
    """
    r = int(resolution)
    if r <= 0:
        return rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, n)

    m = 2 ** (r - 1)  # cells per side
    h = 1.0 / (2 * m)  # cell half-width = 1 / 2^r

    # Uniformly pick a cell (i, j) for each point.
    i = rng.integers(0, m, size=n)
    j = rng.integers(0, m, size=n)
    cx = (i + 0.5) / m  # cell-centre x
    cy = (j + 0.5) / m  # cell-centre y

    # Uniform position t along a diagonal, and a random diagonal (+/-).
    t = rng.uniform(-h, h, size=n)
    s = rng.choice(np.array([-1.0, 1.0]), size=n)

    x = cx + t
    y = cy + s * t
    # Guard against tiny FP excursions outside the unit square.
    np.clip(x, 0.0, 1.0, out=x)
    np.clip(y, 0.0, 1.0, out=y)
    return x, y


def default_resolutions() -> np.ndarray:
    """Default sweep: ``0`` (null) then BEX resolutions ``1..8``.

    Salahub's Figure goes to 10; the cross structure has already dissolved
    into near-independence by 7-8 at the sample sizes we use, so 0..8 captures
    the full rise-and-collapse of the quasi-power curve.
    """
    return np.arange(0, 9, dtype=int)
