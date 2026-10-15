"""Partitions of the unit square with vectorised cell labelling.

Every partition exposes ``K`` (cell count), ``areas`` (length-K, sums to 1) and
``codes(u, v)``: cell index of the points ``(u, v)`` where ``u`` has shape ``(n,)``
and ``v`` any shape broadcastable to ``(..., n)`` — so a ``(B, n)`` array of permuted
``v`` labels all B permutations in one call.

Geometries
----------
square  : regular k x k grid, analytic binning (identical to ``src_pug_grid``).
rect    : axis-aligned k1 x k2 grid with 2:1 aspect (cells twice as wide as tall),
          k1 = round(k / sqrt 2), k2 = 2 k1, so the cell count is within ~12% of k*k.
rotated : the k x k grid rotated by 45 degrees about the centre of the square, cells
          clipped to the square (exact polygon areas); boundary fragments smaller than
          half a cell are merged into their inward lattice neighbour. Not equal-area:
          the Pearson statistic uses the measured areas.
golden  : recursive equal-area bisection with cut directions ``(l+1) * golden_angle mod pi``
          at level ``l``; ``2**depth`` convex cells, exact tree-walk labelling.
power   : centroidal equal-area power diagram with Halton seeds (see ``power_fit.py``);
          exact weighted-distance argmin labelling, areas measured on a 1024^2 raster.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
_MULTI_K = (3, 5, 8, 12)  # the ladder of src_pug_grid density-multi


# ----------------------------------------------------------------- base
class Partition:
    K: int
    areas: np.ndarray
    label: str
    equal_area: bool = (
        True  # exact by construction (square, golden); False for power diagrams
    )

    def codes(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class SquareGrid(Partition):
    def __init__(self, k: int):
        self.k, self.K = k, k * k
        self.areas = np.full(self.K, 1.0 / self.K)
        self.label = f"square k={k}"

    def codes(self, u, v):
        k = self.k
        cu = np.clip(np.ceil(u * k).astype(np.int64) - 1, 0, k - 1)
        cv = np.clip(np.ceil(v * k).astype(np.int64) - 1, 0, k - 1)
        return cu * k + cv


# ----------------------------------------------------------------- golden bisection
def _polygon_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip(P, n, c):
    out, m = [], len(P)
    for i in range(m):
        a, b = P[i], P[(i + 1) % m]
        da, db = n @ a - c, n @ b - c
        if da <= 0:
            out.append(a)
        if (da < 0 < db) or (db < 0 < da):
            out.append(a + da / (da - db) * (b - a))
    return np.array(out) if out else np.empty((0, 2))


def _bisect_offset(P, n):
    proj = P @ n
    lo, hi, target = proj.min(), proj.max(), _polygon_area(P) / 2
    for _ in range(60):
        c = 0.5 * (lo + hi)
        if _polygon_area(_clip(P, n, c)) < target:
            lo = c
        else:
            hi = c
    return 0.5 * (lo + hi)


class GoldenBisection(Partition):
    """Binary tree of half-plane cuts; node at level l has offset offsets[l][node]."""

    def __init__(self, depth: int):
        self.depth, self.K = depth, 2**depth
        self.normals = np.zeros((depth, 2))
        self.offsets = []
        cells = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]
        for level in range(depth):
            theta = ((level + 1) * GOLDEN_ANGLE) % math.pi
            n = np.array([math.cos(theta), math.sin(theta)])
            self.normals[level] = n
            offs, nxt = np.zeros(len(cells)), []
            for j, P in enumerate(cells):
                c = _bisect_offset(P, n)
                offs[j] = c
                nxt += [_clip(P, n, c), _clip(P, -n, -c)]  # child 0: n.p <= c
            self.offsets.append(offs)
            cells = nxt
        self.cells = cells
        self.areas = np.array([_polygon_area(P) for P in cells])
        self.areas /= self.areas.sum()
        self.label = f"golden depth={depth}"

    def codes(self, u, v):
        u = np.asarray(u)
        v = np.asarray(v)
        node = np.zeros(np.broadcast(u, v).shape, dtype=np.int64)
        for level in range(self.depth):
            nx, ny = self.normals[level]
            side = (nx * u + ny * v > self.offsets[level][node]).astype(np.int64)
            node = 2 * node + side
        return node


# ----------------------------------------------------------------- power diagram
class PowerDiagram(Partition):
    def __init__(self, K: int):
        from .power_fit import get_power_diagram

        self.seeds, self.w, self.areas = get_power_diagram(K)
        self.K = K
        self.equal_area = False
        self.S = self.seeds.astype(np.float32)
        self.bias = ((self.S**2).sum(1) - self.w.astype(np.float32))[None, :]
        self.label = f"power K={K}"

    def codes(self, u, v):
        u, v = np.broadcast_arrays(
            np.asarray(u, dtype=np.float32), np.asarray(v, dtype=np.float32)
        )
        shape = u.shape
        P = np.column_stack([u.ravel(), v.ravel()])
        out = np.empty(len(P), dtype=np.int64)
        chunk = max(1, 4_000_000 // self.K)
        for i in range(0, len(P), chunk):
            D = -2.0 * (P[i : i + chunk] @ self.S.T) + self.bias
            out[i : i + chunk] = D.argmin(axis=1)
        return out.reshape(shape)


# ----------------------------------------------------------------- rectangles
class RectGrid(Partition):
    """Axis-aligned k1 x k2 grid: k1 columns along u, k2 rows along v (equal area)."""

    def __init__(self, k1: int, k2: int):
        self.k1, self.k2, self.K = k1, k2, k1 * k2
        self.areas = np.full(self.K, 1.0 / self.K)
        self.label = f"rect {k1}x{k2}"

    def codes(self, u, v):
        cu = np.clip(np.ceil(u * self.k1).astype(np.int64) - 1, 0, self.k1 - 1)
        cv = np.clip(np.ceil(v * self.k2).astype(np.int64) - 1, 0, self.k2 - 1)
        return cu * self.k2 + cv


# ----------------------------------------------------------------- rotated grid
_UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
_SQUARE_HALFPLANES = (  # (normal, offset): the square is {p : n.p <= c for all}
    (np.array([-1.0, 0.0]), 0.0),
    (np.array([1.0, 0.0]), 1.0),
    (np.array([0.0, -1.0]), 0.0),
    (np.array([0.0, 1.0]), 1.0),
)


class RotatedGrid(Partition):
    """k x k lattice of side 1/k rotated by ``theta`` about (1/2, 1/2), clipped to the
    unit square. Lattice origin at the centre. Fragments with clipped area below
    ``merge_frac`` of a full cell are merged into the neighbouring lattice cell one
    step towards the centre (repeatedly, until a large-enough cell is reached)."""

    def __init__(self, k: int, theta: float = math.pi / 4, merge_frac: float = 0.5):
        self.k, self.theta, self.merge_frac = k, theta, merge_frac
        s = 1.0 / k
        c, sn = math.cos(theta), math.sin(theta)
        self.R = np.array([[c, -sn], [sn, c]])  # original = centre + R @ rotated
        m = int(math.ceil(math.sqrt(0.5) / s)) + 1
        self.s, self.m, M = s, m, 2 * m
        raw = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                x0, y0 = (i - m) * s, (j - m) * s
                P = np.array([[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]])
                P = P @ self.R.T + 0.5
                for nrm, off in _SQUARE_HALFPLANES:
                    P = _clip(P, nrm, off)
                    if len(P) < 3:
                        P = np.empty((0, 2))
                        break
                raw[i, j] = _polygon_area(P) if len(P) >= 3 else 0.0
        self.raw_areas = raw
        thr = merge_frac * s * s
        target = -np.ones((M, M), dtype=np.int64)
        big = np.argwhere(raw >= thr)
        for idx, (i, j) in enumerate(big):
            target[i, j] = idx
        self.n_full = len(big)
        self.n_merged = 0
        for i in range(M):
            for j in range(M):
                if target[i, j] >= 0 or raw[i, j] == 0.0:
                    continue
                ii, jj = i, j
                for _ in range(M):
                    di = -int(np.sign(ii - m + 0.5))
                    dj = -int(np.sign(jj - m + 0.5))
                    cands = [(ii + di, jj), (ii, jj + dj)]
                    ii, jj = max(cands, key=lambda t: raw[t])
                    if target[ii, jj] >= 0:
                        break
                target[i, j] = target[ii, jj]
                self.n_merged += 1
        # zero-area lattice cells (never hit by points inside the square) fall back to
        # the nearest inward neighbour too, so any clipped index maps somewhere.
        for i in range(M):
            for j in range(M):
                if target[i, j] < 0:
                    ii, jj = i, j
                    while target[ii, jj] < 0:
                        ii -= int(np.sign(ii - m + 0.5))
                        jj -= int(np.sign(jj - m + 0.5))
                    target[i, j] = target[ii, jj]
        self.lut = target
        self.K = int(target.max()) + 1
        areas = np.zeros(self.K)
        np.add.at(areas, target.ravel(), raw.ravel())
        self.areas = areas / areas.sum()
        self.equal_area = False
        self.label = f"rotated k={k}"

    def codes(self, u, v):
        u, v = np.broadcast_arrays(
            np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        )
        du, dv = u - 0.5, v - 0.5
        (c, sn) = self.R[0, 0], self.R[1, 0]
        px = c * du + sn * dv  # R^T (p - centre)
        py = -sn * du + c * dv
        M = 2 * self.m
        i = np.clip(np.floor(px / self.s).astype(np.int64) + self.m, 0, M - 1)
        j = np.clip(np.floor(py / self.s).astype(np.int64) + self.m, 0, M - 1)
        return self.lut[i, j]


# ----------------------------------------------------------------- ladders
def ladder_ks(n: int) -> list[int]:
    """The density-multi ladder: sorted({3,5,8,12 : k < floor(sqrt n)} | {floor(sqrt n)})."""
    k_sqrt = int(math.floor(math.sqrt(n)))
    return sorted({k for k in _MULTI_K if k < k_sqrt} | {k_sqrt})


@lru_cache(maxsize=None)
def make_partition(geometry: str, k: int) -> Partition:
    """Partition of the given family with cell count matched to k*k (exact for square,
    power and the interior of rotated; 2 round(k/sqrt2)^2 for rect; nearest power of two
    for golden)."""
    if geometry == "square":
        return SquareGrid(k)
    if geometry == "rect":
        k1 = max(1, round(k / math.sqrt(2.0)))
        return RectGrid(k1, 2 * k1)
    if geometry == "rotated":
        return RotatedGrid(k)
    if geometry == "golden":
        return GoldenBisection(max(1, round(math.log2(k * k))))
    if geometry == "power":
        return PowerDiagram(k * k)
    raise ValueError(f"unknown geometry {geometry!r}")


# ------------------------------------------------ single-partition resolution profile
# Nominal resolutions k (cells ~ k^2) of the per-family power-vs-resolution sweep
# (paper "shape" section: one curve per geometry, x = cell count K, fixed n). The
# cap is k <= floor(sqrt n), i.e. K <= n (at least one expected point per cell) --
# the same ceiling as the MRPT ladder's top rung. Each family is evaluated at its
# own achievable cell counts; the exactly matched counts (16, 64, 256 for square /
# golden / power) are a subset, not a requirement.
_PROFILE_KS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22)
_PROFILE_POWER_KS = (4, 9, 16, 25, 36, 49, 64, 81, 100, 144, 196, 256, 324, 400, 484)


def profile_Ks(geometry: str, n: int) -> list[int]:
    """Cell counts of the resolution profile of one family at sample size n (K <= n)."""
    k_max = int(math.floor(math.sqrt(n)))
    if geometry == "square":
        return [k * k for k in _PROFILE_KS if k <= k_max]
    if geometry == "rotated":
        # boundary merging maps neighbouring nominal k to the same cell count
        # (k=3,4 -> 12 cells; k=6,7 -> 40): keep the first nominal K of each
        out, seen = [], set()
        for k in _PROFILE_KS:
            if k > k_max:
                break
            K_actual = make_partition_K("rotated", k * k).K
            if K_actual not in seen:
                seen.add(K_actual)
                out.append(k * k)
        return out
    if geometry == "rect":
        return [2 * k1 * k1 for k1 in range(2, 16) if 2 * k1 * k1 <= n]
    if geometry == "golden":
        return [2**d for d in range(2, 10) if 2**d <= n]
    if geometry == "power":
        return [K for K in _PROFILE_POWER_KS if K <= n]
    raise ValueError(f"unknown geometry {geometry!r}")


@lru_cache(maxsize=None)
def make_partition_K(geometry: str, K: int) -> Partition:
    """Partition of the given family with (nominal) cell count K.

    square: k = sqrt(K) (K must be a perfect square); rect: k1 x 2k1 with K = 2 k1^2;
    rotated: the k = sqrt(K) grid rotated (actual count differs after merging);
    golden: depth log2(K) (K a power of two); power: exactly K cells.
    """
    if geometry == "square":
        k = round(math.sqrt(K))
        assert k * k == K, K
        return SquareGrid(k)
    if geometry == "rect":
        k1 = round(math.sqrt(K / 2))
        assert 2 * k1 * k1 == K, K
        return RectGrid(k1, 2 * k1)
    if geometry == "rotated":
        k = round(math.sqrt(K))
        assert k * k == K, K
        return RotatedGrid(k)
    if geometry == "golden":
        d = round(math.log2(K))
        assert 2**d == K, K
        return GoldenBisection(d)
    if geometry == "power":
        return PowerDiagram(K)
    raise ValueError(f"unknown geometry {geometry!r}")


def partitions_for(geometry: str, n: int, mode: str, single_k: int) -> list[Partition]:
    ks = ladder_ks(n) if mode == "ladder" else [single_k]
    parts, seen = [], set()
    for k in ks:
        P = make_partition(geometry, k)
        if P.label not in seen:  # golden rounding can map two rungs to one depth
            parts.append(P)
            seen.add(P.label)
    return parts
