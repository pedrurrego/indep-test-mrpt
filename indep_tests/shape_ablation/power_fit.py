"""Fit equal-area power diagrams (weighted Voronoi) with Halton seeds + Lloyd steps.

Weights w solve the semi-discrete optimal-transport problem: cell_i = {p : |p-s_i|^2 - w_i
minimal} with area(cell_i) = 1/K. They are fitted by ascent on the concave Kantorovich dual,
``w_i += step * (1/K - area_i)``, with areas measured on a raster. Lloyd steps move seeds to
cell centroids (compact, hexagon-like interiors). Results are cached in ``cache/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE = Path(__file__).parent / "cache"
FIT_RES = 512
AREA_RES = 1024


def halton(n: int, base: int) -> np.ndarray:
    out = np.zeros(n)
    for i in range(n):
        f, r, k = 1.0, 0.0, i + 1
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        out[i] = r
    return out


def power_labels(
    seeds: np.ndarray, w: np.ndarray, px: np.ndarray, py: np.ndarray
) -> np.ndarray:
    """argmin_i |p - s_i|^2 - w_i  ==  argmin_i (|s_i|^2 - w_i) - 2 p.s_i  (|p|^2 dropped)."""
    P = np.column_stack([px.ravel(), py.ravel()]).astype(np.float32)
    S = seeds.astype(np.float32)
    D = -2.0 * (P @ S.T) + ((S**2).sum(1) - w.astype(np.float32))[None, :]
    return D.argmin(axis=1).reshape(np.shape(px))


def _raster(res: int):
    g = (np.arange(res) + 0.5) / res
    PX, PY = np.meshgrid(g, g, indexing="ij")
    return PX.ravel(), PY.ravel()


def _areas(seeds, w, res: int) -> np.ndarray:
    """Cell areas measured on a res x res raster, in row chunks to bound memory."""
    K = len(seeds)
    g = (np.arange(res) + 0.5) / res
    counts = np.zeros(K)
    step = max(1, 2_000_000 // (res * K))
    for r0 in range(0, res, step):
        PX, PY = np.meshgrid(g[r0 : r0 + step], g, indexing="ij")
        lab = power_labels(seeds, w, PX.ravel(), PY.ravel())
        counts += np.bincount(lab, minlength=K)
    return counts / (res * res)


def fit_weights(seeds, px, py, w=None, iters=3000, step=0.2, tol=3e-3):
    K, target = len(seeds), 1.0 / len(seeds)
    w = np.zeros(K) if w is None else w.copy()
    for _ in range(iters):
        lab = power_labels(seeds, w, px, py)
        area = np.bincount(lab, minlength=K) / len(px)
        if np.abs(area * K - 1).max() < tol:
            break
        w += step * (target - area)
    return w


def fit_power_diagram(K: int, lloyd_steps: int = 5, fit_res: int = FIT_RES):
    px, py = _raster(fit_res)
    seeds = np.column_stack([halton(K, 2), halton(K, 3)])
    w = None
    for _ in range(lloyd_steps):
        w = fit_weights(seeds, px, py, w, iters=600)
        lab = power_labels(seeds, w, px, py)
        cnt = np.bincount(lab, minlength=K)[:, None]
        seeds = (
            np.column_stack([np.bincount(lab, px, K), np.bincount(lab, py, K)]) / cnt
        )
    w = fit_weights(seeds, px, py, w)
    area = _areas(seeds, w, AREA_RES)
    return seeds, w, area


def get_power_diagram(K: int):
    """(seeds, w, areas) for an equal-area centroidal power diagram with K cells (cached)."""
    f = CACHE / f"power_K{K}_fit{FIT_RES}_lloyd5.npz"
    if f.exists():
        z = np.load(f)
        return z["seeds"], z["w"], z["area"]
    CACHE.mkdir(exist_ok=True)
    seeds, w, area = fit_power_diagram(K)
    np.savez(f, seeds=seeds, w=w, area=area)
    return seeds, w, area


if __name__ == "__main__":
    import sys
    import time

    Ks = [int(a) for a in sys.argv[1:]] or [9, 16, 25, 49, 64, 100, 144, 196, 484]
    for K in Ks:
        t0 = time.perf_counter()
        _, _, area = get_power_diagram(K)
        print(
            f"K={K:4d}  max|area*K-1|={np.abs(area * K - 1).max():.2e}  {time.perf_counter() - t0:.0f}s",
            flush=True,
        )
