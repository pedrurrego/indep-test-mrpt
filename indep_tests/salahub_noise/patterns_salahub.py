"""Pattern generators for the Salahub & Oldford (2025) noise-vs-power benchmark.

Ported from `scripts/noise.R` in https://github.com/Salahub/recursive-binning.

Each generator has signature `(n, sigma, rng) -> PatternSample` and returns the
clean pattern, the observed (noisy) pattern, and the sample R^2 used as the
x-axis metric in the paper (R^2 = var(y_clean) / var(y_obs), so 1 - R^2 is the
noise fraction of the total variance).

Noise convention matches the paper:
- Non-functional patterns add i.i.d. N(0, sigma^2) noise to BOTH x and y
  (the point cloud is genuinely 2D, not a function of x).
- Functional patterns add noise only to y (x is the free variable).
- rBEX is treated as non-functional.

The 2D "die" patterns (`four`, `five`) split n into equal chunks; for
reproducibility match the paper, use n divisible by 20. Non-divisible n
still works — the last chunk absorbs the remainder.

Provenance
----------
Reference: Salahub, C. & Oldford, R. W. (2025). Recursive random binning to
detect and display pairwise dependence. Statistical Analysis and Data Mining
18(5):e70042 (Section 5, noise benchmark).
Upstream code: ``scripts/noise.R`` in https://github.com/Salahub/recursive-binning
(no license stated in that repository).
This module: the paper's pattern definitions and noise convention written in
Python, with the R script as reference. Distributed under GPL-3 (LICENSE).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PatternSample:
    """One realization of a pattern, clean and noisy."""

    x_clean: np.ndarray
    y_clean: np.ndarray
    x: np.ndarray
    y: np.ndarray
    rsq: float


PatternFn = Callable[[int, float, np.random.Generator], PatternSample]

_REGISTRY: dict[str, PatternFn] = {}

# Shared coordinate range used by every pattern in noise.R.
_RNG_LO = 0.0
_RNG_HI = 6.0 * np.pi


def register(name: str):
    def decorator(fn: PatternFn) -> PatternFn:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_pattern(name: str) -> PatternFn:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown pattern '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_patterns() -> list[str]:
    return sorted(_REGISTRY.keys())


def _rsq(y_clean: np.ndarray, y_obs: np.ndarray) -> float:
    var_obs = float(np.var(y_obs))
    if var_obs == 0.0:
        return 1.0
    return float(np.var(y_clean)) / var_obs


def _noise_xy(
    x_clean: np.ndarray,
    y_clean: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> PatternSample:
    """Add independent N(0, sigma^2) noise to both x and y."""
    x_obs = x_clean + rng.normal(0.0, sigma, size=x_clean.shape)
    y_obs = y_clean + rng.normal(0.0, sigma, size=y_clean.shape)
    return PatternSample(x_clean, y_clean, x_obs, y_obs, _rsq(y_clean, y_obs))


def _noise_y(
    x_clean: np.ndarray,
    y_clean: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> PatternSample:
    """Add noise to y only (functional patterns). x is passed through unchanged."""
    y_obs = y_clean + rng.normal(0.0, sigma, size=y_clean.shape)
    return PatternSample(x_clean, y_clean, x_clean.copy(), y_obs, _rsq(y_clean, y_obs))


# --- Non-functional patterns (noise.R: exFuns) ---


@register("sine")
def _sine(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    x = rng.uniform(_RNG_LO, _RNG_HI, n)
    y = np.sin(x) * (_RNG_HI - _RNG_LO) / 2.0 + _RNG_LO
    return _noise_xy(x, y, sigma, rng)


@register("nonco")
def _nonco(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    """Non-coexistence: either x=0 with y random, or y=0 with x random."""
    n_half = n // 2
    x = np.concatenate([np.zeros(n_half), rng.uniform(_RNG_LO, _RNG_HI, n - n_half)])
    y = rng.uniform(_RNG_LO, _RNG_HI, n) * (x == 0.0)
    return _noise_xy(x, y, sigma, rng)


@register("lines")
def _two_lines(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    x = rng.uniform(_RNG_LO, _RNG_HI, n)
    u = (rng.uniform(size=n) >= 0.5).astype(int)
    slopes = np.array([0.2, 1.0])[u]
    y = x * slopes
    return _noise_xy(x, y, sigma, rng)


@register("x")
def _x_pattern(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    x = rng.uniform(_RNG_LO, _RNG_HI, n)
    u = (rng.uniform(size=n) >= 0.5).astype(int)
    signs = np.array([1.0, -1.0])[u]
    offsets = np.array([0.0, float(x.max())])[u]
    y = x * signs + offsets
    return _noise_xy(x, y, sigma, rng)


@register("ellipse")
def _ellipse(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    x = rng.uniform(_RNG_LO, _RNG_HI, n)
    u = (rng.uniform(size=n) >= 0.5).astype(int)
    scl = (_RNG_HI - _RNG_LO) / 2.0
    mean = (_RNG_LO + _RNG_HI) / 2.0
    radicand = np.clip(1.0 - ((x - mean) / scl) ** 2, 0.0, None)
    signs = np.array([1.0, -1.0])[u]
    y = signs * scl * np.sqrt(radicand)
    return _noise_xy(x, y, sigma, rng)


@register("four")
def _four(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    pos1 = _RNG_LO + 0.25 * (_RNG_HI - _RNG_LO)
    pos2 = _RNG_LO + 0.75 * (_RNG_HI - _RNG_LO)
    n2 = n // 2
    n4 = n // 4
    x = np.concatenate([np.full(n2, pos1), np.full(n - n2, pos2)])
    y = np.concatenate(
        [
            np.full(n4, pos1),
            np.full(n2 - n4, pos2),
            np.full(n4, pos1),
            np.full(n - n2 - n4, pos2),
        ]
    )
    return _noise_xy(x, y, sigma, rng)


@register("five")
def _five(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    pos1 = _RNG_LO + 0.25 * (_RNG_HI - _RNG_LO)
    pos2 = _RNG_LO + 0.75 * (_RNG_HI - _RNG_LO)
    mid = (_RNG_LO + _RNG_HI) / 2.0
    n5 = n // 5
    x = np.concatenate(
        [np.full(2 * n5, pos1), np.full(n5, mid), np.full(n - 3 * n5, pos2)]
    )
    y = np.concatenate(
        [
            np.full(n5, pos1),
            np.full(n5, pos2),
            np.full(n5, mid),
            np.full(n5, pos1),
            np.full(n - 4 * n5, pos2),
        ]
    )
    return _noise_xy(x, y, sigma, rng)


# --- Functional patterns (noise.R: funcFuns) ---


@register("line")
def _line(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    x = rng.uniform(_RNG_LO, _RNG_HI, n)
    y = x.copy()
    return _noise_y(x, y, sigma, rng)


@register("parabola")
def _parabola(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    x = rng.uniform(_RNG_LO, _RNG_HI, n)
    mean = (_RNG_LO + _RNG_HI) / 2.0
    y = (_RNG_HI - _RNG_LO) * (x - mean) ** 2 / (mean**2)
    return _noise_y(x, y, sigma, rng)


@register("cubic")
def _cubic(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    x = rng.uniform(_RNG_LO, _RNG_HI, n)
    mean = (_RNG_LO + _RNG_HI) / 2.0
    scl = (_RNG_HI - _RNG_LO) / 2.0
    xadj = x - mean
    y = (
        (_RNG_HI - _RNG_LO)
        * (20.0 * xadj**3 / scl**3 - 6.0 * xadj**2 / scl**2 + xadj / scl)
        / 42.0
    )
    return _noise_y(x, y, sigma, rng)


@register("exponential")
def _exponential(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    x = rng.uniform(_RNG_LO, _RNG_HI, n)
    y = (_RNG_HI - _RNG_LO) * np.exp(x) / np.exp(_RNG_HI - _RNG_LO)
    return _noise_y(x, y, sigma, rng)


# --- Adversarial pattern: rBEX (noise.R: rBEX) ---


@register("rbex")
def _rbex(
    n: int,
    sigma: float,
    rng: np.random.Generator,
    res: int = 4,
) -> PatternSample:
    """X-shapes sampled in cells of a regular 2^(res-1) x 2^(res-1) grid on [0,1]^2."""
    ncells = 2 ** (res - 1)
    edges = np.linspace(0.0, 1.0, ncells + 1)
    xbin = rng.integers(0, ncells, n)
    ybin = rng.integers(0, ncells, n)
    xlo, xhi = edges[xbin], edges[xbin + 1]
    ylo, yhi = edges[ybin], edges[ybin + 1]
    x = rng.uniform(xlo, xhi, n)
    u = rng.integers(0, 2, n)
    t = (x - xlo) / (xhi - xlo)
    upslope = u == 0
    y = np.where(upslope, ylo + (yhi - ylo) * t, yhi - (yhi - ylo) * t)
    return _noise_xy(x, y, sigma, rng)
