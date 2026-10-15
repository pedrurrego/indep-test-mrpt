"""Pattern generators from Zhang (2019), Table 1.

Ported from Zhang, K. (2019). "BET on Independence." Journal of the American
Statistical Association, 114(528), 1620-1637. Table 1 on p. 1631.

These are the six patterns cited by Salahub & Oldford (2025) for their §5
quasi-power figure (Figure 11). Salahub's paper explicitly states (§5):

    "The simulation patterns are those first appearing in [Ma & Mao] and used
     again ... in [Zhang 2019]. The latter set of parameters are used here."

Noise schedule (Zhang Table 1 caption):
    At each noise level l = 1, ..., 10,
        eps, eps', eps'' ~iid N(0, (l/40)^2).
    Auxiliary variables:
        U   ~ Uniform[0, 1]
        th  ~ Uniform[-pi, pi]
        W   ~ MultiBern({1, 2, 3}, each 1/3)
        V1  ~ Bern({2, 4}, each 1/2)
        V2  ~ MultiBern({1, 3, 5}, each 1/3)
        G1, G2 ~iid N(0, 1/4)

Signature mirrors `patterns_salahub.py`:
    (n, sigma, rng) -> PatternSample(x_clean, y_clean, x, y, rsq)
where `sigma = l / 40` is the noise scale at level l.

Unlike the Salahub patterns, there is no clean universal x_clean / y_clean
decomposition here — noise enters at the generative stage (e.g. X = cos(th)
+ 2.5*eps mixes signal and noise). We store the "signal component" as the
clean arrays where it's meaningful; rsq is left as NaN because the paper
does not use R^2 as an axis (it uses the discrete level l).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PatternSample:
    x_clean: np.ndarray
    y_clean: np.ndarray
    x: np.ndarray
    y: np.ndarray
    rsq: float


PatternFn = Callable[[int, float, np.random.Generator], PatternSample]

_REGISTRY: dict[str, PatternFn] = {}


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


@register("linear")
def _linear(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    u = rng.uniform(0.0, 1.0, n)
    eps = rng.normal(0.0, sigma, n)
    x = u
    y = u + 6.0 * eps
    return PatternSample(u, u, x, y, float("nan"))


@register("parabolic")
def _parabolic(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    u = rng.uniform(0.0, 1.0, n)
    eps = rng.normal(0.0, sigma, n)
    x = u
    y_clean = (x - 0.5) ** 2
    y = y_clean + 1.5 * eps
    return PatternSample(x, y_clean, x, y, float("nan"))


@register("circular")
def _circular(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    th = rng.uniform(-np.pi, np.pi, n)
    eps = rng.normal(0.0, sigma, n)
    eps2 = rng.normal(0.0, sigma, n)
    x_clean = np.cos(th)
    y_clean = np.sin(th)
    x = x_clean + 2.5 * eps
    y = y_clean + 2.5 * eps2
    return PatternSample(x_clean, y_clean, x, y, float("nan"))


@register("sine")
def _sine(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    u = rng.uniform(0.0, 1.0, n)
    eps = rng.normal(0.0, sigma, n)
    x = u
    y_clean = np.sin(4.0 * np.pi * x)
    y = y_clean + 8.0 * eps
    return PatternSample(x, y_clean, x, y, float("nan"))


@register("checkerboard")
def _checkerboard(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    w = rng.choice(np.array([1, 2, 3]), size=n, replace=True)
    v1 = rng.choice(np.array([2, 4]), size=n, replace=True)
    v2 = rng.choice(np.array([1, 3, 5]), size=n, replace=True)
    eps = rng.normal(0.0, sigma, n)
    eps2 = rng.normal(0.0, sigma, n)
    x_clean = w.astype(float)
    x = x_clean + eps
    y_clean = np.where(w == 2, v1, v2).astype(float)
    y = y_clean + 4.0 * eps2
    return PatternSample(x_clean, y_clean, x, y, float("nan"))


@register("local")
def _local(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    g1 = rng.normal(0.0, 0.5, n)
    g2 = rng.normal(0.0, 0.5, n)
    eps = rng.normal(0.0, sigma, n)
    in_box = (g1 >= 0.0) & (g1 <= 1.0) & (g2 >= 0.0) & (g2 <= 1.0)
    x = g1
    y = np.where(in_box, g1 + eps, g2)
    return PatternSample(g1, np.where(in_box, g1, g2), x, y, float("nan"))


@register("null")
def _null(n: int, sigma: float, rng: np.random.Generator) -> PatternSample:
    """Independent X, Y for empirical Type-I error.

    Sigma is ignored — under the null, the noise is the only thing.
    Used to populate the leftmost ("null") tick on the paper's quasi-power
    figure. Run this pattern with `--levels 1` (any single level is fine,
    the result is independent of sigma).
    """
    x = rng.uniform(0.0, 1.0, n)
    y = rng.uniform(0.0, 1.0, n)
    return PatternSample(x, y, x, y, float("nan"))


def sigma_for_level(level: int) -> float:
    """Zhang schedule: sigma_l = l / 40 for l = 1, ..., 10."""
    return level / 40.0


def default_levels() -> np.ndarray:
    """The ten discrete noise levels used in the paper's Figure 11."""
    return np.arange(1, 11, dtype=int)
