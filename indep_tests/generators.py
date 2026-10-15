"""Registry of data generators for independence test experiments."""

from typing import Callable

import numpy as np

GeneratorFn = Callable[[int, np.random.Generator], tuple[np.ndarray, np.ndarray]]

_REGISTRY: dict[str, GeneratorFn] = {}


def register(name: str) -> Callable[[GeneratorFn], GeneratorFn]:
    """Decorator to register a data generator."""

    def decorator(fn: GeneratorFn) -> GeneratorFn:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_generator(name: str) -> GeneratorFn:
    """Look up a generator by name. Raises KeyError if not found."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown generator '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_generators() -> list[str]:
    """Return sorted list of registered generator names."""
    return sorted(_REGISTRY.keys())


# --- Built-in generators ---


@register("indep-normal")
def _indep_normal(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Independent standard normal marginals."""
    X = rng.standard_normal(n)
    Y = rng.standard_normal(n)
    return X, Y


@register("indep-uniform")
def _indep_uniform(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Independent uniform [0,1] marginals."""
    X = rng.uniform(0, 1, n)
    Y = rng.uniform(0, 1, n)
    return X, Y


@register("indep-expon")
def _indep_expon(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Independent standard exponential marginals."""
    X = rng.standard_exponential(n)
    Y = rng.standard_exponential(n)
    return X, Y


@register("dep-normal-0.3")
def _dep_normal_03(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Bivariate normal with rho=0.3."""
    rho = 0.3
    data = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    return data[:, 0], data[:, 1]


@register("dep-yanagimoto")
def _dep_yanagimoto(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Yanagimoto parallelogram counterexample."""
    from pug.utils_yanagimoto import yanagimoto_inside_and_vertical_sweep_param

    X = rng.uniform(0, 1, n)
    Y = rng.uniform(0, 1, n)
    _, _, _, Y_swept = yanagimoto_inside_and_vertical_sweep_param(
        X, Y, base=0.3, height=0.6, x0=0.3, y0=0.1
    )
    return X, Y_swept


@register("dep-sine-4pi")
def _dep_sine_4pi(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sine wave dependence with frequency 4pi."""
    X = rng.uniform(-1, 1, n)
    Y = np.sin(4 * np.pi * X)
    return X, Y


@register("dep-sine-16pi")
def _dep_sine_16pi(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sine wave dependence with frequency 16pi."""
    X = rng.uniform(-1, 1, n)
    Y = np.sin(16 * np.pi * X)
    return X, Y


def _spiral(
    n: int,
    rng: np.random.Generator,
    turns: float,
    noise: float = 1.0,
    r0: float = 1.0,
    growth: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Noisy Archimedean spiral with configurable turns and noise.

    ``theta`` is drawn i.i.d. uniform on ``[0, 2*pi*turns]`` (until 2026-09-13 it was an
    evenly spaced ``linspace`` design, which made the cell counts of grid statistics
    under-dispersed and the rejection rate fall below alpha at some resolutions).
    """
    theta = rng.uniform(0, turns * 2 * np.pi, n)
    r = r0 + growth * theta + rng.normal(0, noise, n)
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    return X, Y


@register("dep-spiral-2-turns")
def _dep_spiral_2(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Noisy Archimedean spiral, 2 turns."""
    return _spiral(n, rng, turns=2)


@register("dep-spiral-3-turns")
def _dep_spiral_3(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Noisy Archimedean spiral, 3 turns."""
    return _spiral(n, rng, turns=3)


@register("dep-modular-5")
def _dep_modular_5(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Complete dependence Y = 5X mod 1. From Junker et al. 2021, Section 4.3."""
    X = rng.uniform(0, 1, n)
    Y = (5 * X) % 1
    return X, Y


@register("dep-modular-10")
def _dep_modular_10(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Complete dependence Y = 10X mod 1. From Junker et al. 2021, Section 4.3."""
    X = rng.uniform(0, 1, n)
    Y = (10 * X) % 1
    return X, Y


def _checkerboard(
    n: int, rng: np.random.Generator, k: int, delta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Checkerboard dependence on [0,1]^2.

    Black cells (i+j even) have density 1+delta, white cells 1-delta.
    delta=0 is independence, delta=1 puts all mass on black cells.
    """
    cells_i, cells_j = np.meshgrid(range(k), range(k), indexing="ij")
    cells_i = cells_i.ravel()
    cells_j = cells_j.ravel()

    is_black = (cells_i + cells_j) % 2 == 0
    weights = np.where(is_black, 1 + delta, 1 - delta)
    probs = weights / weights.sum()

    indices = rng.choice(k * k, size=n, p=probs)

    X = (cells_i[indices] + rng.uniform(size=n)) / k
    Y = (cells_j[indices] + rng.uniform(size=n)) / k

    return X, Y


@register("dep-checkerboard-k5-d1")
def _dep_checkerboard_k5_d1(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Checkerboard dependence, k=5 grid, delta=1 (all mass on black cells)."""
    return _checkerboard(n, rng, k=5, delta=1.0)
