"""Noise-sweep experiment runner for quasi-power benchmarks.

Source-agnostic: takes a pattern callable `(n, sigma, rng) -> PatternSample`
so the same runner works for both `patterns_salahub.py` (continuous sigma)
and `patterns_zhang.py` (discrete levels l = 1..10, sigma = l/40).

For a given test and pattern, sweeps a grid of sigmas, runs R replications
at each sigma, and records per-replication (sigma, rsq, p_value, statistic).

`rejection_rate_by_sigma` aggregates records into the y-axis of a quasi-power
curve: fraction of replications with p-value <= alpha, at each sigma.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Protocol

import numpy as np

from indep_tests.base import IndependenceTest


class _PatternSample(Protocol):
    x: np.ndarray
    y: np.ndarray
    rsq: float


class PatternFn(Protocol):
    def __call__(
        self, n: int, sigma: float, rng: np.random.Generator
    ) -> _PatternSample: ...


@dataclass
class NoiseSweepRecord:
    """One replication of one (test, pattern, sigma) combination."""

    test_name: str
    pattern: str
    n: int
    sigma: float
    rep: int
    rsq: float
    p_value: float
    statistic: float


@dataclass
class NoiseSweepResult:
    """All replications for a given (test, pattern) pair."""

    test_name: str
    pattern: str
    n: int
    R: int
    B: int
    sigmas: np.ndarray
    records: list[NoiseSweepRecord] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def run_noise_sweep(
    test: IndependenceTest,
    pattern_fn: PatternFn,
    pattern_name: str,
    n: int,
    sigmas: Iterable[float],
    R: int = 200,
    B: int = 1000,
    seed: int = 0,
) -> NoiseSweepResult:
    """Sweep sigma and collect per-replication p-values.

    Parameters
    ----------
    test : IndependenceTest
        Configured test instance.
    pattern_fn : PatternFn
        Callable `(n, sigma, rng) -> PatternSample` (see `patterns_salahub`
        or `patterns_zhang` for the two registries).
    pattern_name : str
        Display name stored on each record.
    n : int
        Sample size per replication.
    sigmas : Iterable[float]
        Noise levels to sweep. For the Zhang schedule use `l / 40` for
        l = 1, ..., 10 (see `patterns_zhang.sigma_for_level`).
    R : int
        Replications per sigma. Defaults to 200 (Salahub §5).
    B : int
        Permutation / bootstrap replicates per test call.
    seed : int
        Base seed. Replication i at sigma index j uses seed + j * R + i.
    """
    sigmas_arr = np.asarray(list(sigmas), dtype=float)
    records: list[NoiseSweepRecord] = []

    t0 = time.perf_counter()
    for j, sigma in enumerate(sigmas_arr):
        for i in range(R):
            rep_seed = seed + j * R + i
            rng = np.random.default_rng(rep_seed)
            sample = pattern_fn(n, float(sigma), rng)
            result = test.test(sample.x, sample.y, B=B, seed=rep_seed)
            records.append(
                NoiseSweepRecord(
                    test_name=test.name,
                    pattern=pattern_name,
                    n=n,
                    sigma=float(sigma),
                    rep=i,
                    rsq=float(sample.rsq),
                    p_value=result.p_value,
                    statistic=result.statistic,
                )
            )
    elapsed = time.perf_counter() - t0

    return NoiseSweepResult(
        test_name=test.name,
        pattern=pattern_name,
        n=n,
        R=R,
        B=B,
        sigmas=sigmas_arr,
        records=records,
        elapsed_seconds=elapsed,
    )


def rejection_rate_by_sigma(
    result: NoiseSweepResult,
    alpha: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Quasi-power y-axis: fraction of reps rejecting H0 at each sigma.

    Returns
    -------
    sigmas : np.ndarray, shape (L,)
        Sorted unique sigmas present in the records.
    power : np.ndarray, shape (L,)
        Rejection rate at each sigma, i.e. mean(p <= alpha).
    """
    sigmas = np.array([r.sigma for r in result.records])
    pvals = np.array([r.p_value for r in result.records])
    unique = np.sort(np.unique(sigmas))
    power = np.array([float(np.mean(pvals[sigmas == s] <= alpha)) for s in unique])
    return unique, power


def records_to_dict(result: NoiseSweepResult) -> list[dict]:
    """Flatten records into plain dicts for JSON / pandas."""
    return [r.__dict__ for r in result.records]
