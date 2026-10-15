"""Resolution-sweep runner for the rBEX quasi-power benchmark.

Structurally the analogue of ``salahub_noise.experiment`` but the swept axis is
the BEX *resolution* (integer) instead of a noise sigma. For a given test it
draws ``R`` independent rBEX samples at each resolution, runs the test, and
records the per-replication ``(resolution, p_value, statistic)``.

``rejection_rate_by_resolution`` collapses the records into the y-axis of the
quasi-power curve: the fraction of replications rejecting ``H0`` at ``alpha``,
per resolution. On the null anchor (resolution 0, independent uniform) this is
the empirical *size*; on resolutions ``>= 1`` it is *power* against ``BEX_d``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from indep_tests.base import IndependenceTest

from .patterns import rbex_sample


@dataclass
class ResolutionSweepRecord:
    """One replication of one (test, resolution) combination."""

    test_name: str
    n: int
    resolution: int
    rep: int
    p_value: float
    statistic: float


@dataclass
class ResolutionSweepResult:
    """All replications for a given test across the resolution sweep."""

    test_name: str
    n: int
    R: int
    B: int
    resolutions: np.ndarray
    records: list[ResolutionSweepRecord] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def run_resolution_sweep(
    test: IndependenceTest,
    n: int,
    resolutions: Iterable[int],
    R: int = 200,
    B: int = 500,
    seed: int = 1234,
) -> ResolutionSweepResult:
    """Sweep rBEX resolution and collect per-replication p-values.

    Parameters
    ----------
    test : IndependenceTest
        Configured test instance.
    n : int
        Sample size per replication.
    resolutions : Iterable[int]
        BEX resolutions to sweep; ``0`` is the independent-uniform null anchor.
    R : int
        Replications per resolution. Salahub §5 uses 200.
    B : int
        Permutation / bootstrap replicates per test call (ignored by
        parametric tests).
    seed : int
        Base seed. Replication ``i`` at resolution index ``j`` uses
        ``seed + j * R + i`` for both data generation and the test call.
    """
    res_arr = np.asarray(list(resolutions), dtype=int)
    records: list[ResolutionSweepRecord] = []

    t0 = time.perf_counter()
    for j, resolution in enumerate(res_arr):
        for i in range(R):
            rep_seed = seed + j * R + i
            rng = np.random.default_rng(rep_seed)
            x, y = rbex_sample(n, int(resolution), rng)
            result = test.test(x, y, B=B, seed=rep_seed)
            records.append(
                ResolutionSweepRecord(
                    test_name=test.name,
                    n=n,
                    resolution=int(resolution),
                    rep=i,
                    p_value=result.p_value,
                    statistic=result.statistic,
                )
            )
    elapsed = time.perf_counter() - t0

    return ResolutionSweepResult(
        test_name=test.name,
        n=n,
        R=R,
        B=B,
        resolutions=res_arr,
        records=records,
        elapsed_seconds=elapsed,
    )


def rejection_rate_by_resolution(
    result: ResolutionSweepResult,
    alpha: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Quasi-power y-axis: fraction of reps rejecting H0 at each resolution.

    Returns
    -------
    resolutions : np.ndarray, shape (L,)
        Sorted unique resolutions present in the records.
    power : np.ndarray, shape (L,)
        Rejection rate at each resolution, i.e. ``mean(p <= alpha)``.
    """
    res = np.array([r.resolution for r in result.records])
    pvals = np.array([r.p_value for r in result.records])
    unique = np.sort(np.unique(res))
    power = np.array([float(np.mean(pvals[res == u] <= alpha)) for u in unique])
    return unique, power


def records_to_dict(result: ResolutionSweepResult) -> list[dict]:
    """Flatten records into plain dicts for JSON / pandas."""
    return [r.__dict__ for r in result.records]
