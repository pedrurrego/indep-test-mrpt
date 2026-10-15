"""Trutschnig QAD (Quantification of Asymmetric Dependence) test."""

from typing import Literal

import numpy as np

from indep_tests.base import IndependenceTest, TestResult
from pug_trutschnig.utils import qad as _qad_full
from pug_trutschnig.utils import qad_null_stats

# Distribution-free null cache, keyed by (n, resolution, B).  Valid for
# continuous (tie-free) data only, where the qad permutation null does not
# depend on the observed sample — same pattern as heller_bins / cmiel_ledwina.
_NULL_CACHE: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}


def _get_cached_null(n: int, resolution: int, B: int) -> dict[str, np.ndarray]:
    key = (n, resolution, B)
    null = _NULL_CACHE.get(key)
    if null is None:
        # Deterministic, call-order-independent seed so every replication
        # that hits this configuration shares one null table.
        seed_seq = np.random.SeedSequence([n, resolution, B])
        null = qad_null_stats(n, resolution, B, np.random.default_rng(seed_seq))
        _NULL_CACHE[key] = null
    return null


class QADTest(IndependenceTest):
    """QAD rank-based test using checkerboard copula aggregation.

    Parameters
    ----------
    resolution : int | None
        Resolution for checkerboard. None = auto (floor(sqrt(min unique ranks))).
    return_param : str
        Which statistic to report: "max_dependence", "q_xy", or "q_yx".
    cache_null : bool
        Reuse one permutation null per ``(n, resolution, B)`` configuration
        (the null is distribution-free for continuous data, so it is shared
        across Monte-Carlo replications; ~B-fold faster in benchmarks).
        Automatically falls back to the per-call permutation loop when the
        data contain ties, where the null is data-dependent.  Set to False
        to always permute per call (the original behaviour).
    """

    def __init__(
        self,
        resolution: int | None = None,
        return_param: Literal["max_dependence", "q_xy", "q_yx"] = "max_dependence",
        cache_null: bool = True,
    ) -> None:
        self.resolution = resolution
        self.return_param = return_param
        self.cache_null = cache_null

    @property
    def name(self) -> str:
        return "QAD"

    def statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        result = _qad_full(
            np.asarray(x),
            np.asarray(y),
            resolution=self.resolution,
            p_value=False,
        )
        return float(result[self.return_param])

    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        B: int = 1000,
        seed: int | None = None,
    ) -> TestResult:
        x = np.asarray(x)
        y = np.asarray(y)
        n = len(x)

        has_ties = len(np.unique(x)) < n or len(np.unique(y)) < n

        if self.cache_null and not has_ties:
            result = _qad_full(x, y, resolution=self.resolution, p_value=False)
            null = _get_cached_null(n, int(result["resolution"]), B)
            for param in ("q_xy", "q_yx", "max_dependence"):
                # Same right-tail convention as qad()'s p_value=True path.
                result[f"p_{param}"] = np.mean(null[param] >= result[param])
        else:
            # qad() uses np.random global state for permutations
            if seed is not None:
                np.random.seed(seed)
            result = _qad_full(
                x,
                y,
                resolution=self.resolution,
                p_value=True,
                nperm=B,
            )

        stat_value = float(result[self.return_param])
        p_key = f"p_{self.return_param}"
        p_value = float(result[p_key])

        return TestResult(
            statistic=stat_value,
            p_value=p_value,
            n=int(result["n"]),
            B=B,
            test_name=self.name,
            extras={
                "q_xy": result["q_xy"],
                "q_yx": result["q_yx"],
                "max_dependence": result["max_dependence"],
                "asymmetry": result["asymmetry"],
                "resolution": result["resolution"],
            },
        )
