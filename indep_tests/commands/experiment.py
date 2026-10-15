"""Logic for running independence test experiments."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from indep_tests.base import IndependenceTest
from indep_tests.generators import get_generator

_DEFAULT_RESULTS_DIR = Path(__file__).parents[1] / "results"


@dataclass
class ExperimentResult:
    """Result from running an experiment at one sample size."""

    n: int
    rejection_rate: float
    R: int
    B: int
    elapsed_seconds: float = 0.0
    p_values: list[float] = field(default_factory=list)
    extras: list[dict] = field(default_factory=list)


def run_single_experiment(
    test: IndependenceTest,
    generator_name: str,
    n: int,
    alpha: float = 0.05,
    R: int = 100,
    B: int = 500,
    seed: int = 0,
    collect_p_values: bool = False,
) -> ExperimentResult:
    """Run R replications of a test at sample size n.

    Parameters
    ----------
    test : IndependenceTest
        The test instance (already configured).
    generator_name : str
        Name of the data generator in the registry.
    n : int
        Sample size per replication.
    alpha : float
        Significance level.
    R : int
        Number of Monte Carlo replications.
    B : int
        Bootstrap replicates per test call.
    seed : int
        Base random seed. Each replication uses seed + i.
    collect_p_values : bool
        Whether to store all p-values in the result.

    Returns
    -------
    ExperimentResult
    """
    generator = get_generator(generator_name)
    rejections = 0
    p_values = []
    extras = []

    t0 = time.perf_counter()

    for i in range(R):
        rep_seed = seed + i
        rng = np.random.default_rng(rep_seed)
        x, y = generator(n, rng)

        result = test.test(x, y, B=B, seed=rep_seed + R)

        if result.p_value < alpha:
            rejections += 1

        if collect_p_values:
            p_values.append(result.p_value)
            if result.extras:
                extras.append(result.extras)

    elapsed = time.perf_counter() - t0

    return ExperimentResult(
        n=n,
        rejection_rate=rejections / R,
        R=R,
        B=B,
        elapsed_seconds=elapsed,
        p_values=p_values if collect_p_values else [],
        extras=extras if collect_p_values else [],
    )


def run_power_curve(
    test: IndependenceTest,
    generator_name: str,
    sample_sizes: list[int],
    alpha: float = 0.05,
    R: int = 100,
    B: int = 500,
    seed: int = 0,
    collect_p_values: bool = False,
) -> list[ExperimentResult]:
    """Run an experiment across multiple sample sizes (power curve).

    Returns
    -------
    list[ExperimentResult]
        One result per sample size, in order.
    """
    results = []
    for n in sample_sizes:
        result = run_single_experiment(
            test=test,
            generator_name=generator_name,
            n=n,
            alpha=alpha,
            R=R,
            B=B,
            seed=seed,
            collect_p_values=collect_p_values,
        )
        print(
            f"  {test.name} | n={n:>5d} | rejection_rate={result.rejection_rate:.4f} "
            f"| {result.elapsed_seconds:.1f}s"
        )
        results.append(result)
    return results


def build_output_filename(
    generator: str, test: str, config: dict, B: int, R: int
) -> str:
    """Build an auto-generated filename from experiment parameters."""
    parts = [generator, test]
    if config:
        config_str = "+".join(f"{k}={v}" for k, v in sorted(config.items()))
        parts.append(config_str)
    parts.append(f"B{B}")
    parts.append(f"R{R}")
    return "_".join(parts) + ".json"


def resolve_output_path(
    output: str | None, generator: str, test: str, config: dict, B: int, R: int
) -> Path | None:
    """Resolve the output path from the --output flag.

    - None (flag absent)      -> return None (print to stdout)
    - "" (flag with no value) -> default dir + auto filename
    - a directory path        -> that dir + auto filename
    - a .json file path       -> use as-is
    """
    if output is None:
        return None

    filename = build_output_filename(generator, test, config, B, R)

    if output == "":
        return _DEFAULT_RESULTS_DIR / filename

    p = Path(output)
    if p.suffix == ".json":
        return p

    return p / filename


def save_results(
    results: list[ExperimentResult],
    test_name: str,
    generator: str,
    alpha: float,
    R: int,
    B: int,
    seed: int,
    output_path: Path,
) -> None:
    """Serialize experiment results to a JSON file."""
    output_data = {
        "test": test_name,
        "generator": generator,
        "alpha": alpha,
        "R": R,
        "B": B,
        "seed": seed,
        "results": [
            {
                "n": r.n,
                "rejection_rate": r.rejection_rate,
                "elapsed_seconds": round(r.elapsed_seconds, 2),
                **({"p_values": r.p_values} if r.p_values else {}),
                **({"extras": r.extras} if r.extras else {}),
            }
            for r in results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def results_to_dict(
    results: list[ExperimentResult],
    test_name: str,
    generator: str,
    alpha: float,
    R: int,
    B: int,
    seed: int,
) -> dict:
    """Convert experiment results to a dict (for stdout printing)."""
    return {
        "test": test_name,
        "generator": generator,
        "alpha": alpha,
        "R": R,
        "B": B,
        "seed": seed,
        "results": [
            {
                "n": r.n,
                "rejection_rate": r.rejection_rate,
                "elapsed_seconds": round(r.elapsed_seconds, 2),
                **({"p_values": r.p_values} if r.p_values else {}),
                **({"extras": r.extras} if r.extras else {}),
            }
            for r in results
        ],
    }
