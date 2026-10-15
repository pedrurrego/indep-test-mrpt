"""Compare chi-squared p-values vs bootstrap p-values for the PUG-Grid density test.

Reports:
  * Type-I error (calibration) under several null generators.
  * Power under several dependent generators.
  * Median wall-clock per call.

Run with:  uv run python -m indep_tests.pug_grid_pearson_chi_squared.experiment
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from indep_tests.generators import get_generator
from indep_tests.pug_grid import PUGGridTest
from indep_tests.pug_grid_pearson_chi_squared import PUGGridPearsonChiSquaredTest

ALPHA = 0.05
N_REPS = 300
B_BOOTSTRAP = 300


@dataclass
class Row:
    generator: str
    n: int
    test: str
    rejection_rate: float
    median_time_ms: float


def _run_cell(test, generator_name: str, n: int, n_reps: int) -> tuple[float, float]:
    gen = get_generator(generator_name)
    rejections = 0
    times: list[float] = []
    for i in range(n_reps):
        rng = np.random.default_rng(seed=10_000 + i)
        x, y = gen(n, rng)
        t0 = time.perf_counter()
        r = test.test(x, y, B=B_BOOTSTRAP, seed=i)
        times.append(time.perf_counter() - t0)
        if r.p_value <= ALPHA:
            rejections += 1
    return rejections / n_reps, float(np.median(times) * 1000)


def main() -> None:
    null_generators = ["indep-normal", "indep-uniform", "indep-expon"]
    alt_generators = [
        "dep-normal-0.3",
        "dep-sine-4pi",
        "dep-sine-16pi",
        "dep-checkerboard-k5-d1",
        "dep-modular-5",
    ]
    sample_sizes = [100, 300, 1000]

    tests: dict[str, object] = {
        "ChiSq-density": PUGGridPearsonChiSquaredTest(norm="density"),
        "Boot-density": PUGGridTest(norm="density"),
        "ChiSq-multi": PUGGridPearsonChiSquaredTest(norm="density-multi"),
        "Boot-multi": PUGGridTest(norm="density-multi"),
    }

    rows: list[Row] = []
    for n in sample_sizes:
        for gname in null_generators + alt_generators:
            for tname, test in tests.items():
                rate, t_ms = _run_cell(test, gname, n, N_REPS)
                rows.append(Row(gname, n, tname, rate, t_ms))

    # Print as markdown tables
    print(f"## Calibration (target Type I = {ALPHA})\n")
    _print_table(rows, null_generators, sample_sizes, list(tests.keys()), show="rate")
    print("\n## Power\n")
    _print_table(rows, alt_generators, sample_sizes, list(tests.keys()), show="rate")
    print("\n## Median wall-clock per test call (ms)\n")
    _print_table(
        rows,
        null_generators[:1] + alt_generators[:1],
        sample_sizes,
        list(tests.keys()),
        show="time",
    )


def _print_table(
    rows: list[Row], gens: list[str], ns: list[int], tests: list[str], show: str
) -> None:
    by_key = {(r.generator, r.n, r.test): r for r in rows}
    header_cells = ["generator", "n", *tests]
    print("| " + " | ".join(header_cells) + " |")
    print("|" + "|".join("---" for _ in header_cells) + "|")
    for g in gens:
        for n in ns:
            cells = [g, str(n)]
            for t in tests:
                r = by_key.get((g, n, t))
                if r is None:
                    cells.append("-")
                elif show == "rate":
                    cells.append(f"{r.rejection_rate:.3f}")
                else:
                    cells.append(f"{r.median_time_ms:.1f}")
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
