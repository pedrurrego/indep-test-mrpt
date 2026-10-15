"""Log-log slope of single-call cost vs n, per test: ``uv run python -m indep_tests.timing.fit``.

Prints, for each test, the OLS exponent of ``t ~ n^a`` over ``n >= min_n`` for
the statistic-only and full-call series, next to the largest-n timings. This
is the empirical column of the paper's cost table (subsec:time).
"""

from __future__ import annotations

import sys

import numpy as np

from indep_tests.paper_figures import config
from indep_tests.paper_figures.data import load_timing, loglog_slope


def main(B: int = config.TIMING_B, min_n: int = config.TIMING_FIT_MIN_N) -> None:
    meta, series = load_timing(B)
    print(f"B={B}, fit over n>={min_n}, machine={meta['machine'].get('cpu')}")
    print(
        f"{'test':26s} {'n_max':>6s} {'stat(n_max)':>12s} {'call(n_max)':>12s} "
        f"{'warm(n_max)':>12s} {'a_stat':>7s} {'a_call':>7s} {'a_warm':>7s}"
    )
    f = lambda v: "--" if v is None else f"{v:.2f}"  # noqa: E731

    def last(n, t, scale, unit):
        ok = np.isfinite(t)
        if not ok.any():
            return f"{'--':>12s}"
        return f"{t[ok][-1] * scale:9.2f} {unit}(n={int(n[ok][-1])})"

    for s in config.TIMING_TESTS:
        if s.key not in series:
            continue
        ser = series[s.key]
        a1 = loglog_slope(ser["n"], ser["stat"], min_n=min_n)
        a2 = loglog_slope(ser["n"], ser["call"], min_n=min_n)
        a3 = loglog_slope(ser["n"], ser["warm"], min_n=min_n)
        print(
            f"{s.key:26s} {int(ser['n'][-1]):6d} "
            f"{last(ser['n'], ser['stat'], 1e3, 'ms')} "
            f"{last(ser['n'], ser['call'], 1, 's ')} "
            f"{last(ser['n'], ser['warm'], 1e3, 'ms')} "
            f"{f(a1):>7s} {f(a2):>7s} {f(a3):>7s}"
        )


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*args)
