"""Run the timing sweep: ``uv run python -m indep_tests.timing [--B 500] [--cap 90]``.

Single-threaded BLAS is enforced here, not in ``tests.py``: ``indep_tests``
imports numpy on package import, and Accelerate/OpenBLAS read their thread
variables when they load, so setting them from Python afterwards has no
effect.  If any of the variables is not ``"1"`` the driver re-executes itself
with all of them pinned before any import happens.
"""

from __future__ import annotations

import os
import sys

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

if any(os.environ.get(v) != "1" for v in THREAD_VARS):
    _env = {**os.environ, **{v: "1" for v in THREAD_VARS}}
    os.execve(
        sys.executable,
        [sys.executable, "-m", "indep_tests.timing", *sys.argv[1:]],
        _env,
    )

import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Annotated  # noqa: E402

import numpy as np  # noqa: E402
import typer  # noqa: E402

from indep_tests.timing.tests import paper_tests, warm_tests  # noqa: E402

app = typer.Typer(help=__doc__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_NS = "50,100,200,500,1000,2000,4000,10000,100000,1000000,10000000"


def _machine() -> dict:
    cpu = platform.processor() or ""
    if sys.platform == "darwin":
        try:
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:  # pragma: no cover
            pass
    import scipy

    cfg = np.show_config(mode="dicts")["Build Dependencies"]
    return {
        "cpu": cpu,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas": cfg["blas"]["name"],
        "threads": {v: os.environ.get(v) for v in THREAD_VARS},
    }


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=RESULTS_DIR
        ).strip()
    except Exception:  # pragma: no cover
        return None


def _params(t) -> dict:
    """Public constructor attributes of a test instance, for the metadata."""
    return {"class": type(t).__name__} | {
        k: v for k, v in vars(t).items() if not k.startswith("_")
    }


def _timed(fn, repeats: int, cap: float, budget: float) -> list[float]:
    """Wall-clock seconds of up to ``repeats`` calls.

    Stops early after one call over ``cap`` seconds, or once the calls made so
    far add up to more than ``budget`` seconds (so a five-minute call is made
    once, not three times).
    """
    out: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
        if out[-1] > cap or sum(out) > budget:
            break
    return out


def _predicted(history: list[tuple[int, float]], n: int) -> float | None:
    """Log-log extrapolation of a cost to ``n`` from the last two measurements.

    ``None`` with fewer than two points.  A single decade step can turn a
    46 s call into a 10^4 s one (Cmiel-Ledwina at n = 4000 -> 10^5), so the
    sweep skips a call whose predicted cost exceeds ``--max-predicted``
    instead of discovering that by running it.
    """
    if len(history) < 2:
        return None
    (n0, t0), (n1, t1) = history[-2], history[-1]
    if t0 <= 0 or t1 <= 0 or n1 == n0:
        return None
    slope = np.log(t1 / t0) / np.log(n1 / n0)
    return float(t1 * (n / n1) ** max(slope, 0.0))


@app.command()
def run(
    B: Annotated[
        int, typer.Option("--B", help="Permutation / MC draws per call.")
    ] = 500,
    cap: Annotated[
        float,
        typer.Option(
            help="Drop a test from larger n once one cold call exceeds this many seconds."
        ),
    ] = 90.0,
    warm_fill_cap: Annotated[
        float,
        typer.Option(
            help="Drop the warm series of a test from larger n once filling its null cache exceeds this many seconds."
        ),
    ] = 900.0,
    sample_sizes: Annotated[str, typer.Option("--sample-sizes")] = DEFAULT_NS,
    stat_repeats: Annotated[
        int, typer.Option(help="Repeats of statistic() (median).")
    ] = 5,
    call_repeats: Annotated[
        int, typer.Option(help="Repeats of each test() call, cold and warm (median).")
    ] = 3,
    seed: Annotated[int, typer.Option()] = 0,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            help="Directory for the per-test JSON files <test>_B{B}.json (default results/)."
        ),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(
            help="Re-time only this test (label in tests.py); its <test>_B{B}.json is rewritten (previous run metadata kept in meta.reruns) and the other files are untouched."
        ),
    ] = None,
    skip: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated labels (tests.py) to leave out of the sweep entirely; recorded in meta.skipped."
        ),
    ] = None,
    repeat_budget: Annotated[
        float,
        typer.Option(
            help="Stop repeating a timed quantity at one (test, n) once its repeats add up to more than this many seconds."
        ),
    ] = 60.0,
    max_predicted: Annotated[
        float,
        typer.Option(
            help="Skip a cold call or a warm-null fill whose log-log extrapolation from the last two measurements exceeds this many seconds (recorded as skipped_predicted / warm_skipped_predicted)."
        ),
    ] = 3600.0,
    no_warm: Annotated[
        bool,
        typer.Option("--no-warm", help="Skip the reused-null (warm) series entirely."),
    ] = False,
    extend: Annotated[
        bool,
        typer.Option(
            "--extend",
            help="With --only: keep the rows already in the test's file for sample sizes not in --sample-sizes (new rows replace same-n rows) and seed the extrapolation guards from them.",
        ),
    ] = False,
) -> None:
    """Time one statistic evaluation and one full test() call per (test, n).

    One JSON per test, ``<output_dir>/<test>_B{B}.json`` (like ``results/`` of
    the power harness): ``meta`` holds the run parameters and machine plus that
    test's constructor attributes, ``rows`` its per-``n`` timings.  Files are
    rewritten after every row, so a killed sweep leaves every completed test
    readable.

    Every test sees the same independent bivariate normal sample at each n
    (cost is data-independent for the rank tests; FES skips sparse windows, so
    the independent sample is its most expensive input).  Three quantities per
    (test, n): ``stat`` (one statistic evaluation), ``call`` (a *cold* call:
    statistic + B null draws, null caches disabled) and, for the four tests
    whose null law depends on n only, ``warm`` (a call that reuses a stored
    null table; the untimed fill is reported as ``warm_fill_seconds``).  All
    individual repeat timings are kept next to their medians.
    """
    ns = [int(s) for s in sample_sizes.split(",")]
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tests = paper_tests()
    warm = {} if no_warm else warm_tests()
    skipped = [t for t in (skip or "").split(",") if t]
    for name in skipped:
        if name not in tests:
            raise typer.BadParameter(
                f"unknown test {name!r} in --skip; choices: {', '.join(tests)}"
            )
        tests.pop(name)
        warm.pop(name, None)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if only is not None:
        if only not in tests:
            raise typer.BadParameter(
                f"unknown test {only!r}; choices: {', '.join(tests)}"
            )
        tests = {only: tests[only]}
        warm = {k: v for k, v in warm.items() if k == only}
    run_meta = {
        "B": B,
        "cap_seconds": cap,
        "warm_fill_cap_seconds": warm_fill_cap,
        "repeat_budget_seconds": repeat_budget,
        "max_predicted_seconds": max_predicted,
        "sample_sizes": ns,
        "stat_repeats": stat_repeats,
        "call_repeats": call_repeats,
        "seed": seed,
        "skipped": skipped,
        "no_warm": no_warm,
        "started": started,
        "git": _git_revision(),
        "machine": _machine(),
    }
    if extend and only is None:
        raise typer.BadParameter("--extend requires --only")
    paths = {name: out_dir / f"{name}_B{B}.json" for name in tests}
    metas: dict[str, dict] = {}
    kept_rows: dict[str, list[dict]] = {name: [] for name in tests}
    for name, t in tests.items():
        meta = {
            **run_meta,
            "test": name,
            "class": type(t).__name__,
            "params": _params(t),
            "warm_params": _params(warm[name]) if name in warm else None,
        }
        if paths[name].exists():  # keep the history of what this file replaced
            payload = json.load(open(paths[name]))
            prev = payload.get("meta", {})
            meta["reruns"] = prev.get("reruns", []) + [
                {k: prev.get(k) for k in ("started", "git", "sample_sizes")}
            ]
            if extend:
                kept_rows[name] = [r for r in payload["rows"] if r["n"] not in ns]
                meta["sample_sizes"] = sorted(
                    {r["n"] for r in kept_rows[name]} | set(ns)
                )
                meta["extended"] = {"started": started, "sample_sizes": ns}
        metas[name] = meta
    rng = np.random.default_rng(seed)
    rows_by_test: dict[str, list[dict]] = {name: [] for name in tests}
    alive = dict.fromkeys(tests, True)
    alive_warm = dict.fromkeys(warm, True)
    cold_hist: dict[str, list[tuple[int, float]]] = {k: [] for k in tests}
    fill_hist: dict[str, list[tuple[int, float]]] = {k: [] for k in warm}
    for name, rows_prev in kept_rows.items():  # seed the guards from kept rows
        for r in sorted(rows_prev, key=lambda r: r["n"]):
            if r.get("call_seconds") is not None:
                cold_hist[name].append((r["n"], r["call_seconds"]))
            if name in fill_hist and r.get("warm_fill_seconds") is not None:
                fill_hist[name].append((r["n"], r["warm_fill_seconds"]))
    for n in ns:
        x, y = rng.normal(size=n), rng.normal(size=n)
        for name, t in tests.items():
            if not alive[name] and not alive_warm.get(name, False):
                continue
            row: dict = {"test": name, "n": n}
            pred_cold = _predicted(cold_hist[name], n) if alive[name] else None
            if alive[name] and pred_cold is not None and pred_cold > max_predicted:
                alive[name] = False
                row |= {"skipped_predicted": True, "predicted_call_seconds": pred_cold}
            if alive[name]:
                ts = _timed(lambda: t.statistic(x, y), stat_repeats, cap, repeat_budget)
                tc: list[float] = []
                for r in range(call_repeats):  # distinct seed per cold repeat
                    t0 = time.perf_counter()
                    t.test(x, y, B=B, seed=seed + r)
                    tc.append(time.perf_counter() - t0)
                    if tc[-1] > cap or sum(tc) > repeat_budget:
                        break
                stat, call = float(np.median(ts)), float(np.median(tc))
                alive[name] = max(stat, call) <= cap
                cold_hist[name].append((n, call))
                row |= {
                    "stat_seconds": stat,
                    "call_seconds": call,
                    "stat_seconds_all": ts,
                    "call_seconds_all": tc,
                    "censored_after": not alive[name],
                }
            else:
                row |= {
                    "stat_seconds": None,
                    "call_seconds": None,
                    "stat_seconds_all": [],
                    "call_seconds_all": [],
                    "censored_after": True,
                }
            pred_fill = (
                _predicted(fill_hist[name], n) if alive_warm.get(name, False) else None
            )
            if pred_fill is not None and pred_fill > min(warm_fill_cap, max_predicted):
                alive_warm[name] = False
                row |= {
                    "warm_skipped_predicted": True,
                    "predicted_fill_seconds": pred_fill,
                }
            if alive_warm.get(name, False):
                w = warm[name]
                t0 = time.perf_counter()
                w.test(x, y, B=B, seed=seed)  # fills the null cache (untimed)
                fill = time.perf_counter() - t0
                fill_hist[name].append((n, fill))
                tw = _timed(
                    lambda: w.test(x, y, B=B, seed=seed),
                    call_repeats,
                    cap,
                    repeat_budget,
                )
                warm_med = float(np.median(tw))
                alive_warm[name] = fill <= warm_fill_cap and warm_med <= cap
                row |= {
                    "warm_seconds": warm_med,
                    "warm_seconds_all": tw,
                    "warm_fill_seconds": fill,
                    "warm_censored_after": not alive_warm[name],
                }
            else:
                row |= {
                    "warm_seconds": None,
                    "warm_seconds_all": [],
                    "warm_fill_seconds": None,
                }
            rows_by_test[name].append(row)
            fmt = lambda v, s: "     --   " if v is None else f"{v * s:10.3f}"  # noqa: E731
            flags = []
            if row.get("censored_after") and row["stat_seconds"] is not None:
                flags.append("cold cap reached, cold dropped")
            if row.get("warm_censored_after"):
                flags.append("warm dropped")
            if row.get("skipped_predicted"):
                flags.append(
                    f"cold skipped, predicted {row['predicted_call_seconds']:.0f} s"
                )
            if row.get("warm_skipped_predicted"):
                flags.append(
                    f"warm skipped, predicted fill {row['predicted_fill_seconds']:.0f} s"
                )
            typer.echo(
                f"{name:24s} n={n:5d}  stat={fmt(row['stat_seconds'], 1e3)} ms"
                f"  call={fmt(row['call_seconds'], 1)} s"
                f"  warm={fmt(row['warm_seconds'], 1e3)} ms"
                + (f"  [{'; '.join(flags)}]" if flags else "")
            )
            json.dump(
                {
                    "meta": metas[name],
                    "rows": sorted(
                        kept_rows[name] + rows_by_test[name], key=lambda r: r["n"]
                    ),
                },
                open(paths[name], "w"),
                indent=1,
                default=str,
            )
    typer.echo(f"Wrote {len(paths)} files to {out_dir}")


if __name__ == "__main__":
    app()
