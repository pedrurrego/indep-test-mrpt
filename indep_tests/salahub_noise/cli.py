"""Typer CLI for the Salahub quasi-power noise sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from indep_tests.cli import _TEST_NAMES, _get_test_class

from .commands import (
    _DEFAULT_RESULTS_DIR,
    discover_patterns,
    load_null_rates,
    load_results_for_pattern,
    resolve_output_path,
    save_result,
)
from .experiment import run_noise_sweep
from .patterns_zhang import default_levels, get_pattern, list_patterns, sigma_for_level
from .plotting import plot_quasi_power_pattern

app = typer.Typer(
    help="Salahub quasi-power noise-sweep benchmark (Zhang 2019 patterns)."
)

_DEFAULT_CHARTS_DIR = _DEFAULT_RESULTS_DIR / "charts"


@app.command()
def run(
    test: Annotated[str, typer.Option(help="Test name (see `list-tests`).")],
    pattern: Annotated[str, typer.Option(help="Pattern name (see `list-patterns`).")],
    n: Annotated[int, typer.Option(help="Sample size per replication.")] = 500,
    R: Annotated[int, typer.Option(help="Replications per noise level.")] = 200,
    B: Annotated[
        int, typer.Option(help="Bootstrap/permutation replicates per test call.")
    ] = 1000,
    alpha: Annotated[
        float, typer.Option(help="Significance level stored in JSON metadata.")
    ] = 0.01,
    seed: Annotated[int, typer.Option(help="Base random seed.")] = 1234,
    levels: Annotated[
        str,
        typer.Option(help="Comma-separated integer noise levels. Default: 1..10."),
    ] = "",
    test_config: Annotated[
        str, typer.Option(help="JSON dict of test constructor kwargs.")
    ] = "{}",
    output: Annotated[
        Optional[str],
        typer.Option(
            help="Save result JSON. Empty → default dir + auto name; "
            "dir → auto name in that dir; .json path → use as-is; "
            "omit → print to stdout."
        ),
    ] = None,
) -> None:
    """Run one (test, pattern) noise sweep and save per-replication records."""
    cls = _get_test_class(test)
    config = json.loads(test_config)
    test_instance = cls(**config)

    if levels.strip():
        level_list = [int(s.strip()) for s in levels.split(",") if s.strip()]
    else:
        level_list = default_levels().tolist()
    sigmas = [sigma_for_level(lev) for lev in level_list]

    pattern_fn = get_pattern(pattern)

    typer.echo(f"Test: {test_instance.name}")
    typer.echo(f"Pattern: {pattern}")
    typer.echo(f"n={n}, R={R}, B={B}, alpha={alpha}, seed={seed}")
    typer.echo(f"levels: {level_list}")
    typer.echo("-" * 60)

    result = run_noise_sweep(
        test=test_instance,
        pattern_fn=pattern_fn,
        pattern_name=pattern,
        n=n,
        sigmas=sigmas,
        R=R,
        B=B,
        seed=seed,
    )

    typer.echo(
        f"Elapsed: {result.elapsed_seconds:.1f}s ({len(result.records)} records)"
    )

    output_path = resolve_output_path(output, pattern, test, config, n, R, B)
    if output_path is None:
        from .experiment import records_to_dict

        typer.echo(json.dumps({"records": records_to_dict(result)}, indent=2))
    else:
        save_result(result, alpha=alpha, seed=seed, output_path=output_path)
        typer.echo(f"Saved: {output_path}")


@app.command()
def plot(
    pattern: Annotated[str, typer.Option(help="Pattern name to plot.")],
    results_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory of result JSON files."),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory to write charts into."),
    ] = None,
    alpha: Annotated[float, typer.Option(help="Rejection threshold.")] = 0.01,
) -> None:
    """Plot one quasi-power chart for `pattern`, overlaying every test found."""
    rdir = results_dir or _DEFAULT_RESULTS_DIR
    odir = output_dir or _DEFAULT_CHARTS_DIR
    per_test = load_results_for_pattern(rdir, pattern)
    if not per_test:
        raise typer.BadParameter(f"No results found for pattern '{pattern}' in {rdir}")
    null_rates = load_null_rates(rdir, alpha=alpha)
    path = plot_quasi_power_pattern(
        pattern=pattern,
        results_by_test=per_test,
        out_path=odir / f"quasi-power-{pattern}.png",
        alpha=alpha,
        null_rates=null_rates,
    )
    typer.echo(f"Saved: {path}  (tests: {', '.join(sorted(per_test))})")


@app.command("plot-all")
def plot_all(
    results_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory of result JSON files."),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory to write charts into."),
    ] = None,
    alpha: Annotated[float, typer.Option(help="Rejection threshold.")] = 0.01,
) -> None:
    """Regenerate one quasi-power chart per pattern found in results_dir."""
    rdir = results_dir or _DEFAULT_RESULTS_DIR
    odir = output_dir or _DEFAULT_CHARTS_DIR
    patterns = discover_patterns(rdir)
    if not patterns:
        raise typer.BadParameter(f"No result JSONs found in {rdir}")
    null_rates = load_null_rates(rdir, alpha=alpha)
    for pat in patterns:
        per_test = load_results_for_pattern(rdir, pat)
        if not per_test:
            continue
        path = plot_quasi_power_pattern(
            pattern=pat,
            results_by_test=per_test,
            out_path=odir / f"quasi-power-{pat}.png",
            alpha=alpha,
            null_rates=null_rates,
        )
        typer.echo(f"  {pat}: {path}")


@app.command("list-tests")
def list_tests_cmd() -> None:
    """List every independence test name accepted by `--test`."""
    typer.echo("Available tests:")
    for name in _TEST_NAMES:
        typer.echo(f"  {name}")


@app.command("list-patterns")
def list_patterns_cmd() -> None:
    """List every Zhang-2019 pattern name accepted by `--pattern`."""
    typer.echo("Available patterns:")
    for name in list_patterns():
        typer.echo(f"  {name}")


if __name__ == "__main__":
    app()
