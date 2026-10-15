"""Typer CLI for the rBEX resolution-sweep quasi-power benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from indep_tests.cli import _TEST_NAMES, _get_test_class

from .commands import (
    _DEFAULT_RESULTS_DIR,
    load_all_results,
    resolve_output_path,
    save_result,
)
from .experiment import records_to_dict, run_resolution_sweep
from .patterns import default_resolutions
from .plotting import plot_pattern_grid, plot_quasi_power_resolution

app = typer.Typer(
    help="rBEX (bisection expanding cross) quasi-power benchmark "
    "(Zhang 2019 / Salahub 2025)."
)

_DEFAULT_CHARTS_DIR = _DEFAULT_RESULTS_DIR / "charts"


def _parse_resolutions(resolutions: str) -> list[int]:
    if resolutions.strip():
        return [int(s.strip()) for s in resolutions.split(",") if s.strip()]
    return default_resolutions().tolist()


@app.command()
def run(
    test: Annotated[str, typer.Option(help="Test name (see `list-tests`).")],
    n: Annotated[int, typer.Option(help="Sample size per replication.")] = 2000,
    R: Annotated[int, typer.Option(help="Replications per resolution.")] = 200,
    B: Annotated[
        int, typer.Option(help="Bootstrap/permutation replicates per test call.")
    ] = 500,
    alpha: Annotated[
        float, typer.Option(help="Significance level stored in JSON metadata.")
    ] = 0.01,
    seed: Annotated[int, typer.Option(help="Base random seed.")] = 1234,
    resolutions: Annotated[
        str,
        typer.Option(help="Comma-separated resolutions (0 = null). Default: 0..8."),
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
    """Run one test across the rBEX resolution sweep and save per-rep records."""
    cls = _get_test_class(test)
    config = json.loads(test_config)
    test_instance = cls(**config)

    res_list = _parse_resolutions(resolutions)

    typer.echo(f"Test: {test_instance.name}")
    typer.echo(f"n={n}, R={R}, B={B}, alpha={alpha}, seed={seed}")
    typer.echo(f"resolutions: {res_list}")
    typer.echo("-" * 60)

    result = run_resolution_sweep(
        test=test_instance,
        n=n,
        resolutions=res_list,
        R=R,
        B=B,
        seed=seed,
    )

    typer.echo(
        f"Elapsed: {result.elapsed_seconds:.1f}s ({len(result.records)} records)"
    )

    output_path = resolve_output_path(output, test, config, n, R, B)
    if output_path is None:
        typer.echo(json.dumps({"records": records_to_dict(result)}, indent=2))
    else:
        save_result(result, alpha=alpha, seed=seed, output_path=output_path)
        typer.echo(f"Saved: {output_path}")


@app.command()
def plot(
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
    """Plot one quasi-power chart overlaying every test found in results_dir."""
    rdir = results_dir or _DEFAULT_RESULTS_DIR
    odir = output_dir or _DEFAULT_CHARTS_DIR
    per_test = load_all_results(rdir)
    if not per_test:
        raise typer.BadParameter(f"No result JSONs found in {rdir}")
    path = plot_quasi_power_resolution(
        results_by_test=per_test,
        out_path=odir / "quasi-power-rbex.png",
        alpha=alpha,
    )
    typer.echo(f"Saved: {path}  (tests: {', '.join(sorted(per_test))})")


@app.command("plot-pattern")
def plot_pattern(
    n: Annotated[int, typer.Option(help="Sample size per panel.")] = 2000,
    resolutions: Annotated[
        str, typer.Option(help="Comma-separated resolutions to render.")
    ] = "0,1,2,3,4,5,6",
    output_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory to write the figure into."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Base random seed.")] = 0,
) -> None:
    """Render a strip of rBEX scatterplots (sampler sanity check)."""
    odir = output_dir or _DEFAULT_CHARTS_DIR
    res_list = _parse_resolutions(resolutions)
    path = plot_pattern_grid(
        resolutions=res_list,
        n=n,
        out_path=odir / f"rbex-patterns-N{n}.png",
        seed=seed,
    )
    typer.echo(f"Saved: {path}")


@app.command("list-tests")
def list_tests_cmd() -> None:
    """List every independence test name accepted by `--test`."""
    typer.echo("Available tests:")
    for name in _TEST_NAMES:
        typer.echo(f"  {name}")


if __name__ == "__main__":
    app()
