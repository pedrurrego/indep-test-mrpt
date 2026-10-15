"""Command-line interface for running independence test experiments."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from indep_tests.registry import get_test_class, test_names

app = typer.Typer(help="Independence test experiments.")

_TEST_NAMES = test_names()
_DEFAULT_SAMPLE_SIZES = "20,50,100,200,500"


def _parse_sample_sizes(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",")]


def _get_test_class(name: str):
    """Lazy import via the registry (avoids loading all tests upfront)."""
    try:
        return get_test_class(name)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from None


@app.command()
def run(
    test: Annotated[str, typer.Option(help="Test name.")],
    generator: Annotated[str, typer.Option(help="Data generator name.")],
    sample_sizes: Annotated[
        str,
        typer.Option("--sample-sizes", help="Comma-separated sample sizes."),
    ] = _DEFAULT_SAMPLE_SIZES,
    alpha: Annotated[float, typer.Option(help="Significance level.")] = 0.05,
    R: Annotated[int, typer.Option(help="Monte Carlo replications.")] = 100,
    B: Annotated[int, typer.Option(help="Bootstrap replicates per test.")] = 500,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 1234,
    test_config: Annotated[
        str, typer.Option(help="JSON string of test constructor kwargs.")
    ] = "{}",
    collect_p_values: Annotated[
        bool,
        typer.Option(
            "--collect-p-values", help="Store per-replication p-values and extras."
        ),
    ] = False,
    output: Annotated[
        Optional[str],
        typer.Option(
            help="Save results to file. Directory: auto-name. "
            "File path (.json): use as-is. Omit to print to stdout."
        ),
    ] = None,
) -> None:
    """Run an independence test experiment across sample sizes."""
    from indep_tests.commands.experiment import (
        resolve_output_path,
        results_to_dict,
        run_power_curve,
        save_results,
    )

    cls = _get_test_class(test)
    config = json.loads(test_config)
    test_instance = cls(**config)
    parsed_sizes = _parse_sample_sizes(sample_sizes)

    print(f"Test: {test_instance.name}")
    print(f"Generator: {generator}")
    print(f"Sample sizes: {parsed_sizes}")
    print(f"R={R}, B={B}, alpha={alpha}, seed={seed}")
    print("-" * 60)

    results = run_power_curve(
        test=test_instance,
        generator_name=generator,
        sample_sizes=parsed_sizes,
        alpha=alpha,
        R=R,
        B=B,
        seed=seed,
        collect_p_values=collect_p_values,
    )

    output_path = resolve_output_path(output, generator, test, config, B, R)

    if output_path:
        save_results(
            results, test_instance.name, generator, alpha, R, B, seed, output_path
        )
        print(f"\nResults saved to {output_path}")
    else:
        output_data = results_to_dict(
            results, test_instance.name, generator, alpha, R, B, seed
        )
        print("\n" + json.dumps(output_data, indent=2))


@app.command()
def plot(
    generator: Annotated[str, typer.Option(help="Generator name to filter results.")],
    results_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory containing JSON result files."),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option(
            help="Save figure to file. No value: auto-name in results/charts/. "
            "Directory: auto-name in that dir. File path: use as-is. "
            "Omit to show interactively."
        ),
    ] = None,
    alpha: Annotated[
        float, typer.Option(help="Significance level reference line.")
    ] = 0.05,
) -> None:
    """Plot a power curve from saved experiment results."""
    from indep_tests.commands.plotting import (
        _DEFAULT_RESULTS_DIR,
        plot_power_curve,
        resolve_chart_output_path,
    )

    output_path = resolve_chart_output_path(output, generator)

    plot_power_curve(
        generator=generator,
        results_dir=results_dir or _DEFAULT_RESULTS_DIR,
        output=output_path,
        alpha=alpha,
    )


@app.command("plot-generators")
def plot_generators(
    output: Annotated[
        Optional[str],
        typer.Option(
            help="Save figure to file. No value: auto-name in paper/figures/. "
            "Directory: auto-name in that dir. File path: use as-is. "
            "Omit to show interactively."
        ),
    ] = None,
    n: Annotated[int, typer.Option(help="Number of points per generator.")] = 500,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 1234,
) -> None:
    """Plot a 5x3 grid showing samples from all data generators."""
    from indep_tests.commands.plotting import (
        plot_all_generators,
        resolve_generators_chart_output_path,
    )

    output_path = resolve_generators_chart_output_path(output)

    plot_all_generators(output=output_path, n=n, seed=seed)


@app.command("list-generators")
def list_generators() -> None:
    """List available data generators."""
    from indep_tests.generators import list_generators

    print("Available generators:")
    for name in list_generators():
        print(f"  {name}")


@app.command("list-tests")
def list_tests() -> None:
    """List available independence tests."""
    print("Available tests:")
    for name in _TEST_NAMES:
        print(f"  {name}")


if __name__ == "__main__":
    app()
