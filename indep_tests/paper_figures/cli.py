"""Typer CLI for building the paper's figures.

Run with::

    uv run python -m indep_tests.paper_figures build-all
    uv run python -m indep_tests.paper_figures build power-curves
    uv run python -m indep_tests.paper_figures list

Figures default to vector PDF written into ``paper/figures/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from . import config
from .data import PAPER_FIGURES_DIR
from .registry import FIGURES

app = typer.Typer(help="Build publication-ready figures for the paper.")


def _resolve_output(output: str | None, filename: str, fmt: str) -> Path:
    """Resolve ``--output`` into a concrete file path.

    - omitted            -> paper/figures/<default name>.<fmt>
    - a directory        -> <dir>/<default name>.<fmt>
    - a file path        -> use as-is (its own suffix wins)
    """
    default_name = Path(filename).with_suffix(f".{fmt}").name
    if output is None:
        return PAPER_FIGURES_DIR / default_name
    p = Path(output)
    if p.suffix:
        return p
    return p / default_name


@app.command()
def build(
    name: Annotated[str, typer.Argument(help="Figure name (see `list`).")],
    output: Annotated[
        str | None,
        typer.Option(help="Output file or directory. Default: paper/figures/."),
    ] = None,
    fmt: Annotated[
        str, typer.Option(help="Vector/raster format when auto-naming.")
    ] = "pdf",
    seed: Annotated[int, typer.Option(help="Seed for the generator gallery.")] = 1234,
    alpha: Annotated[
        float | None,
        typer.Option(help="Significance level (default: 0.05 power, 0.01 noise)."),
    ] = None,
    results_dir: Annotated[
        Path | None,
        typer.Option(help="Override the results directory for data figures."),
    ] = None,
    B: Annotated[
        int | None,
        typer.Option(
            "--B",
            help="all data figures: permutation budget the figure "
            "is pinned to (default: config.POWER_CURVE_BUDGET / LADDER_BUDGET).",
        ),
    ] = None,
    R: Annotated[
        int | None,
        typer.Option(
            "--R",
            help="all data figures: replication count the figure "
            "is pinned to (default: config.POWER_CURVE_BUDGET / LADDER_BUDGET).",
        ),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="type-one / power-curves / noise / rbex: fail instead of warn when a curated test has "
            "no result file at the pinned budget.",
        ),
    ] = False,
) -> None:
    """Build one figure by name."""
    if name not in FIGURES:
        raise typer.BadParameter(
            f"Unknown figure '{name}'. Choices: {', '.join(FIGURES)}"
        )
    spec = FIGURES[name]
    filename = spec.filename
    if name in ("power-curves", "type-one"):
        # The default name carries the budget the figure is pinned to.
        namer = (
            config.power_curves_filename
            if name == "power-curves"
            else config.type_one_filename
        )
        filename = namer(
            B if B is not None else config.POWER_CURVE_BUDGET["B"],
            R if R is not None else config.POWER_CURVE_BUDGET["R"],
            alpha if alpha is not None else config.POWER_CURVE_ALPHA,
        )
    elif name == "ladder-ablation":
        filename = config.ladder_ablation_filename(
            B if B is not None else config.LADDER_BUDGET["B"],
            R if R is not None else config.LADDER_BUDGET["R"],
            alpha if alpha is not None else config.LADDER_ALPHA,
        )
    elif name == "ladder-choice":
        filename = config.ladder_choice_filename(
            B if B is not None else config.LADDER_BUDGET["B"],
            R if R is not None else config.LADDER_BUDGET["R"],
            alpha if alpha is not None else config.LADDER_ALPHA,
        )
    elif name == "noise":
        filename = config.noise_filename(
            B if B is not None else config.NOISE_BUDGET["B"],
            R if R is not None else config.NOISE_BUDGET["R"],
            alpha if alpha is not None else config.NOISE_ALPHA,
        )
    elif name == "shape-ablation":
        filename = config.shape_ablation_filename(
            B if B is not None else config.SHAPE_BUDGET["B"],
            R if R is not None else config.SHAPE_BUDGET["R"],
            alpha if alpha is not None else config.SHAPE_ALPHA,
        )
    elif name.startswith("shape-profile-n"):
        filename = config.shape_profile_filename(
            int(name.rsplit("n", 1)[1]),
            B if B is not None else config.SHAPE_PROFILE_BUDGET["B"],
            R if R is not None else config.SHAPE_PROFILE_BUDGET["R"],
            alpha if alpha is not None else config.SHAPE_ALPHA,
        )
    elif name == "timing":
        filename = config.timing_filename(B if B is not None else config.TIMING_B)
    elif name == "rbex":
        filename = config.rbex_filename(
            B if B is not None else config.RBEX_BUDGET["B"],
            R if R is not None else config.RBEX_BUDGET["R"],
            alpha if alpha is not None else config.RBEX_ALPHA,
        )
    out_path = _resolve_output(output, filename, fmt)

    kwargs: dict = {}
    if name in ("generators", "generators-ranks", "noise-gallery", "rbex-gallery"):
        kwargs["seed"] = seed
    elif name == "shape-catalogue":
        pass  # pure illustration: no budget, no results directory
    else:  # data figures: type-one, power-curves, noise
        if alpha is not None:
            kwargs["alpha"] = alpha
        if results_dir is not None:
            kwargs["results_dir"] = results_dir
    if name in (
        "type-one",
        "power-curves",
        "ladder-ablation",
        "ladder-choice",
        "noise",
        "rbex",
        "timing",
        "shape-ablation",
        "shape-profile-n200",
        "shape-profile-n500",
    ):
        if B is not None:
            kwargs["B"] = B
        if R is not None:
            kwargs["R"] = R
    if name in (
        "type-one",
        "power-curves",
        "noise",
        "rbex",
        "shape-ablation",
        "shape-profile-n200",
        "shape-profile-n500",
    ):
        kwargs["strict"] = strict

    written = spec.builder(out_path, **kwargs)
    typer.echo(f"Wrote {written}")


@app.command("build-all")
def build_all(
    output_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory for all figures. Default: paper/figures/."),
    ] = None,
    fmt: Annotated[str, typer.Option(help="Output format.")] = "pdf",
) -> None:
    """Build every registered figure with default settings."""
    out_dir = output_dir or PAPER_FIGURES_DIR
    for spec in FIGURES.values():
        out_path = out_dir / Path(spec.filename).with_suffix(f".{fmt}").name
        written = spec.builder(out_path)
        typer.echo(f"  {spec.name}: {written}")


@app.command("list")
def list_figures() -> None:
    """List the registered figures."""
    typer.echo("Available figures:")
    for spec in FIGURES.values():
        typer.echo(f"  {spec.name:<14} -> {spec.filename}")
        typer.echo(f"  {'':<14}    {spec.description}")


if __name__ == "__main__":
    app()
