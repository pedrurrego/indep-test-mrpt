"""Typer CLI for the resolution-ladder ablation (paper subsec:ladder).

Runs ``pug-grid norm=density`` at each fixed resolution ``k`` of the ladder
over the ablation generators, plus (by default) the two baseline rows the
figure needs at the *same* Monte-Carlo budget: the ``floor(sqrt(n))`` default
(``norm=density``, no explicit ``k``) and the multi-resolution ladder
(``norm=density-multi``).  Jobs run in parallel processes.  Every run goes
through :func:`indep_tests.commands.experiment.run_power_curve`, so seeding,
naming and JSON layout are identical to ``python -m indep_tests run``.

    uv run python -m indep_tests.pug_grid_density_multi run                 # B=500 R=100
    uv run python -m indep_tests.pug_grid_density_multi run --B 2000 --R 1000 --jobs 8
    uv run python -m indep_tests.pug_grid_density_multi run --dry-run
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer

from indep_tests.commands.experiment import (
    _DEFAULT_RESULTS_DIR,
    build_output_filename,
)

app = typer.Typer(help="Resolution-ladder ablation for PUG-Grid-DENSITY-MULTI.")

HERE = Path(__file__).resolve().parent
DEFAULT_ABLATION_DIR = HERE / "ladder_ablation"

# The nine dependent generators of the paper + the three independent ones.
ABLATION_GENERATORS = [
    "dep-normal-0.3",
    "dep-sine-4pi",
    "dep-sine-16pi",
    "dep-modular-5",
    "dep-modular-10",
    "dep-checkerboard-k5-d1",
    "dep-spiral-2-turns",
    "dep-spiral-3-turns",
    "dep-yanagimoto",
    "indep-uniform",
    "indep-normal",
    "indep-expon",
]
DEFAULT_KS = [3, 5, 8, 12]
DEFAULT_SAMPLE_SIZES = [20, 50, 100, 200, 500]


@dataclass(frozen=True)
class Job:
    """One ``pug-grid`` power-curve run and where its JSON goes."""

    generator: str
    config: dict = field(hash=False)
    output_path: Path
    sample_sizes: tuple[int, ...]
    alpha: float
    R: int
    B: int
    seed: int

    @property
    def label(self) -> str:
        cfg = "+".join(f"{k}={v}" for k, v in sorted(self.config.items()))
        return f"{self.generator} [{cfg}]"


def _run_job(job: Job) -> tuple[str, float]:
    """Worker: run one power curve and save it. Module-level for pickling."""
    from indep_tests.commands.experiment import run_power_curve, save_results
    from indep_tests.pug_grid import PUGGridTest

    test = PUGGridTest(**job.config)
    t0 = time.perf_counter()
    results = run_power_curve(
        test=test,
        generator_name=job.generator,
        sample_sizes=list(job.sample_sizes),
        alpha=job.alpha,
        R=job.R,
        B=job.B,
        seed=job.seed,
    )
    save_results(
        results,
        test.name,
        job.generator,
        job.alpha,
        job.R,
        job.B,
        job.seed,
        job.output_path,
    )
    return job.label, time.perf_counter() - t0


def _parse_ints(text: str) -> list[int]:
    return [int(s) for s in text.split(",") if s.strip()]


def _parse_ladders(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def build_jobs(
    *,
    generators: list[str],
    ks: list[int],
    sample_sizes: list[int],
    alpha: float,
    R: int,
    B: int,
    seed: int,
    ablation_dir: Path,
    results_dir: Path,
    with_baselines: bool,
    ladders: list[str] = (),
) -> list[Job]:
    """Enumerate (generator × config) jobs with their auto-named output files.

    ``ladders`` are alternative ``density-multi`` ladder specs (e.g. ``"3-sqrt"``,
    see ``src_pug_grid.pug_independence_test.resolve_ladder``); their JSONs go to
    ``ablation_dir`` as ``{gen}_pug-grid_ladder={spec}+norm=density-multi_B{B}_R{R}.json``.
    The production ladder is the ``norm=density-multi`` baseline in ``results_dir``.
    """
    configs: list[tuple[dict, Path]] = [
        ({"norm": "density", "k": k}, ablation_dir) for k in ks
    ]
    configs += [
        ({"norm": "density-multi", "ladder": spec}, ablation_dir) for spec in ladders
    ]
    if with_baselines:
        configs += [
            ({"norm": "density"}, results_dir),  # floor(sqrt(n)) default
            ({"norm": "density-multi"}, results_dir),  # the ladder
        ]
    jobs = []
    for generator in generators:
        for config, out_dir in configs:
            filename = build_output_filename(generator, "pug-grid", config, B, R)
            jobs.append(
                Job(
                    generator=generator,
                    config=config,
                    output_path=out_dir / filename,
                    sample_sizes=tuple(sample_sizes),
                    alpha=alpha,
                    R=R,
                    B=B,
                    seed=seed,
                )
            )
    return jobs


@app.command()
def run(
    B: Annotated[int, typer.Option("--B", help="Permutations per test.")] = 500,
    R: Annotated[int, typer.Option("--R", help="Monte-Carlo replications.")] = 100,
    seed: Annotated[int, typer.Option(help="Base seed (harness convention).")] = 1234,
    alpha: Annotated[float, typer.Option(help="Significance level.")] = 0.05,
    sample_sizes: Annotated[
        str, typer.Option("--sample-sizes", help="Comma-separated sample sizes.")
    ] = ",".join(map(str, DEFAULT_SAMPLE_SIZES)),
    ks: Annotated[
        str, typer.Option(help="Comma-separated fixed grid resolutions ('' for none).")
    ] = ",".join(map(str, DEFAULT_KS)),
    ladders: Annotated[
        str,
        typer.Option(
            help="Comma-separated alternative density-multi ladder specs, rungs "
            "joined by '-' (e.g. '3-sqrt,3-5-8-12'). Default: none."
        ),
    ] = "",
    generators: Annotated[
        str,
        typer.Option(help="Comma-separated generator names, or 'all' (default set)."),
    ] = "all",
    jobs: Annotated[int, typer.Option(help="Parallel worker processes.")] = 6,
    with_baselines: Annotated[
        bool,
        typer.Option(
            "--with-baselines/--no-baselines",
            help="Also run the sqrt(n) (norm=density) and density-multi rows "
            "into --results-dir at the same budget.",
        ),
    ] = True,
    force: Annotated[
        bool, typer.Option("--force", help="Re-run jobs whose output file exists.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List the jobs and exit.")
    ] = False,
    ablation_dir: Annotated[
        Path, typer.Option(help="Where fixed-k JSONs go.")
    ] = DEFAULT_ABLATION_DIR,
    results_dir: Annotated[
        Path, typer.Option(help="Where the baseline JSONs go.")
    ] = _DEFAULT_RESULTS_DIR,
) -> None:
    """Run the fixed-k sweep (and baselines) at one Monte-Carlo budget."""
    gens = ABLATION_GENERATORS if generators == "all" else generators.split(",")
    all_jobs = build_jobs(
        generators=[g.strip() for g in gens],
        ks=_parse_ints(ks),
        ladders=_parse_ladders(ladders),
        sample_sizes=_parse_ints(sample_sizes),
        alpha=alpha,
        R=R,
        B=B,
        seed=seed,
        ablation_dir=ablation_dir,
        results_dir=results_dir,
        with_baselines=with_baselines,
    )
    todo = [j for j in all_jobs if force or not j.output_path.exists()]
    skipped = len(all_jobs) - len(todo)

    typer.echo(
        f"ladder ablation: B={B} R={R} seed={seed} alpha={alpha} "
        f"sizes={sample_sizes} ks={ks} generators={len(gens)} "
        f"jobs={len(todo)} to run ({skipped} already on disk), workers={jobs}"
    )
    if dry_run:
        for j in all_jobs:
            state = "skip" if j not in todo else "run "
            typer.echo(f"  {state} {j.label:50s} -> {j.output_path}")
        return
    if not todo:
        typer.echo("Nothing to do (use --force to re-run).")
        return

    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_job, j): j for j in todo}
        for fut in as_completed(futures):
            label, secs = fut.result()
            done += 1
            typer.echo(f"[{done}/{len(todo)}] {label:50s} {secs:7.1f}s")
    typer.echo(f"Done in {(time.perf_counter() - t0) / 60:.1f} min.")


@app.command("list-jobs")
def list_jobs(
    B: Annotated[int, typer.Option("--B")] = 500,
    R: Annotated[int, typer.Option("--R")] = 100,
) -> None:
    """Show which of the figure's inputs exist on disk at a budget."""
    for j in build_jobs(
        generators=ABLATION_GENERATORS,
        ks=DEFAULT_KS,
        sample_sizes=DEFAULT_SAMPLE_SIZES,
        alpha=0.05,
        R=R,
        B=B,
        seed=1234,
        ablation_dir=DEFAULT_ABLATION_DIR,
        results_dir=_DEFAULT_RESULTS_DIR,
        with_baselines=True,
    ):
        mark = "ok     " if j.output_path.exists() else "MISSING"
        typer.echo(f"  {mark} {j.output_path.name}")
