"""Compare density-multi ladder variants against the production ladder.

Reads, at one Monte-Carlo budget:

* the production ladder ``{gen}_pug-grid_norm=density-multi_B{B}_R{R}.json`` and the
  ``floor(sqrt n)`` single grid ``{gen}_pug-grid_norm=density_B{B}_R{R}.json`` from
  ``indep_tests/results/``;
* the fixed-k single grids ``..._k={k}+norm=density_...`` and every alternative ladder
  ``..._ladder={spec}+norm=density-multi_...`` from ``ladder_ablation/``.

Prints, per sample size, the mean and worst-case rejection rate over the paper's
nine-generator panel (``paper_figures.config.LADDER_PANEL``) for every variant, the
size on the three null generators, and a per-generator table at the requested sizes.

    uv run python -m indep_tests.pug_grid_density_multi.compare_ladders compare
    uv run python -m indep_tests.pug_grid_density_multi.compare_ladders compare --B 2000 --R 1000 --n 200 --n 500
    uv run python -m indep_tests.pug_grid_density_multi.compare_ladders paper-table
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import typer

from indep_tests.paper_figures.config import LADDER_PANEL

HERE = Path(__file__).resolve().parent
ABLATION = HERE / "ladder_ablation"
RESULTS = HERE.parent / "results"
NULLS = ["indep-uniform", "indep-normal", "indep-expon"]
PRODUCTION = "3-5-8-12-sqrt"

app = typer.Typer(add_completion=False)


def _rates(path: Path) -> dict[int, float]:
    d = json.loads(path.read_text())
    return {r["n"]: r["rejection_rate"] for r in d["results"]}


def load_variants(B: int, R: int) -> dict[str, dict[str, dict[int, float]]]:
    """variant name -> generator -> n -> rejection rate."""
    gens = list(LADDER_PANEL) + NULLS
    variants: dict[str, dict[str, dict[int, float]]] = {}

    def add(name: str, pattern: str) -> None:
        table = {}
        for g in gens:
            p = pattern.format(gen=g, B=B, R=R)
            path = Path(p)
            if path.exists():
                table[g] = _rates(path)
        if table:
            variants[name] = table

    for k in (3, 5, 8, 12):
        add(
            f"k={k}",
            str(ABLATION / f"{{gen}}_pug-grid_k={k}+norm=density_B{{B}}_R{{R}}.json"),
        )
    add("k=sqrt", str(RESULTS / "{gen}_pug-grid_norm=density_B{B}_R{R}.json"))
    add(
        f"ladder {PRODUCTION} (production)",
        str(RESULTS / "{gen}_pug-grid_norm=density-multi_B{B}_R{R}.json"),
    )
    specs = sorted(
        {
            m.group(1)
            for f in ABLATION.glob(
                f"*_pug-grid_ladder=*+norm=density-multi_B{B}_R{R}.json"
            )
            if (m := re.search(r"_ladder=([^+]+)\+norm=density-multi_", f.name))
        }
    )
    for spec in specs:
        add(
            f"ladder {spec}",
            str(
                ABLATION
                / f"{{gen}}_pug-grid_ladder={spec}+norm=density-multi_B{{B}}_R{{R}}.json"
            ),
        )
    return variants


@app.command()
def compare(
    B: Annotated[int, typer.Option("--B")] = 2000,
    R: Annotated[int, typer.Option("--R")] = 1000,
    n: Annotated[
        list[int] | None,
        typer.Option("--n", help="Sizes for the per-generator tables."),
    ] = None,
) -> None:
    variants = load_variants(B, R)
    if not variants:
        raise typer.Exit("no result files found at this budget")
    sizes = sorted({m for t in variants.values() for g in t.values() for m in g})
    per_gen_sizes = n or [200, 500]
    names = list(variants)
    w = max(len(v) for v in names) + 2

    print(f"Budget B={B} R={R}; panel = {len(LADDER_PANEL)} dependent generators\n")
    for stat, fn in (("mean", lambda xs: sum(xs) / len(xs)), ("worst", min)):
        print(f"{stat.upper()} power over the panel, by n")
        print(" " * w + "".join(f"{m:>8d}" for m in sizes))
        for name in names:
            t = variants[name]
            row = []
            for m in sizes:
                xs = [t[g][m] for g in LADDER_PANEL if g in t and m in t[g]]
                row.append(
                    f"{fn(xs):>8.3f}" if len(xs) == len(LADDER_PANEL) else f"{'-':>8}"
                )
            print(f"{name:<{w}}" + "".join(row))
        print()

    print("SIZE (mean over the three nulls), by n")
    print(" " * w + "".join(f"{m:>8d}" for m in sizes))
    for name in names:
        t = variants[name]
        row = []
        for m in sizes:
            xs = [t[g][m] for g in NULLS if g in t and m in t[g]]
            row.append(
                f"{sum(xs) / len(xs):>8.3f}" if len(xs) == len(NULLS) else f"{'-':>8}"
            )
        print(f"{name:<{w}}" + "".join(row))
    print()

    gw = max(len(g) for g in LADDER_PANEL + NULLS) + 2
    for m in per_gen_sizes:
        print(f"PER-GENERATOR rejection rate at n={m}")
        header = " " * gw + "".join(f"{nm[:14]:>16}" for nm in names)
        print(header)
        for g in LADDER_PANEL + NULLS:
            cells = []
            for name in names:
                v = variants[name].get(g, {}).get(m)
                cells.append(f"{v:>16.3f}" if v is not None else f"{'-':>16}")
            print(f"{g:<{gw}}" + "".join(cells))
        print()


LADDER_ROWS = [  # (variant key in load_variants, LaTeX label)
    (
        "ladder 3-5-8-12-sqrt (production)",
        r"$\{3,5,8,12,\lfloor\sqrt n\rfloor\}$ (production)",
    ),
    ("ladder 3-sqrt", r"$\{3,\lfloor\sqrt n\rfloor\}$"),
    ("ladder 3-5-8-12", r"$\{3,5,8,12\}$"),
    ("ladder all", r"$\{2,\dots,\lfloor\sqrt n\rfloor\}$"),
]


@app.command()
def paper_table(
    B: Annotated[int, typer.Option("--B")] = 2000,
    R: Annotated[int, typer.Option("--R")] = 1000,
    sizes: Annotated[
        str, typer.Option(help="Comma-separated n columns.")
    ] = "100,200,500",
) -> None:
    """Print the LaTeX rows of the ladder-choice table (mean / worst / size by n)."""
    variants = load_variants(B, R)
    ns = [int(s) for s in sizes.split(",")]
    for key, label in LADDER_ROWS:
        t = variants[key]
        cells = []
        for m in ns:
            xs = [t[g][m] for g in LADDER_PANEL]
            nulls = [t[g][m] for g in NULLS]
            cells += [
                f"{sum(xs) / len(xs):.2f}",
                f"{min(xs):.2f}",
                f"{sum(nulls) / len(nulls):.3f}",
            ]
        print(f"                {label:<58} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    app()
