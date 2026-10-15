"""Logic for plotting independence test results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_TAB10 = plt.cm.tab10.colors
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]

_DEFAULT_RESULTS_DIR = Path(__file__).parents[1] / "results"
_DEFAULT_CHARTS_DIR = _DEFAULT_RESULTS_DIR / "charts"
_PAPER_FIGURES_DIR = Path(__file__).parents[2] / "paper" / "figures"

# Order matches the paper's narrative arc: independent → monotone smooth →
# high-frequency → geometric / categorical.
_GENERATOR_DISPLAY_ORDER = [
    "indep-uniform",
    "indep-normal",
    "indep-expon",
    "dep-normal-0.3",
    "dep-yanagimoto",
    "dep-sine-4pi",
    "dep-sine-16pi",
    "dep-modular-5",
    "dep-modular-10",
    "dep-spiral-2-turns",
    "dep-spiral-3-turns",
    "dep-checkerboard-k5-d1",
]

_GENERATOR_DISPLAY_LABELS = {
    "indep-uniform": "Uniform (independent)",
    "indep-normal": "Normal (independent)",
    "indep-expon": "Exponential (independent)",
    "dep-normal-0.3": "Bivariate normal, ρ=0.3",
    "dep-yanagimoto": "Yanagimoto parallelogram",
    "dep-sine-4pi": "Sine (4π)",
    "dep-sine-16pi": "Sine (16π)",
    "dep-modular-5": "Modular (5x mod 1)",
    "dep-modular-10": "Modular (10x mod 1)",
    "dep-spiral-2-turns": "Spiral (2 turns)",
    "dep-spiral-3-turns": "Spiral (3 turns)",
    "dep-checkerboard-k5-d1": "Checkerboard (5×5)",
}


def resolve_chart_output_path(output: str | None, generator: str) -> Path | None:
    """Resolve the output path for a chart.

    - None (flag absent)      -> return None (show interactively)
    - "" (flag with no value) -> default charts dir + auto filename
    - a directory path        -> that dir + auto filename
    - a file path             -> use as-is
    """
    if output is None:
        return None

    filename = f"{generator}.png"

    if output == "":
        return _DEFAULT_CHARTS_DIR / filename

    p = Path(output)
    if p.suffix:
        return p

    return p / filename


def _load_results_for_generator(generator: str, results_dir: Path) -> list[dict]:
    """Load all JSON result files matching a generator name."""
    pattern = f"{generator}_*.json"
    files = sorted(results_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No result files found for generator '{generator}' in {results_dir}"
        )

    data = []
    for f in files:
        with open(f) as fh:
            data.append(json.load(fh))
    return data


def plot_power_curve(
    generator: str,
    results_dir: Path = _DEFAULT_RESULTS_DIR,
    output: Path | None = None,
    alpha: float = 0.05,
) -> None:
    """Plot rejection rate vs sample size for all tests run on a generator.

    Parameters
    ----------
    generator : str
        Generator name to filter result files (e.g. "dep-sine-16pi").
    results_dir : Path
        Directory containing JSON result files.
    output : Path | None
        If given, save the figure to this path. Otherwise, show interactively.
    alpha : float
        Significance level — drawn as a horizontal reference line.
    """
    all_data = _load_results_for_generator(generator, results_dir)

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.set_title(f"Power curve: {generator}")
    ax.set_xlabel("n")
    ax.set_ylabel("Rejection rate")
    ax.set_ylim(0, 1)
    ax.set_yticks([i * 0.1 for i in range(11)])

    for i, experiment in enumerate(all_data):
        test_name = experiment["test"]
        ns = [r["n"] for r in experiment["results"]]
        rates = [r["rejection_rate"] for r in experiment["results"]]
        color = _TAB10[i % len(_TAB10)]
        marker = _MARKERS[i // len(_TAB10) % len(_MARKERS)]
        ax.plot(ns, rates, marker=marker, color=color, label=test_name)

    ax.axhline(y=alpha, color="gray", linestyle="--", linewidth=0.8, label=f"α={alpha}")
    handles, labels = ax.get_legend_handles_labels()
    # Move α line (last entry) to the front
    handles = [handles[-1]] + handles[:-1]
    labels = [labels[-1]] + labels[:-1]
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize="small")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        print(f"Saved to {output}")
    else:
        plt.show()


def resolve_generators_chart_output_path(output: str | None) -> Path | None:
    """Resolve the output path for the all-generators chart.

    - None (flag absent)      -> return None (show interactively)
    - "" (flag with no value) -> default paper figures dir + auto filename
    - a directory path        -> that dir + auto filename
    - a file path             -> use as-is
    """
    if output is None:
        return None

    filename = "all_generators.png"

    if output == "":
        return _PAPER_FIGURES_DIR / filename

    p = Path(output)
    if p.suffix:
        return p

    return p / filename


def plot_all_generators(
    output: Path | None = None,
    n: int = 500,
    seed: int = 1234,
) -> None:
    """Plot a 5x3 grid of scatter plots, one sample per generator.

    Parameters
    ----------
    output : Path | None
        If given, save the figure to this path. Otherwise, show interactively.
    n : int
        Number of points to sample from each generator.
    seed : int
        Random seed used for every generator (independently re-seeded so the
        figure is reproducible regardless of iteration order).
    """
    from indep_tests.generators import get_generator

    n_rows, n_cols = 5, 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 14), layout="constrained")

    for ax, name in zip(axes.flat, _GENERATOR_DISPLAY_ORDER):
        rng = np.random.default_rng(seed)
        X, Y = get_generator(name)(n, rng)
        ax.scatter(X, Y, s=10, alpha=0.7, color=_TAB10[0])
        ax.set_title(_GENERATOR_DISPLAY_LABELS[name], fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)

    for ax in axes.flat[len(_GENERATOR_DISPLAY_ORDER) :]:
        ax.set_visible(False)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()
