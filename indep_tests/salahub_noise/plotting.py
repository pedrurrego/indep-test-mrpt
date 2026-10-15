"""Plotting for noise-sweep results.

Two figure styles are supported:

1. `plot_median_p` / `plot_median_log_p` — median p-value vs (1 - R^2),
   mirroring the orphaned Salahub appendix figures (`performance.png`,
   `logPperformance.png`) driven by `scripts/noise.R`.

2. `plot_quasi_power` — rejection rate at alpha vs discrete noise level,
   reproducing the layout of Salahub & Oldford (2025) Figure 11:
   2 x 3 panel grid with one pattern per panel and test curves overlaid.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .experiment import NoiseSweepResult, rejection_rate_by_sigma

# Color x marker grid so >10 overlaid tests stay distinguishable (matches the
# power charts in ``indep_tests/commands/plotting.py``): color cycles every 10
# entries, the marker advances every 10 -> 100 distinct (color, marker) pairs.
_TAB10 = plt.cm.tab10.colors
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]


def median_by_bin(
    x: np.ndarray,
    y: np.ndarray,
    breaks: int | Sequence[float] = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate R's `medianPlot`: bin x into intervals, return (bin_center, median_y)."""
    if isinstance(breaks, int):
        edges = np.linspace(np.nanmin(x), np.nanmax(x), breaks + 1)
    else:
        edges = np.asarray(breaks, dtype=float)
    bin_idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    centers = 0.5 * (edges[:-1] + edges[1:])

    medians = np.full(len(centers), np.nan)
    for k in range(len(centers)):
        vals = y[bin_idx == k]
        if vals.size > 0:
            medians[k] = float(np.median(vals))

    mask = ~np.isnan(medians)
    return centers[mask], medians[mask]


def _extract(result: NoiseSweepResult) -> tuple[np.ndarray, np.ndarray]:
    rsq = np.array([r.rsq for r in result.records])
    pval = np.array([r.p_value for r in result.records])
    return 1.0 - rsq, pval


def plot_median_p(
    results: list[NoiseSweepResult],
    out_path: str | Path,
    breaks: int | Sequence[float] = 100,
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
    title: str | None = None,
) -> Path:
    """Plot median p-value vs (1 - R^2), one curve per result (Salahub style)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for result in results:
        x, p = _extract(result)
        xs, med = median_by_bin(x, p, breaks=breaks)
        ax.plot(xs, med, marker="o", markersize=3, linewidth=1, label=result.pattern)
    ax.set_xlabel(r"$1 - R^2$")
    ax.set_ylabel("Median $p$")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_median_log_p(
    results: list[NoiseSweepResult],
    out_path: str | Path,
    breaks: int | Sequence[float] = 50,
    xlim: tuple[float, float] = (0.0, 0.2),
    ylim: tuple[float, float] = (-250.0, 0.0),
    title: str | None = None,
) -> Path:
    """Plot median log10(p) vs (1 - R^2) on a low-noise zoom (Salahub style)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for result in results:
        x, p = _extract(result)
        log_p = np.log10(np.clip(p, 1e-300, None))
        xs, med = median_by_bin(x, log_p, breaks=breaks)
        ax.plot(xs, med, marker="o", markersize=3, linewidth=1, label=result.pattern)
    ax.set_xlabel(r"$1 - R^2$")
    ax.set_ylabel(r"Median $\log_{10} p$")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_quasi_power_pattern(
    pattern: str,
    results_by_test: dict[str, NoiseSweepResult],
    out_path: str | Path,
    alpha: float = 0.01,
    x_as_level: bool = True,
    null_rates: dict[str, float] | None = None,
) -> Path:
    """Single quasi-power chart for one pattern with all tests overlaid.

    Matches the style of `indep_tests/results/charts/*.png`:
    - Title: "Quasi-power curve: <pattern>"
    - X-axis: noise level l (if `x_as_level`, else sigma)
    - Y-axis: rejection rate at alpha
    - One curve per test, dashed alpha line

    If `null_rates` is provided, each test's null rejection rate is drawn
    at the leftmost x=0 tick (relabelled "null"), connected to its main
    curve. Mirrors Salahub & Oldford (2025) Figure 11.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    null_rates = null_rates or {}

    for i, (test_label, sweep) in enumerate(
        sorted(results_by_test.items(), key=lambda kv: kv[0])
    ):
        color = _TAB10[i % len(_TAB10)]
        marker = _MARKERS[i // len(_TAB10) % len(_MARKERS)]
        sigmas, power = rejection_rate_by_sigma(sweep, alpha=alpha)
        xs = sigmas * 40.0 if x_as_level else sigmas

        if test_label in null_rates and x_as_level:
            xs_full = np.concatenate([[0.0], xs])
            power_full = np.concatenate([[null_rates[test_label]], power])
            ax.plot(
                xs_full,
                power_full,
                marker=marker,
                color=color,
                markersize=4,
                linewidth=1.2,
                label=test_label,
            )
        else:
            ax.plot(
                xs,
                power,
                marker=marker,
                color=color,
                markersize=4,
                linewidth=1.2,
                label=test_label,
            )

    ax.axhline(alpha, linestyle="--", color="gray", linewidth=0.8, label=f"α={alpha}")

    if null_rates and x_as_level:
        ticks = list(range(0, 11))
        labels = ["null"] + [str(i) for i in range(1, 11)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

    ax.set_title(f"Quasi-power curve: {pattern}")
    ax.set_xlabel("noise level $l$" if x_as_level else r"$\sigma$")
    ax.set_ylabel("Rejection rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=4,
        frameon=True,
        fontsize=8,
    )
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_quasi_power(
    results_by_test: dict[str, list[NoiseSweepResult]],
    out_dir: str | Path,
    alpha: float = 0.01,
    patterns: Sequence[str] | None = None,
    x_as_level: bool = True,
    filename_prefix: str = "quasi-power-",
) -> list[Path]:
    """Write one quasi-power PNG per pattern into `out_dir`, all tests overlaid.

    Parameters
    ----------
    results_by_test : dict[str, list[NoiseSweepResult]]
        Keys are test labels; values are sweeps for that test across patterns.
    out_dir : path-like
        Directory to write `<filename_prefix><pattern>.png` into.
    alpha : float
        Rejection threshold.
    patterns : sequence of pattern names, optional
        Order and subset of patterns to plot. Defaults to the sorted union
        of patterns found in the results.
    x_as_level : bool
        If True, use integer noise level l (sigma * 40); else raw sigma.
    filename_prefix : str
        Prepended to each pattern name when constructing the filename.

    Returns
    -------
    list of Path, one per pattern written.
    """
    if patterns is None:
        all_patterns = set()
        for rs in results_by_test.values():
            all_patterns.update(r.pattern for r in rs)
        patterns = sorted(all_patterns)

    by_pattern: dict[str, dict[str, NoiseSweepResult]] = {}
    for test_label, sweeps in results_by_test.items():
        for sweep in sweeps:
            by_pattern.setdefault(sweep.pattern, {})[test_label] = sweep

    out_dir = Path(out_dir)
    written: list[Path] = []
    for pattern in patterns:
        per_test = by_pattern.get(pattern, {})
        if not per_test:
            continue
        path = out_dir / f"{filename_prefix}{pattern}.png"
        written.append(
            plot_quasi_power_pattern(
                pattern, per_test, path, alpha=alpha, x_as_level=x_as_level
            )
        )
    return written
