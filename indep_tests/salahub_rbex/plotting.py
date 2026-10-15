"""Plotting for the rBEX resolution sweep.

Two figure styles:

1. ``plot_quasi_power_resolution`` — rejection rate at ``alpha`` vs BEX
   resolution, reproducing Salahub & Oldford (2025) Fig. quasi_power_rBEX:
   x-axis ``null, 1, 2, ..., N`` (null = independent-uniform size anchor),
   y-axis "fraction <= level", one curve per test, dashed ``alpha`` line.

2. ``plot_pattern_grid`` — a strip of rBEX scatterplots, one per resolution,
   mirroring the paper's rBEX_pattern figure (a visual sanity check that the
   sampler reproduces the manifold).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .experiment import ResolutionSweepResult, rejection_rate_by_resolution
from .patterns import rbex_sample

# Color x marker grid so >10 overlaid tests stay distinguishable (matches the
# power charts in ``indep_tests/commands/plotting.py``): color cycles every 10
# entries, the marker advances every 10 -> 100 distinct (color, marker) pairs.
_TAB10 = plt.cm.tab10.colors
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]


def _res_tick_labels(resolutions: Sequence[int]) -> list[str]:
    return ["null" if int(r) == 0 else str(int(r)) for r in resolutions]


def plot_quasi_power_resolution(
    results_by_test: dict[str, ResolutionSweepResult],
    out_path: str | Path,
    alpha: float = 0.01,
    title: str | None = None,
) -> Path:
    """Quasi-power chart: rejection rate vs resolution, all tests overlaid.

    ``resolution == 0`` is rendered under the "null" tick (independent-uniform
    size); resolutions ``>= 1`` are BEX levels (power).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    all_res: set[int] = set()
    for i, (test_label, sweep) in enumerate(
        sorted(results_by_test.items(), key=lambda kv: kv[0])
    ):
        color = _TAB10[i % len(_TAB10)]
        marker = _MARKERS[i // len(_TAB10) % len(_MARKERS)]
        res, power = rejection_rate_by_resolution(sweep, alpha=alpha)
        all_res.update(int(r) for r in res)
        ax.plot(
            res,
            power,
            marker=marker,
            color=color,
            markersize=5,
            linewidth=1.4,
            label=test_label,
        )

    ax.axhline(alpha, linestyle="--", color="gray", linewidth=0.8, label=f"α={alpha}")

    ticks = sorted(all_res)
    ax.set_xticks(ticks)
    ax.set_xticklabels(_res_tick_labels(ticks))

    n_values = sorted({s.n for s in results_by_test.values()})
    n_label = ", ".join(str(v) for v in n_values)
    ax.set_title(title or f"rBEX quasi-power (N = {n_label}, level = {alpha})")
    ax.set_xlabel("null, and rBEX resolution")
    ax.set_ylabel(r"fraction $\leq$ level (rejection rate)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pattern_grid(
    resolutions: Sequence[int],
    n: int,
    out_path: str | Path,
    seed: int = 0,
) -> Path:
    """Render one rBEX scatter per resolution (paper's rBEX_pattern figure)."""
    resolutions = [int(r) for r in resolutions]
    ncol = len(resolutions)
    fig, axes = plt.subplots(1, ncol, figsize=(2.0 * ncol, 2.2))
    if ncol == 1:
        axes = [axes]
    for ax, r in zip(axes, resolutions):
        rng = np.random.default_rng(seed + r)
        x, y = rbex_sample(n, r, rng)
        ax.scatter(x, y, s=2, color="black", alpha=0.6, edgecolors="none")
        ax.set_title("null" if r == 0 else f"res = {r}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle(f"rBEX patterns (N = {n})", fontsize=11)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out
