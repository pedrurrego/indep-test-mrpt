"""LaTeX rows for ``tab:zhang_noise`` (Section benchmark, noise sweep).

One row per noise level ``l = 1..10``; one column per test of
:data:`config.NOISE_TESTS` (the exact roster, order and labels of the noise
figure); each entry is the rejection rate at ``alpha`` averaged over the six
patterns of :data:`config.NOISE_PANEL`. A final row averages over all sixty
(pattern, level) cells. The proposed test's columns (both calibrations,
keys ``PUG-Grid*`` in :data:`config.NOISE_TESTS`) are in bold, as in the
manuscript's other tables.

Usage::

    uv run python -m indep_tests.paper_figures.noise_table        # LaTeX rows
    uv run python -m indep_tests.paper_figures.noise_table --plain  # aligned text

Reads the same files at the same budget as the figure, so the table cannot
drift from it: a test missing at the budget is an error here (the figure only
warns).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import config
from .data import NOISE_RESULTS_DIR, load_noise_results_for_pattern

LEVELS = tuple(range(1, 11))


def noise_table_matrix(
    *,
    alpha: float = config.NOISE_ALPHA,
    results_dir: Path = NOISE_RESULTS_DIR,
    B=None,
    R: int | None = None,
) -> tuple[list[str], np.ndarray]:
    """``(labels, M)`` with ``M[level_index, test_index]`` the mean rejection
    rate over the patterns of :data:`config.NOISE_PANEL`."""
    labels = [t.label for t in config.NOISE_TESTS]
    cells = np.full((len(config.NOISE_PANEL), len(LEVELS), len(labels)), np.nan)
    for i, pattern in enumerate(config.NOISE_PANEL):
        curves, _, _ = load_noise_results_for_pattern(
            pattern, alpha, results_dir, B=B, R=R
        )
        for j, style in enumerate(config.NOISE_TESTS):
            if style.key not in curves:
                raise FileNotFoundError(
                    f"no noise result for {style.key!r} on {pattern!r} at the "
                    f"figure budget (R={R or config.NOISE_BUDGET['R']}, "
                    f"B in {B or config.NOISE_BUDGET['B']}, alpha={alpha})"
                )
            levels, power = curves[style.key]
            by_level = dict(zip(np.rint(levels).astype(int), power))
            missing = [lv for lv in LEVELS if lv not in by_level]
            if missing:
                raise ValueError(f"{style.key!r}/{pattern!r}: levels {missing} absent")
            cells[i, :, j] = [by_level[lv] for lv in LEVELS]
    return labels, cells.mean(axis=0)


def _is_proposed(style) -> bool:
    """Both calibrations of the proposed test (permutation and chi^2) are bold."""
    return style.key.startswith("PUG-Grid")


def _fmt_row(values: np.ndarray, bold: bool = True) -> list[str]:
    out = [f"{v:.2f}" for v in values]
    if bold:
        out = [
            rf"\textbf{{{v}}}" if _is_proposed(style) else v
            for v, style in zip(out, config.NOISE_TESTS)
        ]
    return out


def latex_rows(labels: list[str], M: np.ndarray) -> str:
    lines = []
    for lv, row in zip(LEVELS, M):
        lines.append(f"                {lv:<4}& " + " & ".join(_fmt_row(row)) + r" \\")
    lines.append(r"                \midrule")
    lines.append(
        "                Mean& " + " & ".join(_fmt_row(M.mean(axis=0))) + r" \\"
    )
    return "\n".join(lines)


def plain(labels: list[str], M: np.ndarray) -> str:
    w = max(len(s) for s in labels) + 2
    head = f"{'l':>4}" + "".join(f"{s:>{w}}" for s in labels)
    body = [
        f"{lv:>4}" + "".join(f"{v:>{w}.2f}" for v in row) for lv, row in zip(LEVELS, M)
    ]
    body.append(f"{'mean':>4}" + "".join(f"{v:>{w}.2f}" for v in M.mean(axis=0)))
    return "\n".join([head, *body])


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    labels, M = noise_table_matrix()
    print(plain(labels, M) if "--plain" in argv else latex_rows(labels, M))


if __name__ == "__main__":
    main()
