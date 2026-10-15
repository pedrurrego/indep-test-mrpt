"""LaTeX rows of ``tab:shape_profile``: the ladder-free summary of the resolution profiles.

For each family and n in {200, 500}: the *envelope* (best power over the family's cell
counts, per generator; then mean and worst over the nine paper generators) and the power
at the family's count nearest to 64 cells (square 64, rect 72, rotated 60, golden 64,
power 64; mean and worst over generators). Reads ``results_kprofile/`` at the power budget.

    uv run python -m indep_tests.shape_ablation.profile_table
"""

from __future__ import annotations

import numpy as np

from indep_tests.paper_figures import config, data

FAMILIES = [
    ("square", r"\textbf{Square grid}"),
    ("rect", r"Rectangles $2{:}1$"),
    ("rotated", r"Square rotated $45^\circ$"),
    ("golden", r"Golden bisection"),
    ("power", r"Power diagram"),
]
NS = (200, 500)
TARGET = 64


def summarize(geo: str, n: int, B: int, R: int):
    peaks, at64, k64 = [], [], None
    for g in config.SHAPE_PANEL:
        Ks, r = data.load_shape_profile_at_n(g, geo, n=n, B=B, R=R)
        Ks, r = np.array(Ks), np.array(r)
        peaks.append(r.max())
        j = int(np.argmin(np.abs(Ks - TARGET)))
        at64.append(r[j])
        k64 = int(Ks[j])
    return np.mean(peaks), np.min(peaks), np.mean(at64), np.min(at64), k64


def main():
    B, R = config.SHAPE_PROFILE_BUDGET["B"], config.SHAPE_PROFILE_BUDGET["R"]
    print(f"% tab:shape_profile rows, B={B} R={R}, panel={config.SHAPE_PANEL}")
    for geo, label in FAMILIES:
        cells = []
        for n in NS:
            pm, pw, am, aw, k64 = summarize(geo, n, B, R)
            cells += [f"{pm:.2f}", f"{pw:.2f}", f"{am:.2f}", f"{aw:.2f}"]
        cells_tex = " & ".join(cells)
        if geo == "square":
            cells_tex = " & ".join(rf"\textbf{{{c}}}" for c in cells)
        print(f"                {label:28s} & {k64:3d} & {cells_tex} \\\\")


if __name__ == "__main__":
    main()
