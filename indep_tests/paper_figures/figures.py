"""Builders for the paper's figures.

Each ``build_*`` function renders one publication figure and writes it to
``out_path`` (vector PDF by default — the format is taken from the suffix). They
share the styling in :mod:`.style` and the curated metadata in :mod:`.config`,
and they read only *already-computed* results via :mod:`.data`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config, data
from .style import paper_style


def _save(fig, out_path: Path, *, metadata: dict[str, str] | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, metadata=metadata)
    plt.close(fig)
    return out_path


def _budget_stamp(B, R: int, alpha: float) -> tuple[str, str]:
    """Return the (mathtext, plain) forms of the Monte-Carlo budget line that
    is drawn on a figure and written into its PDF metadata. ``B`` may be a
    tuple of per-test permutation counts (noise benchmark), rendered as
    ``B = 500 / 1000 (per test)``."""
    if isinstance(B, tuple) and len(set(B)) > 1:
        b_tex = " / ".join(str(b) for b in sorted(set(B))) + "$ (per test)"
        b_plain = "/".join(str(b) for b in sorted(set(B))) + " (per test)"
    else:
        b = B[0] if isinstance(B, tuple) else B
        b_tex, b_plain = f"{b}$", str(b)
    tex = rf"$R = {R}$ replications, $B = {b_tex} permutations, $\alpha = {alpha:g}$"
    plain = f"R={R} replications, B={b_plain} permutations, alpha={alpha:g}"
    return tex, plain


class PanelGrid:
    """Figure + ``(n_rows, n_cols)`` axes grid with every spacing fixed in inches.

    An explicit :class:`~matplotlib.gridspec.GridSpec` (no layout engine): the
    column gap, row gap, margins and panel aspect come from ``config.PANEL_*``,
    so all multi-panel figures share the same geometry by construction, and the
    figure height is derived from the panel width. The top margin holds the
    panel titles plus the figure legend; call :meth:`legend` (which sizes that
    margin from the number of legend rows) instead of ``fig.legend``.
    Inner columns hide their y tick labels (``sharey``).
    """

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        *,
        box_aspect: float = config.PANEL_BOX_ASPECT,
        sharey: bool = True,
        margin_left_in: float = config.PANEL_MARGIN_LEFT_IN,
        margin_right_in: float = config.PANEL_MARGIN_RIGHT_IN,
        margin_bottom_in: float = config.PANEL_MARGIN_BOTTOM_IN,
        vgap_in: float = config.PANEL_VGAP_IN,
    ) -> None:
        self.n_rows, self.n_cols = n_rows, n_cols
        self._margin_left = margin_left_in
        self._margin_right = margin_right_in
        self._margin_bottom = margin_bottom_in
        self._vgap = vgap_in
        W = config.TEXTWIDTH_IN
        gap = config.PANEL_GAP_IN
        self.panel_w = (W - margin_left_in - margin_right_in) / n_cols - gap * (
            n_cols - 1
        ) / n_cols
        self.panel_h = box_aspect * self.panel_w
        self.fig = plt.figure(figsize=(W, 1.0))
        self.gs = GridSpec(n_rows, n_cols, figure=self.fig)
        self.axes = np.empty((n_rows, n_cols), dtype=object)
        for r in range(n_rows):
            for c in range(n_cols):
                share = self.axes[0, 0] if (sharey and (r or c)) else None
                self.axes[r, c] = self.fig.add_subplot(self.gs[r, c], sharey=share)
                if c and sharey:
                    self.axes[r, c].tick_params(labelleft=False)
        self._apply(n_legend_rows=0)

    def _apply(self, *, n_legend_rows: int) -> None:
        W = config.TEXTWIDTH_IN
        top = config.PANEL_TITLE_IN + n_legend_rows * config.PANEL_LEGEND_ROW_IN
        bottom = self._margin_bottom
        H = top + self.n_rows * self.panel_h + (self.n_rows - 1) * self._vgap + bottom
        self.fig.set_size_inches(W, H)
        self.gs.update(
            left=self._margin_left / W,
            right=1.0 - self._margin_right / W,
            bottom=bottom / H,
            top=1.0 - top / H,
            wspace=config.PANEL_GAP_IN / self.panel_w,
            hspace=self._vgap / self.panel_h,
        )

    def legend(self, handles, *, ncol: int, **kwargs) -> None:
        """Figure legend centred above the titles; resizes the top margin."""
        n_legend_rows = -(-len(handles) // ncol)
        self._apply(n_legend_rows=n_legend_rows)
        self.fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            borderaxespad=0.15,
            ncol=ncol,
            **kwargs,
        )

    def labels(self, xlabel: str, ylabel: str) -> None:
        self.fig.supxlabel(xlabel, fontsize=9, y=0.012)
        self.fig.supylabel(ylabel, fontsize=9, x=0.012)


def _legend_handles(styles, *, with_alpha: float | None = None) -> list[Line2D]:
    """Proxy legend handles in a fixed curated order (independent of panels)."""
    handles = [
        Line2D(
            [],
            [],
            color=s.color,
            marker=s.marker,
            linewidth=s.linewidth,
            linestyle=s.linestyle,
            markersize=5,
            label=s.label,
        )
        for s in styles
    ]
    if with_alpha is not None:
        handles.append(
            Line2D(
                [],
                [],
                color="gray",
                linestyle="--",
                linewidth=0.8,
                label=rf"$\alpha={with_alpha:g}$",
            )
        )
    return handles


# --- Figure 1: generator gallery ---------------------------------------------


def build_generator_gallery(
    out_path: Path, *, n: int = 500, seed: int = 1234, ranks: bool = False
) -> Path:
    """Gallery of scatter plots, one sample per generator (Fig. generators).

    With ``ranks=True`` the same samples are shown as pseudo-observations
    ``(R_i/n, S_i/n)`` --- the rank-based tests' actual input --- on
    ``[0, 1]^2`` (Fig. generators_ranks). Same seed => same points.

    Same fixed-inch :class:`PanelGrid` geometry as the power figures, three
    columns, with *square* panels of identical size. The tests are rank-based,
    so the raw axis scales carry no information: each panel is stretched to its
    own data range (a small pad) instead of forcing an equal data aspect, which
    would shrink panels with non-square ranges (exponential / normal margins,
    the sine curves) to different sizes.
    """
    names = config.GENERATOR_DISPLAY_ORDER
    n_cols = 3
    n_rows = -(-len(names) // n_cols)  # ceil: one panel per generator
    pad = 0.03

    with paper_style():
        grid = PanelGrid(
            n_rows,
            n_cols,
            box_aspect=1.0,
            sharey=False,
            margin_left_in=config.PANEL_MARGIN_RIGHT_IN,  # no y labels: symmetric
            margin_bottom_in=config.PANEL_MARGIN_RIGHT_IN,  # no x labels / stamp
            vgap_in=config.PANEL_TITLE_IN + 0.08,  # only the next title in between
        )
        fig, axes = grid.fig, grid.axes
        ax: Axes
        for ax, name in zip(axes.flat, names):
            x, y = data.sample_generator(name, n=n, seed=seed)
            if ranks:
                x, y = data.pseudo_observations(x, y)
            ax.scatter(
                x,
                y,
                s=6,
                alpha=0.6,
                color=config.BLUE,
                edgecolors="none",
                rasterized=True,
            )
            ax.set_title(config.GENERATOR_DISPLAY_LABELS[name])
            for lim, v in ((ax.set_xlim, x), (ax.set_ylim, y)):
                lo, hi = (0.0, 1.0) if ranks else (float(np.min(v)), float(np.max(v)))
                span = hi - lo
                lim(lo - pad * span, hi + pad * span)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        for ax in axes.flat[len(names) :]:
            ax.set_visible(False)
        return _save(fig, out_path)


def _scatter_gallery_panel(
    ax: Axes, x, y, *, unit_square: bool, size: float = 4.0, pad: float = 0.03
) -> None:
    """One gallery scatter panel: no ticks, own data range (or ``[0,1]^2``)."""
    ax.scatter(
        x, y, s=size, alpha=0.6, color=config.BLUE, edgecolors="none", rasterized=True
    )
    for lim, v in ((ax.set_xlim, x), (ax.set_ylim, y)):
        lo, hi = (0.0, 1.0) if unit_square else (float(np.min(v)), float(np.max(v)))
        span = hi - lo
        lim(lo - pad * span, hi + pad * span)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def build_noise_gallery(
    out_path: Path,
    *,
    n: int = config.NOISE_GALLERY_N,
    levels: tuple[int, ...] = config.NOISE_GALLERY_LEVELS,
    seed: int = 1234,
) -> Path:
    """Gallery of the six Zhang (2019) noise patterns (Fig. noise_gallery).

    One row per pattern of ``config.NOISE_PANEL``, one column per noise level
    in ``levels`` (left = least noise), one sample of size ``n`` per panel. The
    same seed is used in every panel of a row, so the columns differ only by
    the noise scale ``sigma = l / 40`` --- the reader sees the *same* points
    being progressively blurred. Unlike :func:`build_generator_gallery` the
    axes are shown (raw ``X``/``Y`` values, a few ticks) and the limits are
    shared along a row, so the growth of the noise is visible on the axes.
    """
    from matplotlib.ticker import MaxNLocator

    names = config.NOISE_PANEL
    n_rows, n_cols = len(names), len(levels)
    pad = 0.03
    with paper_style():
        grid = PanelGrid(
            n_rows,
            n_cols,
            sharey=False,
            margin_left_in=0.62,  # pattern label + y tick labels
            margin_bottom_in=0.22,  # x tick labels of the bottom row only
        )
        axes = grid.axes
        for r, name in enumerate(names):
            samples = [
                data.sample_noise_pattern(name, n=n, level=level, seed=seed)
                for level in levels
            ]
            xs = np.concatenate([x for x, _ in samples])
            ys = np.concatenate([y for _, y in samples])
            for c, (level, (x, y)) in enumerate(zip(levels, samples)):
                ax: Axes = axes[r, c]
                ax.scatter(
                    x,
                    y,
                    s=4,
                    alpha=0.6,
                    color=config.BLUE,
                    edgecolors="none",
                    rasterized=True,
                )
                for lim, v in ((ax.set_xlim, xs), (ax.set_ylim, ys)):
                    lo, hi = float(np.min(v)), float(np.max(v))
                    span = hi - lo
                    lim(lo - pad * span, hi + pad * span)
                ax.xaxis.set_major_locator(MaxNLocator(4))
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.grid(False)
                if r == 0:
                    ax.set_title(rf"$l = {level}$")
                if c == 0:
                    ax.set_ylabel(config.NOISE_PATTERN_LABELS[name], labelpad=3)
                else:
                    ax.tick_params(labelleft=False)
        return _save(grid.fig, out_path)


def build_rbex_gallery(
    out_path: Path,
    *,
    ns: tuple[int, ...] = tuple(config.RBEX_PANEL),
    resolutions: tuple[int, ...] = config.RBEX_GALLERY_RESOLUTIONS,
    seed: int = 1234,
) -> Path:
    """Gallery of Zhang's bisection expanding cross (Fig. rbex_gallery).

    One row per sample size in ``ns`` (the two sizes of the rBEX benchmark),
    one column per BEX resolution ``d`` in ``resolutions``: a uniform sample of
    ``N`` points on ``BEX_d``, the ``2^{d-1} x 2^{d-1}`` lattice of diagonal
    crosses. Panels are ``[0,1]^2`` (the margins are uniform by construction),
    so no rank transform is needed to read them as copula samples; the
    unit axes are shown with ticks at 0, 0.5 and 1.
    """
    n_rows, n_cols = len(ns), len(resolutions)
    pad = 0.03
    with paper_style():
        grid = PanelGrid(
            n_rows,
            n_cols,
            box_aspect=1.0,
            sharey=False,
            margin_left_in=0.56,  # row labels "n = 2000" + y tick labels
            margin_bottom_in=0.20,  # x tick labels of the bottom row
            vgap_in=0.34,  # x tick labels + next title
        )
        axes = grid.axes
        for r, N in enumerate(ns):
            for c, d in enumerate(resolutions):
                ax: Axes = axes[r, c]
                x, y = data.sample_rbex(N, d, seed=seed + d)
                # Smaller markers at the larger N so the lattice stays legible.
                ax.scatter(
                    x,
                    y,
                    s=4.0 * 500 / N,
                    alpha=0.6,
                    color=config.BLUE,
                    edgecolors="none",
                    rasterized=True,
                )
                ax.set_xlim(-pad, 1 + pad)
                ax.set_ylim(-pad, 1 + pad)
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([0, 0.5, 1])
                ax.grid(False)
                if r == 0:
                    ax.set_title(rf"$d = {d}$")
                if c == 0:
                    ax.set_ylabel(rf"$n = {N}$", labelpad=3)
                else:
                    ax.tick_params(labelleft=False)
        return _save(grid.fig, out_path)


# --- Type I error: independent generators (subsec:type1) ---------------------


def build_type_one_error(
    out_path: Path,
    *,
    generators=None,
    tests=None,
    results_dir: Path = data.POWER_RESULTS_DIR,
    alpha: float = 0.05,
    B: int | None = None,
    R: int | None = None,
    strict: bool = False,
) -> Path:
    """Type I error figure: rejection rate vs n on the independent generators.

    The y-axis is zoomed near ``alpha`` (the null curves would be invisible on a
    0-1 power axis) and the +/-2 Monte-Carlo-SE calibration band is shaded, so a
    test sitting inside the band is compatible with correct calibration.

    Like the power figure, every curve is read at one *exact* Monte-Carlo
    budget ``(B, R, alpha)`` (default :data:`config.POWER_CURVE_BUDGET`, the
    same runs), that budget is stamped bottom-right and in the PDF metadata,
    and the calibration band uses this ``R``. A curated test with no file at
    the budget is left out and reported on stderr (``strict=True`` raises).
    """
    generators = generators or config.TYPE_I_PANEL
    tests = tests or config.TYPE_I_TESTS
    B = config.POWER_CURVE_BUDGET["B"] if B is None else B
    R = config.POWER_CURVE_BUDGET["R"] if R is None else R
    n_cols = min(3, len(generators))
    n_rows = -(-len(generators) // n_cols)  # ceil: one panel per generator
    used: list = []
    missing: list[tuple[str, str]] = []
    max_rate = alpha

    # MC standard error of a rejection rate near alpha at R replications.
    se = (alpha * (1.0 - alpha) / R) ** 0.5
    band_lo, band_hi = alpha - 2 * se, alpha + 2 * se

    with paper_style():
        grid = PanelGrid(n_rows, n_cols)
        fig, axes = grid.fig, grid.axes
        ax: Axes
        for ax, generator in zip(axes.flat, generators):
            curves = data.load_power_results_at_budget(
                generator, B=B, R=R, alpha=alpha, results_dir=results_dir
            )
            for style in tests:
                if style.key not in curves:
                    missing.append((generator, style.key))
                    continue
                if style not in used:
                    used.append(style)
                ns, rates = curves[style.key]
                max_rate = max(max_rate, float(rates.max()))
                ax.plot(
                    ns,
                    rates,
                    color=style.color,
                    marker=style.marker,
                    linewidth=style.linewidth,
                    linestyle=style.linestyle,
                    markersize=4,
                    zorder=style.zorder,
                )
            ax.axhspan(band_lo, band_hi, color="gray", alpha=0.15, zorder=0)
            ax.axhline(alpha, color="gray", linestyle="--", linewidth=0.8, zorder=1)
            ax.set_title(config.GENERATOR_DISPLAY_LABELS.get(generator, generator))
            ax.set_xscale("log")
            ax.set_xticks([20, 50, 100, 200, 500])
            ax.set_xticklabels(["20", "50", "100", "200", "500"])
            ax.minorticks_off()
        for ax in axes.flat[len(generators) :]:
            ax.set_visible(False)

        top = min(0.5, max(0.12, max_rate * 1.3))
        for ax in axes.flat[: len(generators)]:
            ax.set_ylim(0.0, top)

        if missing:
            lines = [f"  {g:<24} {t}" for g, t in missing]
            msg = (
                f"type-one: {len(missing)} curve(s) have no result file at "
                f"B={B}, R={R}, alpha={alpha:g} and were left out:\n" + "\n".join(lines)
            )
            if strict:
                plt.close(fig)
                raise FileNotFoundError(msg)
            print(msg, file=sys.stderr)

        stamp_tex, stamp_plain = _budget_stamp(B, R, alpha)
        grid.labels("sample size $n$", "rejection rate")
        fig.text(
            0.995,
            0.004,
            stamp_tex,
            ha="right",
            va="bottom",
            fontsize=6,
            color="0.45",
        )
        ordered = [s for s in tests if s in used]
        handles = _legend_handles(ordered, with_alpha=alpha)
        handles.append(
            Patch(facecolor="gray", alpha=0.3, label=r"$\alpha \pm 2\,\mathrm{SE}$")
        )
        grid.legend(handles, ncol=4, columnspacing=1.2)
        return _save(
            fig,
            out_path,
            metadata={
                "Subject": f"Type I error vs n; {stamp_plain}",
                "Keywords": stamp_plain,
            },
        )


# --- Power vs sample size: dependent generators (subsec:power-n) --------------


def build_power_curves(
    out_path: Path,
    *,
    generators=None,
    tests=None,
    results_dir: Path = data.POWER_RESULTS_DIR,
    alpha: float = 0.05,
    B: int | None = None,
    R: int | None = None,
    strict: bool = False,
) -> Path:
    """Multi-panel power-vs-n figure: one panel per generator, curated tests.

    Every curve is read at one *exact* Monte-Carlo budget ``(B, R, alpha)``
    (default :data:`config.POWER_CURVE_BUDGET`) and that budget is stamped on
    the figure and in the PDF metadata. A curated test with no result file at
    the budget is left out of the panel and reported on stderr; with
    ``strict=True`` it raises instead, so a paper build cannot silently ship an
    incomplete figure.
    """
    generators = generators or config.POWER_CURVE_PANEL
    tests = tests or config.POWER_CURVE_TESTS
    B = config.POWER_CURVE_BUDGET["B"] if B is None else B
    R = config.POWER_CURVE_BUDGET["R"] if R is None else R
    n_cols = 3
    n_rows = -(-len(generators) // n_cols)  # ceil: one panel per generator
    used: list = []
    missing: list[tuple[str, str]] = []

    with paper_style():
        grid = PanelGrid(n_rows, n_cols)
        fig, axes = grid.fig, grid.axes
        ax: Axes
        for ax, generator in zip(axes.flat, generators):
            curves = data.load_power_results_at_budget(
                generator, B=B, R=R, alpha=alpha, results_dir=results_dir
            )
            for style in tests:
                if style.key not in curves:
                    missing.append((generator, style.key))
                    continue
                if style not in used:
                    used.append(style)
                ns, rates = curves[style.key]
                ax.plot(
                    ns,
                    rates,
                    color=style.color,
                    marker=style.marker,
                    linewidth=style.linewidth,
                    linestyle=style.linestyle,
                    markersize=4,
                    zorder=style.zorder,
                )
            ax.axhline(alpha, color="gray", linestyle="--", linewidth=0.8, zorder=1)
            ax.set_title(config.GENERATOR_DISPLAY_LABELS.get(generator, generator))
            ax.set_xscale("log")
            ax.set_xticks([20, 50, 100, 200, 500])
            ax.set_xticklabels(["20", "50", "100", "200", "500"])
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.minorticks_off()
        for ax in axes.flat[len(generators) :]:
            ax.set_visible(False)

        if missing:
            lines = [f"  {g:<24} {t}" for g, t in missing]
            msg = (
                f"power-curves: {len(missing)} curve(s) have no result file at "
                f"B={B}, R={R}, alpha={alpha:g} and were left out:\n" + "\n".join(lines)
            )
            if strict:
                plt.close(fig)
                raise FileNotFoundError(msg)
            print(msg, file=sys.stderr)

        stamp_tex, stamp_plain = _budget_stamp(B, R, alpha)
        grid.labels("sample size $n$", "rejection rate")
        fig.text(
            0.995,
            0.004,
            stamp_tex,
            ha="right",
            va="bottom",
            fontsize=6,
            color="0.45",
        )
        ordered = [s for s in tests if s in used]
        grid.legend(
            _legend_handles(ordered, with_alpha=alpha), ncol=4, columnspacing=1.2
        )
        return _save(
            fig,
            out_path,
            metadata={
                "Subject": f"Empirical power vs n; {stamp_plain}",
                "Keywords": stamp_plain,
            },
        )


# --- Power vs noise: Zhang patterns (subsec:power-noise) ---------------------


def build_noise_curves(
    out_path: Path,
    *,
    patterns=None,
    tests=None,
    results_dir: Path = data.NOISE_RESULTS_DIR,
    alpha: float = config.NOISE_ALPHA,
    B=None,
    R: int | None = None,
    strict: bool = False,
) -> Path:
    """Multi-panel power-vs-noise figure: one panel per Zhang pattern.

    Every curve is read at one *exact* budget ``(B, R, alpha)`` (default
    :data:`config.NOISE_BUDGET`; ``B`` may be a tuple because the benchmark
    used a per-test permutation count). The B values actually drawn are
    stamped bottom-right and written into the PDF metadata; a curated test
    with no file at the budget is left out and reported on stderr
    (``strict=True`` raises).
    """
    patterns = patterns or config.NOISE_PANEL
    tests = tests or config.NOISE_TESTS
    B = config.NOISE_BUDGET["B"] if B is None else B
    R = config.NOISE_BUDGET["R"] if R is None else R
    n_rows, n_cols = 2, 3
    used: list = []
    missing: list[tuple[str, str]] = []
    drawn_B: set[int] = set()

    with paper_style():
        grid = PanelGrid(n_rows, n_cols)
        fig, axes = grid.fig, grid.axes
        ax: Axes
        for ax, pattern in zip(axes.flat, patterns):
            curves, null_rates, budgets = data.load_noise_results_for_pattern(
                pattern, alpha=alpha, results_dir=results_dir, B=B, R=R
            )
            for style in tests:
                if style.key not in curves:
                    missing.append((pattern, style.key))
                    continue
                drawn_B.add(budgets[style.key])
                if style not in used:
                    used.append(style)
                levels, power = curves[style.key]
                # Prepend the null rejection rate at level 0 ("null" tick).
                if style.key in null_rates:
                    levels = np.concatenate([[0.0], levels])
                    power = np.concatenate([[null_rates[style.key]], power])
                ax.plot(
                    levels,
                    power,
                    color=style.color,
                    marker=style.marker,
                    linewidth=style.linewidth,
                    linestyle=style.linestyle,
                    markersize=4,
                    zorder=style.zorder,
                )
            ax.axhline(alpha, color="gray", linestyle="--", linewidth=0.8, zorder=1)
            ax.set_title(pattern)
            ax.set_xticks(range(0, 11, 2))
            ax.set_xticklabels(["null", "2", "4", "6", "8", "10"])
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for ax in axes.flat[len(patterns) :]:
            ax.set_visible(False)

        if missing:
            lines = [f"  {p:<14} {t}" for p, t in missing]
            msg = (
                f"noise: {len(missing)} curve(s) have no result file at "
                f"B in {B}, R={R}, alpha={alpha:g} and were left out:\n"
                + "\n".join(lines)
            )
            if strict:
                plt.close(fig)
                raise FileNotFoundError(msg)
            print(msg, file=sys.stderr)

        stamp_tex, stamp_plain = _budget_stamp(tuple(sorted(drawn_B)), R, alpha)
        grid.labels("noise level $l$", "rejection rate")
        fig.text(
            0.995,
            0.004,
            stamp_tex,
            ha="right",
            va="bottom",
            fontsize=6,
            color="0.45",
        )
        ordered = [s for s in tests if s in used]
        grid.legend(
            _legend_handles(ordered, with_alpha=alpha), ncol=3, columnspacing=1.2
        )
        return _save(
            fig,
            out_path,
            metadata={
                "Subject": f"Empirical power vs noise level; {stamp_plain}",
                "Keywords": stamp_plain,
            },
        )


# --- Ladder composition: power vs n for four Bonferroni ladders (subsec:ladder)


def build_ladder_choice(
    out_path: Path,
    *,
    generators=None,
    ladders=None,
    alpha: float = config.LADDER_ALPHA,
    B: int | None = None,
    R: int | None = None,
) -> Path:
    """Power-vs-n curves of four resolution ladders, one panel per generator.

    The raw information behind the paper's ladder-composition table: for each
    dependent generator of the nine-generator panel, the rejection rate of the
    production ladder and of the three alternatives (endpoints, no adaptive
    top, dense) at every sample size of the sweep, all combined by Bonferroni
    on the same samples and permutations. Same panel grammar as
    :func:`build_power_curves`. Read at the single budget ``(B, R)`` (default
    ``config.LADDER_BUDGET``), stamped on the figure and in the PDF metadata.
    """
    generators = generators or config.LADDER_PANEL
    ladders = ladders or config.LADDER_CHOICE_LADDERS
    B = B if B is not None else config.LADDER_BUDGET["B"]
    R = R if R is not None else config.LADDER_BUDGET["R"]
    n_cols = 3
    n_rows = -(-len(generators) // n_cols)

    with paper_style():
        grid = PanelGrid(n_rows, n_cols)
        fig, axes = grid.fig, grid.axes
        ax: Axes
        for ax, generator in zip(axes.flat, generators):
            for style in ladders:
                ns, rates = data.load_ladder_choice_curve(
                    generator, style.key, B=B, R=R
                )
                ax.plot(
                    ns,
                    rates,
                    color=style.color,
                    linestyle=style.linestyle,
                    marker=style.marker,
                    markersize=4,
                    linewidth=style.linewidth,
                    zorder=style.zorder,
                )
            ax.axhline(alpha, color="gray", linestyle="--", linewidth=0.8, zorder=1)
            ax.set_title(config.GENERATOR_DISPLAY_LABELS.get(generator, generator))
            ax.set_xscale("log")
            ax.set_xticks([20, 50, 100, 200, 500])
            ax.set_xticklabels(["20", "50", "100", "200", "500"])
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.minorticks_off()
        for ax in axes.flat[len(generators) :]:
            ax.set_visible(False)

        stamp_tex, stamp_plain = _budget_stamp(B, R, alpha)
        grid.labels("sample size $n$", "rejection rate")
        fig.text(
            0.995,
            0.004,
            stamp_tex,
            ha="right",
            va="bottom",
            fontsize=6,
            color="0.45",
        )
        grid.legend(
            _legend_handles(ladders, with_alpha=alpha), ncol=5, columnspacing=1.2
        )
        return _save(
            fig,
            out_path,
            metadata={
                "Subject": f"Ladder composition: power vs n for four ladders; {stamp_plain}",
                "Keywords": stamp_plain,
            },
        )


# --- Resolution-ladder ablation: power vs single grid k (subsec:ladder) ------


def build_ladder_ablation(
    out_path: Path,
    *,
    generators=None,
    n: int = 500,
    secondary_ns: tuple[int, ...] = config.LADDER_SECONDARY_NS,
    alpha: float = config.LADDER_ALPHA,
    B: int | None = None,
    R: int | None = None,
) -> Path:
    """Power-vs-resolution profile per generator, ladder as reference line.

    One panel per dependent generator of the paper's nine-generator panel. The
    blue curve is the single fixed-resolution density statistic across the
    evaluated grids, plotted at their actual resolution k on a logarithmic axis;
    the horizontal line is the multi-resolution ladder on the same data. The peak of
    the profile moving from left (coarse) to right (fine) across panels is the
    resolution flip; the reference line hugging the peak everywhere is the hedge.

    Solid lines are the sample size ``n``; each entry of ``secondary_ns`` adds
    the same two series at a smaller sample size in the same colours with a
    more broken line style (dashed with open circles, then dotted with open
    squares), so each panel shows how the profile and the ladder move with
    ``n``. Each series runs over the fixed rungs ``config.LADDER_FIXED_KS``
    that lie below its own ``floor(sqrt(n))`` and ends at that top rung (22, 14,
    10 at n = 500, 200, 100), so a fixed rung at or above the cap (k = 12 at
    n = 100) is outside the ladder (Eq. K-ladder) and is not drawn for that
    series. Labelled ticks are the rungs of the primary ``n``. Pass
    ``secondary_ns=()`` for the single-``n`` figure.

    Every rung is read at the single Monte-Carlo budget ``(B, R)`` (default
    ``config.LADDER_BUDGET``); the budget is stamped bottom-right on the figure
    and written into the PDF metadata, and the default filename carries it.
    """
    generators = generators or config.LADDER_PANEL
    B = B if B is not None else config.LADDER_BUDGET["B"]
    R = R if R is not None else config.LADDER_BUDGET["R"]
    n_cols = 3
    n_rows = -(-len(generators) // n_cols)
    sqrt_k = None

    with paper_style():
        grid = PanelGrid(n_rows, n_cols)
        fig, axes = grid.fig, grid.axes
        ax: Axes
        # (n, line style, marker, filled?, ladder line width); the primary n is
        # drawn last so it sits on top.
        secondary_styles = [("--", "o", False, 1.6), (":", "s", False, 1.4)]
        if len(secondary_ns) > len(secondary_styles):
            raise ValueError("at most two secondary sample sizes are supported")
        series = [(n_i, *style) for n_i, style in zip(secondary_ns, secondary_styles)][
            ::-1
        ] + [(n, "-", "o", True, 2.2)]
        fixed_ks = list(config.LADDER_FIXED_KS)
        sqrt_ks: dict[int, int] = {}
        for ax, generator in zip(axes.flat, generators):
            for order, (n_i, linestyle, marker, filled, multi_lw) in enumerate(series):
                sqrt_k_i = int(np.floor(np.sqrt(n_i)))
                admissible = [k for k in fixed_ks if k < sqrt_k_i]
                ks, singles, multi = data.load_ladder_ablation_at_n(
                    generator, n=n_i, fixed_ks=admissible, B=B, R=R
                )
                sqrt_ks[n_i] = ks[-1]
                ax.axhline(
                    multi,
                    color=config.LADDER_MULTI_COLOR,
                    linestyle=linestyle,
                    linewidth=multi_lw,
                    zorder=4 + order,
                )
                ax.plot(
                    ks,
                    singles,
                    color=config.LADDER_SINGLE_COLOR,
                    linestyle=linestyle,
                    linewidth=1.3 if filled else 1.0,
                    marker=marker,
                    markersize=4,
                    markerfacecolor=config.LADDER_SINGLE_COLOR if filled else "white",
                    zorder=3 + order,
                )
            sqrt_k = sqrt_ks[n]
            tick_ks = [k for k in fixed_ks if k < sqrt_k] + [sqrt_k]
            ax.axhline(alpha, color="0.65", linestyle="-", linewidth=0.7, zorder=1)
            ax.set_title(config.GENERATOR_DISPLAY_LABELS.get(generator, generator))
            ax.set_xscale("log")
            ax.set_xticks(tick_ks)
            ax.set_xticklabels([str(k) for k in tick_ks])
            ax.set_xlim(min(tick_ks) / 1.15, max(tick_ks) * 1.15)
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.minorticks_off()
        for ax in axes.flat[len(generators) :]:
            ax.set_visible(False)

        all_ns = [n, *secondary_ns]
        sqrt_list = ", ".join(str(sqrt_ks[n_i]) for n_i in all_ns)
        n_list = ", ".join(str(n_i) for n_i in all_ns)
        xlabel = (
            rf"grid resolution $k$ (each series ends at its own"
            rf" $\lfloor\sqrt{{n}}\rfloor = {sqrt_list}$)"
        )
        grid.labels(xlabel, "rejection rate")
        handles = []
        for n_i, linestyle, marker, filled, multi_lw in series[::-1]:
            handles += [
                Line2D(
                    [],
                    [],
                    color=config.LADDER_SINGLE_COLOR,
                    linestyle=linestyle,
                    marker=marker,
                    markerfacecolor=config.LADDER_SINGLE_COLOR if filled else "white",
                    markersize=5,
                    label=f"single fixed grid, $n = {n_i}$",
                ),
                Line2D(
                    [],
                    [],
                    color=config.LADDER_MULTI_COLOR,
                    linestyle=linestyle,
                    linewidth=min(multi_lw + 0.3, 2.5),
                    label=f"MRPT, $n = {n_i}$",
                ),
            ]
        handles.append(
            Line2D(
                [],
                [],
                color="0.65",
                linestyle="-",
                linewidth=0.7,
                label=rf"$\alpha={alpha:g}$",
            )
        )
        grid.legend(handles, ncol=4 if secondary_ns else 3, columnspacing=1.2)
        stamp_tex, stamp_plain = _budget_stamp(B, R, alpha)
        fig.text(
            0.995,
            0.004,
            stamp_tex,
            ha="right",
            va="bottom",
            fontsize=6,
            color="0.45",
        )
        return _save(
            fig,
            out_path,
            metadata={
                "Subject": (
                    f"Ladder ablation at n={n} (solid)"
                    + (f", secondary n={list(secondary_ns)}" if secondary_ns else "")
                    + f"; {stamp_plain}"
                ),
                "Keywords": f"B={B}, R={R}, n={n_list}, alpha={alpha:g}",
            },
        )


# --- rBEX quasi-power vs resolution (subsec:power-rbex) -----------------------


def build_rbex_curves(
    out_path: Path,
    *,
    ns=None,
    tests=None,
    results_dir: Path = data.RBEX_RESULTS_DIR,
    alpha: float = config.RBEX_ALPHA,
    B=None,
    R: int | None = None,
    strict: bool = False,
) -> Path:
    """Two-panel rBEX quasi-power figure: one panel per sample size.

    X-axis is the BEX resolution with ``0`` relabelled "null" (independent
    uniform, i.e. size); resolutions ``>= 1`` are power against ``BEX_d``. Every
    curve collapses back to ``alpha`` as ``d`` grows (Zhang's non-uniform
    consistency), the power cliff sliding right with ``n``.

    Every curve is read at one *exact* budget ``(B, R, alpha)`` (default
    :data:`config.RBEX_BUDGET`; ``B`` may be a tuple because the sweeps used a
    per-test permutation count). The B values actually drawn are stamped
    bottom-right and written into the PDF metadata; a curated test with no
    file at the budget is left out and reported on stderr (``strict=True``
    raises). Shares the ``PanelGrid`` geometry of the other data figures.
    """
    ns = ns or config.RBEX_PANEL
    tests = tests or config.RBEX_TESTS
    B = config.RBEX_BUDGET["B"] if B is None else B
    R = config.RBEX_BUDGET["R"] if R is None else R
    used: list = []
    missing: list[tuple[int, str]] = []
    drawn_B: set[int] = set()

    with paper_style():
        grid = PanelGrid(1, len(ns))
        fig, axes = grid.fig, grid.axes
        for ax, n in zip(axes.flat, ns):
            curves, budgets = data.load_rbex_results_for_n(
                n, alpha=alpha, results_dir=results_dir, B=B, R=R
            )
            ticks: set[int] = set()
            for style in tests:
                if style.key not in curves:
                    missing.append((n, style.key))
                    continue
                drawn_B.add(budgets[style.key])
                if style not in used:
                    used.append(style)
                res, power = curves[style.key]
                ticks.update(int(r) for r in res)
                ax.plot(
                    res,
                    power,
                    color=style.color,
                    marker=style.marker,
                    linewidth=style.linewidth,
                    linestyle=style.linestyle,
                    markersize=4,
                    zorder=style.zorder,
                )
            ax.axhline(alpha, color="gray", linestyle="--", linewidth=0.8, zorder=1)
            ax.set_title(rf"$n = {n}$")
            sorted_ticks = sorted(ticks)
            ax.set_xticks(sorted_ticks)
            ax.set_xticklabels(["null" if t == 0 else str(t) for t in sorted_ticks])
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        if missing:
            lines = [f"  N={n:<6} {t}" for n, t in missing]
            msg = (
                f"rbex: {len(missing)} curve(s) have no result file at "
                f"B in {B}, R={R}, alpha={alpha:g} and were left out:\n"
                + "\n".join(lines)
            )
            if strict:
                plt.close(fig)
                raise FileNotFoundError(msg)
            print(msg, file=sys.stderr)

        stamp_tex, stamp_plain = _budget_stamp(tuple(sorted(drawn_B)), R, alpha)
        grid.labels("null, and rBEX resolution", "rejection rate")
        fig.text(
            0.995,
            0.004,
            stamp_tex,
            ha="right",
            va="bottom",
            fontsize=6,
            color="0.45",
        )
        ordered = [s for s in tests if s in used]
        grid.legend(
            _legend_handles(ordered, with_alpha=alpha), ncol=3, columnspacing=1.2
        )
        return _save(
            fig,
            out_path,
            metadata={
                "Subject": f"rBEX quasi-power vs resolution; {stamp_plain}",
                "Keywords": stamp_plain,
            },
        )


# --- Cell-geometry (partition shape) ablation (subsec:shape-ablation) --------


def build_shape_ablation(
    out_path: Path,
    *,
    generators=None,
    geometries=None,
    results_dir: Path = data.SHAPE_RESULTS_DIR,
    alpha: float = config.SHAPE_ALPHA,
    B: int | None = None,
    R: int | None = None,
    strict: bool = False,
) -> Path:
    """Power vs n, one panel per generator, one curve per cell geometry.

    Every curve is the multi-resolution *ladder* of one partition family
    (``indep_tests/shape_ablation``, test ``pug-partition``, mode ``ladder``),
    read at one exact budget ``(B, R, alpha)`` (default
    :data:`config.SHAPE_BUDGET`, the power figure's) and stamped on the figure
    and in the PDF metadata; a geometry with no file at the budget is left out
    and reported on stderr (``strict=True`` raises). Same panel geometry as
    :func:`build_power_curves` so the two figures read together.
    """
    generators = generators or config.SHAPE_PANEL
    geometries = geometries or config.SHAPE_GEOMETRIES
    B = config.SHAPE_BUDGET["B"] if B is None else B
    R = config.SHAPE_BUDGET["R"] if R is None else R
    n_cols = 3
    n_rows = -(-len(generators) // n_cols)
    used: list = []
    missing: list[tuple[str, str]] = []

    with paper_style():
        grid = PanelGrid(n_rows, n_cols)
        fig, axes = grid.fig, grid.axes
        ax: Axes
        for ax, generator in zip(axes.flat, generators):
            curves = data.load_power_results_at_budget(
                generator, B=B, R=R, alpha=alpha, results_dir=results_dir
            )
            for style in geometries:
                if style.key not in curves:
                    missing.append((generator, style.key))
                    continue
                if style not in used:
                    used.append(style)
                ns, rates = curves[style.key]
                ax.plot(
                    ns,
                    rates,
                    color=style.color,
                    marker=style.marker,
                    linewidth=style.linewidth,
                    markersize=4,
                    zorder=style.zorder,
                )
            ax.axhline(alpha, color="gray", linestyle="--", linewidth=0.8, zorder=1)
            ax.set_title(config.GENERATOR_DISPLAY_LABELS.get(generator, generator))
            ax.set_xscale("log")
            ax.set_xticks([20, 50, 100, 200, 500])
            ax.set_xticklabels(["20", "50", "100", "200", "500"])
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.minorticks_off()
        for ax in axes.flat[len(generators) :]:
            ax.set_visible(False)

        if missing:
            lines = [f"  {g:<24} {t}" for g, t in missing]
            msg = (
                f"shape-ablation: {len(missing)} curve(s) have no result file at "
                f"B={B}, R={R}, alpha={alpha:g} and were left out:\n" + "\n".join(lines)
            )
            if strict:
                plt.close(fig)
                raise FileNotFoundError(msg)
            print(msg, file=sys.stderr)

        stamp_tex, stamp_plain = _budget_stamp(B, R, alpha)
        grid.labels("sample size $n$", "rejection rate")
        fig.text(
            0.995, 0.004, stamp_tex, ha="right", va="bottom", fontsize=6, color="0.45"
        )
        ordered = [s for s in geometries if s in used]
        grid.legend(
            _legend_handles(ordered, with_alpha=alpha), ncol=3, columnspacing=1.2
        )
        return _save(
            fig,
            out_path,
            metadata={
                "Subject": f"Cell-geometry ablation, power vs n; {stamp_plain}",
                "Keywords": stamp_plain,
            },
        )


def build_shape_profile(
    out_path: Path,
    *,
    n: int = 500,
    generators=None,
    geometries=None,
    results_dir: Path = data.SHAPE_PROFILE_DIR,
    alpha: float = config.SHAPE_ALPHA,
    B: int | None = None,
    R: int | None = None,
    strict: bool = False,
) -> Path:
    """Power of a single partition vs its cell count, one curve per geometry.

    The ladder-free shape ablation: at a fixed sample size ``n``, each curve is
    one partition family evaluated at its own achievable cell counts up to
    ``k <= floor(sqrt n)`` (``K <= n``); the light vertical guides mark the
    counts at which square, golden and power are exactly matched. Same panel
    grammar as :func:`build_ladder_ablation`, which is the next section's
    figure and whose square profile is this figure's square curve. Read at one
    exact budget ``(B, R, alpha)`` (default :data:`config.SHAPE_PROFILE_BUDGET`);
    a family with no file at the budget is left out and reported on stderr
    (``strict=True`` raises).
    """
    generators = generators or config.SHAPE_PANEL
    geometries = geometries or config.SHAPE_PROFILE_GEOMETRIES
    B = config.SHAPE_PROFILE_BUDGET["B"] if B is None else B
    R = config.SHAPE_PROFILE_BUDGET["R"] if R is None else R
    n_cols = 3
    n_rows = -(-len(generators) // n_cols)
    used: list = []
    missing: list[tuple[str, str]] = []
    k_max = int(np.floor(np.sqrt(n)))
    # Powers of four below the cap, then the cap itself so the reader can see
    # the largest cell count tested (196 at n=200, 484 at n=500).
    ticks = [K for K in (4, 16, 64, 256) if K < k_max * k_max] + [k_max * k_max]

    with paper_style():
        grid = PanelGrid(n_rows, n_cols)
        fig, axes = grid.fig, grid.axes
        ax: Axes
        for ax, generator in zip(axes.flat, generators):
            for K in config.SHAPE_PROFILE_MATCHED_KS:
                if K <= n:
                    ax.axvline(K, color="0.85", linewidth=0.6, zorder=0)
            for style in geometries:
                Ks, rates = data.load_shape_profile_at_n(
                    generator, style.family, n=n, B=B, R=R, results_dir=results_dir
                )
                if not Ks:
                    missing.append((generator, style.family))
                    continue
                if style not in used:
                    used.append(style)
                ax.plot(
                    Ks,
                    rates,
                    color=style.color,
                    marker=style.marker,
                    linewidth=style.linewidth,
                    markersize=3.5,
                    zorder=style.zorder,
                )
            ax.axhline(alpha, color="gray", linestyle="--", linewidth=0.8, zorder=1)
            ax.set_title(config.GENERATOR_DISPLAY_LABELS.get(generator, generator))
            ax.set_xscale("log", base=2)
            ax.set_xlim(3, k_max * k_max * 1.15)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(K) for K in ticks])
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.minorticks_off()
        for ax in axes.flat[len(generators) :]:
            ax.set_visible(False)

        if missing:
            lines = [f"  {g:<24} {t}" for g, t in missing]
            msg = (
                f"shape-profile n={n}: {len(missing)} curve(s) have no result file at "
                f"B={B}, R={R} and were left out:\n" + "\n".join(lines)
            )
            if strict:
                plt.close(fig)
                raise FileNotFoundError(msg)
            print(msg, file=sys.stderr)

        stamp_tex, stamp_plain = _budget_stamp(B, R, alpha)
        grid.labels(
            rf"cell count $K$ of the single partition ($n = {n}$)",
            "rejection rate",
        )
        fig.text(
            0.995, 0.004, stamp_tex, ha="right", va="bottom", fontsize=6, color="0.45"
        )
        ordered = [s for s in geometries if s in used]
        grid.legend(
            _legend_handles(ordered, with_alpha=alpha), ncol=3, columnspacing=1.2
        )
        return _save(
            fig,
            out_path,
            metadata={
                "Subject": f"Cell-geometry resolution profile at n={n}; {stamp_plain}",
                "Keywords": f"n={n}, {stamp_plain}",
            },
        )


def build_shape_catalogue(
    out_path: Path,
    *,
    ks=None,
    geometries=None,
    raster: int = 500,
    edge_px: int = 3,
) -> Path:
    """Catalogue of the partition families of the shape ablation.

    One column per family (order of :data:`config.SHAPE_GEOMETRIES`), one row
    per ladder rung in ``ks`` (default :data:`config.SHAPE_CATALOGUE_KS`). Each
    panel is the unit square with the cell boundaries drawn from a label
    raster (``raster`` x ``raster`` pixels, boundaries ``edge_px`` pixels wide),
    so every family — including the argmin-defined power diagram — is rendered
    the same way. Reads no experiment results.
    """
    from indep_tests.shape_ablation.partitions import make_partition

    ks = tuple(ks or config.SHAPE_CATALOGUE_KS)
    geometries = geometries or config.SHAPE_GEOMETRIES
    n_rows, n_cols = len(ks), len(geometries)
    W = config.TEXTWIDTH_IN
    gap = 0.08
    panel = (W - gap * (n_cols - 1)) / n_cols
    title_in = 0.30
    H = n_rows * panel + (n_rows - 1) * (gap + title_in) + title_in
    g = (np.arange(raster) + 0.5) / raster
    U, V = np.meshgrid(g, g)  # V rows (y), U columns (x)

    with paper_style():
        fig = plt.figure(figsize=(W, H))
        gs = GridSpec(
            n_rows,
            n_cols,
            figure=fig,
            left=0,
            right=1,
            bottom=0,
            top=1 - title_in / H,
            wspace=gap / panel,
            hspace=(gap + title_in) / panel,
        )
        for r, k in enumerate(ks):
            for c, style in enumerate(geometries):
                ax = fig.add_subplot(gs[r, c])
                P = make_partition(style.family, k)
                L = P.codes(U.ravel(), V.ravel()).reshape(U.shape)
                edge = np.zeros_like(L, dtype=bool)
                edge[1:, :] |= L[1:, :] != L[:-1, :]
                edge[:, 1:] |= L[:, 1:] != L[:, :-1]
                # thicken to `edge_px` pixels so the lines survive resampling
                thick = edge.copy()
                for d in range(1, edge_px):
                    thick[d:, :] |= edge[:-d, :]
                    thick[:, d:] |= edge[:, :-d]
                edge = thick
                rgba = np.zeros(L.shape + (4,))
                rgba[edge, 3] = 1.0
                ax.imshow(
                    rgba,
                    origin="lower",
                    extent=(0, 1, 0, 1),
                    interpolation="nearest",
                    zorder=2,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.8)
                title = f"{style.label}" if r == 0 else ""
                ax.set_title(
                    (title + "\n" if title else "") + f"{P.K} cells",
                    fontsize=8,
                    pad=2,
                )
        return _save(
            fig,
            out_path,
            metadata={"Subject": "Partition families of the cell-geometry ablation"},
        )


# --- Figure: ladder mechanism (Sec. 2, proposed test) ------------------------


def build_ladder_mechanism(
    out_path: Path,
    *,
    n: int = 500,
    seed: int = 1234,
    generators: tuple[str, str] = ("dep-normal-0.3", "dep-sine-4pi"),
    shade: str = "signed",
    annotate: bool = False,
) -> Path:
    """2x3 cell-departure maps: what the single-resolution statistic sees.

    One sample from a weak bivariate normal and from the sine-4pi pattern,
    rank-transformed to pseudo-observations (ranks / n, overlaid as points)
    and binned on the k x k grids of three ladder rungs. Every cell is shaded
    by its standardized count departure (N_ij - n/k^2)/sqrt(n/k^2) under a
    diverging blue-white-vermillion map; the single-resolution statistic is
    the sum of these squared departures. Illustrates Sec. 3 (Eq.
    statistic-single-k); reads no experiment results.

    ``shade="signed"`` colours cells by the signed standardized departure
    z_ij (diverging map); ``shade="squared"`` by the per-cell summand
    z_ij^2 of the statistic (sequential map, sign lost). ``annotate=True``
    prints the statistic T_k^n of the drawn sample in each panel.
    """
    from matplotlib.colors import LinearSegmentedColormap

    generators = list(generators)
    if shade not in {"signed", "squared"}:
        raise ValueError(f"shade must be 'signed' or 'squared', got {shade!r}")
    resolutions = [3, 5, int(np.floor(np.sqrt(n)))]

    # Diverging map between the house Okabe-Ito poles (deficit blue, excess
    # vermillion) through white, matching the paper's palette and CVD-safe.
    if shade == "signed":
        cmap = LinearSegmentedColormap.from_list(
            "departure", [config.BLUE, "#FFFFFF", config.VERMILLION]
        )
        vmin, vmax = -3.0, 3.0  # standardized departures clipped at +-3 sigma
    else:
        cmap = LinearSegmentedColormap.from_list(
            "summand", ["#FFFFFF", config.VERMILLION]
        )
        vmin, vmax = 0.0, 9.0  # z^2 clipped at 3^2

    with paper_style():
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(config.TEXTWIDTH_IN, config.TEXTWIDTH_IN * 0.63),
            layout="constrained",
        )
        mesh = None
        for row, gen in enumerate(generators):
            x, y = data.sample_generator(gen, n=n, seed=seed)
            # Pseudo-observations (R_i/n, S_i/n): both the cell counts and the
            # overlaid points live in rank space, never the raw (X, Y) scale.
            u, v = data.pseudo_observations(x, y)
            for col, k in enumerate(resolutions):
                ax = axes[row, col]
                bin_x = np.clip(np.ceil(u * k).astype(int) - 1, 0, k - 1)
                bin_y = np.clip(np.ceil(v * k).astype(int) - 1, 0, k - 1)
                counts = np.zeros((k, k))
                np.add.at(counts, (bin_x, bin_y), 1)
                expected = n / k**2
                z = (counts - expected) / np.sqrt(expected)
                shaded = z if shade == "signed" else z**2
                edges = np.linspace(0, 1, k + 1)
                mesh = ax.pcolormesh(
                    edges,
                    edges,
                    shaded.T,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    rasterized=False,
                    zorder=1,
                )
                for e in edges[1:-1]:
                    ax.axvline(e, color="0.75", linewidth=0.3, zorder=2)
                    ax.axhline(e, color="0.75", linewidth=0.3, zorder=2)
                ax.scatter(
                    u, v, s=2.5, color="black", alpha=0.45, linewidths=0, zorder=3
                )
                if annotate:
                    ax.text(
                        0.03,
                        0.97,
                        rf"$T_k^n = {np.sum(z**2):.0f}$",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        bbox=dict(
                            facecolor="white", alpha=0.85, edgecolor="none", pad=1.5
                        ),
                        zorder=4,
                    )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect("equal", "box")
                ax.grid(False)
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([0, 0.5, 1])
                if row == 0:
                    title = (
                        rf"$k = {k}$"
                        if k != resolutions[-1]
                        else rf"$k = \lfloor\sqrt{{n}}\rfloor = {k}$"
                    )
                    ax.set_title(title)
                if row < len(generators) - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(r"$\tilde U$")
                if col == 0:
                    # Axis variable next to the ticks; generator name further left.
                    ax.set_ylabel(r"$\tilde V$")
                    ax.annotate(
                        config.GENERATOR_DISPLAY_LABELS[gen],
                        xy=(-0.36, 0.5),
                        xycoords="axes fraction",
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=8,
                        annotation_clip=False,
                    )
                else:
                    ax.set_yticklabels([])
        cbar = fig.colorbar(
            mesh,
            ax=axes,
            shrink=0.85,
            pad=0.02,
            extend="both" if shade == "signed" else "max",
            aspect=30,
        )
        cbar.set_label(
            r"$(N_{ij} - n/k^2)\,/\,\sqrt{n/k^2}$"
            if shade == "signed"
            else r"$(N_{ij} - n/k^2)^2\,/\,(n/k^2)$",
            fontsize=8,
        )
        cbar.ax.tick_params(labelsize=7)
        return _save(fig, out_path)


# --- Computational cost vs n (subsec:time) -----------------------------------


def _n_tick_label(n: int) -> str:
    """``4000`` stays ``4000``; exact powers of ten from 10^4 up become ``$10^k$``."""
    k = round(np.log10(n)) if n > 0 else 0
    if n >= 10_000 and 10**k == n:
        return rf"$10^{{{k}}}$"
    return str(n)


def build_timing(
    out_path: Path,
    *,
    B: int | None = None,
    results_dir: Path | None = None,
    alpha: float | None = None,  # accepted for CLI uniformity; unused
) -> Path:
    """Cold-call wall-clock time versus ``n`` on log-log axes, one panel.

    One full ``test()`` call at the pinned ``B`` (statistic + calibration) with
    the distribution-free null caches disabled, so every test pays its own
    null.  The statistic-only and reused-null series in the JSON are not drawn
    (a reused-null call is the statistic plus a lookup, so they coincide; the
    paper quotes them in the text).  Reference slopes ``n``, ``n^2`` and
    ``n^3`` are thin dotted guides.  Data come from
    ``indep_tests/timing/results/<test>_B{B}.json`` (one file per test).
    """
    del alpha
    B = B if B is not None else config.TIMING_B
    meta, series = data.load_timing(B, results_dir)
    styles = [
        s
        for s in config.TIMING_TESTS
        if s.key in series and np.isfinite(series[s.key]["call"]).any()
    ]

    with paper_style():
        grid = PanelGrid(1, 1)
        fig, axes = grid.fig, grid.axes
        ax: Axes = axes.flat[0]
        all_n = np.unique(np.concatenate([s["n"] for s in series.values()]))
        ys = []
        for s in styles:
            ser = series[s.key]
            ok = np.isfinite(ser["call"])
            ys.append(ser["call"][ok])
            ax.plot(
                ser["n"][ok],
                ser["call"][ok],
                color=s.color,
                marker=s.marker,
                markersize=4,
                linewidth=s.linewidth,
                linestyle=s.linestyle,
                zorder=s.zorder,
            )
        # reference slopes, anchored at the largest n and drawn over the top two
        # decades only (an n^3 guide spanning 50..10^7 would drag the axis to
        # 1e-13 and flatten every series)
        n1 = all_n[-1]
        n0 = max(all_n[0], n1 / 100)
        y_all = np.concatenate(ys)
        y_anchor = 10 ** np.floor(np.log10(y_all.max()))
        ax.set_ylim(y_all.min() / 3, y_all.max() * 3)
        for p, lab in ((1, r"$n$"), (2, r"$n^2$"), (3, r"$n^3$")):
            ax.plot(
                [n0, n1],
                [y_anchor * (n0 / n1) ** p, y_anchor],
                color="0.6",
                linestyle=":",
                linewidth=0.7,
                zorder=1,
            )
            n_mid = np.sqrt(n0 * n1)
            ax.annotate(
                lab,
                (n_mid, y_anchor * (n_mid / n1) ** p),
                fontsize=6,
                color="0.45",
                ha="center",
                va="bottom",
                xytext=(0, 1),
                textcoords="offset points",
                zorder=1,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(all_n)
        ax.set_xticklabels([_n_tick_label(int(n)) for n in all_n], rotation=45)
        ax.minorticks_off()
        grid.labels(r"sample size $n$", f"seconds per call, $B = {B}$ (log scale)")
        grid.legend(_legend_handles(styles), ncol=4, columnspacing=1.0)
        threads = meta["machine"].get("threads", {})
        thread_note = (
            "single thread"
            if threads and all(v == "1" for v in threads.values())
            else "threads unpinned"
        )
        stamp = (
            f"{meta['machine'].get('cpu', '')}, {thread_note}, medians of up to "
            f"{meta['call_repeats']} calls"
        )
        fig.text(0.995, 0.004, stamp, ha="right", va="bottom", fontsize=6, color="0.45")
        return _save(
            fig,
            out_path,
            metadata={"Subject": f"Computational cost vs n; B={B}; {stamp}"},
        )
