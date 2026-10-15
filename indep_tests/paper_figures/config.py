"""Canonical presentation metadata for the paper figures.

Single source of truth for everything that is a *manuscript* concern rather than
an experiment concern: the generator display order/labels (the paper's narrative
arc), the curated test sets + per-test styling for each figure, the
representative generators/patterns shown in the multi-panel figures, and the
physical figure dimensions matched to the ``sn-jnl`` text block.

Keeping this here (not in ``indep_tests/commands/plotting.py``) is deliberate:
the experiment CLI is for fast, open-ended iteration; the paper figures are a
fixed, pinned set rendered with publication styling.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# --- Physical dimensions (inches), matched to the sn-jnl single-column text block.
# \textwidth in the default sn-jnl layout is ~16 cm.
TEXTWIDTH_IN = 6.3
COLWIDTH_IN = 3.1

# --- Okabe-Ito colourblind-safe palette. -------------------------------------
ORANGE = "#E69F00"  # dCov
SKY = "#56B4E9"  # cmiel/ledwina
GREEN = "#009E73"  # qad
YELLOW = "#F0E442"  # heller
BLUE = "#0072B2"  # zhang-bet
VERMILLION = "#D55E00"  # ours
PURPLE = "#CC79A7"  # ma-mao-fes
BLACK = "#000000"  # salahub

# --- Generator gallery: order matches the paper's narrative arc --------------
# (independent baselines -> monotone smooth -> high-frequency -> geometric /
# categorical). This is the same arc Fig. \ref{fig:generators} narrates.
GENERATOR_DISPLAY_ORDER = [
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

GENERATOR_DISPLAY_LABELS = {
    "indep-uniform": "Uniform (independent)",
    "indep-normal": "Normal (independent)",
    "indep-expon": "Exponential (independent)",
    "dep-normal-0.3": r"Bivariate normal, $\rho=0.3$",
    "dep-yanagimoto": "Yanagimoto parallelogram",
    "dep-sine-4pi": r"$\sin(4\pi X)$",
    "dep-sine-16pi": r"$\sin(16\pi X)$",
    "dep-modular-5": r"Modular ($5x \,\mathrm{mod}\, 1$)",
    "dep-modular-10": r"Modular ($10x \,\mathrm{mod}\, 1$)",
    "dep-spiral-2-turns": "Spiral (2 turns)",
    "dep-spiral-3-turns": "Spiral (3 turns)",
    "dep-checkerboard-k5-d1": r"Checkerboard ($5\times5$)",
}


@dataclass(frozen=True)
class TestStyle:
    """How one test is drawn in a paper figure.

    ``key`` is the exact ``test`` field stored in the result JSON; ``label`` is
    the (shorter, paper-facing) name shown in legends.
    """

    # Must match a test's `.name` exactly (folds in config: e.g. dCov_auto-True,
    # PUG-Grid-DENSITY-MULTI). It's the `test` field in results/*.json — list
    # valid keys with: sorted({json.load(open(f))["test"] for f in results/*.json}).
    # A key with no matching JSON is silently dropped from the figure (no error).
    key: str
    label: str
    family: str
    color: str
    marker: str
    linewidth: float = 1.3
    zorder: int = 2
    linestyle: str = "-"


# --- Empirical Type I error: independent generators (subsec:type1) -----------
# Size, not power — the rejection rate should hug alpha. Curate which tests
# appear here exactly as for the power/noise figures.
TYPE_I_TESTS = [
    TestStyle(
        "PUG-Grid-DENSITY-MULTI",
        "MRPT",
        "density",
        VERMILLION,
        "o",
        linewidth=2.5,
        zorder=5,
    ),
    # Same statistic and ladder, closed-form chi^2_{(k-1)^2} calibration per rung
    # (``PUGGridPearsonChiSquaredTest``; added 2026-10-12). Same seeds as the
    # permutation curve; its files carry B=2000 nominally, like the other
    # closed-form tests'. Same colour and dashes as in the timing figure.
    TestStyle(
        "PUG-Grid-ChiSq-DENSITY-MULTI",
        r"MRPT ($\chi^2$ calibration)",
        "density",
        VERMILLION,
        "o",
        linewidth=1.3,
        zorder=6,  # above the solid MRPT line, so the dashes stay visible where both coincide
        linestyle="--",
    ),
    TestStyle("ZhangBETs-dmax4", "Zhang-BET", "binning", BLUE, "s"),
    TestStyle("QAD", "GJT-QAD", "published", GREEN, "^"),
    TestStyle("Ma-Mao-FES", "Ma-Mao-FES", "binning", PURPLE, "*"),
    TestStyle("dCov_auto-True", "SRB-dCov", "kernel", ORANGE, "P"),
    TestStyle(
        "Salahub-Binning-depth8-p_simple",
        "Salahub-Oldford-p_simple",
        "rank",
        BLACK,
        "X",
    ),
    TestStyle("Heller-ADP-LR-MinP", "Heller-ADP", "density", YELLOW, "D"),
    TestStyle("Cmiel-Ledwina-Tn", "Cmiel-Ledwina-Tn", "cdf", SKY, "v"),
]

# The three independent generators (Uniform, Normal, Exponential margins).
TYPE_I_PANEL = ["indep-uniform", "indep-normal", "indep-expon"]


# --- Empirical power vs sample size: dependent generators (subsec:power-n) ----
# Curated representatives, one or two per test family, so the multi-panel figure
# stays legible. PUG-Density-Multi (the proposed test) is drawn heaviest.
# `dCov_auto-True` is the chi^2-calibrated dCov, the recommended default.
POWER_CURVE_TESTS = [
    TestStyle(
        "PUG-Grid-DENSITY-MULTI",
        "MRPT",
        "density",
        VERMILLION,
        "o",
        linewidth=2.5,
        zorder=5,
    ),
    # Same statistic and ladder, closed-form chi^2_{(k-1)^2} calibration per rung
    # (``PUGGridPearsonChiSquaredTest``; added 2026-10-12). Same seeds as the
    # permutation curve; its files carry B=2000 nominally, like the other
    # closed-form tests'. Same colour and dashes as in the timing figure.
    TestStyle(
        "PUG-Grid-ChiSq-DENSITY-MULTI",
        r"MRPT ($\chi^2$ calibration)",
        "density",
        VERMILLION,
        "o",
        linewidth=1.3,
        zorder=6,  # above the solid MRPT line, so the dashes stay visible where both coincide
        linestyle="--",
    ),
    TestStyle("ZhangBETs-dmax4", "Zhang-BET", "binning", BLUE, "s"),
    TestStyle("QAD", "GJT-QAD", "published", GREEN, "^"),
    TestStyle("Ma-Mao-FES", "Ma-Mao-FES", "binning", PURPLE, "*"),
    TestStyle("dCov_auto-True", "SRB-dCov", "kernel", ORANGE, "P"),
    # Key is the name the harness records *today* (`SalahubBinningTest.name`,
    # CLI default p_method="simple"). Older B500/R100 files carry the legacy
    # name `Salahub-Binning-max_depth-8` and are no longer matched here.
    TestStyle(
        "Salahub-Binning-depth8-p_simple",
        "Salahub-Oldford-p_simple",
        "rank",
        BLACK,
        "X",
    ),
    TestStyle("Heller-ADP-LR-MinP", "Heller-ADP", "density", YELLOW, "D"),
    TestStyle("Cmiel-Ledwina-Tn", "Cmiel-Ledwina-Tn", "cdf", SKY, "v"),
]

# Monte-Carlo budget the power figure is *pinned* to. Only result files whose
# recorded (B, R, alpha) match exactly are drawn, and the values are stamped on
# the figure + PDF metadata, so a curve can never come from a lower-budget run.
# B = permutations/bootstrap replicates per test call (ignored by the parametric
# tests, whose files still carry the value in the name), R = replications per
# (generator, n) cell.
POWER_CURVE_BUDGET = {"B": 2000, "R": 1000}
POWER_CURVE_ALPHA = 0.05


def power_curves_filename(B: int, R: int, alpha: float = POWER_CURVE_ALPHA) -> str:
    """Default output name of the power figure, mirroring the results-file
    convention ``..._B{B}_R{R}.json`` so the budget is visible in the filename
    (and a re-run at a new budget never overwrites the previous figure).
    ``main.tex`` must include this exact name."""
    return f"power_curves_B{B}_R{R}_alpha{alpha:g}.pdf"


def type_one_filename(B: int, R: int, alpha: float = POWER_CURVE_ALPHA) -> str:
    """Default output name of the Type I figure; same convention (and the same
    budget) as :func:`power_curves_filename`. ``main.tex`` must include it."""
    return f"type_one_error_B{B}_R{R}_alpha{alpha:g}.pdf"


# Dependent generators only — Type I (independent) is its own figure above.
POWER_CURVE_PANEL = [
    "dep-normal-0.3",  # monotone smooth (weak)
    "dep-yanagimoto",  # non-linear smooth
    "dep-sine-4pi",  # high-frequency
    "dep-sine-16pi",  # high-frequency
    "dep-modular-5",  # high-frequency
    "dep-modular-10",  # high-frequency
    "dep-spiral-2-turns",  # geometric
    "dep-spiral-3-turns",  # geometric
    "dep-checkerboard-k5-d1",  # categorical / structured
]

# --- Empirical power vs noise: Zhang patterns (subsec:power-noise) -----------
# Curated to match POWER_CURVE_TESTS (same tests, same colours/markers/labels)
# so the two figures read as one story. Reconciliations vs the power figure:
#   * Salahub is the closed-form `p_simple` variant, as in the type-one/power/rBEX
#     figures (its noise null at alpha=0.01 is 0.006, so calibration is no reason
#     for the permutation variant, whose sweeps remain on disk).
NOISE_TESTS = [
    TestStyle(
        "PUG-Grid-DENSITY-MULTI",
        "MRPT",
        "density",
        VERMILLION,
        "o",
        linewidth=2.5,
        zorder=5,
    ),
    # Same statistic and ladder, closed-form chi^2_{(k-1)^2} calibration per rung
    # (``PUGGridPearsonChiSquaredTest``; added 2026-10-12). Same samples as the
    # permutation curve (same seeds); its files carry B=1000 nominally, like
    # Zhang-BET's. Same colour and dashes as in the timing figure.
    TestStyle(
        "PUG-Grid-ChiSq-DENSITY-MULTI",
        r"MRPT ($\chi^2$ calibration)",
        "density",
        VERMILLION,
        "o",
        linewidth=1.3,
        zorder=6,  # above the solid MRPT line, so the dashes stay visible where both coincide
        linestyle="--",
    ),
    TestStyle("ZhangBETs-dmax4", "Zhang-BET", "binning", BLUE, "s"),
    TestStyle("QAD", "GJT-QAD", "published", GREEN, "^"),
    TestStyle("Ma-Mao-FES", "Ma-Mao-FES", "binning", PURPLE, "*"),
    TestStyle("dCov_auto-True", "SRB-dCov", "kernel", ORANGE, "P"),
    TestStyle(
        "Salahub-Binning-depth8-p_simple",
        "Salahub-Oldford-p_simple",
        "rank",
        BLACK,
        "X",
    ),
    TestStyle("Heller-ADP-LR-MinP", "Heller-ADP", "density", YELLOW, "D"),
    TestStyle("Cmiel-Ledwina-Tn", "Cmiel-Ledwina-Tn", "cdf", SKY, "v"),
]

NOISE_PANEL = ["linear", "parabolic", "circular", "sine", "checkerboard", "local"]

# Noise-pattern gallery (Sec. benchmark, noise): one column per pattern, one row
# per displayed noise level, n = 500 as in the benchmark itself.
NOISE_GALLERY_LEVELS = (1, 3, 5)
NOISE_GALLERY_N = 500
NOISE_PATTERN_LABELS = {
    "linear": "Linear",
    "parabolic": "Parabolic",
    "circular": "Circular",
    "sine": "Sine",
    "checkerboard": "Checkerboard",
    "local": "Local",
}

# Zhang noise level l = sigma * 40 (so sigma in [0.025, 0.25] maps to l in 1..10).
NOISE_LEVEL_FACTOR = 40.0

# Monte-Carlo budget the noise figure is pinned to. The Salahub-noise benchmark
# was run at R=200 with a per-test permutation count: B=1000 for MRP, QAD,
# Heller-ADP, Cmiel-Ledwina, B=500 for Ma-Mao-FES, dCov and
# Salahub; Zhang-BET is closed-form (its files carry B=1000 nominally). So
# ``B`` is the *set* of admissible values. Files at any other (R, B,
# alpha) are ignored; the stamp lists the B values actually drawn. Unifying B
# means re-running the B=500 tests (``python -m indep_tests.salahub_noise``).
NOISE_ALPHA = 0.01
NOISE_BUDGET = {"B": (500, 1000), "R": 200}


def _fmt_B(B) -> str:
    return "-".join(str(b) for b in sorted(set(B))) if isinstance(B, tuple) else str(B)


def noise_filename(B, R: int, alpha: float = NOISE_ALPHA) -> str:
    """Default output name of the noise figure; same convention as
    :func:`power_curves_filename`, with a ``B500-1000`` range when the pinned
    budget admits several per-test B values. ``main.tex`` must include it."""
    return f"noise_power_B{_fmt_B(B)}_R{R}_alpha{alpha:g}.pdf"


# --- Resolution-ladder ablation: power vs single grid k (subsec:ladder) ------
# One panel per dependent generator of the paper's nine-generator panel (same
# set and order as POWER_CURVE_PANEL, so the two figures read as one story).
# Each panel shows the empirical power of the single fixed-resolution density
# statistic across the five evaluated grids, with the multi-resolution ladder
# drawn as a horizontal reference line.
LADDER_PANEL = list(POWER_CURVE_PANEL)

# The fixed single-grid resolutions of the ablation; floor(sqrt(n)) is appended
# at load time from the norm=density (default-k) results.
LADDER_FIXED_KS = [3, 5, 8, 12]

# Monte-Carlo budget the ladder-ablation figure reads. All three sources (the
# fixed-k sweep in pug_grid_density_multi/ladder_ablation/, and the sqrt(n) and
# density-multi rows in results/) must exist at exactly this (B, R). Override
# per build with `--B/--R`; regenerate the sweep with
# `uv run python -m indep_tests.pug_grid_density_multi run --B ... --R ...`.


# Multi-panel line-chart figures (type-one, power-curves, noise, ladder) share
# one geometry, every spacing fixed in inches and laid out by an explicit
# GridSpec (``figures.PanelGrid``): the column gap, the row gap, the margins
# and the panel box aspect are identical in every figure by construction, and
# the figure height follows from the panel width. (Under constrained layout the
# gap was whatever the tick labels and the row height left over, and it varied
# from 0.13 to 0.91 in across figures.)
PANEL_BOX_ASPECT = 0.8  # panel height / panel width
PANEL_GAP_IN = 0.25  # horizontal gap between panel columns
PANEL_VGAP_IN = 0.38  # vertical gap between rows (x tick labels + next title)
PANEL_MARGIN_LEFT_IN = 0.52  # supylabel + y tick labels
PANEL_MARGIN_RIGHT_IN = 0.12  # overhang of the last x tick label
PANEL_MARGIN_BOTTOM_IN = 0.45  # x tick labels + supxlabel + budget stamp
PANEL_TITLE_IN = 0.24  # panel title above the top row
PANEL_LEGEND_ROW_IN = 0.19  # one row of the figure legend, above the titles

LADDER_BUDGET = {"B": 2000, "R": 1000}
LADDER_ALPHA = 0.05

# Smaller sample sizes drawn alongside the solid n=500 series in every panel of
# the ladder-ablation figure, in order of decreasing n: the first is dashed with
# open circles, the second dotted with open squares. Each series ends at its
# own floor(sqrt(n)) (14, 10); a fixed rung at or above that cap (k=12 at
# n=100) is outside the ladder and not drawn. () -> single-n figure.
LADDER_SECONDARY_NS: tuple[int, ...] = (200, 100)


def ladder_ablation_filename(B: int, R: int, alpha: float = LADDER_ALPHA) -> str:
    """Default output name of the ladder-ablation figure; same convention as
    :func:`power_curves_filename` so the budget is visible in the filename and
    a re-run at a new budget never overwrites the previous figure.
    ``main.tex`` must include this exact name."""
    return f"ladder_ablation_B{B}_R{R}_alpha{alpha:g}.pdf"


LADDER_SINGLE_COLOR = BLUE
LADDER_MULTI_COLOR = VERMILLION


# --- Ladder-composition sensitivity: power vs n for four ladders (subsec:ladder)
# One panel per LADDER_PANEL generator, x = n, one curve per ladder combined by
# Bonferroni on the same samples and permutations. ``spec`` is the
# ``ladder=`` token of the result filename in pug_grid_density_multi/
# ladder_ablation/ (None = the production ladder, read from results/ as
# ``norm=density-multi``). Budget: LADDER_BUDGET.
LADDER_CHOICE_LADDERS = [
    TestStyle(
        "production",
        r"$\{3,5,8,12,\lfloor\sqrt{n}\rfloor\}$ (MRPT)",
        "ladder",
        VERMILLION,
        "o",
        linewidth=2.0,
        zorder=4,
    ),
    TestStyle(
        "3-sqrt",
        r"$\{3,\lfloor\sqrt{n}\rfloor\}$",
        "ladder",
        BLUE,
        "s",
        linestyle="--",
        zorder=3,
    ),
    TestStyle(
        "3-5-8-12",
        r"$\{3,5,8,12\}$",
        "ladder",
        BLACK,
        "^",
        linestyle="-.",
        zorder=3,
    ),
    TestStyle(
        "all",
        r"$\{2,\dots,\lfloor\sqrt{n}\rfloor\}$",
        "ladder",
        GREEN,
        "D",
        linestyle=":",
        zorder=3,
    ),
]


def ladder_choice_filename(B: int, R: int, alpha: float = LADDER_ALPHA) -> str:
    """Default output name of the ladder-composition figure (budget in the
    name, as for :func:`ladder_ablation_filename`)."""
    return f"ladder_choice_B{B}_R{R}_alpha{alpha:g}.pdf"


# --- rBEX quasi-power vs resolution: Zhang's bisection expanding cross --------
# One panel per sample size. The proposed density-multi is drawn heaviest; the
# hero/Salahub/QAD colours match the other figures, Heller reuses its type-I
# purple, and BET (rBEX's home-turf test) takes the free blue slot.
RBEX_TESTS = [
    TestStyle(
        "PUG-Grid-DENSITY-MULTI",
        "MRPT",
        "density",
        VERMILLION,
        "o",
        linewidth=2.5,
        zorder=5,
    ),
    # Same statistic and ladder, closed-form chi^2_{(k-1)^2} calibration per rung
    # (``PUGGridPearsonChiSquaredTest``; added 2026-10-13). Same seeds as the
    # permutation curve; its files carry B=500 nominally, like the other
    # closed-form tests'. Same colour and dashes as in the timing figure.
    TestStyle(
        "PUG-Grid-ChiSq-DENSITY-MULTI",
        r"MRPT ($\chi^2$ calibration)",
        "density",
        VERMILLION,
        "o",
        linewidth=1.3,
        zorder=6,  # above the solid MRPT line, so the dashes stay visible where both coincide
        linestyle="--",
    ),
    TestStyle("ZhangBETs-dmax4", "Zhang-BET", "binning", BLUE, "s"),
    TestStyle("QAD", "GJT-QAD", "published", GREEN, "^"),
    TestStyle("Ma-Mao-FES", "Ma-Mao-FES", "binning", PURPLE, "*"),
    TestStyle("dCov_auto-True", "SRB-dCov", "kernel", ORANGE, "P"),
    TestStyle(
        "Salahub-Binning-depth8-p_simple",
        "Salahub-Oldford-p_simple",
        "rank",
        BLACK,
        "X",
    ),
    TestStyle("Heller-ADP-LR-MinP", "Heller-ADP", "density", YELLOW, "D"),
    TestStyle("Cmiel-Ledwina-Tn", "Cmiel-Ledwina-Tn", "cdf", SKY, "v"),
]

# One panel per sample size (small -> large, left to right).
RBEX_PANEL = [500, 2000]

# rBEX gallery: one row per sample size, one column per BEX resolution d.
RBEX_GALLERY_RESOLUTIONS = (1, 2, 3, 4, 5)

# Monte-Carlo budget the rBEX figure is pinned to (Salahub & Oldford's protocol:
# alpha=0.01, R=200). Per-test permutation count: B=2000 for Heller-ADP (cached
# null), B=500 for the rest; Zhang-BET and Salahub p_simple are closed-form
# (files carry B=500 nominally). ``B`` is the set of admissible values, as for the noise
# figure; the stamp lists the B values actually drawn.
RBEX_ALPHA = 0.01
RBEX_BUDGET = {"B": (500, 2000), "R": 200}


def rbex_filename(B, R: int, alpha: float = RBEX_ALPHA) -> str:
    """Default output name of the rBEX figure; same convention as
    :func:`noise_filename`. ``main.tex`` must include it."""
    return f"rbex_quasi_power_B{_fmt_B(B)}_R{R}_alpha{alpha:g}.pdf"


# --- Cell-geometry (partition shape) ablation (subsec:shape-ablation) --------
# Same nine-generator panel and Monte-Carlo budget as the power figure. Each
# curve is the *ladder* (MRPT rungs mapped to the family) of one cell geometry
# from ``indep_tests/shape_ablation`` (test ``pug-partition``); the square arm
# is MRPT itself (exact tie-breaking). ``family`` holds the geometry key used by
# the partition catalogue figure. Results: ``shape_ablation/results_paper/``.
SHAPE_PANEL = list(POWER_CURVE_PANEL)
SHAPE_BUDGET = dict(POWER_CURVE_BUDGET)
SHAPE_ALPHA = POWER_CURVE_ALPHA
SHAPE_GEOMETRIES = [
    TestStyle(
        "PUG-Partition-square-MULTI",
        "square (MRPT)",
        "square",
        VERMILLION,
        "o",
        linewidth=2.5,
        zorder=5,
    ),
    TestStyle("PUG-Partition-rect-MULTI", "rectangles 2:1", "rect", BLUE, "s"),
    TestStyle(
        "PUG-Partition-rotated-MULTI",
        r"square rotated $45^\circ$",
        "rotated",
        GREEN,
        "^",
    ),
    TestStyle("PUG-Partition-golden-MULTI", "golden bisection", "golden", PURPLE, "v"),
    TestStyle("PUG-Partition-power-MULTI", "power diagram", "power", BLACK, "D"),
]
# Rungs drawn in the partition catalogue (one row per rung, one column per family).
SHAPE_CATALOGUE_KS = (5, 8)


def shape_ablation_filename(B: int, R: int, alpha: float = SHAPE_ALPHA) -> str:
    """Default output name of the shape-ablation figure; same convention as
    :func:`power_curves_filename`. ``main.tex`` must include this exact name."""
    return f"shape_ablation_B{B}_R{R}_alpha{alpha:g}.pdf"


# --- Per-family resolution profile (subsec:shape-ablation, single partition) --
# One curve per cell geometry: power of a *single* partition of that family as a
# function of its cell count K at a fixed sample size n (the ladder plays no
# part). Sweep: ``python -m indep_tests.shape_ablation.kprofile`` ->
# ``shape_ablation/results_kprofile/``; cap k <= floor(sqrt n), i.e. K <= n.
# Same budget as the power figure. The square label drops "(MRPT)" here because
# a single square grid is not the proposed test; MRPT is its ladder.
SHAPE_PROFILE_NS = (200, 500)
SHAPE_PROFILE_BUDGET = dict(POWER_CURVE_BUDGET)
SHAPE_PROFILE_MATCHED_KS = (16, 64, 256)  # exact for square / golden / power
SHAPE_PROFILE_GEOMETRIES = [
    replace(s, label="square grid") if s.family == "square" else s
    for s in SHAPE_GEOMETRIES
]


def shape_profile_filename(n: int, B: int, R: int, alpha: float = SHAPE_ALPHA) -> str:
    """Default output name of a resolution-profile figure at sample size n."""
    return f"shape_profile_n{n}_B{B}_R{R}_alpha{alpha:g}.pdf"


# --- Computational cost vs n (subsec:computational-time) --------------------
# Single-call wall-clock time of each Section-4 test, measured by
# ``python -m indep_tests.timing`` (results in ``indep_tests/timing/results/``).
# Same colours/markers/labels as the power figures.
TIMING_TESTS = [
    TestStyle("MRP", "MRPT", "density", VERMILLION, "o", linewidth=2.5, zorder=5),
    # Same statistic, closed-form chi^2_{(k-1)^2} calibration (E = 1): the cost
    # of the statistic alone. Timing figure only (2026-09-27); the paper's power
    # results are permutation-calibrated.
    TestStyle(
        "MRP-chi2",
        r"MRPT ($\chi^2$ calibration)",
        "density",
        VERMILLION,
        "o",
        linewidth=1.3,
        zorder=4,
        linestyle="--",
    ),
    TestStyle("Zhang-BET", "Zhang-BET", "binning", BLUE, "s"),
    TestStyle("GJT-QAD", "GJT-QAD", "published", GREEN, "^"),
    TestStyle("Ma-Mao-FES", "Ma-Mao-FES", "binning", PURPLE, "*"),
    TestStyle("SRB-dCov", "SRB-dCov", "kernel", ORANGE, "P"),
    TestStyle(
        "Salahub-Oldford-p_simple", "Salahub-Oldford-p_simple", "rank", BLACK, "X"
    ),
    TestStyle("Heller-ADP", "Heller-ADP", "density", YELLOW, "D"),
    TestStyle("Cmiel-Ledwina-Tn", "Cmiel-Ledwina-Tn", "cdf", SKY, "v"),
]
# Tests whose null law depends on n only: they get a "warm" (reused-null) series.
TIMING_WARM_TESTS = ("MRP", "GJT-QAD", "Heller-ADP", "Cmiel-Ledwina-Tn")

# Permutation / Monte-Carlo draws per timed call; the figure is pinned to it.
TIMING_B = 2000
# Smallest n entering the log-log slope fits (below it, fixed overheads dominate).
TIMING_FIT_MIN_N = 500


def timing_filename(B: int = TIMING_B) -> str:
    """Default output name of the cost figure. ``main.tex`` must include it."""
    return f"timing_B{B}.pdf"
