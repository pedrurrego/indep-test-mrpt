"""Declarative registry of paper figures.

Each figure is a pinned target: a stable name, a default output filename under
``paper/figures/``, a one-line description, and the builder that renders it.
``build-all`` walks this dict, so regenerating the whole figure set for a
resubmission is deterministic.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import config, figures


@dataclass(frozen=True)
class FigureSpec:
    name: str
    filename: str
    description: str
    builder: Callable[..., Path]


FIGURES: dict[str, FigureSpec] = {
    "generators": FigureSpec(
        "generators",
        "generators.pdf",
        "Gallery of every paper generator (one sample each), 3 columns.",
        figures.build_generator_gallery,
    ),
    "generators-ranks": FigureSpec(
        "generators-ranks",
        "generators_ranks.pdf",
        "Same gallery as pseudo-observations (ranks / n) on [0,1]^2.",
        functools.partial(figures.build_generator_gallery, ranks=True),
    ),
    "type-one": FigureSpec(
        "type-one",
        config.type_one_filename(**config.POWER_CURVE_BUDGET),
        "Sec. 3.1 Type I error: rejection rate vs n on the independent generators.",
        figures.build_type_one_error,
    ),
    "power-curves": FigureSpec(
        "power-curves",
        config.power_curves_filename(**config.POWER_CURVE_BUDGET),
        "Sec. 3.2 power vs sample size: dependent generators x curated tests.",
        figures.build_power_curves,
    ),
    "noise": FigureSpec(
        "noise",
        config.noise_filename(**config.NOISE_BUDGET),
        "Sec. 3.3 power vs noise level on the Zhang (2019) patterns.",
        figures.build_noise_curves,
    ),
    "noise-gallery": FigureSpec(
        "noise-gallery",
        "noise_gallery.pdf",
        "Sec. 4.3 illustration: the six Zhang (2019) noise patterns at n=500, "
        "one row per noise level l in {1, 3, 5}.",
        figures.build_noise_gallery,
    ),
    "rbex-gallery": FigureSpec(
        "rbex-gallery",
        "rbex_gallery.pdf",
        "Sec. 4.4 illustration: Zhang's BEX at resolutions d=1..5, one row per "
        "sample size N in {500, 2000}.",
        figures.build_rbex_gallery,
    ),
    "rbex": FigureSpec(
        "rbex",
        config.rbex_filename(**config.RBEX_BUDGET),
        "Sec. 3.4 quasi-power vs BEX resolution (Zhang's cross), one panel per N.",
        figures.build_rbex_curves,
    ),
    "ladder-ablation": FigureSpec(
        "ladder-ablation",
        config.ladder_ablation_filename(**config.LADDER_BUDGET),
        "Sec. 4 ladder ablation: power vs single grid k at n=500 (solid), n=200 "
        "(dashed) and n=100 (dotted), one panel per dependent generator, ladder as "
        "reference line.",
        figures.build_ladder_ablation,
    ),
    "ladder-choice": FigureSpec(
        "ladder-choice",
        config.ladder_choice_filename(**config.LADDER_BUDGET),
        "Sec. 4 ladder composition: power vs n of the production ladder and three "
        "alternatives (endpoints, no top, dense), one panel per dependent generator.",
        figures.build_ladder_choice,
    ),
    "ladder-mechanism": FigureSpec(
        "ladder-mechanism",
        "ladder_mechanism.pdf",
        "Sec. 3 illustration of what the statistic sees: pseudo-observations and "
        "standardized cell departures at three ladder rungs on two archetypes.",
        figures.build_ladder_mechanism,
    ),
    "shape-catalogue": FigureSpec(
        "shape-catalogue",
        "shape_catalogue.pdf",
        "Sec. 3 shape ablation: the five partition families at two ladder rungs.",
        figures.build_shape_catalogue,
    ),
    "shape-ablation": FigureSpec(
        "shape-ablation",
        config.shape_ablation_filename(**config.SHAPE_BUDGET),
        "Sec. 3 shape ablation: power vs n per generator, one curve per cell "
        "geometry (ladder mode).",
        figures.build_shape_ablation,
    ),
    "shape-profile-n200": FigureSpec(
        "shape-profile-n200",
        config.shape_profile_filename(200, **config.SHAPE_PROFILE_BUDGET),
        "Sec. 3 shape ablation: power of a single partition vs its cell count at "
        "n=200, one curve per cell geometry (no ladder).",
        functools.partial(figures.build_shape_profile, n=200),
    ),
    "shape-profile-n500": FigureSpec(
        "shape-profile-n500",
        config.shape_profile_filename(500, **config.SHAPE_PROFILE_BUDGET),
        "Sec. 3 shape ablation: power of a single partition vs its cell count at "
        "n=500, one curve per cell geometry (no ladder).",
        functools.partial(figures.build_shape_profile, n=500),
    ),
    "timing": FigureSpec(
        "timing",
        config.timing_filename(config.TIMING_B),
        "Sec. 3 computational cost: single-call seconds vs n, statistic-only and full call.",
        figures.build_timing,
    ),
}
