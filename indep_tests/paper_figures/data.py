"""Data access for the paper figures.

Reads the *existing* experiment outputs — never re-runs experiments:

- power-vs-n curves from ``indep_tests/results/*.json``;
- power-vs-noise sweeps from ``indep_tests/salahub_noise/results/*.json``
  (reusing that package's own loaders);
- fresh scatter samples straight from the generator registry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from indep_tests.generators import get_generator

_REPO_ROOT = Path(__file__).resolve().parents[2]
POWER_RESULTS_DIR = _REPO_ROOT / "indep_tests" / "results"
NOISE_RESULTS_DIR = _REPO_ROOT / "indep_tests" / "salahub_noise" / "results"
RBEX_RESULTS_DIR = _REPO_ROOT / "indep_tests" / "salahub_rbex" / "results"
LADDER_ABLATION_DIR = (
    _REPO_ROOT / "indep_tests" / "pug_grid_density_multi" / "ladder_ablation"
)
SHAPE_RESULTS_DIR = _REPO_ROOT / "indep_tests" / "shape_ablation" / "results_paper"
SHAPE_PROFILE_DIR = _REPO_ROOT / "indep_tests" / "shape_ablation" / "results_kprofile"
PAPER_FIGURES_DIR = _REPO_ROOT / "paper" / "figures"


def load_power_results_at_budget(
    generator: str,
    *,
    B: int,
    R: int,
    alpha: float = 0.05,
    results_dir: Path = POWER_RESULTS_DIR,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{test_name: (ns, rejection_rates)}`` for one generator at an
    *exact* Monte-Carlo budget.

    A file is used only if the ``B``, ``R`` and ``alpha`` recorded *inside* it match. Tests
    with no file at this budget are simply absent from the result, so the caller
    can report them. This is what lets a figure honestly claim "every curve here
    was run with (B, R, alpha)" — mixed budgets can never sneak into one panel.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(Path(results_dir).glob(f"{generator}_*.json")):
        with open(path) as fh:
            payload = json.load(fh)
        if payload.get("generator") != generator:
            continue
        if int(payload.get("B", -1)) != B or int(payload.get("R", -1)) != R:
            continue
        if not np.isclose(float(payload.get("alpha", -1.0)), alpha):
            continue
        test = payload["test"]
        if test in out:
            raise ValueError(
                f"Two result files for {generator!r}/{test!r} at B={B}, R={R}, "
                f"alpha={alpha}; second one: {path.name}"
            )
        rows = sorted(payload["results"], key=lambda r: r["n"])
        ns = np.array([r["n"] for r in rows], dtype=float)
        rates = np.array([r["rejection_rate"] for r in rows], dtype=float)
        out[test] = (ns, rates)
    return out


def _noise_sweeps_at_budget(
    pattern: str, *, B, R: int, alpha: float, results_dir: Path
) -> dict:
    """``{test_name: NoiseSweepResult}`` for one pattern at an exact budget.

    ``B`` is an int or a tuple of admissible ints (the benchmark used a
    per-test permutation count). ``R`` and ``alpha`` must match exactly; both
    are read from *inside* the file, not from its name.
    """
    from indep_tests.salahub_noise.commands import _FILENAME_RE, load_result

    admissible_B = set(B) if isinstance(B, tuple) else {int(B)}
    out: dict = {}
    for path in sorted(Path(results_dir).glob(f"{pattern}_*.json")):
        m = _FILENAME_RE.match(path.name)
        if not m or m.group("pattern") != pattern:
            continue
        with open(path) as fh:
            payload = json.load(fh)
        if (
            int(payload.get("R", -1)) != R
            or int(payload.get("B", -1)) not in admissible_B
        ):
            continue
        if not np.isclose(float(payload.get("alpha", -1.0)), alpha):
            continue
        sweep = load_result(path)
        if sweep.test_name in out:
            raise ValueError(
                f"Two noise result files for {pattern!r}/{sweep.test_name!r} at "
                f"R={R}, B in {sorted(admissible_B)}, alpha={alpha}; second: {path.name}"
            )
        out[sweep.test_name] = sweep
    return out


def load_noise_results_for_pattern(
    pattern: str,
    alpha: float,
    results_dir: Path = NOISE_RESULTS_DIR,
    *,
    B=None,
    R: int | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float], dict[str, int]]:
    """Return per-test ``(levels, rejection_rates)``, per-test null rates and
    the per-test permutation count ``B`` actually used, all at one *exact*
    Monte-Carlo budget (default :data:`config.NOISE_BUDGET`).

    Reuses the ``salahub_noise`` rejection-rate convention so the curves match
    that benchmark's own charts. Tests with no file at the budget are absent.
    """
    from indep_tests.salahub_noise.experiment import rejection_rate_by_sigma

    from .config import NOISE_BUDGET, NOISE_LEVEL_FACTOR

    B = NOISE_BUDGET["B"] if B is None else B
    R = NOISE_BUDGET["R"] if R is None else R
    sweeps = _noise_sweeps_at_budget(
        pattern, B=B, R=R, alpha=alpha, results_dir=Path(results_dir)
    )
    nulls = _noise_sweeps_at_budget(
        "null", B=B, R=R, alpha=alpha, results_dir=Path(results_dir)
    )

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    budgets: dict[str, int] = {}
    for test_name, sweep in sweeps.items():
        sigmas, power = rejection_rate_by_sigma(sweep, alpha=alpha)
        levels = np.asarray(sigmas, dtype=float) * NOISE_LEVEL_FACTOR
        curves[test_name] = (levels, power)
        budgets[test_name] = int(sweep.B)
    null_rates: dict[str, float] = {}
    for test_name, sweep in nulls.items():
        pvals = np.array([r.p_value for r in sweep.records])
        if pvals.size:
            null_rates[test_name] = float(np.mean(pvals <= alpha))
    return curves, null_rates, budgets


def load_rbex_results_for_n(
    n: int,
    alpha: float,
    results_dir: Path = RBEX_RESULTS_DIR,
    *,
    B=None,
    R: int | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    """Return per-test ``(resolutions, rejection_rates)`` for one sample size,
    plus the per-test permutation count ``B`` actually used, at one *exact*
    Monte-Carlo budget (default :data:`config.RBEX_BUDGET`; ``B`` may be a
    tuple of admissible per-test values).

    Reads the ``salahub_rbex`` sweep JSONs and reuses that package's own
    ``rejection_rate_by_resolution`` so the calibration convention matches its
    charts. Only sweeps whose recorded ``n``, ``R``, ``alpha`` match and whose
    ``B`` is admissible are kept; resolution 0 is the independent-uniform
    "null" (size) anchor. Tests with no file at the budget are absent.
    """
    from indep_tests.salahub_rbex.commands import _FILENAME_RE, load_result
    from indep_tests.salahub_rbex.experiment import rejection_rate_by_resolution

    from .config import RBEX_BUDGET

    B = RBEX_BUDGET["B"] if B is None else B
    R = RBEX_BUDGET["R"] if R is None else R
    admissible_B = set(B) if isinstance(B, tuple) else {int(B)}

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    budgets: dict[str, int] = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        if not _FILENAME_RE.match(path.name):
            continue
        with open(path) as fh:
            payload = json.load(fh)
        if int(payload.get("n", -1)) != n or int(payload.get("R", -1)) != R:
            continue
        if int(payload.get("B", -1)) not in admissible_B:
            continue
        if not np.isclose(float(payload.get("alpha", -1.0)), alpha):
            continue
        sweep = load_result(path)
        if sweep.test_name in out:
            raise ValueError(
                f"Two rBEX result files for {sweep.test_name!r} at n={n}, R={R}, "
                f"B in {sorted(admissible_B)}, alpha={alpha}; second: {path.name}"
            )
        res, power = rejection_rate_by_resolution(sweep, alpha=alpha)
        out[sweep.test_name] = (
            np.asarray(res, dtype=float),
            np.asarray(power, dtype=float),
        )
        budgets[sweep.test_name] = int(sweep.B)
    return out, budgets


def _rejection_rate_at_n(path: Path, n: int) -> float:
    with open(path) as fh:
        payload = json.load(fh)
    for row in payload["results"]:
        if row["n"] == n:
            return float(row["rejection_rate"])
    raise KeyError(f"no n={n} row in {path}")


def load_ladder_ablation_at_n(
    generator: str,
    n: int,
    fixed_ks: list[int],
    B: int = 500,
    R: int = 100,
    ablation_dir: Path = LADDER_ABLATION_DIR,
    results_dir: Path = POWER_RESULTS_DIR,
) -> tuple[list[int], list[float], float]:
    """Return ``(ks, single_grid_powers, multi_power)`` at one sample size.

    The single-grid sequence is the fixed resolutions from the ladder-ablation
    sweep (``k={K}+norm=density``) followed by the ``floor(sqrt(n))`` default
    (the ``norm=density`` results, which carry no explicit ``k``); the multi
    value comes from the ``norm=density-multi`` results. All three sources are
    read at exactly the budget ``(B, R)`` (the ``_B{B}_R{R}.json`` suffix), so
    a rung can never silently come from a different Monte-Carlo budget; a
    missing file raises ``FileNotFoundError`` naming it.
    """
    suffix = f"_B{B}_R{R}.json"
    ks, powers = [], []
    for k in fixed_ks:
        path = ablation_dir / f"{generator}_pug-grid_k={k}+norm=density{suffix}"
        ks.append(k)
        powers.append(_rejection_rate_at_n(path, n))
    sqrt_k = int(np.floor(np.sqrt(n)))
    ks.append(sqrt_k)
    powers.append(
        _rejection_rate_at_n(
            results_dir / f"{generator}_pug-grid_norm=density{suffix}", n
        )
    )
    multi = _rejection_rate_at_n(
        results_dir / f"{generator}_pug-grid_norm=density-multi{suffix}", n
    )
    return ks, powers, multi


def load_ladder_choice_curve(
    generator: str,
    spec: str,
    B: int,
    R: int,
    ablation_dir: Path = LADDER_ABLATION_DIR,
    results_dir: Path = POWER_RESULTS_DIR,
) -> tuple[list[int], list[float]]:
    """``(ns, rejection_rates)`` of one Bonferroni ladder on one generator.

    ``spec`` is the ``ladder=`` token of the ladder-ablation sweep
    (``3-sqrt``, ``3-5-8-12``, ``all``); ``"production"`` reads the paper's
    ladder from the ``norm=density-multi`` file in ``results/``. Read at exactly
    the budget ``(B, R)``; a missing file raises ``FileNotFoundError``.
    """
    suffix = f"_B{B}_R{R}.json"
    if spec == "production":
        path = results_dir / f"{generator}_pug-grid_norm=density-multi{suffix}"
    else:
        path = ablation_dir / (
            f"{generator}_pug-grid_ladder={spec}+norm=density-multi{suffix}"
        )
    with path.open() as fh:
        payload = json.load(fh)
    rows = sorted(payload["results"], key=lambda r: r["n"])
    return [int(r["n"]) for r in rows], [float(r["rejection_rate"]) for r in rows]


def load_shape_profile_at_n(
    generator: str,
    geometry: str,
    n: int,
    B: int,
    R: int,
    results_dir: Path = SHAPE_PROFILE_DIR,
) -> tuple[list[int], list[float]]:
    """``(cell_counts, rejection_rates)`` of one family's single-partition profile.

    Reads every ``{generator}_pug-partition_geometry={geometry}+mode=single+
    single_K={K}_B{B}_R{R}.json`` of the sweep and keeps the rows at sample size
    ``n`` (files whose nominal K exceeds n carry no such row and are skipped).
    The x value is the partition's *actual* cell count (differs from the nominal
    K only for the rotated grid, whose boundary merging changes the count).
    """
    from indep_tests.shape_ablation.partitions import make_partition_K

    pattern = f"{generator}_pug-partition_geometry={geometry}+mode=single+single_K=*_B{B}_R{R}.json"
    points: dict[int, float] = {}
    for path in results_dir.glob(pattern):
        nominal = int(path.name.split("single_K=")[1].split("_")[0])
        with open(path) as fh:
            rows = json.load(fh)["results"]
        rate = next((float(r["rejection_rate"]) for r in rows if r["n"] == n), None)
        if rate is None:
            continue
        points[make_partition_K(geometry, nominal).K] = rate
    Ks = sorted(points)
    return Ks, [points[K] for K in Ks]


def sample_generator(name: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw one ``(X, Y)`` sample of size ``n`` from a registered generator."""
    rng = np.random.default_rng(seed)
    return get_generator(name)(n, rng)


def sample_noise_pattern(
    name: str, n: int, level: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """One sample of a Zhang (2019) noise pattern at discrete noise level
    ``level`` (sigma = level / 40), for the noise-pattern gallery. Same seed
    => same points; the level only changes the noise scale."""
    from indep_tests.salahub_noise.patterns_zhang import get_pattern, sigma_for_level

    rng = np.random.default_rng(seed)
    sample = get_pattern(name)(n, sigma_for_level(level), rng)
    return np.asarray(sample.x, dtype=float), np.asarray(sample.y, dtype=float)


def sample_rbex(n: int, resolution: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One uniform sample of ``n`` points on Zhang's ``BEX_resolution`` (the
    rBEX pattern of Salahub & Oldford), for the rBEX gallery."""
    from indep_tests.salahub_rbex.patterns import rbex_sample

    rng = np.random.default_rng(seed)
    return rbex_sample(n, resolution, rng)


def pseudo_observations(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank-transform a sample to the paper's pseudo-observations ``(R_i/n, S_i/n)``
    (Eq. empcopula; average ranks for ties)."""
    n = len(x)
    return rankdata(x) / n, rankdata(y) / n


# --- Computational cost sweep (subsec:time) ----------------------------------

TIMING_RESULTS_DIR = _REPO_ROOT / "indep_tests" / "timing" / "results"


def load_timing(
    B: int, results_dir: Path | None = None
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Read the per-test files ``<test>_B{B}.json`` written by ``python -m indep_tests.timing``.

    Every file directly in ``results_dir`` (sub-directories such as
    ``superseded/`` are ignored) whose name ends in ``_B{B}.json`` contributes
    one test.  Returns ``(meta, series)`` with ``meta`` the run parameters of
    the most recently started file (plus ``meta["files"]``, the per-test
    ``started``/``git`` stamps, so mixed reruns stay visible) and
    ``series[test] = {"n", "stat", "call", "warm"}`` as arrays sorted by ``n``
    (seconds; ``stat`` is one statistic evaluation, ``call`` one cold
    ``test()`` call at the pinned ``B``, ``warm`` one call reusing a stored null
    table -- NaN where not measured).
    """
    root = results_dir or TIMING_RESULTS_DIR
    paths = sorted(root.glob(f"*_B{B}.json"))
    if not paths:
        raise FileNotFoundError(f"no *_B{B}.json in {root}")
    nan = float("nan")
    series: dict[str, dict[str, list]] = {}
    metas: list[dict] = []
    for path in paths:
        with open(path) as fh:
            payload = json.load(fh)
        meta = payload["meta"]
        if meta["B"] != B:
            raise ValueError(f"{path} was run at B={meta['B']}, not {B}")
        metas.append(meta)
        for row in payload["rows"]:
            s = series.setdefault(
                row["test"], {"n": [], "stat": [], "call": [], "warm": []}
            )
            s["n"].append(row["n"])
            s["stat"].append(
                nan if row["stat_seconds"] is None else row["stat_seconds"]
            )
            s["call"].append(
                nan if row["call_seconds"] is None else row["call_seconds"]
            )
            w = row.get("warm_seconds")
            s["warm"].append(nan if w is None else w)
    latest = max(metas, key=lambda m: m.get("started", ""))
    meta = {
        k: v
        for k, v in latest.items()
        if k not in ("test", "class", "params", "warm_params", "reruns")
    }
    meta["files"] = {
        m["test"]: {"started": m.get("started"), "git": m.get("git")} for m in metas
    }
    out = {}
    for test, s in series.items():
        order = np.argsort(s["n"])
        out[test] = {k: np.asarray(v, dtype=float)[order] for k, v in s.items()}
    return meta, out


def loglog_slope(n: np.ndarray, t: np.ndarray, *, min_n: int) -> float | None:
    """OLS slope of log t on log n over ``n >= min_n`` (None if < 2 points)."""
    m = (n >= min_n) & np.isfinite(t)
    if m.sum() < 2:
        return None
    return float(np.polyfit(np.log(n[m]), np.log(t[m]), 1)[0])
