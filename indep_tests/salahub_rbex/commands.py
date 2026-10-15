"""IO helpers for the salahub_rbex CLI.

Kept separate from ``experiment.py`` so the runner stays pure (no JSON, no
filenames) and the CLI layer handles serialization + path conventions.

Filename convention for one (test, config) sweep:
    <test>[_<config>]_n<N>_R<R>_B<B>.json

The pattern is always rBEX, so — unlike ``salahub_noise`` — the filename keys
on the *test* (and its config), and every JSON in ``results/`` is a full
resolution sweep for one test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .experiment import ResolutionSweepResult, records_to_dict

_DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


def _config_token(config: dict) -> str:
    return "+".join(f"{k}={v}" for k, v in sorted(config.items()))


def build_output_filename(test: str, config: dict, n: int, R: int, B: int) -> str:
    parts = [test]
    if config:
        parts.append(_config_token(config))
    parts.extend([f"n{n}", f"R{R}", f"B{B}"])
    return "_".join(parts) + ".json"


def resolve_output_path(
    output: str | None,
    test: str,
    config: dict,
    n: int,
    R: int,
    B: int,
) -> Path | None:
    """Resolve --output into a concrete file path.

    - None     -> None (caller prints to stdout)
    - ""       -> default dir + auto filename
    - .json    -> use as-is
    - dir      -> dir + auto filename
    """
    if output is None:
        return None
    filename = build_output_filename(test, config, n, R, B)
    if output == "":
        return _DEFAULT_RESULTS_DIR / filename
    p = Path(output)
    if p.suffix == ".json":
        return p
    return p / filename


def save_result(
    result: ResolutionSweepResult, alpha: float, seed: int, output_path: Path
) -> None:
    """Serialize a ResolutionSweepResult to JSON, including per-rep records."""
    payload = {
        "test": result.test_name,
        "pattern": "rbex",
        "n": result.n,
        "R": result.R,
        "B": result.B,
        "alpha": alpha,
        "seed": seed,
        "resolutions": result.resolutions.tolist(),
        "elapsed_seconds": round(result.elapsed_seconds, 2),
        "records": records_to_dict(result),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_result(path: Path) -> ResolutionSweepResult:
    """Rehydrate a ResolutionSweepResult from JSON written by ``save_result``."""
    import numpy as np

    from .experiment import ResolutionSweepRecord

    with open(path) as f:
        payload = json.load(f)

    records = [ResolutionSweepRecord(**rec) for rec in payload["records"]]
    return ResolutionSweepResult(
        test_name=payload["test"],
        n=payload["n"],
        R=payload["R"],
        B=payload["B"],
        resolutions=np.asarray(payload["resolutions"], dtype=int),
        records=records,
        elapsed_seconds=payload.get("elapsed_seconds", 0.0),
    )


_FILENAME_RE = re.compile(r"^.+_n(?P<n>\d+)_R(?P<R>\d+)_B(?P<B>\d+)\.json$")


def load_all_results(results_dir: Path) -> dict[str, ResolutionSweepResult]:
    """Load every sweep JSON in ``results_dir``, keyed by legend label.

    The label is the payload ``test_name``; when several sample sizes are
    present it is disambiguated to ``"<test_name> (N=<n>)"`` so sweeps at
    different ``n`` can be overlaid on the same chart without colliding.
    """
    sweeps = []
    for path in sorted(results_dir.glob("*.json")):
        if not _FILENAME_RE.match(path.name):
            continue
        sweeps.append(load_result(path))

    multi_n = len({s.n for s in sweeps}) > 1
    out: dict[str, ResolutionSweepResult] = {}
    for s in sweeps:
        label = f"{s.test_name} (N={s.n})" if multi_n else s.test_name
        out[label] = s
    return out
