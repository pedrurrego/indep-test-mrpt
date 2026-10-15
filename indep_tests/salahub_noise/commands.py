"""IO helpers for the salahub_noise CLI.

Kept separate from `experiment.py` so the runner stays pure (no JSON, no
filenames) and the CLI layer handles serialization + path conventions.

Filename convention for one (pattern, test) run:
    <pattern>_<test>_n<N>_R<R>_B<B>.json

This mirrors `indep_tests/commands/experiment.py:build_output_filename`
but for the noise-sweep setting (fixed n, swept sigma) rather than the
power-curve setting (varied n, fixed pattern).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .experiment import NoiseSweepResult, records_to_dict

_DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


def build_output_filename(
    pattern: str, test: str, config: dict, n: int, R: int, B: int
) -> str:
    parts = [pattern, test]
    if config:
        parts.append("+".join(f"{k}={v}" for k, v in sorted(config.items())))
    parts.extend([f"n{n}", f"R{R}", f"B{B}"])
    return "_".join(parts) + ".json"


def resolve_output_path(
    output: str | None,
    pattern: str,
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
    filename = build_output_filename(pattern, test, config, n, R, B)
    if output == "":
        return _DEFAULT_RESULTS_DIR / filename
    p = Path(output)
    if p.suffix == ".json":
        return p
    return p / filename


def save_result(
    result: NoiseSweepResult, alpha: float, seed: int, output_path: Path
) -> None:
    """Serialize a NoiseSweepResult to JSON, including per-rep records."""
    payload = {
        "test": result.test_name,
        "pattern": result.pattern,
        "n": result.n,
        "R": result.R,
        "B": result.B,
        "alpha": alpha,
        "seed": seed,
        "sigmas": result.sigmas.tolist(),
        "elapsed_seconds": round(result.elapsed_seconds, 2),
        "records": records_to_dict(result),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_result(path: Path) -> NoiseSweepResult:
    """Rehydrate a NoiseSweepResult from a JSON file written by `save_result`."""
    import numpy as np

    from .experiment import NoiseSweepRecord

    with open(path) as f:
        payload = json.load(f)

    records = [NoiseSweepRecord(**rec) for rec in payload["records"]]
    return NoiseSweepResult(
        test_name=payload["test"],
        pattern=payload["pattern"],
        n=payload["n"],
        R=payload["R"],
        B=payload["B"],
        sigmas=np.asarray(payload["sigmas"], dtype=float),
        records=records,
        elapsed_seconds=payload.get("elapsed_seconds", 0.0),
    )


_FILENAME_RE = re.compile(
    r"^(?P<pattern>[^_]+)_(?P<test>.+?)"
    r"(?:_(?P<config>[^_]+=[^_]+(?:\+[^_]+=[^_]+)*))?"
    r"_n(?P<n>\d+)_R(?P<R>\d+)_B(?P<B>\d+)\.json$"
)


def load_results_for_pattern(
    results_dir: Path, pattern: str
) -> dict[str, NoiseSweepResult]:
    """Load every JSON in `results_dir` whose filename starts with `<pattern>_`.

    Returns
    -------
    dict[str, NoiseSweepResult]
        Keyed by test label (from the loaded payload, not from the filename).
    """
    out: dict[str, NoiseSweepResult] = {}
    for path in sorted(results_dir.glob(f"{pattern}_*.json")):
        m = _FILENAME_RE.match(path.name)
        if not m or m.group("pattern") != pattern:
            continue
        sweep = load_result(path)
        out[sweep.test_name] = sweep
    return out


def discover_patterns(results_dir: Path) -> list[str]:
    """List every non-null pattern with at least one result JSON in `results_dir`."""
    patterns: set[str] = set()
    for path in results_dir.glob("*.json"):
        m = _FILENAME_RE.match(path.name)
        if m and m.group("pattern") != "null":
            patterns.add(m.group("pattern"))
    return sorted(patterns)


def load_null_rates(results_dir: Path, alpha: float) -> dict[str, float]:
    """Load every `null_*.json` in `results_dir` and return rejection rate per test.

    Keyed on test_name (from the JSON payload, not the filename) so the
    returned dict aligns with the keys returned by `load_results_for_pattern`.
    """
    import numpy as np

    out: dict[str, float] = {}
    for path in sorted(results_dir.glob("null_*.json")):
        m = _FILENAME_RE.match(path.name)
        if not m or m.group("pattern") != "null":
            continue
        sweep = load_result(path)
        pvals = np.array([r.p_value for r in sweep.records])
        if pvals.size:
            out[sweep.test_name] = float(np.mean(pvals <= alpha))
    return out
