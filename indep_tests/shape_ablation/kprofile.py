"""Per-family resolution profile: power of a *single* partition vs its cell count K.

The paper's shape section (one curve per geometry, x = cell count, fixed n). For every
geometry and every K in ``partitions.profile_Ks(geometry, 500)`` (the cap is
k <= floor(sqrt n), i.e. K <= n), one ``pug-partition mode=single single_K=K`` job per
generator of the paper panel, at the power budget (R=1000, B=2000, seed 1234,
alpha 0.05). Sample sizes: n in {200, 500} when K <= 200, n = 500 only otherwise.
Outputs go to ``results_kprofile/``; jobs whose JSON exists are skipped, so the sweep is
resumable. Power-diagram jobs wait for every needed ``cache/power_K*.npz`` to exist
(fit them first with ``python -m indep_tests.shape_ablation.power_fit K ...``) so that
parallel workers never trigger a fit.

    uv run python -m indep_tests.shape_ablation.kprofile            # all, 7 workers
    uv run python -m indep_tests.shape_ablation.kprofile --jobs 4 --geometries square golden
    uv run python -m indep_tests.shape_ablation.kprofile --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .partitions import profile_Ks
from .power_fit import CACHE, FIT_RES

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "results_kprofile"
GENERATORS = [
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
GEOMETRIES = [
    "square",
    "rect",
    "rotated",
    "golden",
    "power",
]  # power last: waits for fits
N_MAX, N_SMALL = 500, 200
R, B, SEED, ALPHA = 1000, 2000, 1234, 0.05


def jobs(geometries):
    out = []
    for geo in geometries:
        for K in profile_Ks(geo, N_MAX):
            ns = f"{N_SMALL},{N_MAX}" if K <= N_SMALL else f"{N_MAX}"
            for gen in GENERATORS:
                fname = f"{gen}_pug-partition_geometry={geo}+mode=single+single_K={K}_B{B}_R{R}.json"
                out.append((geo, K, gen, ns, OUT / fname))
    return out


def run_one(job):
    geo, K, gen, ns, path = job
    if path.exists():
        return f"skip {path.name}"
    cfg = f'{{"geometry":"{geo}","mode":"single","single_K":{K}}}'
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "indep_tests",
        "run",
        "--test",
        "pug-partition",
        "--generator",
        gen,
        "--sample-sizes",
        ns,
        "--alpha",
        str(ALPHA),
        "--r",
        str(R),
        "--b",
        str(B),
        "--seed",
        str(SEED),
        "--test-config",
        cfg,
        "--output",
        str(OUT),
    ]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    status = "OK  " if res.returncode == 0 and path.exists() else "FAIL"
    return f"{status} {path.name} ({time.time() - t0:.0f}s)" + (
        "" if status == "OK  " else "\n" + res.stderr[-800:]
    )


def wait_for_power_cache(Ks, poll=30):
    missing = lambda: [  # noqa: E731
        K for K in Ks if not (CACHE / f"power_K{K}_fit{FIT_RES}_lloyd5.npz").exists()
    ]
    while missing():
        print(f"waiting for power-diagram fits: K={missing()}", flush=True)
        time.sleep(poll)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--geometries", nargs="*", default=GEOMETRIES)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    todo = jobs(a.geometries)
    pending = [j for j in todo if not j[4].exists()]
    print(f"{len(todo)} jobs, {len(pending)} pending, {a.jobs} workers", flush=True)
    if a.dry_run:
        for j in pending:
            print(j[4].name, "n=" + j[3])
        return
    OUT.mkdir(exist_ok=True)
    non_power = [j for j in pending if j[0] != "power"]
    power = [j for j in pending if j[0] == "power"]
    with ThreadPoolExecutor(a.jobs) as ex:
        for line in ex.map(run_one, non_power):
            print(line, flush=True)
        # power-diagram jobs in increasing K, each K waiting only for its own fit
        for K in sorted({j[1] for j in power}):
            wait_for_power_cache([K])
            for line in ex.map(run_one, [j for j in power if j[1] == K]):
                print(line, flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
