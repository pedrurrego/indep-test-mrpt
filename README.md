# indep-test-mrpt

Code and results for the paper

> Urrego-Gómez, P. and Ortiz, S. *A Rank-Based Multi-Resolution Test of
> Independence on a Ladder of Square Grids.* Manuscript, 2026.

The repository contains the proposed test (MRPT, a multi-resolution Pearson
statistic on the empirical copula), the seven comparators of the simulation
study, the harness that produced every figure and table of the manuscript, and
the result files at the manuscript's Monte-Carlo budgets. Figures and tables
can therefore be rebuilt in seconds without re-running the experiments, and
every experiment can be re-run from the command line.

In the code and in the result files MRPT appears under its development name
**PUG-Grid**: the test label is `PUG-Grid-DENSITY-MULTI` (`MRP` in the timing
files), its χ² calibration is `PUG-Grid-ChiSq-DENSITY-MULTI`, and the
cell-shape variant is `PUG-Partition-*`.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Run every command
from the repository root: the packages are imported from the working directory.

```bash
uv sync                                   # creates .venv from uv.lock
uv run --with pytest pytest src_pug_grid  # unit tests of the MRPT implementation
```

The `Makefile` needs GNU Make 4 (`gmake` on macOS).

## Rebuild the manuscript's figures and tables

```bash
uv run python -m indep_tests.paper_figures build-all   # -> paper/figures/*.pdf (about 15 s)
uv run python -m indep_tests.paper_figures list        # figure names and descriptions
uv run python -m indep_tests.paper_figures build power-curves
```

The builders read only the result files under `indep_tests/`. Each data
figure is pinned to the budget `(B, R, alpha)` of the manuscript, which is
stamped bottom-right on the figure and written into the file name; result
files at any other budget are ignored (pass `--strict` to `build` to fail
instead of warn when a curve is missing).

| Manuscript table | Command |
|---|---|
| Noise benchmark (`tab:zhang_noise`) | `uv run python -m indep_tests.paper_figures.noise_table` |
| Ladder choice (`tab:ladder_choice`) | `uv run python -m indep_tests.pug_grid_density_multi.compare_ladders paper-table` |
| Cell-shape profile (`tab:shape_profile`) | `uv run python -m indep_tests.shape_ablation.profile_table` |
| Computational cost (`tab:time-complexity`) | `uv run python -m indep_tests.timing.fit` |

The power summary table has no script: it is the mean and the minimum, over
the nine dependent generators, of the rejection rates stored in the power
result files (`indep_tests/results/*_B2000_R1000.json`).

## Tests

| CLI name | Label in the paper | Implementation |
|---|---|---|
| `pug-grid` | MRPT | `src_pug_grid/` |
| `pug-grid-chi2` | MRPT (χ² calibration) | `indep_tests/pug_grid_pearson_chi_squared/` |
| `pug-partition` | MRPT on alternative cell shapes (cell-geometry section) | `indep_tests/shape_ablation/` |
| `qad` | GJT-QAD, Junker, Griessenberger & Trutschnig (2021) | `pug_trutschnig/` |
| `dcov` | SRB-dCov, Székely, Rizzo & Bakirov (2007), via `hyppo` | `indep_tests/dcov.py` |
| `heller-bins` | Heller-ADP, Heller, Heller, Kaufman, Brill & Gorfine (2016) | `indep_tests/heller_bins/` |
| `ma-mao-fes` | Ma-Mao-FES, Ma & Mao (2019) | `indep_tests/ma_mao_fes.py` |
| `salahub-binning` | Salahub-Oldford, Salahub & Oldford (2025) | `indep_tests/salahub_binning/` |
| `cmiel-ledwina` | Ćmiel-Ledwina T_n, Ćmiel & Ledwina (2026) | `indep_tests/cmiel_ledwina_test/` |
| `zhang-bet` | Zhang-BET, Zhang (2019) | `indep_tests/zhang_bet.py` |

Every test sits behind the same interface (`indep_tests/base.py`):
`IndependenceTest.test(x, y, B=..., seed=...)` returns a `TestResult` with the
statistic and p-value. Constructor options are passed on the command line as
JSON (`--test-config`); the `_key=value` part of each result file name records
the options that produced it.

```python
import numpy as np
from indep_tests.pug_grid import PUGGridTest

rng = np.random.default_rng(0)
x = rng.uniform(size=500)
y = np.sin(4 * np.pi * x) + 0.3 * rng.normal(size=500)
print(PUGGridTest(norm="density-multi").test(x, y, B=2000, seed=1))
```

## Configurations and budgets used in the manuscript

The harness defaults are not always the manuscript's settings. The table
gives, per test, the `--test-config` and the permutation count `B` used in
each benchmark; a result file is only picked up by the figure builders if its
label and budget match.

| CLI name | `--test-config` | Power vs n (`B`) | Noise (`B`) | rBEX (`B`) |
|---|---|---|---|---|
| `pug-grid` | `{"norm": "density-multi"}` | 2000 | 1000 | 500 |
| `pug-grid-chi2` | `{"norm": "density-multi"}` | closed form | closed form | closed form |
| `qad` | default | 2000 | 1000 | 500 |
| `dcov` | `{"auto": true}` | 2000 | 500 | 500 |
| `heller-bins` | default | 2000 | 1000 | 2000 |
| `ma-mao-fes` | default; noise: `{"K1": 4, "K2": 4}` | 2000 | 500 | 500 |
| `salahub-binning` | `{"max_depth": 8}`; noise, rBEX: add `"p_method": "simple"` | 2000 | 500 | 500 |
| `cmiel-ledwina` | default | 2000 | 1000 | 500 |
| `zhang-bet` | default | closed form | closed form | closed form |

Common settings: power vs n uses `n ∈ {20, 50, 100, 200, 500}`, `R = 1000`,
`alpha = 0.05`; the noise benchmark `n = 500`, `R = 200`, `alpha = 0.01`; the
rBEX benchmark `n ∈ {500, 2000}`, `R = 200`, `alpha = 0.01`; seed 1234
throughout. Closed-form tests ignore `B` (their files carry the nominal value
of the benchmark). QAD, Heller-ADP and Ćmiel-Ledwina draw one null table per
`(n, B)` configuration and reuse it across the `R` replications.

## Re-running the experiments

```bash
uv run python -m indep_tests list-tests
uv run python -m indep_tests list-generators

# Size / power vs n: one test, one generator, manuscript budget
uv run python -m indep_tests run --test pug-grid --generator dep-sine-4pi \
    --test-config '{"norm": "density-multi"}' \
    --sample-sizes 20,50,100,200,500 --r 1000 --b 2000 --alpha 0.05 --seed 1234 \
    --output indep_tests/results

gmake run TEST=qad B=2000 R=1000 -j4                     # one test, every generator
gmake noise-run TEST=qad -j6                             # noise benchmark
gmake rbex-run TEST=qad -j4                              # rBEX benchmark
uv run python -m indep_tests.pug_grid_density_multi run --B 2000 --R 1000 --jobs 4   # ladder ablation
uv run python -m indep_tests.shape_ablation.kprofile --jobs 4                         # cell-shape profile
uv run python -m indep_tests.timing --B 2000 --cap 3600 --max-predicted 10800         # timing sweep
```

Without `--output`, `run` prints the results instead of writing a file. The
Makefile targets write into the result directories listed below; the ladder
and timing CLIs default to pilot budgets, so pass the flags shown. The shipped
timing sweep timed QAD, Heller-ADP and Ćmiel-Ledwina separately
(`--only LABEL`) up to `n = 4000`.

Results are written next to the ones shipped here (`indep_tests/results/`,
`indep_tests/salahub_noise/results/`, `indep_tests/salahub_rbex/results/`,
`indep_tests/pug_grid_density_multi/ladder_ablation/`,
`indep_tests/shape_ablation/results_kprofile/`, `indep_tests/timing/results/`),
auto-named `{generator}_{test}[_{options}]_B{B}_R{R}.json`. Replication `i`
draws data with `seed + i` and runs the test with `seed + i + R`. The full
sweeps at the manuscript's budgets take CPU-days; the shipped files are the
runs used in the paper.

## Layout

```
src_pug_grid/                 MRPT: statistic, resolution ladder, permutation null
pug_trutschnig/               QAD comparator
pug/utils_yanagimoto.py       helper for the Yanagimoto generator
indep_tests/
  base.py, registry.py        common interface and test registry
  generators.py               data generators of the power benchmark
  cli.py, commands/           `python -m indep_tests` (run / plot)
  <test>.py, <test>/          comparator implementations
  pug_grid_pearson_chi_squared/  closed-form (chi-squared) calibration of MRPT
  results/                    size / power vs n result files
  salahub_noise/              noise benchmark (patterns, CLI, results)
  salahub_rbex/               rBEX benchmark (patterns, CLI, results)
  pug_grid_density_multi/     resolution-ladder ablation (CLI, results)
  shape_ablation/             cell-shape study (partitions, CLI, results)
  timing/                     single-call cost sweep (CLI, results)
  paper_figures/              figure and table builders
paper/figures/                the figures as included in the manuscript
```

## License

GPL-3.0-or-later (see `LICENSE`). The proposed test and the harness are the
authors' own code. Heller-ADP and Salahub-Oldford are ports of GPL R packages
(`HHG`, `AssocBin`); Zhang-BET and Ma-Mao-FES were implemented from their
papers with the authors' R code as reference; the remaining comparators were
implemented from their papers. Each comparator module ends its docstring with
a *Provenance* section giving the paper, the upstream code and its license.
