"""rBEX quasi-power benchmark: the bisection expanding cross.

Zhang, K. (2019). "BET on Independence." JASA 114(528), 1620-1637 — §1.1
introduces the *bisection expanding cross* (BEX), a fractal family of degenerate
distributions ``BEX_d`` with Uniform[0,1] margins whose joint law approaches
independence as the resolution ``d`` grows, even though every level sits at TV
distance 1 from independence. It is his motivating example for non-uniform
consistency: no test is uniformly consistent against all BEX levels.

Salahub, C. and Oldford, R. W. (2025). "Recursive random binning to detect and
display pairwise dependence." §5.3 samples ``N`` points uniformly on ``BEX_d``
(the **rBEX pattern** at *resolution* ``d``) and plots quasi-power vs
``null, 1, ..., 10``, where "null" is independent-uniform data.

This subpackage reproduces that experiment for the tests in ``indep_tests/`` —
in particular the headline comparison the user asked for: Salahub's recursive
random binning (``salahub-binning``) vs. our ``pug-grid`` DENSITY-MULTI. Because
BEX is a 45-degree *rotated* grid, it is a sharp probe of whether an
axis-aligned multi-resolution density test can recover geometric dependence.

Contents:
    patterns.py    - `rbex_sample(n, resolution, rng)` + `resolution_to_grid`
    experiment.py  - `run_resolution_sweep` + `rejection_rate_by_resolution`
    commands.py    - JSON save/load + filename conventions
    plotting.py    - `plot_quasi_power_resolution`, `plot_pattern_grid`
    cli.py         - Typer app (`run`, `plot`, `plot-pattern`, `list-tests`)
    results/       - data and figures from replication runs

Run it with `uv run python -m indep_tests.salahub_rbex`.
"""
