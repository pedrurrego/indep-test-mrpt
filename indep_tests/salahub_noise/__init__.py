"""Replication of the noise-vs-power benchmarks from Salahub & Oldford (2025).

Salahub, C. and Oldford, R. W. (2025). "Recursive random binning to detect
and display pairwise dependence." Statistical Analysis and Data Mining: An
ASA Data Science Journal. arXiv:2311.08561v2.

This subpackage benchmarks the tests in `indep_tests/` under the same
noise schedules and patterns used in Salahub's paper, so our comparisons
are directly comparable to theirs.

Two distinct benchmarks are supported; they came from different sources
and target different figures:

1. **Quasi-power curves (paper's Figure 11).**
   Patterns from Zhang (2019) JASA Table 1: linear, parabolic, circular,
   sine, checkerboard, local. Discrete noise levels l = 1, ..., 10 with
   sigma_l = l / 40. X-axis is the noise level, Y-axis is the empirical
   rejection rate at alpha (= 0.01 in the paper).
   - Patterns: `patterns_zhang.py`
   - Plotting: `plotting.plot_quasi_power`

2. **Median-p vs (1 - R^2) curves (orphaned paper appendix).**
   Ported from `scripts/noise.R` in the companion repo
   (https://github.com/Salahub/recursive-binning). Eleven patterns with
   a continuous sigma sweep. This figure is referenced by an appendix
   that is not `\\input`'d in the published arXiv build; kept here for
   completeness.
   - Patterns: `patterns_salahub.py`
   - Plotting: `plotting.plot_median_p`, `plotting.plot_median_log_p`

Contents:
    patterns_zhang.py    - 6 Zhang-2019 patterns + `sigma_for_level`
    patterns_salahub.py  - 11 Salahub noise.R patterns
    experiment.py        - `run_noise_sweep` (source-agnostic)
                           + `rejection_rate_by_sigma`
    plotting.py          - `plot_quasi_power` (Figure 11 layout)
                           + `plot_median_p`, `plot_median_log_p`
    results/             - data and figures from replication runs
"""
