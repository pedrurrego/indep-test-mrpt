"""Null-distribution approximations for recursive random binning.

All formulas come from Appendix D of Salahub & Oldford (2025) and are used in
the R package ``AssocBin`` for the continuous-vs-continuous case.

Provenance
----------
Reference: Salahub & Oldford (2025), see ``binning.py``.
Upstream code: R package ``AssocBin`` (CRAN), GPL (>= 3).
This module: port of the p-value methods from the upstream R code.
Distributed under GPL-3 (LICENSE).
"""

from __future__ import annotations

from typing import Literal

from scipy.stats import chi2, gamma

PMethod = Literal["simple", "fitted", "gamma"]


def simple_df(K: int) -> float:
    """Mnemonic degrees of freedom: (sqrt(K) - 1)^2."""
    import math

    return (math.sqrt(K) - 1.0) ** 2


def fitted_df(K: int) -> float:
    """Least-squares fitted degrees of freedom from the paper."""
    import math

    return (math.sqrt(K) - 0.8576945) ** 2


def gamma_params(df: float) -> tuple[float, float]:
    """Shape and scale of the Gamma approximation, parameterised by df."""
    import math

    shape = (0.1199774 + 0.7214124 * math.sqrt(df)) ** 2
    scale = math.exp(0.4329157 + (1.0 - 0.9571741) * math.log(df))
    return shape, scale


def p_value(stat: float, K: int, method: PMethod = "simple") -> float:
    """Upper-tail p-value under the selected null approximation."""
    if K < 2:
        return 1.0
    if method == "simple":
        df = simple_df(K)
        if df <= 0:
            return 1.0
        return float(chi2.sf(stat, df))
    if method == "fitted":
        df = fitted_df(K)
        if df <= 0:
            return 1.0
        return float(chi2.sf(stat, df))
    if method == "gamma":
        df = simple_df(K)
        if df <= 0:
            return 1.0
        shape, scale = gamma_params(df)
        return float(gamma.sf(stat, a=shape, scale=scale))
    raise ValueError(f"Unknown p-value method: {method!r}")
