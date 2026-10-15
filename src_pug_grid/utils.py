"""Pseudo-observations and grid binning for MRPT."""

import numpy as np
from scipy import stats


def calculate_ecdf(sample: np.ndarray) -> np.ndarray:
    """Calculate ECDF values F(x_i) for each point in a 1D array."""
    assert sample.ndim == 1, (
        f"Input array must be one-dimensional, got {sample.ndim}-dimensional array instead."
    )
    res = stats.ecdf(sample)
    return res.cdf.evaluate(sample)


def calculate_copula_values(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform bivariate data to pseudo-observations via marginal ECDFs.

    Returns
    -------
    u : np.ndarray, shape (n,)
        F_x(x_i) for each observation.
    v : np.ndarray, shape (n,)
        F_y(y_i) for each observation.
    """
    assert X.ndim == 2, (
        f"Input array must be 2-dimensional, got {X.ndim}-dimensional array instead."
    )
    u = calculate_ecdf(X[:, 0])
    v = calculate_ecdf(X[:, 1])
    return u, v


def bin_counts(u: np.ndarray, v: np.ndarray, k: int) -> np.ndarray:
    """Bin pseudo-observations into a k x k grid of the unit square.

    Each observation (u_i, v_i) in (0, 1]^2 is assigned to cell
    (ceil(u_i * k) - 1, ceil(v_i * k) - 1).

    Parameters
    ----------
    u, v : np.ndarray, shape (n,)
        Pseudo-observations in (0, 1].
    k : int
        Grid resolution (number of strips per axis).

    Returns
    -------
    counts : np.ndarray, shape (k, k)
        counts[i, j] = number of observations in cell (i, j).
    """
    bin_x = np.clip(np.ceil(u * k).astype(int) - 1, 0, k - 1)
    bin_y = np.clip(np.ceil(v * k).astype(int) - 1, 0, k - 1)
    counts = np.zeros((k, k), dtype=float)
    np.add.at(counts, (bin_x, bin_y), 1)
    return counts
