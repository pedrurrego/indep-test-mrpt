"""
Implementation of the quantile-based asymmetric dependence (qad) measure.

Based on:
- Trutschnig, W. (2011). On a strong metric on the space of copulas and its induced
  dependence measure. Journal of Mathematical Analysis and Applications, 384(2), 690-705.
- Junker, R.R., Griessenberger, F., Trutschnig, W. (2021). Estimating scale-invariant
  directed dependence of bivariate distributions. Computational Statistics and Data Analysis, 153, 107058.

The qad measure quantifies directed dependence between two random variables X and Y:
- qad(X,Y) ∈ [0,1]
- qad(X,Y) = 0 iff X and Y are independent
- qad(X,Y) = 1 iff Y is a function of X (complete dependence)
- qad(X,Y) ≠ qad(Y,X) in general (asymmetry)

Provenance
----------
Reference: Junker, R. R., Griessenberger, F. & Trutschnig, W. (2021). Estimating
scale-invariant directed dependence of bivariate distributions. Computational
Statistics & Data Analysis 153:107058; Trutschnig, W. (2011). On a strong metric
on the space of copulas and its induced dependence measure. J. Math. Anal. Appl.
384(2):690-705.
Upstream code: R package ``qad`` (CRAN; https://github.com/griefl/qad), GPL-2.
This module: written in Python from the papers; the kernel-integral formulas
were validated against the authors' package. Distributed under GPL-3 (LICENSE).
"""

from typing import Any

import numpy as np
from scipy import stats


def build_checkerboard_weights(
    X: np.ndarray, Y: np.ndarray, resolution: int
) -> np.ndarray:
    """
    Build the mass matrix of the empirical checkerboard copula.

    This function computes the empirical checkerboard copula by aggregating
    the empirical copula into N×N squares. It handles ties properly.

    Parameters
    ----------
    X : np.ndarray
        Ranks of the first variable (1-indexed)
    Y : np.ndarray
        Ranks of the second variable (1-indexed)
    resolution : int
        Number of strips for the checkerboard aggregation

    Returns
    -------
    np.ndarray
        Matrix of shape (resolution, resolution) containing the mass distribution
    """
    sample_size = min(len(X), len(Y))
    result = np.zeros((resolution, resolution))

    # Count occurrences of each rank (handle ties)
    x_range = np.bincount(X.astype(int))[1:]  # Skip index 0 since ranks are 1-indexed
    y_range = np.bincount(Y.astype(int))[1:]

    # Count occurrences of each pair of ranks
    pairs = {}
    for i in range(sample_size):
        rx = x_range[int(X[i]) - 1]
        ry = y_range[int(Y[i]) - 1]

        if rx > 1 and ry > 1:
            key = (X[i], Y[i])
            if key not in pairs:
                pairs[key] = 0
            pairs[key] += 1

    # Build the checkerboard weights
    for i in range(sample_size):
        rx = x_range[int(X[i]) - 1]
        ry = y_range[int(Y[i]) - 1]

        # Handle pairs
        if rx > 1 and ry > 1:
            key = (X[i], Y[i])
            rp = pairs.get(key, 0)
            pairs[key] = 0  # Mark as processed
        else:
            rp = 1

        if rp != 0:
            # Calculate which checkerboard squares this observation contributes to
            x_upper = int(np.ceil(X[i] / sample_size * resolution))
            x_lower = int(max(np.ceil((X[i] - rx) / sample_size * resolution), 1))
            y_upper = int(np.ceil(Y[i] / sample_size * resolution))
            y_lower = int(max(np.ceil((Y[i] - ry) / sample_size * resolution), 1))

            for x in range(x_lower, x_upper + 1):
                lambda_x = min(X[i], x / resolution * sample_size) - max(
                    X[i] - rx, (x - 1) / resolution * sample_size
                )

                for y in range(y_lower, y_upper + 1):
                    lambda_y = min(Y[i], y / resolution * sample_size) - max(
                        Y[i] - ry, (y - 1) / resolution * sample_size
                    )

                    result[x - 1, y - 1] += (lambda_x * lambda_y * rp) / (
                        sample_size * rx * ry
                    )

    return np.round(result, 15)  # Round to avoid numerical errors


def build_checkerboard_weights_vectorized(
    X: np.ndarray, Y: np.ndarray, resolution: int
) -> np.ndarray:
    """
    Vectorized variant of :func:`build_checkerboard_weights`.

    Uses a fully vectorized fast path when neither marginal has ties, and
    delegates to :func:`build_checkerboard_weights` otherwise. The output
    matches the original function to floating-point precision.

    Parameters
    ----------
    X : np.ndarray
        Ranks of the first variable (1-indexed).
    Y : np.ndarray
        Ranks of the second variable (1-indexed).
    resolution : int
        Number of strips for the checkerboard aggregation.

    Returns
    -------
    np.ndarray
        Matrix of shape (resolution, resolution) containing the mass distribution.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    n = min(len(X), len(Y))

    x_range = np.bincount(X.astype(int), minlength=n + 1)[1:]
    y_range = np.bincount(Y.astype(int), minlength=n + 1)[1:]

    if x_range.max() <= 1 and y_range.max() <= 1:
        return _build_cb_no_ties(X, Y, n, resolution)
    return build_checkerboard_weights(X, Y, resolution)


def _build_cb_no_ties(X: np.ndarray, Y: np.ndarray, n: int, N: int) -> np.ndarray:
    """
    Vectorized checkerboard build for the no-ties case.

    Each observation contributes mass 1/n spread uniformly over a
    (1/n) x (1/n) rectangle that straddles at most 2 cells per axis,
    giving up to 4 cell contributions accumulated via ``np.add.at``.
    """
    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)
    scale = N / n

    # 1-indexed cells containing X/n and (X-1)/n on the unit interval.
    # Clip the upper bound to N to guard against floating-point drift in ceil()
    # when X*scale should equal N exactly (e.g. X == n).
    x_upper = np.minimum(np.ceil(X * scale).astype(np.intp), N)
    x_lower = np.maximum(np.ceil((X - 1) * scale).astype(np.intp), 1)
    y_upper = np.minimum(np.ceil(Y * scale).astype(np.intp), N)
    y_lower = np.maximum(np.ceil((Y - 1) * scale).astype(np.intp), 1)

    # Cell boundary between x_lower and x_upper, expressed in rank units
    t_x = x_lower / scale
    t_y = y_lower / scale

    # Unified lambda formula:
    #   single-cell obs (x_lower == x_upper): t_x >= X, so low = 1, high = 0
    #   two-cell obs: X-1 < t_x < X, so low and high are both positive and sum to 1
    split_x = np.minimum(X, t_x)
    split_y = np.minimum(Y, t_y)
    lambda_x_low = split_x - (X - 1)
    lambda_x_high = X - split_x
    lambda_y_low = split_y - (Y - 1)
    lambda_y_high = Y - split_y

    result = np.zeros((N, N))
    inv_n = 1.0 / n

    # Scatter-accumulate the 4 corner contributions; zero entries are harmless.
    np.add.at(result, (x_lower - 1, y_lower - 1), lambda_x_low * lambda_y_low * inv_n)
    np.add.at(result, (x_lower - 1, y_upper - 1), lambda_x_low * lambda_y_high * inv_n)
    np.add.at(result, (x_upper - 1, y_lower - 1), lambda_x_high * lambda_y_low * inv_n)
    np.add.at(result, (x_upper - 1, y_upper - 1), lambda_x_high * lambda_y_high * inv_n)

    return np.round(result, 15)


def local_kernel_integral(A: np.ndarray, x: int, y: int, N: int, y_sum: float) -> float:
    """
    Compute the local kernel integral for the D1 distance calculation.

    Parameters
    ----------
    A : np.ndarray
        Mass matrix of the checkerboard copula
    x : int
        Row index (1-indexed)
    y : int
        Column index (1-indexed)
    N : int
        Resolution
    y_sum : float
        Cumulative sum up to previous column

    Returns
    -------
    float
        Local kernel integral value
    """
    ak = A[x - 1, y - 1]
    d = (y_sum + ak - y * ak) * N
    k = (ak * N) * N

    i_D = d
    k_D = k - 1

    # Check if the linear function crosses zero
    if (i_D + k_D * (y - 1) / N) * (i_D + k_D * y / N) >= 0:
        # No crossing - simple integral
        return abs(
            i_D / N + k_D / 2.0 * ((y * y) / (N * N) - ((y - 1) * (y - 1)) / (N * N))
        )
    else:
        # Crossing zero - split integral. Formulas follow Junker et al. (2021), validated against the qad package.
        y0 = i_D / -k_D
        I1 = abs(
            i_D * (y0 - (y - 1) / N)
            + k_D / 2.0 * (y0 * y0 - ((y - 1) * (y - 1)) / (N * N))
        )
        I2 = abs(i_D * (y / N - y0) + k_D / 2.0 * ((y * y) / (N * N) - y0 * y0))
        return I1 + I2


def D1_Pi(A: np.ndarray, resolution: int) -> float:
    """
    Compute the D1 distance between a checkerboard copula and the independence copula.

    This is the core calculation for the qad measure. It computes the distance
    between the conditional distributions of the checkerboard copula and the
    uniform distribution (independence).

    Parameters
    ----------
    A : np.ndarray
        Mass matrix of the checkerboard copula
    resolution : int
        Resolution of the checkerboard

    Returns
    -------
    float
        D1 distance value
    """
    result = 0.0

    for x in range(resolution):
        y_sum = 0.0
        for y in range(resolution):
            result += local_kernel_integral(A, x + 1, y + 1, resolution, y_sum)
            y_sum += A[x, y]

    return result / resolution


def D1_Pi_vectorized(A: np.ndarray, resolution: int) -> float:
    """
    Vectorized equivalent of :func:`D1_Pi`.

    Computes the D1 distance between a checkerboard copula and the
    independence copula fully in numpy, avoiding the nested Python loop
    in the original. Matches :func:`D1_Pi` to floating-point precision.

    Parameters
    ----------
    A : np.ndarray
        Mass matrix of the checkerboard copula, shape (N, N).
    resolution : int
        Resolution of the checkerboard.

    Returns
    -------
    float
        D1 distance value.
    """
    N = resolution
    A = np.asarray(A, dtype=np.float64)

    # y_sum[x, y] = sum of A[x, 0..y-1]  (exclusive row-wise cumsum)
    # Build via prepended zeros + cumsum of the leading N-1 columns so the
    # sum matches the sequential "y_sum += A[x, y]" order of D1_Pi bit-for-bit
    # (computing np.cumsum(A) - A would introduce a subtraction error that
    # can flip the no_cross classification when lo*hi is near zero).
    y_sum_mat = np.zeros_like(A)
    if N > 1:
        y_sum_mat[:, 1:] = np.cumsum(A[:, :-1], axis=1)

    # 1-indexed column label used inside local_kernel_integral
    y_idx = np.arange(1, N + 1, dtype=np.float64)

    # Match D1_Pi's float operation order bit-for-bit:
    #   d = (y_sum + ak - y*ak) * N        (old)
    #   k = (ak * N) * N                    (old — two multiplications, NOT A*(N*N))
    # The (ak*N)*N vs A*(N*N) distinction matters: both are mathematically
    # equal but differ by 1 ulp per cell, which can flip the no_cross test
    # when lo*hi is near zero (e.g. on exact-rank permutation inputs).
    i_D = (y_sum_mat + A - y_idx * A) * N
    k_D = (A * N) * N - 1.0

    # Linear function f(t) = i_D + k_D * t on t in [(y-1)/N, y/N]
    lo = i_D + k_D * (y_idx - 1.0) / N
    hi = i_D + k_D * y_idx / N
    no_cross = lo * hi >= 0.0

    # No sign change: |integral of f| over the cell
    integral_no_cross = np.abs(i_D / N + k_D * (2.0 * y_idx - 1.0) / (2.0 * N * N))

    # Sign change: split at y0 = -i_D / k_D. k_D != 0 wherever this branch
    # is selected (k_D == 0 gives a constant f, handled by no_cross).
    # Formulas follow Junker et al. (2021), validated against the qad package (local_kernel_integral).
    with np.errstate(divide="ignore", invalid="ignore"):
        y0 = i_D / -k_D
        I1 = np.abs(
            i_D * (y0 - (y_idx - 1.0) / N)
            + k_D / 2.0 * (y0 * y0 - ((y_idx - 1.0) * (y_idx - 1.0)) / (N * N))
        )
        I2 = np.abs(
            i_D * (y_idx / N - y0) + k_D / 2.0 * ((y_idx * y_idx) / (N * N) - y0 * y0)
        )
    integral_cross = I1 + I2

    cell = np.where(no_cross, integral_no_cross, integral_cross)
    return float(cell.sum()) / N


def ECBC(x: np.ndarray, y: np.ndarray, resolution: int | None = None) -> np.ndarray:
    """
    Compute the Empirical Checkerboard Copula (ECBC).

    Parameters
    ----------
    x : np.ndarray
        First variable
    y : np.ndarray
        Second variable
    resolution : int | None
        Resolution for the checkerboard. If None, uses floor(sqrt(min(unique values)))

    Returns
    -------
    np.ndarray
        Mass matrix of the empirical checkerboard copula
    """
    if resolution is None:
        sample_size = min(len(np.unique(x)), len(np.unique(y)))
        resolution = int(np.floor(sample_size**0.5))
    else:
        if resolution > len(x):
            resolution = len(x)
        elif resolution < 1:
            resolution = 1

    # Compute ranks (1-indexed, using maximum rank for ties)
    X = stats.rankdata(x, method="max")
    Y = stats.rankdata(y, method="max")

    return build_checkerboard_weights(X, Y, resolution)


def qad(
    x: np.ndarray,
    y: np.ndarray,
    resolution: int | None = None,
    p_value: bool = True,
    nperm: int = 1000,
    p_value_asymmetry: bool = False,
    nboot: int = 1000,
    remove_00: bool = False,
) -> dict[str, Any]:
    """
    Quantification of Asymmetric Dependence.

    Computes the directed dependence measures q(X,Y) and q(Y,X) which quantify
    how much information one variable provides about the other.

    Parameters
    ----------
    x : np.ndarray
        First variable
    y : np.ndarray
        Second variable
    resolution : int | None
        Resolution for checkerboard aggregation. If None, uses floor(sqrt(min unique ranks))
    p_value : bool
        Whether to compute p-value for independence test via permutation
    nperm : int
        Number of permutations for p-value calculation
    p_value_asymmetry : bool
        Whether to compute p-value for asymmetry via bootstrap
    nboot : int
        Number of bootstrap samples
    remove_00 : bool
        Whether to remove (0,0) entries

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'q_xy': q(X,Y) - influence of X on Y
        - 'q_yx': q(Y,X) - influence of Y on X
        - 'max_dependence': max(q(X,Y), q(Y,X))
        - 'asymmetry': q(X,Y) - q(Y,X)
        - 'mass_matrix': mass distribution of the empirical checkerboard copula
        - 'resolution': used resolution
        - 'n': sample size
        - 'p_q_xy': p-value for q(X,Y) (if p_value=True)
        - 'p_q_yx': p-value for q(Y,X) (if p_value=True)
        - 'p_max_dependence': p-value for max dependence (if p_value=True)
        - 'p_asymmetry': p-value for asymmetry (if p_value_asymmetry=True)
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    # Optionally remove (0, 0) entries
    if remove_00:
        mask = ~((x == 0) & (y == 0))
        x = x[mask]
        y = y[mask]

    if len(x) == 0:
        raise ValueError("No valid data points after filtering")

    # Calculate default resolution
    if resolution is None:
        sample_size = min(len(np.unique(x)), len(np.unique(y)))
        resolution = int(np.floor(sample_size**0.5))
    else:
        sample_size = None
        resolution = int(np.floor(resolution))

    # Compute ranks (1-indexed, maximum rank for ties)
    X = stats.rankdata(x, method="max")
    Y = stats.rankdata(y, method="max")

    # Build checkerboard copula (fast path when no ties; falls back otherwise)
    mass_matrix = build_checkerboard_weights_vectorized(X, Y, resolution)

    # Compute dependence measures
    zeta1 = 3 * D1_Pi_vectorized(mass_matrix, resolution)
    zeta1_t = 3 * D1_Pi_vectorized(mass_matrix.T, resolution)
    max_dependence = max(zeta1, zeta1_t)
    asymmetry = zeta1 - zeta1_t

    result = {
        "q_xy": zeta1,
        "q_yx": zeta1_t,
        "max_dependence": max_dependence,
        "asymmetry": asymmetry,
        "mass_matrix": mass_matrix,
        "resolution": resolution,
        "n": len(x),
    }

    # Compute p-values via permutation test
    if p_value:
        # Count ties for proper permutation.
        # n, rx, ry depend only on the observed ranks — constant across permutations.
        n = len(X)
        x_range = np.bincount(X.astype(int))[1:]
        y_range = np.bincount(Y.astype(int))[1:]
        rx = x_range[X.astype(int) - 1]
        ry = y_range[Y.astype(int) - 1]

        zeta1_perm = np.zeros(nperm)
        zeta1_t_perm = np.zeros(nperm)
        max_dep_perm = np.zeros(nperm)

        for i in range(nperm):
            # Permutation that preserves tie structure
            u_new = (X - rx * np.random.uniform(0, 1, n)) / n
            v_new = (Y - ry * np.random.uniform(0, 1, n)) / n

            sample_perm = np.concatenate([u_new, v_new])
            np.random.shuffle(sample_perm)

            x1_perm = sample_perm[:n]
            x2_perm = sample_perm[n:]

            # The jittered values are continuous (distinct a.s.), so ranks with
            # method="max" reduce to ordinal ranks: argsort(argsort) + 1.  This
            # avoids scipy.stats.rankdata's per-call overhead.
            X1 = x1_perm.argsort().argsort() + 1
            X2 = x2_perm.argsort().argsort() + 1

            mass_perm = build_checkerboard_weights_vectorized(X1, X2, resolution)

            zeta1_perm[i] = 3 * D1_Pi_vectorized(mass_perm, resolution)
            zeta1_t_perm[i] = 3 * D1_Pi_vectorized(mass_perm.T, resolution)
            max_dep_perm[i] = max(zeta1_perm[i], zeta1_t_perm[i])

        result["p_q_xy"] = np.mean(zeta1_perm >= zeta1)
        result["p_q_yx"] = np.mean(zeta1_t_perm >= zeta1_t)
        result["p_max_dependence"] = np.mean(max_dep_perm >= max_dependence)

    # Compute p-value for asymmetry via bootstrap
    if p_value_asymmetry:
        n = len(X)
        zeta1_boot = np.zeros(nboot)
        zeta1_t_boot = np.zeros(nboot)
        asymmetry_boot = np.zeros(nboot)

        N = resolution
        mass_matrix0 = mass_matrix.copy()

        for i in range(nboot):
            # Sample from empirical checkerboard copula
            ru_new = np.random.choice(N, n, replace=True)

            # For each ru, sample rv according to conditional distribution
            rv_new = []
            for k in np.unique(ru_new):
                count = np.sum(ru_new == k)
                probs = N * mass_matrix0[k, :]
                probs = probs / probs.sum() if probs.sum() > 0 else np.ones(N) / N
                rv_new.extend(np.random.choice(N, count, replace=True, p=probs))

            ru_new = np.sort(ru_new)
            rv_new = np.array(rv_new)

            u_new = np.random.uniform(0, 1 / N, n) + ru_new / N
            v_new = np.random.uniform(0, 1 / N, n) + rv_new / N

            x1 = stats.rankdata(u_new, method="max")
            x2 = stats.rankdata(v_new, method="max")

            mass_boot = build_checkerboard_weights_vectorized(x1, x2, resolution)

            zeta1_boot[i] = 3 * D1_Pi_vectorized(mass_boot, resolution)
            zeta1_t_boot[i] = 3 * D1_Pi_vectorized(mass_boot.T, resolution)
            asymmetry_boot[i] = zeta1_boot[i] - zeta1_t_boot[i]

        # Two-sided test for asymmetry
        result["p_asymmetry"] = 2 * min(
            np.mean(asymmetry_boot >= asymmetry), np.mean(asymmetry_boot <= asymmetry)
        )

    # Warning for low resolution
    if resolution <= 3:
        import warnings

        warnings.warn(
            "Resolution is less or equal to 3. Results must be interpreted with caution!"
        )

    return result


def qad_null_stats(
    n: int,
    resolution: int,
    nperm: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Permutation null of the qad statistics for continuous (tie-free) data.

    For continuous data the ranks are a permutation of 1..n with all tie
    counts equal to 1, so the permutation null of :func:`qad` depends only on
    ``(n, resolution)`` — not on the data (distribution-free).  This function
    generates that null with the exact same jitter + pooled-shuffle + re-rank
    scheme as the ``p_value=True`` loop in :func:`qad`, but from a local
    ``Generator`` so it can be cached and shared across benchmark
    replications.

    Do NOT use for data with ties: there the null depends on the observed tie
    structure and must be generated by :func:`qad` itself.

    Returns
    -------
    dict with keys ``"q_xy"``, ``"q_yx"``, ``"max_dependence"``, each an
    array of ``nperm`` null statistic values.
    """
    ranks = np.arange(1, n + 1, dtype=np.float64)

    zeta1_perm = np.zeros(nperm)
    zeta1_t_perm = np.zeros(nperm)

    for i in range(nperm):
        # No ties: rx = ry = 1, and the pooled jittered sample is the same in
        # distribution regardless of the rank order, so canonical 1..n ranks
        # generate the identical null.
        u_new = (ranks - rng.uniform(0, 1, n)) / n
        v_new = (ranks - rng.uniform(0, 1, n)) / n

        sample_perm = np.concatenate([u_new, v_new])
        rng.shuffle(sample_perm)

        X1 = sample_perm[:n].argsort().argsort() + 1
        X2 = sample_perm[n:].argsort().argsort() + 1

        mass_perm = build_checkerboard_weights_vectorized(X1, X2, resolution)

        zeta1_perm[i] = 3 * D1_Pi_vectorized(mass_perm, resolution)
        zeta1_t_perm[i] = 3 * D1_Pi_vectorized(mass_perm.T, resolution)

    return {
        "q_xy": zeta1_perm,
        "q_yx": zeta1_t_perm,
        "max_dependence": np.maximum(zeta1_perm, zeta1_t_perm),
    }


def print_qad_results(
    result: dict[str, Any], variable_names: tuple[str, str] = ("X", "Y")
):
    """
    Print qad results in a readable format.

    Parameters
    ----------
    result : dict[str, Any]
        Result dictionary from qad()
    variable_names : tuple[str, str]
        Names of the variables
    """
    x_name, y_name = variable_names

    print("\nQuantification of Asymmetric Dependence:")
    print("=" * 50)
    print(f"\nData: x1 := {x_name}")
    print(f"      x2 := {y_name}")
    print(f"\nSample Size: {result['n']}")
    print(f"Resolution: {result['resolution']} x {result['resolution']}")
    print("\nDependence measures:")
    print(f"  q(x1,x2)       {result['q_xy']:.3f}", end="")
    if "p_q_xy" in result:
        print(f"    p-value: {result['p_q_xy']:.3f}")
    else:
        print()

    print(f"  q(x2,x1)       {result['q_yx']:.3f}", end="")
    if "p_q_yx" in result:
        print(f"    p-value: {result['p_q_yx']:.3f}")
    else:
        print()

    print(f"  max.dependence {result['max_dependence']:.3f}", end="")
    if "p_max_dependence" in result:
        print(f"    p-value: {result['p_max_dependence']:.3f}")
    else:
        print()

    print("\nAsymmetry:")
    print(f"  a = q(x1,x2) - q(x2,x1) = {result['asymmetry']:.3f}", end="")
    if "p_asymmetry" in result:
        print(f"    p-value: {result['p_asymmetry']:.3f}")
    else:
        print()
    print()


# Additional utility functions


def zeta1(x: np.ndarray, y: np.ndarray, resolution: int | None = None) -> float:
    """
    Compute the zeta1 dependence measure (directed dependence from X to Y).

    This is a simplified interface that just returns the q(X,Y) value.

    Parameters
    ----------
    x : np.ndarray
        First variable
    y : np.ndarray
        Second variable
    resolution : int | None
        Resolution for checkerboard aggregation

    Returns
    -------
    float
        The zeta1 dependence measure
    """
    result = qad(x, y, resolution=resolution, p_value=False)
    return result["q_xy"]


def D1(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    resolution: int | None = None,
) -> float:
    """
    Compute the D1 distance between two bivariate distributions.

    Parameters
    ----------
    x1, y1 : np.ndarray
        First bivariate sample
    x2, y2 : np.ndarray
        Second bivariate sample
    resolution : int | None
        Resolution for checkerboard aggregation

    Returns
    -------
    float
        D1 distance
    """
    if resolution is None:
        resolution = int(
            np.floor(
                min(
                    len(np.unique(x1)),
                    len(np.unique(y1)),
                    len(np.unique(x2)),
                    len(np.unique(y2)),
                )
                ** 0.5
            )
        )

    # Build checkerboard copulas for both samples
    X1 = stats.rankdata(x1, method="max")
    Y1 = stats.rankdata(y1, method="max")
    A1 = build_checkerboard_weights(X1, Y1, resolution)

    X2 = stats.rankdata(x2, method="max")
    Y2 = stats.rankdata(y2, method="max")
    A2 = build_checkerboard_weights(X2, Y2, resolution)

    # Compute D1 distance between the two copulas
    return D1_ECBC(A1, A2)


def D1_ECBC(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute D1 distance between two empirical checkerboard copulas.

    Parameters
    ----------
    A, B : np.ndarray
        Mass matrices of two checkerboard copulas (same dimension)

    Returns
    -------
    float
        D1 distance
    """
    if A.shape != B.shape:
        raise ValueError("Checkerboard copulas must have the same dimensions")

    N = A.shape[0]
    result = 0.0

    for i in range(N):
        yA_sum1 = 0.0
        yB_sum1 = 0.0

        for j in range(N):
            yA_sum2 = N * A[i, j] + yA_sum1
            yB_sum2 = N * B[i, j] + yB_sum1

            # Check if conditional distributions cross
            if (yA_sum2 - yB_sum2) * (yA_sum1 - yB_sum1) >= 0:
                # No crossing
                I1 = (yA_sum2 - yA_sum1) / (2 * N) + yA_sum1 / N
                I2 = (yB_sum2 - yB_sum1) / (2 * N) + yB_sum1 / N
                yD = abs(I1 - I2)
            else:
                # Crossing - need to split the integral
                x_intersect = (yA_sum1 - yB_sum1) / (
                    ((yB_sum2 - yB_sum1) - (yA_sum2 - yA_sum1)) / (1 / N)
                )
                y_intersect = yA_sum1 + N * (yA_sum2 - yA_sum1) * x_intersect

                I11 = (
                    (y_intersect - yA_sum1) * x_intersect
                ) / 2 + yA_sum1 * x_intersect
                I12 = (
                    (y_intersect - yB_sum1) * x_intersect
                ) / 2 + yB_sum1 * x_intersect
                I21 = (
                    (yA_sum2 - y_intersect) * (1 / N - x_intersect)
                ) / 2 + y_intersect * (1 / N - x_intersect)
                I22 = (
                    (yB_sum2 - y_intersect) * (1 / N - x_intersect)
                ) / 2 + y_intersect * (1 / N - x_intersect)

                yD = abs(I11 - I12) + abs(I21 - I22)

            result += yD
            yA_sum1 = yA_sum2
            yB_sum1 = yB_sum2

    return result / N
