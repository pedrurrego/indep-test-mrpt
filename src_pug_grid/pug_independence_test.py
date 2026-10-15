"""MRPT: the multi-resolution Pearson test of independence on a ladder of square grids.

The data are mapped to pseudo-observations ``(u, v)`` by the marginal empirical
distribution functions, binned on a regular ``k x k`` grid of the unit square,
and the cell counts are compared with the uniform expectation ``n / k^2``:

    T(k) = sum_ij (c_ij / n - 1 / k^2)^2 ,

an affine image of Pearson's chi-squared statistic for the ``k x k`` table
(``chi^2(k) = n k^2 T(k)``).  The test evaluates ``T`` on a ladder of
resolutions ``K(n)`` (``resolve_ladder``; by default ``{3, 5, 8, 12}`` below
``floor(sqrt n)`` together with ``floor(sqrt n)``), obtains one permutation
p-value per rung from a null shared across rungs, and combines them by
Bonferroni.  ``norm="density"`` runs the single-grid statistic at one ``k``.

Tie handling.  ``T`` is an affine function of the integer ``S = sum(c_ij^2)``
(``T = S / n^2 - 1 / k^2``), so observed and permuted values are compared on
``S`` itself and every null draw whose cell counts are a rearrangement of the
observed ones counts as ``>=`` -- the conservative permutation convention,
exact to the last bit.  The reported ``statistic`` is ``T``.

The permutation null has two interchangeable implementations, selected with
``impl``:

    "vectorized" : all permutations of a block at once via a single
                   ``np.bincount`` (default, ~5-10x faster);
    "loop"       : one permutation at a time -- the readable reference.

Both draw the same permutations from the same seed and reduce the statistic
in the same order, so their statistics and p-values are identical to the last
bit (``test_pug_grid_impl.py``).
"""

from typing import Any, Literal

import numpy as np

from src_pug_grid.utils import bin_counts, calculate_copula_values

Norm = Literal["density", "density-multi"]
Impl = Literal["loop", "vectorized"]

# Ladder specification: rungs joined by "-".  An integer is a fixed resolution,
# kept only while it is below floor(sqrt(n)) (cells with expected count >= 1);
# "sqrt" is the adaptive top rung floor(sqrt(n)).  The default is the ladder of
# the paper: sorted({3,5,8,12 : < floor(sqrt n)} | {floor(sqrt n)}).
DEFAULT_LADDER = "3-5-8-12-sqrt"


def resolve_ladder(spec: str, n: int) -> list[int]:
    """Resolve a ladder specification into the sorted resolutions used at size ``n``.

    ``"3-5-8-12-sqrt"`` (the default) gives ``{3,5,8,12 below floor(sqrt n)} +
    {floor(sqrt n)}``; ``"3-sqrt"`` the sparse endpoints-only ladder; ``"3-5-8-12"``
    a ladder with no adaptive top rung; ``"all"`` the dense ladder ``{2, ..., floor(sqrt n)}``.
    """
    k_sqrt = int(np.floor(np.sqrt(n)))
    rungs: set[int] = set()
    for token in spec.split("-"):
        token = token.strip()
        if not token:
            continue
        if token == "sqrt":
            rungs.add(k_sqrt)
        elif (
            token == "all"
        ):  # the dense ladder: every resolution from 2 to floor(sqrt n)
            rungs.update(range(2, k_sqrt + 1))
        elif token.isdigit():
            k = int(token)
            if k < 2:
                raise ValueError(f"ladder rung must be >= 2, got {k!r}")
            if k < k_sqrt:
                rungs.add(k)
        else:
            raise ValueError(
                f"unknown ladder token {token!r} in {spec!r}; use integers, 'sqrt' or 'all'"
            )
    if not rungs:
        raise ValueError(f"ladder {spec!r} has no admissible rung at n={n}")
    return sorted(rungs)


# --- Statistic ----------------------------------------------------------------


def _density_int(u: np.ndarray, v: np.ndarray, k: int) -> int:
    """``S = sum(c_ij^2)``: the integer core of the statistic at resolution ``k``.

    ``T = S/n^2 - 1/k^2`` is increasing in ``S``, so ``S`` orders observed and
    permuted samples exactly, with no floating-point ties to break.
    """
    counts = bin_counts(u, v, k).astype(np.int64)
    return int(np.sum(counts * counts))


def _density_from_int(S, k: int, n: int):
    """Map ``S = sum(c_ij^2)`` (int or int array) to ``T = sum((c_ij/n - 1/k^2)^2)``."""
    return S / (n * n) - 1.0 / (k * k)


def _compute_density_statistic(u: np.ndarray, v: np.ndarray, k: int, n: int) -> float:
    """``T(k)`` at a single resolution."""
    return float(_density_from_int(_density_int(u, v, k), k, n))


def _compute_multi_density(
    u: np.ndarray, v: np.ndarray, k_values: list[int], n: int
) -> dict[int, float]:
    """``T(k)`` at each resolution of the ladder."""
    return {k: _compute_density_statistic(u, v, k, n) for k in k_values}


# --- Permutation null: two interchangeable implementations ------------------


def _permutation_null_loop(
    u: np.ndarray,
    v: np.ndarray,
    k_values: list[int],
    n: int,
    n_reps: int,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    """Null values of ``S`` at each resolution, one permutation at a time.

    Readable reference implementation of ``_permutation_null_vectorized``:
    same output, ~5-10x slower.  The same permutation is shared across
    resolutions.
    """
    perm_stats = {k: np.zeros(n_reps, dtype=np.int64) for k in k_values}
    for i in range(n_reps):
        v_perm = rng.permutation(v)
        for k in k_values:
            perm_stats[k][i] = _density_int(u, v_perm, k)
    return perm_stats


# Upper bound on the number of int64 elements of the largest temporary that the
# vectorised null holds at once (the (block, n) permutation array and its
# derived cell-code arrays).  2**26 elements = 512 MiB per array, so a cold
# call peaks at roughly 2 GiB whatever n and n_reps are.  Permutations are
# drawn block by block from the same generator, which yields exactly the same
# stream as one (n_reps, n) draw (numpy permutes the rows of an axis=1
# ``permuted`` call sequentially), so the block size never changes a result;
# ``test_pug_grid_impl.py`` checks this.  Every paper-facing run (n <= 4000,
# B <= 2000) fits in a single block.
VECTORIZED_BLOCK_ELEMENTS = 2**26


def _permutation_null_vectorized(
    u: np.ndarray,
    v: np.ndarray,
    k_values: list[int],
    n: int,
    n_reps: int,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    """Null values of ``S`` at each resolution, vectorised over permutations.

    Binning is elementwise, so bin(v[perm]) == bin(v)[perm]: each margin is
    binned once per resolution and only the integer bin indices are permuted.
    The histograms of a block of permutations are computed in a single
    ``np.bincount`` call by offsetting each replicate's cell codes into its
    own block of length k*k.  The same permutations are shared across
    resolutions, matching the loop implementation.  Permutations are
    processed in blocks of at most ``VECTORIZED_BLOCK_ELEMENTS // n``
    replicates, so memory is O(min(n_reps, VECTORIZED_BLOCK_ELEMENTS / n) * n)
    instead of O(n_reps * n); the block size does not affect the result (see
    the constant's note).
    """
    block = max(1, min(n_reps, VECTORIZED_BLOCK_ELEMENTS // max(n, 1)))
    bins = {}
    for k in k_values:
        # bin each margin: maps each value in [0,1] to a bin index in
        # {0, ..., k-1} (u * k is in [0, k])
        bin_x = np.clip(np.ceil(u * k).astype(np.int64) - 1, 0, k - 1)
        bin_y = np.clip(np.ceil(v * k).astype(np.int64) - 1, 0, k - 1)
        bins[k] = (bin_x, bin_y)

    parts: dict[int, list[np.ndarray]] = {k: [] for k in k_values}
    for start in range(0, n_reps, block):
        m = min(block, n_reps - start)
        # (m, n) array of permuted indices
        perms = rng.permuted(np.tile(np.arange(n), (m, 1)), axis=1)
        offsets = np.arange(m)[:, None]
        for k in k_values:
            bin_x, bin_y = bins[k]
            # codes[r, i]: which of the k^2 cells point i falls into under
            # permutation r, shifted into replicate r's own block of codes.
            codes = bin_x[None, :] * k + bin_y[perms]  # (m, n)
            codes += offsets * (k * k)
            counts = np.bincount(codes.ravel(), minlength=m * k * k).reshape(m, k * k)
            parts[k].append(np.sum(counts * counts, axis=1))  # integer S
    return {k: np.concatenate(parts[k]) for k in k_values}


_NULL_IMPLS = {
    "loop": _permutation_null_loop,
    "vectorized": _permutation_null_vectorized,
}


def _permutation_null(
    u: np.ndarray,
    v: np.ndarray,
    k_values: list[int],
    n: int,
    n_reps: int,
    rng: np.random.Generator,
    impl: Impl = "vectorized",
) -> dict[int, np.ndarray]:
    """Dispatch to the selected permutation-null implementation."""
    try:
        null_fn = _NULL_IMPLS[impl]
    except KeyError:
        raise ValueError(
            f"impl must be one of {sorted(_NULL_IMPLS)}, got {impl!r}"
        ) from None
    return null_fn(u, v, k_values, n, n_reps, rng)


# --- Tests ------------------------------------------------------------------


def _indep_test_multi_density(
    X: np.ndarray,
    n_reps: int,
    seed: int,
    impl: Impl = "vectorized",
    ladder: str = DEFAULT_LADDER,
) -> dict[str, Any]:
    """MRPT: the statistic on the ladder, one p-value per rung, Bonferroni."""
    n = len(X)
    k_values = resolve_ladder(ladder, n)

    rng = np.random.default_rng(seed=seed)
    u, v = calculate_copula_values(X)

    # Observed integer statistics S at each resolution
    S_obs = {k: _density_int(u, v, k) for k in k_values}
    T_obs = {k: _density_from_int(S_obs[k], k, n) for k in k_values}

    # Permutation null (integer S per replicate)
    perm_stats = _permutation_null(u, v, k_values, n, n_reps, rng, impl)

    # Per-resolution p-values, compared on the integers (exact ties)
    p_per_k = {}
    for k in k_values:
        p_per_k[k] = float((np.sum(perm_stats[k] >= S_obs[k]) + 1) / (n_reps + 1))

    # Bonferroni correction
    m = len(k_values)
    p_min = min(p_per_k.values())
    p_corrected = min(1.0, m * p_min)

    return {
        "statistic": max(T_obs.values()),
        "p-value": p_corrected,
        "p-value-raw": p_min,
        "k_values": k_values,
        "p_per_k": p_per_k,
        "T_per_k": {k: float(v) for k, v in T_obs.items()},
        "norm": "density-multi",
    }


def indep_test(
    *,
    X: np.ndarray,
    n_reps: int = 500,
    k: int | None = None,
    norm: Norm = "density-multi",
    seed: int = 1234,
    impl: Impl = "vectorized",
    ladder: str = DEFAULT_LADDER,
) -> dict[str, Any]:
    """Run the test.

    Parameters
    ----------
    X : np.ndarray, shape (n, 2)
        Bivariate data matrix.
    n_reps : int
        Number of permutation replicates ``B``.
    norm : str
        ``"density-multi"`` (default): MRPT, the statistic on the resolution
        ladder combined by Bonferroni.  ``"density"``: the statistic on a
        single ``k x k`` grid.
    k : int | None
        Grid resolution for ``norm="density"``; ``floor(sqrt(n))`` if *None*.
        Ignored by ``"density-multi"``.
    seed : int
        Seed of the permutation null.
    impl : str
        Permutation-null implementation, ``"vectorized"`` (default) or
        ``"loop"``; same seed -> identical statistics and p-values.
    ladder : str
        Resolution ladder for ``"density-multi"`` (see :func:`resolve_ladder`);
        ignored by ``"density"``.  The default is the ladder of the paper.

    Returns
    -------
    dict with keys ``statistic`` (observed ``T``), ``p-value`` and ``norm``;
    ``k`` for the single grid, or ``k_values`` / ``p_per_k`` / ``T_per_k`` /
    ``p-value-raw`` (uncorrected minimum) for the ladder.
    """
    assert X.ndim == 2, (
        f"Input array must be 2-dimensional, got {X.ndim}-dimensional array instead."
    )
    if norm == "density-multi":
        return _indep_test_multi_density(X, n_reps, seed, impl=impl, ladder=ladder)
    if norm != "density":
        raise ValueError(f"norm must be 'density' or 'density-multi', got {norm!r}")

    n = len(X)
    if k is None:
        k = int(np.floor(np.sqrt(n)))

    rng = np.random.default_rng(seed=seed)

    # Pseudo-observations are computed once; permuting y permutes v, because
    # the marginal ECDF is invariant to reordering.
    u, v = calculate_copula_values(X)

    S_obs = _density_int(u, v, k)
    perm_stats = _permutation_null(u, v, [k], n, n_reps, rng, impl)[k]
    p_value = (np.sum(perm_stats >= S_obs) + 1) / (n_reps + 1)

    return {
        "statistic": float(_density_from_int(S_obs, k, n)),
        "p-value": float(p_value),
        "k": k,
        "norm": norm,
    }
