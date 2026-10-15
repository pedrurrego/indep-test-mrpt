"""``impl="loop"`` and ``impl="vectorized"`` must agree to the last bit.

Run with ``uv run --with pytest pytest src_pug_grid`` (pytest is not a project dependency).
"""

import numpy as np
import pytest

from src_pug_grid.pug_independence_test import (
    DEFAULT_LADDER,
    indep_test,
    resolve_ladder,
)

_NOISE = [0.05, 0.3, 1.0, 10.0]  # strong -> no dependence, so p-values span (0, 1]


def _cases():
    rng = np.random.default_rng(2026)
    for n in (20, 50, 100, 200, 500):
        for sigma in _NOISE:
            x = rng.random(n)
            y = x + rng.normal(0.0, sigma, n)
            yield n, sigma, np.column_stack((x, y)), int(rng.integers(1_000_000))


@pytest.mark.parametrize(
    "norm,k",
    [
        ("density", None),
        ("density", 3),
        ("density", 12),
        ("density-multi", None),
    ],
)
def test_loop_and_vectorized_agree(norm, k):
    for n, sigma, X, seed in _cases():
        loop = indep_test(X=X, n_reps=200, k=k, norm=norm, seed=seed, impl="loop")
        vec = indep_test(X=X, n_reps=200, k=k, norm=norm, seed=seed, impl="vectorized")
        assert loop["statistic"] == vec["statistic"], (n, sigma, norm, k)
        assert loop["p-value"] == vec["p-value"], (n, sigma, norm, k)
        if norm == "density-multi":
            assert loop["p_per_k"] == vec["p_per_k"], (n, sigma)
            assert loop["k_values"] == vec["k_values"]


def test_unknown_impl_rejected():
    X = np.random.default_rng(0).random((30, 2))
    with pytest.raises(ValueError, match="impl must be one of"):
        indep_test(X=X, n_reps=10, norm="density", impl="numba")


def test_unknown_norm_rejected():
    X = np.random.default_rng(0).random((30, 2))
    with pytest.raises(ValueError, match="norm must be"):
        indep_test(X=X, n_reps=10, norm="l2")


def test_resolve_ladder():
    assert resolve_ladder(DEFAULT_LADDER, 500) == [3, 5, 8, 12, 22]
    assert resolve_ladder(DEFAULT_LADDER, 200) == [3, 5, 8, 12, 14]
    assert resolve_ladder(DEFAULT_LADDER, 100) == [3, 5, 8, 10]
    assert resolve_ladder(DEFAULT_LADDER, 50) == [3, 5, 7]
    assert resolve_ladder(DEFAULT_LADDER, 20) == [3, 4]
    assert resolve_ladder("3-sqrt", 500) == [3, 22]
    assert resolve_ladder("3-sqrt", 20) == [3, 4]
    assert resolve_ladder("3-5-8-12", 500) == [3, 5, 8, 12]
    assert resolve_ladder("12-sqrt-3", 500) == [3, 12, 22]  # order-free, deduplicated
    assert resolve_ladder("all", 500) == list(range(2, 23))
    assert resolve_ladder("all", 20) == [2, 3, 4]
    with pytest.raises(ValueError, match="unknown ladder token"):
        resolve_ladder("3-half", 500)
    with pytest.raises(ValueError, match="no admissible rung"):
        resolve_ladder("12", 100)  # 12 >= floor(sqrt(100)) = 10


def test_default_ladder_matches_omitting_it():
    for n, sigma, X, seed in _cases():
        a = indep_test(X=X, n_reps=200, norm="density-multi", seed=seed)
        b = indep_test(
            X=X, n_reps=200, norm="density-multi", seed=seed, ladder=DEFAULT_LADDER
        )
        assert a == b, (n, sigma)


def test_sparse_ladder_loop_and_vectorized_agree():
    for n, sigma, X, seed in _cases():
        loop = indep_test(
            X=X,
            n_reps=200,
            norm="density-multi",
            seed=seed,
            impl="loop",
            ladder="3-sqrt",
        )
        vec = indep_test(
            X=X,
            n_reps=200,
            norm="density-multi",
            seed=seed,
            impl="vectorized",
            ladder="3-sqrt",
        )
        assert loop == vec, (n, sigma)
        assert vec["k_values"] == resolve_ladder("3-sqrt", n)


@pytest.mark.parametrize(
    "norm,k", [("density", None), ("density", 5), ("density-multi", None)]
)
def test_vectorized_block_size_does_not_change_results(norm, k, monkeypatch):
    """The blocked null (memory bound) must equal the single-block draw bit for bit."""
    import src_pug_grid.pug_independence_test as mod

    for n, sigma, X, seed in _cases():
        monkeypatch.setattr(mod, "VECTORIZED_BLOCK_ELEMENTS", 2**26)  # one block
        one = indep_test(X=X, n_reps=200, k=k, norm=norm, seed=seed, impl="vectorized")
        for block_elems in (n, 7 * n, 1):  # 1, 7 and (1 replicate) per block
            monkeypatch.setattr(mod, "VECTORIZED_BLOCK_ELEMENTS", block_elems)
            many = indep_test(
                X=X, n_reps=200, k=k, norm=norm, seed=seed, impl="vectorized"
            )
            assert one["statistic"] == many["statistic"], (
                n,
                sigma,
                norm,
                k,
                block_elems,
            )
            assert one["p-value"] == many["p-value"], (n, sigma, norm, k, block_elems)
            if norm == "density-multi":
                assert one["p_per_k"] == many["p_per_k"], (n, sigma, block_elems)
