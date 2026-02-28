import numpy as np
import pytest

from quantumhall_matrixelements import get_exchange_kernels, get_exchange_kernels_compressed


def test_materialize_guard_auto_raises():
    Gs = np.array([0.0, 1.0], dtype=float)
    thetas = np.zeros_like(Gs)

    with pytest.raises(MemoryError):
        get_exchange_kernels(
            Gs,
            thetas,
            3,
            method="laguerre",
            nquad=40,
            materialize_limit_bytes=1,
        )


def test_materialize_limit_none_bypasses_guard():
    Gs = np.array([0.0], dtype=float)
    thetas = np.array([0.0], dtype=float)

    X = get_exchange_kernels(
        Gs,
        thetas,
        2,
        method="laguerre",
        nquad=40,
        materialize_limit_bytes=None,
    )
    assert X.shape == (1, 2, 2, 2, 2)


def test_canonical_select_guard_raises():
    Gs = np.array([0.0], dtype=float)
    thetas = np.array([0.0], dtype=float)

    with pytest.raises(MemoryError):
        get_exchange_kernels_compressed(
            Gs,
            thetas,
            2,
            method="laguerre",
            nquad=40,
            canonical_select_max_entries=1,
        )


def test_canonical_select_guard_is_bypassed_for_explicit_select():
    Gs = np.array([0.0], dtype=float)
    thetas = np.array([0.0], dtype=float)

    values, select_list = get_exchange_kernels_compressed(
        Gs,
        thetas,
        2,
        method="laguerre",
        nquad=40,
        select=[(0, 0, 0, 0)],
        canonical_select_max_entries=1,
    )
    assert values.shape == (1, 1)
    assert select_list == [(0, 0, 0, 0)]


def test_get_exchange_kernels_full_matches_compressed_roundtrip():
    Gs = np.array([0.0, 1.0], dtype=float)
    thetas = np.array([0.0, 0.1], dtype=float)
    nmax = 2

    values, sel = get_exchange_kernels_compressed(
        Gs,
        thetas,
        nmax,
        method="laguerre",
        nquad=80,
    )
    from quantumhall_matrixelements._materialize import materialize_full_tensor

    X_round = materialize_full_tensor(values, sel, nmax)
    X_full = get_exchange_kernels(
        Gs,
        thetas,
        nmax,
        method="laguerre",
        nquad=80,
    )
    assert np.allclose(X_full, X_round)
