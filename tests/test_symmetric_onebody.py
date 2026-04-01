import numpy as np
import pytest

from quantumhall_matrixelements import (
    get_central_onebody_matrix_elements_compressed,
    materialize_central_onebody_matrix,
)


def test_central_onebody_exact_selection_rule_and_lll_coulomb_value():
    values, select_list = get_central_onebody_matrix_elements_compressed(
        2,
        2,
        potential="coulomb",
        qmax=35.0,
        nquad=1200,
        select=[(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)],
    )

    assert select_list == [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)]
    assert np.isclose(values[0], np.sqrt(np.pi / 2.0), rtol=1e-6, atol=1e-8)
    assert values[1] == 0.0
    assert values[2] == 0.0


def test_central_onebody_lll_constant_value_and_materialization_symmetry():
    values, select_list = get_central_onebody_matrix_elements_compressed(
        2,
        2,
        potential="constant",
        kappa=2.5,
        qmax=25.0,
        nquad=1000,
        select=[(0, 0, 0, 0), (0, 0, 1, 1)],
    )

    assert np.isclose(values[0], 2.5 / (2.0 * np.pi), rtol=1e-6, atol=1e-8)
    dense = materialize_central_onebody_matrix(values, select_list, 2, 2)

    assert dense.shape == (2, 2, 2, 2)
    assert np.allclose(dense, np.swapaxes(np.swapaxes(dense, 0, 2), 1, 3))


def test_central_onebody_builtin_matches_callable_potentials():
    select = [(0, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0)]
    qmax = 30.0
    nquad = 1200
    kappa = 1.7

    values_coulomb, select_coulomb = get_central_onebody_matrix_elements_compressed(
        3,
        3,
        potential="coulomb",
        kappa=kappa,
        qmax=qmax,
        nquad=nquad,
        select=select,
    )
    values_coulomb_fn, select_coulomb_fn = get_central_onebody_matrix_elements_compressed(
        3,
        3,
        potential=lambda q, pref=kappa: pref * 2.0 * np.pi / q,
        qmax=qmax,
        nquad=nquad,
        select=select,
    )
    assert select_coulomb == select_coulomb_fn == select
    assert np.allclose(values_coulomb, values_coulomb_fn, rtol=1e-10, atol=1e-12)

    values_const, select_const = get_central_onebody_matrix_elements_compressed(
        3,
        3,
        potential="constant",
        kappa=kappa,
        qmax=qmax,
        nquad=nquad,
        select=select,
    )
    values_const_fn, select_const_fn = get_central_onebody_matrix_elements_compressed(
        3,
        3,
        potential=lambda q, pref=kappa: np.full_like(q, pref, dtype=float),
        qmax=qmax,
        nquad=nquad,
        select=select,
    )
    assert select_const == select_const_fn == select
    assert np.allclose(values_const, values_const_fn, rtol=1e-10, atol=1e-12)


def test_central_onebody_canonical_select_is_deterministic_and_materializes_consistently():
    values1, select1 = get_central_onebody_matrix_elements_compressed(
        3,
        3,
        potential="coulomb",
        qmax=20.0,
        nquad=600,
    )
    values2, select2 = get_central_onebody_matrix_elements_compressed(
        3,
        3,
        potential="coulomb",
        qmax=20.0,
        nquad=600,
    )

    assert select1 == select2
    assert np.allclose(values1, values2, rtol=0.0, atol=0.0)

    dense1 = materialize_central_onebody_matrix(values1, select1, 3, 3)
    dense2 = materialize_central_onebody_matrix(values2, select2, 3, 3)
    assert np.allclose(dense1, dense2, rtol=0.0, atol=0.0)


def test_central_onebody_invalid_inputs_fail_loudly():
    with pytest.raises(ValueError):
        get_central_onebody_matrix_elements_compressed(
            2,
            2,
            select=[(0, 0, 2, 0)],
        )

    with pytest.raises(ValueError):
        get_central_onebody_matrix_elements_compressed(
            2,
            2,
            potential="not-a-potential",
        )

    with pytest.raises(ValueError):
        materialize_central_onebody_matrix(np.array([1.0]), [], 2, 2)
