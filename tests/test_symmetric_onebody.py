import numpy as np

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
