import math

import numpy as np

from quantumhall_matrixelements import (
    get_haldane_pseudopotentials,
    get_twobody_disk_from_pseudopotentials_compressed,
    materialize_twobody_disk_tensor,
)


def test_haldane_pseudopotentials_match_lll_coulomb_values():
    values = get_haldane_pseudopotentials(
        5,
        n_ll=0,
        potential="coulomb",
        qmax=35.0,
        nquad=2500,
    )

    expected = np.array([math.gamma(m + 0.5) / (2.0 * math.factorial(m)) for m in range(5)])
    assert np.allclose(values, expected, rtol=1e-6, atol=1e-8)
    assert np.all(np.diff(values) < 0.0)


def test_disk_two_body_low_lying_values_and_selection_rule():
    v_m = np.array([2.0, 0.5, 0.25], dtype=float)
    values, select_list = get_twobody_disk_from_pseudopotentials_compressed(
        v_m,
        2,
        select=[(0, 0, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 0, 0, 1)],
    )

    assert select_list == [(0, 0, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 0, 0, 1)]
    assert np.isclose(values[0], 2.0)
    assert np.isclose(values[1], 0.5 * (2.0 + 0.5))
    assert np.isclose(values[2], 0.5 * (2.0 - 0.5))
    assert values[3] == 0.0


def test_disk_two_body_antisymmetrized_channel_and_materialization_symmetry():
    v_m = np.array([1.75, 0.25, 0.1], dtype=float)
    values, select_list = get_twobody_disk_from_pseudopotentials_compressed(
        v_m,
        2,
        select=[(0, 1, 0, 1), (0, 1, 1, 0)],
        antisymmetrize=True,
    )

    assert np.isclose(values[0], 0.25)
    dense = materialize_twobody_disk_tensor(values, select_list, 2)

    assert dense.shape == (2, 2, 2, 2)
    assert np.allclose(dense, np.swapaxes(np.swapaxes(dense, 0, 2), 1, 3))
