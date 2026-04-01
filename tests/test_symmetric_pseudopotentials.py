import math

import numpy as np
import pytest

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


def test_haldane_pseudopotentials_builtin_matches_callable_potentials():
    kappa = 1.3
    qmax = 30.0
    nquad = 2200

    values_coulomb = get_haldane_pseudopotentials(
        6,
        n_ll=1,
        potential="coulomb",
        kappa=kappa,
        qmax=qmax,
        nquad=nquad,
    )
    values_coulomb_fn = get_haldane_pseudopotentials(
        6,
        n_ll=1,
        potential=lambda q, pref=kappa: pref * 2.0 * np.pi / q,
        qmax=qmax,
        nquad=nquad,
    )
    assert np.allclose(values_coulomb, values_coulomb_fn, rtol=1e-10, atol=1e-12)

    values_const = get_haldane_pseudopotentials(
        6,
        n_ll=1,
        potential="constant",
        kappa=kappa,
        qmax=qmax,
        nquad=nquad,
    )
    values_const_fn = get_haldane_pseudopotentials(
        6,
        n_ll=1,
        potential=lambda q, pref=kappa: np.full_like(q, pref, dtype=float),
        qmax=qmax,
        nquad=nquad,
    )
    assert np.allclose(values_const, values_const_fn, rtol=1e-10, atol=1e-12)


def test_haldane_pseudopotentials_invalid_inputs_fail_loudly():
    with pytest.raises(ValueError):
        get_haldane_pseudopotentials(0)

    with pytest.raises(ValueError):
        get_haldane_pseudopotentials(3, n_ll=-1)

    with pytest.raises(ValueError):
        get_haldane_pseudopotentials(3, potential="not-a-potential")


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


def test_disk_two_body_canonical_select_is_deterministic_and_materializes_consistently():
    v_m = np.array([1.0, 0.4, 0.1], dtype=float)

    values1, select1 = get_twobody_disk_from_pseudopotentials_compressed(v_m, 3)
    values2, select2 = get_twobody_disk_from_pseudopotentials_compressed(v_m, 3)

    assert select1 == select2
    assert np.allclose(values1, values2, rtol=0.0, atol=0.0)

    dense1 = materialize_twobody_disk_tensor(values1, select1, 3)
    dense2 = materialize_twobody_disk_tensor(values2, select2, 3)
    assert np.allclose(dense1, dense2, rtol=0.0, atol=0.0)


def test_disk_two_body_invalid_inputs_fail_loudly():
    with pytest.raises(ValueError):
        get_twobody_disk_from_pseudopotentials_compressed(np.array([]), 2)

    with pytest.raises(ValueError):
        get_twobody_disk_from_pseudopotentials_compressed(
            np.array([1.0]),
            2,
            select=[(0, 0, 2, 0)],
        )

    with pytest.raises(ValueError):
        materialize_twobody_disk_tensor(np.array([1.0]), [], 2)


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
