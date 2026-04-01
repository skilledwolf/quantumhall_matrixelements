import warnings
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

import quantumhall_matrixelements.symmetric.onebody as symmetric_onebody
from quantumhall_matrixelements import (
    get_central_onebody_matrix_elements_compressed,
    materialize_central_onebody_matrix,
)
from quantumhall_matrixelements._ho import logfact_table, precompute_radial_table

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "playground" / "reproduce_macdonald_ritchie_tableII.py"
)
_SPEC = spec_from_file_location("reproduce_macdonald_ritchie_tableII", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load MacDonald-Ritchie helper from {_MODULE_PATH}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


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
        method="quadrature",
        select=select,
    )
    values_coulomb_fn, select_coulomb_fn = get_central_onebody_matrix_elements_compressed(
        3,
        3,
        potential=lambda q, pref=kappa: pref * 2.0 * np.pi / q,
        qmax=qmax,
        nquad=nquad,
        method="quadrature",
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


def test_central_onebody_coulomb_auto_uses_closed_form_backend():
    select = [(0, 0, 0, 0), (2, 2, 1, 1), (6, 7, 4, 5)]
    values_auto, select_auto = get_central_onebody_matrix_elements_compressed(
        7,
        8,
        potential="coulomb",
        select=select,
    )
    values_closed, select_closed = get_central_onebody_matrix_elements_compressed(
        7,
        8,
        potential="coulomb",
        method="closed_form",
        select=select,
    )
    assert select_auto == select_closed == select
    assert np.allclose(values_auto, values_closed, rtol=0.0, atol=0.0)


def test_central_onebody_closed_form_matches_macdonald_ritchie_reference():
    select = [(30, 30, 30, 30), (60, 60, 20, 20), (80, 82, 75, 77), (120, 123, 110, 113)]
    values, select_list = get_central_onebody_matrix_elements_compressed(
        121,
        124,
        potential="coulomb",
        method="closed_form",
        select=select,
    )

    refs = np.array(
        [
            -0.5
            * np.sqrt(np.pi)
            * (-1.0 if (n_row - n_col) % 2 else 1.0)
            * _MODULE.V_coulomb_symmetric_gauge(n_row, n_col, n_row - m_row)
            for n_row, m_row, n_col, _m_col in select_list
        ],
        dtype=float,
    )

    assert np.allclose(values, refs, rtol=0.0, atol=1e-14)


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
        get_central_onebody_matrix_elements_compressed(
            2,
            2,
            potential="constant",
            method="closed_form",
        )

    with pytest.raises(ValueError):
        get_central_onebody_matrix_elements_compressed(
            2,
            2,
            method="not-a-method",
        )

    with pytest.raises(ValueError):
        materialize_central_onebody_matrix(np.array([1.0]), [], 2, 2)


def test_large_index_radial_table_stays_finite():
    radial = precompute_radial_table(np.array([10.0, 25.0, 35.0], dtype=float), logfact_table(450))
    assert np.isfinite(radial).all()
    assert np.nanmax(np.abs(radial)) < 1.0


def test_central_onebody_blocked_path_matches_full_path(monkeypatch: pytest.MonkeyPatch):
    kwargs = dict(
        nmax=4,
        mmax=5,
        potential="coulomb",
        qmax=25.0,
        nquad=800,
        method="quadrature",
        select=[(0, 1, 0, 1), (1, 2, 0, 1), (2, 3, 1, 2), (3, 4, 1, 2)],
    )
    values_full, select_full = get_central_onebody_matrix_elements_compressed(**kwargs)

    monkeypatch.setattr(symmetric_onebody, "_ONEBODY_BLOCK_TARGET_BYTES", 1)
    values_blocked, select_blocked = get_central_onebody_matrix_elements_compressed(**kwargs)

    assert select_blocked == select_full
    assert np.allclose(values_blocked, values_full, rtol=1e-12, atol=1e-12)


def test_central_onebody_large_cutoff_selected_entries_stay_finite_without_warnings():
    select = [(0, 3, 0, 3), (1, 4, 0, 3), (2, 5, 1, 4), (40, 43, 39, 42)]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        values, select_list = get_central_onebody_matrix_elements_compressed(
            260,
            263,
            potential="coulomb",
            qmax=35.0,
            nquad=800,
            method="quadrature",
            select=select,
        )

    assert select_list == select
    assert np.isfinite(values).all()
