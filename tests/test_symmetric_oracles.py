from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from quantumhall_matrixelements import (
    get_central_onebody_matrix_elements_compressed,
    get_guiding_center_form_factors,
    get_haldane_pseudopotentials,
)

pytestmark = pytest.mark.slow


def _segmented_quad_mpmath(integrand, *, qmax: float, segments: int) -> mp.mpf:
    total = mp.mpf("0")
    step = mp.mpf(qmax) / int(segments)
    for idx in range(int(segments)):
        total += mp.quad(integrand, [idx * step, (idx + 1) * step])
    return total


def _ho_radial_mpmath(row: int, col: int, q: mp.mpf) -> mp.mpf:
    x = mp.mpf("0.5") * q * q
    amin = min(row, col)
    aabs = abs(row - col)
    return (
        mp.sqrt(mp.factorial(amin) / mp.factorial(amin + aabs))
        * (q / mp.sqrt(2)) ** aabs
        * mp.laguerre(amin, aabs, x)
        * mp.e ** (-x / 2)
    )


def _guiding_center_form_factor_mpmath(
    m_row: int,
    m_col: int,
    q: float,
    theta: float,
    *,
    sign_magneticfield: int,
    dps: int = 80,
) -> complex:
    mp.mp.dps = int(dps)
    q_mp = mp.mpf(q)
    theta_mp = mp.mpf(theta)
    diff = int(m_row - m_col)
    sigma_gc = -int(sign_magneticfield)
    angular = complex(mp.cos(sigma_gc * diff * theta_mp), mp.sin(sigma_gc * diff * theta_mp))
    return ((1j) ** abs(diff)) * angular * complex(_ho_radial_mpmath(m_row, m_col, q_mp))


def _central_onebody_coulomb_mpmath(
    n_row: int,
    m_row: int,
    n_col: int,
    m_col: int,
    *,
    dps: int = 80,
    qmax: float = 35.0,
    segments: int = 8,
) -> float:
    if (m_row - n_row) != (m_col - n_col):
        return 0.0

    mp.mp.dps = int(dps)

    def integrand(q: mp.mpf) -> mp.mpf:
        return _ho_radial_mpmath(n_row, n_col, q) * _ho_radial_mpmath(m_row, m_col, q)

    value = _segmented_quad_mpmath(integrand, qmax=qmax, segments=segments)
    if (n_row - n_col) % 2:
        value = -value
    return float(value)


def _pseudopotential_coulomb_mpmath(
    m: int,
    *,
    n_ll: int,
    dps: int = 80,
    qmax: float = 35.0,
    segments: int = 8,
) -> float:
    mp.mp.dps = int(dps)
    half = mp.mpf("0.5")

    def integrand(q: mp.mpf) -> mp.mpf:
        x = half * q * q
        t = q * q
        return mp.laguerre(n_ll, 0, x) ** 2 * mp.laguerre(m, 0, t) * mp.e ** (-t)

    return float(_segmented_quad_mpmath(integrand, qmax=qmax, segments=segments))


def _composition_window_error(
    *,
    q: float,
    theta_q: float,
    k: float,
    theta_k: float,
    m_test: int,
    buffer: int,
    sign_magneticfield: int,
) -> float:
    mmax = int(m_test + buffer)
    qx = q * np.cos(theta_q)
    qy = q * np.sin(theta_q)
    kx = k * np.cos(theta_k)
    ky = k * np.sin(theta_k)
    q_plus_k_mag = float(np.hypot(qx + kx, qy + ky))
    q_plus_k_ang = float(np.arctan2(qy + ky, qx + kx))

    factors = get_guiding_center_form_factors(
        np.array([q, k, q_plus_k_mag], dtype=float),
        np.array([theta_q, theta_k, q_plus_k_ang], dtype=float),
        mmax,
        sign_magneticfield=sign_magneticfield,
    )

    sigma_gc = -int(sign_magneticfield)
    q_cross_k = qx * ky - qy * kx
    phase = np.exp(-0.5j * sigma_gc * q_cross_k)
    lhs = factors[0] @ factors[1]
    rhs = phase * factors[2]
    return float(np.linalg.norm((lhs - rhs)[:m_test, :m_test], ord=np.inf))


def test_guiding_center_matches_mpmath_closed_form_oracle():
    qs = np.array([0.25, 1.1, 2.8], dtype=float)
    thetas = np.array([0.1, -0.7, 1.2], dtype=float)

    for sign in (-1, +1):
        out = get_guiding_center_form_factors(qs, thetas, 6, sign_magneticfield=sign)
        for iq, (q, theta) in enumerate(zip(qs, thetas, strict=True)):
            for m_row, m_col in [(0, 0), (1, 0), (2, 1), (4, 2), (5, 5)]:
                ref = _guiding_center_form_factor_mpmath(
                    m_row,
                    m_col,
                    float(q),
                    float(theta),
                    sign_magneticfield=sign,
                )
                assert np.allclose(out[iq, m_row, m_col], ref, rtol=1e-13, atol=1e-13)


def test_guiding_center_protected_window_unitarity_improves_with_buffer():
    q = 1.3
    theta = 0.4
    m_test = 6
    sign = -1

    def protected_error(buffer: int) -> float:
        mmax = m_test + buffer
        factor = get_guiding_center_form_factors(
            np.array([q], dtype=float),
            np.array([theta], dtype=float),
            mmax,
            sign_magneticfield=sign,
        )[0]
        product = factor @ factor.conj().T
        return float(np.linalg.norm(product[:m_test, :m_test] - np.eye(m_test), ord=np.inf))

    err_small = protected_error(4)
    err_large = protected_error(10)

    assert err_large < err_small
    assert err_large < 1e-5


def test_guiding_center_gmp_composition_on_protected_window():
    err_small = _composition_window_error(
        q=0.4,
        theta_q=0.3,
        k=0.7,
        theta_k=-0.2,
        m_test=6,
        buffer=4,
        sign_magneticfield=-1,
    )
    err_large = _composition_window_error(
        q=0.4,
        theta_q=0.3,
        k=0.7,
        theta_k=-0.2,
        m_test=6,
        buffer=10,
        sign_magneticfield=-1,
    )

    assert err_large < err_small
    assert err_large < 1e-10


def test_central_onebody_coulomb_matches_mpmath_oracle():
    select = [(0, 0, 0, 0), (1, 1, 0, 0), (2, 1, 1, 0), (2, 2, 1, 1)]
    values, select_list = get_central_onebody_matrix_elements_compressed(
        3,
        3,
        potential="coulomb",
        qmax=35.0,
        nquad=2000,
        select=select,
    )

    refs = np.array(
        [
            _central_onebody_coulomb_mpmath(*entry, qmax=35.0, segments=8)
            for entry in select_list
        ],
        dtype=float,
    )
    assert np.allclose(values, refs, rtol=1e-10, atol=1e-12)


def test_higher_ll_pseudopotentials_match_mpmath_oracle():
    for n_ll in (1, 2, 3):
        values = get_haldane_pseudopotentials(
            5,
            n_ll=n_ll,
            potential="coulomb",
            qmax=35.0,
            nquad=2500,
        )
        refs = np.array(
            [
                _pseudopotential_coulomb_mpmath(m, n_ll=n_ll, qmax=35.0, segments=8)
                for m in range(5)
            ],
            dtype=float,
        )
        assert np.allclose(values, refs, rtol=1e-10, atol=1e-12)
