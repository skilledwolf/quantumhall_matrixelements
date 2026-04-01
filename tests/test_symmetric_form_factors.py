import numpy as np

from quantumhall_matrixelements import (
    get_factorized_density_form_factors,
    get_form_factors,
    get_guiding_center_form_factors,
)


def test_guiding_center_form_factors_q_zero_is_identity():
    mmax = 4
    qs = np.zeros(3)
    thetas = np.array([0.0, 0.7, -1.2])

    g_gc = get_guiding_center_form_factors(qs, thetas, mmax)

    expected = np.broadcast_to(np.eye(mmax, dtype=np.complex128), g_gc.shape)
    assert np.allclose(g_gc, expected)


def test_guiding_center_form_factors_hermitian_under_q_inversion():
    mmax = 3
    q = 0.8
    theta = 0.37

    g_gc = get_guiding_center_form_factors(
        np.array([q, q]),
        np.array([theta, theta + np.pi]),
        mmax,
    )

    assert np.allclose(g_gc[1], g_gc[0].conj().T, atol=1e-12)


def test_guiding_center_form_factors_known_low_index_values():
    qs = np.array([0.1, 1.0, 2.0])
    thetas = np.array([0.0, np.pi / 4, np.pi / 2])
    mmax = 2

    g_gc = get_guiding_center_form_factors(
        qs,
        thetas,
        mmax,
        sign_magneticfield=-1,
    )

    mag_term = qs / np.sqrt(2.0)
    gauss_term = np.exp(-(qs**2) / 4.0)

    assert np.allclose(g_gc[:, 0, 0], gauss_term)
    assert np.allclose(g_gc[:, 0, 1], 1j * mag_term * gauss_term * np.exp(-1j * thetas))
    assert np.allclose(g_gc[:, 1, 0], 1j * mag_term * gauss_term * np.exp(+1j * thetas))


def test_guiding_center_sign_magneticfield_phase_relation():
    mmax = 4
    qs = np.array([0.4, 1.1])
    thetas = np.array([0.2, -0.6])

    g_neg = get_guiding_center_form_factors(qs, thetas, mmax, sign_magneticfield=-1)
    g_pos = get_guiding_center_form_factors(qs, thetas, mmax, sign_magneticfield=+1)

    idx = np.arange(mmax)
    phase = np.where((idx[:, None] - idx[None, :]) % 2 == 0, 1.0, -1.0)
    expected = np.conj(g_neg) * phase[None, :, :]
    assert np.allclose(g_pos, expected)


def test_factorized_density_form_factors_delegate_to_component_apis():
    qs = np.array([0.3, 0.7])
    thetas = np.array([0.2, -0.1])
    nmax = 3
    mmax = 4

    f_cyc, g_gc = get_factorized_density_form_factors(qs, thetas, nmax, mmax)

    assert np.allclose(f_cyc, get_form_factors(qs, thetas, nmax))
    assert np.allclose(g_gc, get_guiding_center_form_factors(qs, thetas, mmax))


def test_factorized_density_sign_consistency_matches_components():
    qs = np.array([0.5])
    thetas = np.array([0.4])
    nmax = 3
    mmax = 3

    f_neg, g_neg = get_factorized_density_form_factors(
        qs,
        thetas,
        nmax,
        mmax,
        sign_magneticfield=-1,
    )
    f_pos, g_pos = get_factorized_density_form_factors(
        qs,
        thetas,
        nmax,
        mmax,
        sign_magneticfield=+1,
    )

    idx_n = np.arange(nmax)
    phase_n = np.where((idx_n[:, None] - idx_n[None, :]) % 2 == 0, 1.0, -1.0)
    idx_m = np.arange(mmax)
    phase_m = np.where((idx_m[:, None] - idx_m[None, :]) % 2 == 0, 1.0, -1.0)

    assert np.allclose(f_pos, np.conj(f_neg) * phase_n[None, :, :])
    assert np.allclose(g_pos, np.conj(g_neg) * phase_m[None, :, :])
