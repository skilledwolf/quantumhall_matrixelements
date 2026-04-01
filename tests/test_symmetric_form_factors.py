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


def test_factorized_density_form_factors_delegate_to_component_apis():
    qs = np.array([0.3, 0.7])
    thetas = np.array([0.2, -0.1])
    nmax = 3
    mmax = 4

    f_cyc, g_gc = get_factorized_density_form_factors(qs, thetas, nmax, mmax)

    assert np.allclose(f_cyc, get_form_factors(qs, thetas, nmax))
    assert np.allclose(g_gc, get_guiding_center_form_factors(qs, thetas, mmax))
