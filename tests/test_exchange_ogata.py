import numpy as np

from quantumhall_matrixelements import get_exchange_kernels


def test_ogata_analytic_coulomb_limit_zero_G():
    """Ogata backend should match the known Coulomb limit at G=0 (falls back to quadrature)."""
    nmax = 1
    Gs_dimless = np.array([0.0])
    thetas = np.array([0.0])

    expected = np.sqrt(np.pi / 2.0)
    X = get_exchange_kernels(Gs_dimless, thetas, nmax, method="ogata", nquad=1200)
    val = X[0, 0, 0, 0, 0]
    assert np.isclose(val, expected, atol=2e-3)


def test_ogata_vs_hankel_small_n():
    """Check Ogata agrees with Hankel in a regime where Ogata is used."""
    nmax = 2
    Gs_dimless = np.array([2.0, 4.0])
    thetas = np.array([0.1, 0.4])

    X_og = get_exchange_kernels(Gs_dimless, thetas, nmax, method="ogata", chunk_size=64)
    X_hk = get_exchange_kernels(Gs_dimless, thetas, nmax, method="hankel")

    assert np.allclose(X_og, X_hk, rtol=5e-3, atol=5e-3)


def test_ogata_sign_magneticfield_phase_relation():
    """sign_magneticfield=+1 should match the conjugation/phase relation of σ flip."""
    nmax = 2
    Gs_dimless = np.array([3.0])
    thetas = np.array([0.25])

    X_neg = get_exchange_kernels(Gs_dimless, thetas, nmax, method="ogata", sign_magneticfield=-1)
    X_pos = get_exchange_kernels(Gs_dimless, thetas, nmax, method="ogata", sign_magneticfield=+1)

    idx = np.arange(nmax)
    phase = np.where((idx[:, None] - idx[None, :]) % 2 == 0, 1.0, -1.0)
    phase = phase[:, :, None, None] * phase[None, None, :, :]

    expected = np.conj(X_neg) * phase[None, ...]
    assert np.allclose(X_pos, expected)
