import numpy as np

from quantumhall_matrixelements import get_exchange_kernels, get_exchange_kernels_compressed


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


def test_ogata_constant_potential_matches_hankel():
    """Ogata backend should support the built-in 'constant' potential."""
    nmax = 2
    Gs_dimless = np.array([3.0, 5.0])  # ensure Ogata path (k >= kmin_ogata) is used
    thetas = np.array([0.1, 0.4])
    kappa = 0.7

    X_og = get_exchange_kernels(
        Gs_dimless,
        thetas,
        nmax,
        method="ogata",
        potential="constant",
        kappa=kappa,
    )
    X_hk = get_exchange_kernels(
        Gs_dimless,
        thetas,
        nmax,
        method="hankel",
        potential="constant",
        kappa=kappa,
    )

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


def test_ogata_auto_preserves_explicit_select_in_all_fallback_regime():
    """ogata_auto should not expand an explicit select list when all G use fallback."""
    select = [(0, 0, 0, 0)]
    values, select_list = get_exchange_kernels_compressed(
        np.array([0.5]),
        np.array([0.0]),
        3,
        method="ogata",
        select=select,
        ogata_auto=True,
        kmin_ogata=5.0,
    )

    assert values.shape == (1, 1)
    assert select_list == select
