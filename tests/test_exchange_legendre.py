import numpy as np

from quantumhall_matrixelements import get_exchange_kernels, get_exchange_kernels_GaussLegendre


def test_legendre_basic_shape():
    nmax = 2
    Gs_dimless = np.array([0.0, 1.0])
    thetas = np.array([0.0, np.pi])
    X = get_exchange_kernels_GaussLegendre(Gs_dimless, thetas, nmax, nquad=100)
    assert X.shape == (2, nmax, nmax, nmax, nmax)
    assert np.isfinite(X).all()

def test_legendre_vs_hankel_small_n():
    """Verify agreement with Hankel for small n."""
    nmax = 2
    Gs_dimless = np.array([0.5, 1.5])
    thetas = np.array([0.0, 0.2])
    
    X_leg = get_exchange_kernels(
        Gs_dimless,
        thetas,
        nmax,
        method="gausslegendre",
        nquad=500,
        sign_magneticfield=-1,
    )
    X_hk = get_exchange_kernels(
        Gs_dimless,
        thetas,
        nmax,
        method="hankel",
        sign_magneticfield=-1,
    )

    assert np.allclose(X_leg, X_hk, rtol=2e-3, atol=2e-3)

def test_legendre_large_n_stability():
    """Verify that it runs without error for large n."""
    nmax = 15
    Gs_dimless = np.array([1.0])
    thetas = np.array([0.0])
    
    # This should not raise an error
    X = get_exchange_kernels_GaussLegendre(Gs_dimless, thetas, nmax, nquad=500)
    assert np.isfinite(X).all()


def test_legendre_callable_potential_matches_coulomb():
    """Callable potential should reproduce Coulomb when given V(q)=2πκ/q."""
    nmax = 3
    Gs_dimless = np.array([0.3, 1.1])
    thetas = np.array([0.0, 0.7])
    kappa = 1.2
    nquad = 300

    def V_coulomb(q):
        return kappa * 2.0 * np.pi / q

    X_coulomb = get_exchange_kernels_GaussLegendre(
        Gs_dimless, thetas, nmax, potential="coulomb", kappa=kappa, nquad=nquad
    )
    X_callable = get_exchange_kernels_GaussLegendre(
        Gs_dimless, thetas, nmax, potential=V_coulomb, nquad=nquad
    )

    assert np.allclose(X_callable, X_coulomb, rtol=1e-4, atol=1e-4)
