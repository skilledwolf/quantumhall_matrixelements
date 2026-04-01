"""Internal regression tests for the Gauss-Legendre backend module."""
import warnings

import numpy as np

from quantumhall_matrixelements import get_exchange_kernels_compressed
from quantumhall_matrixelements._materialize import materialize_full_tensor
from quantumhall_matrixelements.exchange_legendre import get_exchange_kernels_GaussLegendre


def test_legendre_basic_shape():
    nmax = 2
    Gs_dimless = np.array([0.0, 1.0])
    thetas = np.array([0.0, np.pi])
    values, select = get_exchange_kernels_GaussLegendre(Gs_dimless, thetas, nmax, nquad=100)
    X = materialize_full_tensor(values, select, nmax)
    assert X.shape == (2, nmax, nmax, nmax, nmax)
    assert np.isfinite(X).all()

def test_legendre_vs_hankel_small_n():
    """Verify agreement with Hankel for small n."""
    nmax = 2
    Gs_dimless = np.array([0.5, 1.5])
    thetas = np.array([0.0, 0.2])

    values, select = get_exchange_kernels_GaussLegendre(
        Gs_dimless, thetas, nmax, nquad=500, sign_magneticfield=-1,
    )
    X_leg = materialize_full_tensor(values, select, nmax)

    from quantumhall_matrixelements import get_exchange_kernels
    X_hk = get_exchange_kernels(
        Gs_dimless, thetas, nmax, method="hankel", sign_magneticfield=-1,
    )

    assert np.allclose(X_leg, X_hk, rtol=2e-3, atol=2e-3)

def test_legendre_large_n_stability():
    """Verify that it runs without error for large n."""
    nmax = 15
    Gs_dimless = np.array([1.0])
    thetas = np.array([0.0])

    values, select = get_exchange_kernels_GaussLegendre(Gs_dimless, thetas, nmax, nquad=500)
    X = materialize_full_tensor(values, select, nmax)
    assert np.isfinite(X).all()


def test_legendre_callable_potential_matches_coulomb():
    """Callable potential should reproduce Coulomb when given V(q)=2piK/q."""
    nmax = 3
    Gs_dimless = np.array([0.3, 1.1])
    thetas = np.array([0.0, 0.7])
    kappa = 1.2
    nquad = 8000

    def V_coulomb(q):
        return kappa * 2.0 * np.pi / q

    values_c, sel_c = get_exchange_kernels_GaussLegendre(
        Gs_dimless, thetas, nmax, potential="coulomb", kappa=kappa, nquad=nquad,
    )
    values_f, sel_f = get_exchange_kernels_GaussLegendre(
        Gs_dimless, thetas, nmax, potential=V_coulomb, nquad=nquad,
    )

    assert sel_c == sel_f
    assert np.allclose(values_f, values_c, rtol=1e-4, atol=1e-4)


def test_legendre_large_n_warning_free_matches_laguerre():
    select = [(100, 100, 100, 100)]
    Gs = np.array([0.0, 5.0, 10.0], dtype=float)
    thetas = np.zeros_like(Gs)

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        values_leg, select_leg = get_exchange_kernels_GaussLegendre(
            Gs,
            thetas,
            101,
            nquad=800,
            scale=0.04,
            select=select,
        )

    values_lag, select_lag = get_exchange_kernels_compressed(
        Gs,
        thetas,
        101,
        method="laguerre",
        select=select,
    )

    assert select_leg == select
    assert select_lag == select
    assert np.allclose(values_leg, values_lag, rtol=1e-10, atol=1e-10)
