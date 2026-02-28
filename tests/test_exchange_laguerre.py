import numpy as np
import pytest

from quantumhall_matrixelements import build_fockmatrix_apply, get_exchange_kernels_compressed, get_fockmatrix_constructor
from quantumhall_matrixelements.exchange_laguerre import QuadratureParams, build_exchange_fock_precompute


def _random_hermitian_rho(nG: int, nmax: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(nG, nmax, nmax)) + 1j * rng.normal(size=(nG, nmax, nmax))
    A = 0.5 * (A + np.conjugate(np.transpose(A, (0, 2, 1))))
    norms = np.linalg.norm(A.reshape(nG, -1), axis=1)
    A = A / norms[:, None, None]
    return A.astype(np.complex128)


def test_fast_fock_matches_exchange_kernel_construction_coulomb():
    nmax = 2
    Gs = np.array([0.0, 0.8, 1.7])
    thetas = np.array([0.0, 0.2, -0.4])
    rho = _random_hermitian_rho(len(Gs), nmax, seed=1)

    # Reference from Hankel backend
    fock_ref = get_fockmatrix_constructor(Gs, thetas, nmax, method="hankel")
    F_ref = fock_ref(rho)

    # Fast finite-q quadrature should agree within a looser tolerance.
    params = QuadratureParams(qmax=35.0, N=800)
    fast = build_exchange_fock_precompute(nmax, Gs, thetas, params, sigma=-1.0, kappa=1.0)
    F_fast = fast.exchange_fock(rho)

    assert np.allclose(F_fast, F_ref, rtol=5e-3, atol=5e-3)


def test_fast_fock_include_minus_flag():
    nmax = 1
    Gs = np.array([0.4])
    thetas = np.array([0.1])
    rho = np.array([[[1.0 + 0.0j]]])

    params = QuadratureParams(qmax=40.0, N=1200)
    fast_minus = build_exchange_fock_precompute(nmax, Gs, thetas, params, include_minus=True)
    fast_plus = build_exchange_fock_precompute(nmax, Gs, thetas, params, include_minus=False)

    Fm = fast_minus.exchange_fock(rho)
    Fp = fast_plus.exchange_fock(rho)
    assert np.allclose(Fm, -Fp)


def test_laguerre_backend_values_match_fast_apply():
    nmax = 3
    Gs = np.array([0.0, 0.8, 1.7])
    thetas = np.array([0.0, 0.2, -0.4])
    rho = _random_hermitian_rho(len(Gs), nmax, seed=2)

    params = QuadratureParams(qmax=35.0, N=250)
    fast = build_exchange_fock_precompute(nmax, Gs, thetas, params, sigma=-1.0, kappa=1.0)
    F_fast = fast.exchange_fock(rho)

    values, select = get_exchange_kernels_compressed(
        Gs,
        thetas,
        nmax,
        method="laguerre",
        potential="coulomb",
        kappa=1.0,
        qmax=params.qmax,
        nquad=params.N,
        sign_magneticfield=-1,
    )
    apply = build_fockmatrix_apply(values, select, nmax, convention="standard")
    F_vals = apply(rho, include_minus=True)

    assert np.allclose(F_vals, F_fast, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Adaptive nquad tests
# ---------------------------------------------------------------------------
def test_adaptive_nquad_increases_with_large_G():
    """Adaptive nquad should give good results at large G without manual tuning."""
    nmax = 3
    Gs = np.array([0.0, 5.0, 40.0])
    thetas = np.zeros_like(Gs)
    select = [(0, 0, 0, 0), (1, 1, 1, 1), (1, 0, 1, 0)]

    # With adaptive_nquad=True (default), large G should be resolved
    vals_adapt, sel = get_exchange_kernels_compressed(
        Gs, thetas, nmax, method="laguerre", select=select, adaptive_nquad=True
    )

    # Reference: very high nquad on GL
    vals_ref, _ = get_exchange_kernels_compressed(
        Gs, thetas, nmax, method="laguerre", select=select,
        nquad=3000, adaptive_nquad=False,
    )

    assert np.allclose(vals_adapt, vals_ref, rtol=1e-6, atol=1e-8)


def test_adaptive_nquad_matches_hankel_at_moderate_G():
    """Adaptive laguerre should match Hankel backend at moderate G."""
    nmax = 2
    Gs = np.array([0.5, 3.0, 10.0])
    thetas = np.array([0.0, 0.3, -0.2])

    vals_ff, sel_ff = get_exchange_kernels_compressed(
        Gs, thetas, nmax, method="laguerre", adaptive_nquad=True,
    )
    vals_hk, sel_hk = get_exchange_kernels_compressed(
        Gs, thetas, nmax, method="hankel",
    )
    assert sel_ff == sel_hk
    assert np.allclose(vals_ff, vals_hk, rtol=2e-3, atol=2e-3)


# ---------------------------------------------------------------------------
# Ogata q-space tests
# ---------------------------------------------------------------------------
def test_ogata_q_matches_gl_at_moderate_G():
    """Ogata q-space results should match GL at moderate G values."""
    nmax = 3
    # Use G values that are above kmin_ogata so Ogata is actually used
    Gs = np.array([25.0, 30.0])
    thetas = np.array([0.0, 0.3])
    select = [(0, 0, 0, 0), (1, 1, 1, 1), (1, 0, 1, 0), (2, 1, 2, 1)]

    vals_gl, sel_gl = get_exchange_kernels_compressed(
        Gs, thetas, nmax, method="laguerre", select=select,
        nquad=3000, adaptive_nquad=False, use_ogata=False,
    )
    vals_og, sel_og = get_exchange_kernels_compressed(
        Gs, thetas, nmax, method="laguerre", select=select,
        use_ogata=True, kmin_ogata=20.0,
    )
    assert sel_gl == sel_og
    assert np.allclose(vals_og, vals_gl, rtol=5e-4, atol=1e-7)


def test_ogata_q_matches_hankel_small_nmax():
    """Ogata q-space should match Hankel backend for small nmax."""
    nmax = 2
    Gs = np.array([0.0, 5.0, 25.0])
    thetas = np.array([0.0, 0.2, -0.4])

    vals_og, sel_og = get_exchange_kernels_compressed(
        Gs, thetas, nmax, method="laguerre",
        use_ogata=True, kmin_ogata=10.0,
    )
    vals_hk, sel_hk = get_exchange_kernels_compressed(
        Gs, thetas, nmax, method="hankel",
    )
    assert sel_og == sel_hk
    assert np.allclose(vals_og, vals_hk, rtol=2e-3, atol=2e-3)


def test_ogata_q_hybrid_splits_correctly():
    """With use_ogata, small-|G| uses GL and large-|G| uses Ogata."""
    nmax = 2
    Gs = np.array([0.0, 2.0, 30.0])
    thetas = np.zeros_like(Gs)

    # All via GL (high nquad)
    vals_gl, sel = get_exchange_kernels_compressed(
        Gs, thetas, nmax, method="laguerre",
        nquad=3000, adaptive_nquad=False, use_ogata=False,
    )
    # Hybrid: GL for G<20, Ogata for G>=20
    vals_hyb, _ = get_exchange_kernels_compressed(
        Gs, thetas, nmax, method="laguerre",
        use_ogata=True, kmin_ogata=20.0,
    )
    # Both should agree
    assert np.allclose(vals_hyb, vals_gl, rtol=1e-3, atol=1e-6)


@pytest.mark.slow
def test_ogata_q_large_nmax():
    """Ogata q-space should handle large nmax without overflow."""
    nmax = 50
    Gs = np.array([0.0, 15.0, 30.0])
    thetas = np.zeros_like(Gs)
    select = [(0, 0, 0, 0), (nmax - 1, nmax - 1, nmax - 1, nmax - 1)]

    vals, sel = get_exchange_kernels_compressed(
        Gs, thetas, nmax, method="laguerre", select=select,
        use_ogata=True, kmin_ogata=10.0,
    )
    assert np.isfinite(vals).all()
    assert sel == select
