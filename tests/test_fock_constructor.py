import numpy as np

from quantumhall_matrixelements import (
    get_exchange_kernels,
    get_fockmatrix_constructor,
    get_fockmatrix_constructor_hf,
)


def _random_hermitian_rho(nG: int, nmax: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(nG, nmax, nmax)) + 1j * rng.normal(size=(nG, nmax, nmax))
    A = 0.5 * (A + np.conjugate(np.transpose(A, (0, 2, 1))))
    return A.astype(np.complex128)


def test_fock_constructor_matches_full_tensor_contraction():
    nmax = 3
    Gs = np.array([0.0, 1.2])
    thetas = np.array([0.0, 0.3])

    X_full = get_exchange_kernels(Gs, thetas, nmax, method="laguerre", nquad=400)
    rho = _random_hermitian_rho(len(Gs), nmax, seed=1)

    # Reference: Σ_{n2,m2} = - Σ_{n1,m1} X_{n1,m1,n2,m2} ρ_{n1,m1}
    ref = -np.einsum("gnm,gnmrt->grt", rho, X_full, optimize=True)

    fock = get_fockmatrix_constructor(Gs, thetas, nmax, method="laguerre", nquad=400)
    out = fock(rho)

    assert np.allclose(out, ref, rtol=1e-10, atol=1e-12)


def test_fock_constructor_hf_matches_full_tensor_contraction():
    nmax = 3
    Gs = np.array([0.0, 1.2])
    thetas = np.array([0.0, 0.3])

    X_full = get_exchange_kernels(Gs, thetas, nmax, method="laguerre", nquad=400)
    rng = np.random.default_rng(2)
    rho = rng.normal(size=(len(Gs), nmax, nmax)) + 1j * rng.normal(size=(len(Gs), nmax, nmax))
    rho = rho.astype(np.complex128, copy=False)

    # Reference:
    #   Σ_{n2,n1}(G) = - Σ_{m1,m2} X_{n1,m1,n2,m2}(G) ρ^*_{m2,m1}(G)
    ref = -np.einsum("gpqrs,gsq->grp", X_full, rho.conj(), optimize=True)

    fock = get_fockmatrix_constructor_hf(Gs, thetas, nmax, method="laguerre", nquad=400)
    out_minus = fock(rho, include_minus=True)
    out_plus = fock(rho, include_minus=False)

    assert np.allclose(out_minus, ref, rtol=1e-10, atol=1e-12)
    assert np.allclose(out_minus, -out_plus, rtol=0.0, atol=0.0)


def test_fock_constructor_laguerre_constant_matches_full_tensor_contraction():
    nmax = 3
    Gs = np.array([0.0, 1.2])
    thetas = np.array([0.0, 0.3])
    kappa = 0.4

    X_full = get_exchange_kernels(
        Gs,
        thetas,
        nmax,
        method="laguerre",
        potential="constant",
        kappa=kappa,
        nquad=300,
    )
    rho = _random_hermitian_rho(len(Gs), nmax, seed=5)
    ref = -np.einsum("gnm,gnmrt->grt", rho, X_full, optimize=True)

    fock = get_fockmatrix_constructor(
        Gs,
        thetas,
        nmax,
        method="laguerre",
        potential="constant",
        kappa=kappa,
        nquad=300,
    )
    out = fock(rho)

    assert np.allclose(out, ref, rtol=1e-10, atol=1e-12)
