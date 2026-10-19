import numpy as np
import pytest

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


def test_fock_constructor_laguerre_workspace_guard_uses_fast_path():
    Gs = np.array([0.0, 12.0])
    thetas = np.array([0.0, 0.1])

    with pytest.raises(MemoryError):
        get_fockmatrix_constructor(
            Gs,
            thetas,
            3,
            method="laguerre",
            nquad=80,
            workspace_limit_bytes=1,
        )


def test_fock_constructor_hf_fast_path_matches_compressed_apply():
    """The Numba HF fast path must agree with the compressed ``np.add.at`` path
    that ships in :func:`_build_compressed_constructor_hf` for both
    ``sign_magneticfield`` choices and across small ``nmax``.

    This is the golden-reference test for the 1.3.0 fast HF Fock-apply
    addition: a regression here means the fast path's index pattern,
    phase-table lookup, or sign convention has drifted from the
    documented analytic formula.
    """
    from quantumhall_matrixelements import get_exchange_kernels_compressed
    from quantumhall_matrixelements.fock import _build_compressed_constructor_hf

    rng = np.random.default_rng(123)
    for nmax in (1, 2, 3, 4):
        for sigma in (-1, 1):
            Gs = np.array([0.0, 0.7, 1.2, 2.5])
            thetas = np.array([0.0, -0.6, 0.4, 1.1])
            rho = (
                rng.normal(size=(len(Gs), nmax, nmax))
                + 1j * rng.normal(size=(len(Gs), nmax, nmax))
            ).astype(np.complex128)

            # Slow-but-correct compressed path (current qhf path)
            values, select = get_exchange_kernels_compressed(
                Gs, thetas, nmax, method="laguerre", nquad=300,
                sign_magneticfield=sigma,
            )
            slow = _build_compressed_constructor_hf(values, select, nmax)
            ref = slow(rho)

            # Fast Numba HF path (returned by get_fockmatrix_constructor_hf)
            fast = get_fockmatrix_constructor_hf(
                Gs, thetas, nmax, method="laguerre", nquad=300,
                sign_magneticfield=sigma,
            )
            out = fast(rho)

            assert np.allclose(out, ref, rtol=1e-10, atol=1e-12), (
                f"fast HF path disagrees with compressed apply at "
                f"nmax={nmax}, sigma={sigma}: max diff "
                f"{np.max(np.abs(out - ref)):.3e}"
            )


def test_fock_constructor_laguerre_accepts_compressed_limit_on_fast_path():
    Gs = np.array([0.0, 1.2])
    thetas = np.array([0.0, 0.3])
    rho = _random_hermitian_rho(len(Gs), 3, seed=7)

    ref = get_fockmatrix_constructor(Gs, thetas, 3, method="laguerre", nquad=200)(rho)
    out = get_fockmatrix_constructor(
        Gs,
        thetas,
        3,
        method="laguerre",
        nquad=200,
        compressed_limit_bytes=1,
    )(rho)

    assert np.allclose(out, ref, rtol=1e-10, atol=1e-12)
