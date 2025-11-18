import numpy as np

from quantumhall_matrixelements import get_exchange_kernels
from quantumhall_matrixelements.diagnostic import verify_exchange_kernel_symmetries


def test_exchange_kernel_basic_shape_and_real_N0():
    """Sanity check: small G grid and nmax=2 produce finite kernels."""
    nmax = 2
    # G vectors: G0=(0,0), G+=(1,0), G-=(-1,0)
    Gs_dimless = np.array([0.0, 1.0, 1.0])
    thetas = np.array([0.0, 0.0, np.pi])

    X = get_exchange_kernels(Gs_dimless, thetas, nmax, method="gausslag")
    assert X.shape == (3, nmax, nmax, nmax, nmax)

    # N=0 sectors at G0 should be real (imag ~ 0)
    for n1 in range(nmax):
        for m1 in range(nmax):
            for n2 in range(nmax):
                for m2 in range(nmax):
                    N = n1 - m1 - n2 + m2
                    if N == 0:
                        val = X[0, n1, m1, n2, m2]
                        assert abs(val.imag) < 1e-8

    assert np.isfinite(X).all()


def test_exchange_kernel_g_inversion_symmetry():
    """Use the diagnostic helper to validate G-inversion and internal LL symmetries."""
    nmax = 2
    Gs_dimless = np.array([0.0, 1.0, 1.0])
    thetas = np.array([0.0, 0.0, np.pi])
    # Will raise AssertionError if symmetry fails
    verify_exchange_kernel_symmetries(Gs_dimless, thetas, nmax, rtol=1e-6, atol=1e-8)

