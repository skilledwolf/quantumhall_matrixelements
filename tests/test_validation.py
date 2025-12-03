import numpy as np

from quantumhall_matrixelements import get_exchange_kernels
from quantumhall_matrixelements.diagnostic import verify_exchange_kernel_symmetries


def test_cross_backend_consistency():
    """
    Verify that 'gausslegendre' and 'hankel' backends produce consistent results.
    """
    nmax = 6
    # Use a non-trivial set of G vectors
    # G0=(0,0), G1=(1.5, 0.2), G2=(2.0, pi)
    Gs_dimless = np.array([0.0, 1.5, 2.0])
    thetas = np.array([0.0, 0.2, np.pi])
    
    # Compute with both backends
    X_gl = get_exchange_kernels(
        Gs_dimless, thetas, nmax, method="gausslegendre", sign_magneticfield=-1
    )
    X_hk = get_exchange_kernels(
        Gs_dimless, thetas, nmax, method="hankel", sign_magneticfield=-1
    )
    
    # Check for agreement
    # The Hankel transform can be slightly less precise depending on the grid,
    # but should agree well for standard Coulomb potentials.
    assert np.allclose(X_gl, X_hk, rtol=3e-3, atol=3e-3), \
        "Mismatch between Gauss-Legendre and Hankel backends"

def test_large_n_consistency():
    """
    Verify consistency at larger nmax (e.g. 12) with relaxed tolerance.
    """
    nmax = 12
    Gs_dimless = np.array([0.0, 1.5, 2.0])
    thetas = np.array([0.0, 0.2, np.pi])
    
    X_gl = get_exchange_kernels(
        Gs_dimless, thetas, nmax, method="gausslegendre", sign_magneticfield=-1
    )
    X_hk = get_exchange_kernels(
        Gs_dimless, thetas, nmax, method="hankel", sign_magneticfield=-1
    )
    
    # At nmax=12, differences up to ~3e-3 are acceptable due to quadrature limits
    assert np.allclose(X_gl, X_hk, rtol=3e-3, atol=3e-3), \
        "Mismatch at large nmax exceeded relaxed tolerance"

def test_analytic_coulomb_limit_zero_G():
    """
    Verify the analytic limit for Coulomb interaction at G=0 for lowest Landau level.
    
    Analytic value:
    X_{0000}(0) = \\int d^2q/(2pi)^2 * (2pi/q) * |F_{00}(q)|^2
                = \\int_0^\\inf q dq/(2pi) * (2pi/q) * exp(-q^2/2)
                = \\int_0^\\inf dr * exp(-r^2/2)   (where r=q)
                = sqrt(pi/2)
    """
    nmax = 1
    Gs_dimless = np.array([0.0])
    thetas = np.array([0.0])
    
    # Expected value: sqrt(pi/2)
    expected = np.sqrt(np.pi / 2.0)
    
    # Check Gauss-Legendre
    X_gl = get_exchange_kernels(
        Gs_dimless, thetas, nmax, method="gausslegendre", sign_magneticfield=-1
    )
    val_gl = X_gl[0, 0, 0, 0, 0]
    assert np.isclose(val_gl, expected, atol=5e-4), \
        f"Gauss-Legendre failed analytic limit. Got {val_gl}, expected {expected}"
        
    # Check Hankel
    X_hk = get_exchange_kernels(Gs_dimless, thetas, nmax, method="hankel", sign_magneticfield=-1)
    val_hk = X_hk[0, 0, 0, 0, 0]
    assert np.isclose(val_hk, expected, atol=1e-5), \
        f"Hankel failed analytic limit. Got {val_hk}, expected {expected}"

def test_symmetry_checks_extended():
    """
    Run the symmetry diagnostic on a larger set of G vectors.
    """
    nmax = 3
    # A mix of magnitudes and angles
    Gs_dimless = np.array([0.1, 1.0, 2.5, 5.0])
    thetas = np.array([0.0, np.pi/3, np.pi/2, 3*np.pi/4])
    
    # This function asserts internally if symmetries are violated
    verify_exchange_kernel_symmetries(Gs_dimless, thetas, nmax, rtol=1e-6, atol=1e-8)

def test_gausslegendre_convergence():
    """
    Verify Gauss-Legendre quadrature converges as nquad increases.
    """
    nmax = 2
    Gs_dimless = np.array([1.0])
    thetas = np.array([0.0])

    X_low = get_exchange_kernels(
        Gs_dimless,
        thetas,
        nmax,
        method="gausslegendre",
        nquad=200,
        sign_magneticfield=-1,
    )
    X_high = get_exchange_kernels(
        Gs_dimless,
        thetas,
        nmax,
        method="gausslegendre",
        nquad=400,
        sign_magneticfield=-1,
    )

    assert np.allclose(X_low, X_high, rtol=3e-3, atol=2e-3), \
        "Gauss-Legendre quadrature not converged between nquad=200 and nquad=400"


def test_hankel_callable_potential_matches_coulomb():
    """Hankel backend should respect a user-supplied callable identical to Coulomb."""
    nmax = 2
    Gs_dimless = np.array([0.8, 1.6])
    thetas = np.array([0.0, 0.4])
    kappa = 0.9

    def V_coulomb(q):
        return kappa * 2.0 * np.pi / q

    X_coulomb = get_exchange_kernels(
        Gs_dimless, thetas, nmax, method="hankel", potential="coulomb", kappa=kappa
    )
    X_callable = get_exchange_kernels(
        Gs_dimless, thetas, nmax, method="hankel", potential=V_coulomb
    )

    assert np.allclose(X_callable, X_coulomb, rtol=1e-4, atol=1e-4)
