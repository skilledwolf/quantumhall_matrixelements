"""Exchange kernels via Hankel transforms."""
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
from hankel import HankelTransform
from scipy.special import genlaguerre, rgamma

from .diagnostic import get_exchange_kernels_opposite_field

if TYPE_CHECKING:
    from numpy.typing import NDArray

    ComplexArray = NDArray[np.complex128]
    RealArray = NDArray[np.float64]


def _N_order(n1: int, m1: int, n2: int, m2: int) -> int:
    return ((n1 - m1) + (m2 - n2))

def _parity_factor(N: int) -> int:
    return (-1) ** ((N - abs(N)) // 2)

@cache
def _get_hankel_transformer(order: int) -> HankelTransform:
    """Cached HankelTransform instance for a given Bessel order."""
    return HankelTransform(nu=order, N=6000, h=7e-6)


def _radial_exchange_integrand_rgamma(
    q_magnitudes,
    n1,
    m1,
    n2,
    m2,
    potential: str | callable = "coulomb",
    kappa: float = 1.0,
):
    """Build integrand g(q) for Hankel transform with rgamma-normalization.

    For Coulomb: g(q) = κ F1F2 / q. We parameterize q = √2 r and absorb factors
    into a stable radial base constructed from generalized Laguerres.
    """
    q = np.asarray(q_magnitudes, dtype=float)
    r = q / np.sqrt(2.0)

    Δ1, N1 = abs(n1 - m1), min(n1, m1)
    Δ2, N2 = abs(n2 - m2), min(n2, m2)

    power = Δ1 + Δ2 - 1

    lag1 = genlaguerre(N1, Δ1)
    lag2 = genlaguerre(N2, Δ2)

    nrm1 = np.sqrt(lag1(0))
    nrm2 = np.sqrt(lag2(0))
    nrm0 = np.sqrt(rgamma(1 + Δ1) * rgamma(1 + Δ2))

    z = r * r
    base = np.exp(-z) * nrm0 * (r**power) * (lag1(z) / nrm1) * (lag2(z) / nrm2)

    if potential == "coulomb":
        return (kappa / np.sqrt(2.0)) * base
    if potential == "constant":
        return (kappa / (2 * np.pi)) * (r * base)
    if callable(potential):
        qphys = np.sqrt(2.0) * r
        return (potential(qphys) / (2 * np.pi)) * (r * base)
    raise ValueError("potential must be 'coulomb', 'constant', or callable V(q)")


def get_exchange_kernels_hankel(
    G_magnitudes: RealArray,
    G_angles: RealArray,
    nmax: int,
    *,
    potential: str | callable = "coulomb",
    kappa: float = 1.0,
    sign_magneticfield: int = -1,
) -> ComplexArray:
    """Compute X_{n1,m1,n2,m2}(G) via Hankel transforms (κ=1 convention).

    This backend parametrizes the radial integral via Hankel transforms with
    robust Laguerre-based normalization and explicit control over the Bessel
    order. It is numerically more intensive than the Gauss–Legendre backend
    but can be useful for cross-checks or alternative potentials.

    Parameters
    ----------
    G_magnitudes, G_angles :
        Arrays describing |G| and polar angle θ_G (same shape, no broadcasting).
    nmax :
        Number of Landau levels.
    potential :
        ``'coulomb'`` (default), ``'constant'``, or a callable ``V(q)`` giving
        the interaction in 1/ℓ units.
    kappa :
        Prefactor for Coulomb/constant cases.
    sign_magneticfield :
        Sign of the charge–field product σ = sgn(q B_z). ``-1`` matches the
        package's internal convention; ``+1`` returns the kernels for the
        opposite sign by applying the appropriate complex conjugation and
        phase factors.
    """
    if sign_magneticfield not in (1, -1):
        raise ValueError("sign_magneticfield must be 1 or -1")

    G_magnitudes = np.asarray(G_magnitudes, dtype=float)
    G_angles = np.asarray(G_angles, dtype=float)
    if G_magnitudes.shape != G_angles.shape:
        raise ValueError("G_magnitudes and G_angles must have same shape")

    # Unique |G| to shrink Hankel workload (same radial value for all with same |G|)
    # Note: np.unique sorts; use inverse map to scatter back to original order.
    k_unique, inv_idx = np.unique(G_magnitudes, return_inverse=True)
    nG = G_magnitudes.size

    # Resolve potential
    if callable(potential):
        pot_callable = potential
        pot_key = ("callable", id(potential))
    else:
        pot_name = str(potential).strip().lower()
        if pot_name in {"coulomb", "constant"}:
            pot_callable = pot_name
            pot_key = (pot_name, float(kappa))
        else:
            raise ValueError("potential must be 'coulomb', 'constant', or callable V(q)")

    # Precompute angular phase vectors for all possible N values
    # N = (n1 - m1) - (m2 - n2) ∈ [Nmin, Nmax]
    N_min = -2 * (nmax - 1)
    N_max = +2 * (nmax - 1)
    Ns = np.arange(N_min, N_max + 1, dtype=int)
    phase_by_N: dict[int, ComplexArray] = {}
    for N in Ns:
        phase = -N * G_angles
        phase_by_N[int(N)] = (np.cos(phase) + 1j * np.sin(phase)) * _parity_factor(int(N))

    # Small lookup for internal (i)^(d1+d2), indexed by d1,d2 in [0..nmax-1]
    d_vals = np.arange(nmax, dtype=int)
    phase_internal_table = (1j) ** (d_vals[:, None] + d_vals[None, :])  # (nmax,nmax)

    d_lookup = np.abs(np.subtract.outer(np.arange(nmax), np.arange(nmax)))  # (nmax,nmax)
    # Precompute abs diffs (d) and mins (Nmin) for quick indexing
    d_mat = d_lookup  # alias
    Nmin_mat = np.minimum(
        np.arange(nmax)[:, None].repeat(nmax, axis=1),
        np.arange(nmax)[None, :].repeat(nmax, axis=0),
    )

    # Cache for radial Hankel transforms keyed by (d1,N1,d2,N2,absN,pot_key)
    radial_cache: dict[tuple[int, int, int, int, int, tuple], np.ndarray] = {}

    # Output
    Xs = np.zeros((nG, nmax, nmax, nmax, nmax), dtype=np.complex128)

    # Helper to fetch/compute the radial piece for given indices
    def get_radial_block(n1: int, m1: int, n2: int, m2: int, absN: int) -> np.ndarray:
        d1 = int(d_mat[n1, m1])
        d2 = int(d_mat[n2, m2])
        N1 = int(Nmin_mat[n1, m1])
        N2 = int(Nmin_mat[n2, m2])
        key = (d1, N1, d2, N2, int(absN), pot_key)
        arr = radial_cache.get(key)
        if arr is not None:
            return arr

        def integrand(q):
            return _radial_exchange_integrand_rgamma(
                q, n1, m1, n2, m2, potential=pot_callable, kappa=kappa
            )

        ht = _get_hankel_transformer(absN)
        # Compute only on unique radii, then cache
        arr_unique = ht.transform(integrand, k_unique, ret_err=False)
        radial_cache[key] = arr_unique
        return arr_unique

    # Main assignment: iterate indices but reuse cached radial data and phases
    for n1 in range(nmax):
        for m1 in range(nmax):
            d1 = int(d_mat[n1, m1])
            for n2 in range(nmax):
                for m2 in range(nmax):
                    d2 = int(d_mat[n2, m2])
                    N = _N_order(n1, m1, n2, m2)
                    absN = abs(N)
                    # Radial part (on unique k), then scatter to all G via inv_idx
                    X_radial_unique = get_radial_block(n1, m1, n2, m2, absN)
                    X_radial = X_radial_unique[inv_idx]
                    # Angular/internal phases
                    phase_internal = phase_internal_table[d1, d2]
                    phase_angle = phase_by_N[N]
                    extra_sgn = (-1)**(n2-m2)

                    Xs[:, n1, m1, n2, m2] = phase_internal * phase_angle * X_radial * extra_sgn


    if sign_magneticfield == 1:
        Xs = get_exchange_kernels_opposite_field(Xs)

    return Xs


__all__ = ["get_exchange_kernels_hankel"]
