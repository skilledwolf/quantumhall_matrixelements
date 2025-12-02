"""Exchange kernels via Gauss-Legendre quadrature with rational mapping."""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import scipy.special as sps
from scipy.special import roots_legendre

if TYPE_CHECKING:
    from numpy.typing import NDArray

    ComplexArray = NDArray[np.complex128]
    RealArray = NDArray[np.float64]


def _N_order(n1: int, m1: int, n2: int, m2: int) -> int:
    #return (n1 - m1) - (m2 - n2)# change here 
    return ((n1 - m1) + (m2 - n2))


def _parity_factor(N: int) -> int:
    #"""(-1)^((N+|N|)/2) → (-1)^N for N>=0, and 1 for N<0."""
    """(-1)^((N-|N|)/2) → (-1)^N for N<=0, and 1 for N>0."""# CHANGE HERE NEGATIVE B FIELD
    #return (-1) ** ((N + abs(N)) // 2) # change here
    return (-1) ** ((N - abs(N)) // 2) # CHANGE HERE NEGATIVE B FIELD


@lru_cache(maxsize=None)
def _logfact(n: int) -> float:
    return float(sps.gammaln(n + 1))


def _C_and_indices(n1: int, m1: int, n2: int, m2: int):
    """Constants and Laguerre parameters for f_{n1,m1} * f_{m2,n2}."""
    p, d1 = min(n1, m1), abs(n1 - m1)
    q, d2 = min(m2, n2), abs(m2 - n2)
    logC = 0.5 * ((_logfact(p) - _logfact(p + d1)) + (_logfact(q) - _logfact(q + d2)))
    C = np.exp(logC)
    return C, p, d1, q, d2


@lru_cache(maxsize=None)
def _legendre_nodes_weights_mapped(nquad: int, scale: float):
    """
    Gauss-Legendre nodes/weights mapped from [-1, 1] to [0, inf).
    Mapping: z = scale * (1+x)/(1-x)
    Jacobian: dz/dx = scale * 2/(1-x)^2
    """
    x, w_leg = roots_legendre(nquad)
    denom = 1.0 - x
    z = scale * (1.0 + x) / denom
    w = w_leg * (scale * 2.0 / (denom * denom))
    return z, w


def get_exchange_kernels_GaussLegendre(
    G_magnitudes,
    G_angles,
    nmax: int,
    *,
    potential: str | callable = "coulomb",
    kappa: float = 1.0,
    nquad: int = 1000,
    scale: float = 0.5,
    ell: float = 1.0,
    sigma: int = -1,
) -> "ComplexArray":
    """Compute X_{n1,m1,n2,m2}(G) using Gauss-Legendre quadrature with rational mapping.

    This backend maps the semi-infinite radial integral to the finite interval [-1, 1]
    using the rational mapping z = scale * (1+x)/(1-x). It avoids the numerical instability
    of Gauss-Laguerre quadrature for large quantum numbers while remaining faster than
    Hankel transforms.

    Parameters
    ----------
    G_magnitudes, G_angles :
        Arrays of the same shape describing |G| and polar angle θ_G.
    nmax :
        Number of Landau levels.
    potential :
        ``'coulomb'`` (default) or a callable ``V(q)`` returning the interaction.
    kappa :
        Interaction strength prefactor.
    nquad :
        Number of quadrature points (default 1000).
    scale :
        Mapping scale factor (default 0.5). Controls the distribution of points.
        Smaller values cluster points near the peak of the integrand for large n.
    ell :
        Magnetic length ℓ_B (default 1.0).

    Returns
    -------
    Xs : (nG, nmax, nmax, nmax, nmax) complex array
    """
    if sigma not in (1, -1):
        raise ValueError("sigma must be 1 or -1")
    
    G_magnitudes = np.asarray(G_magnitudes, dtype=float)
    G_angles = np.asarray(G_angles, dtype=float)
    if G_magnitudes.shape != G_angles.shape:
        raise ValueError("G_magnitudes and G_angles must have the same shape.")
    nG = G_magnitudes.size

    Gscaled = G_magnitudes * float(ell)
    Xs = np.zeros((nG, nmax, nmax, nmax, nmax), dtype=np.complex128)

    # Resolve potential
    if callable(potential):
        pot_kind = "callable"
        pot_fn = potential
    else:
        pot_kind = str(potential).strip().lower()
        pot_fn = None

    if pot_kind == "coulomb":
        pass
    elif pot_kind == "callable":
        pass
    else:
        raise ValueError("potential must be 'coulomb' or a callable V(q).")

    # Get mapped grid
    z, w = _legendre_nodes_weights_mapped(nquad, scale)

    # Precompute Bessel functions J_N(sqrt(2z)*G)
    # We cache by absN to avoid recomputing
    J_cache: dict[int, np.ndarray] = {}

    # Cache for Laguerre evaluations
    # We evaluate L_n^d(z) for many n, d.
    # Since n, d are small integers, we can just compute on the fly or use sps.eval_genlaguerre
    # sps.eval_genlaguerre is efficient enough.

    for n1 in range(nmax):
        for m1 in range(nmax):
            for n2 in range(nmax):
                for m2 in range(nmax):
                    N = _N_order(n1, m1, n2, m2)
                    absN = abs(N)
                    C, p, d1, q, d2 = _C_and_indices(n1, m1, n2, m2)

                    # Compute radial integral
                    if potential == "coulomb":
                        # Integrand factor: exp(-z) * z^alpha * L * L * J
                        # alpha = (d1 + d2 - 1) / 2
                        alpha = 0.5 * (d1 + d2 - 1)

                        L1 = sps.eval_genlaguerre(p, d1, z)
                        L2 = sps.eval_genlaguerre(q, d2, z)
                        
                        # Bessel part
                        if absN not in J_cache:
                            arg = np.sqrt(2.0 * z)[None, :] * Gscaled[:, None]
                            J_cache[absN] = sps.jv(absN, arg)
                        J_abs = J_cache[absN]
                        
                        # Full integrand term (excluding J and weights)
                        # exp(-z) handles x->1 decay
                        # z^alpha handles x->-1 behavior
                        term = np.exp(-z) * (z**alpha) * L1 * L2
                        
                        # Sum over quadrature points
                        # J_abs is (nG, nquad), term is (nquad,), w is (nquad,)
                        # Result is (nG,)
                        radial = (J_abs * term) @ w
                        
                        signN = _parity_factor(N)

                        #phase_factor = (1j) ** (d1 - d2) ## changed here
                        phase_factor = (1j) ** (d1 + d2)  #CHANGE HERE NEGATIVE B FIELD 
                        pref = (kappa * C / np.sqrt(2.0)) * phase_factor
                        
                    else:
                        # General/callable potential
                        alpha = 0.5 * (d1 + d2)
                        L1 = sps.eval_genlaguerre(p, d1, z)
                        L2 = sps.eval_genlaguerre(q, d2, z)
                        
                        qvals = np.sqrt(2.0 * z) / float(ell)
                        Veff = pot_fn(qvals) / (2.0 * np.pi * float(ell) ** 2)
                        
                        if absN not in J_cache:
                            arg = np.sqrt(2.0 * z)[None, :] * Gscaled[:, None]
                            J_cache[absN] = sps.jv(absN, arg)
                        J_abs = J_cache[absN]
                        
                        term = np.exp(-z) * (z**alpha) * L1 * L2 * Veff
                        radial = (J_abs * term) @ w
                        
                        signN = _parity_factor(N)
                        #phase_factor = (1j) ** (d1 - d2) ## changed here
                        phase_factor = (1j) ** (d1 + d2) # CHANGED HERE NEGATIVE B FIELD 
                        pref = C * phase_factor

                    phase = np.exp(-1j * N * G_angles) #CHANGED HERE NEGATIVE B FIELD 
                    #phase = np.exp(1j * N * G_angles)

                    extra_sgn = (-1)**(n2-m2) # CHANGED HERE NEGATIVE B FIELD  

                    Xs[:, n1, m1, n2, m2] = (pref * phase) * (signN * radial) * extra_sgn # CHANGED HERE NEGATIVE B FIELD

    if sigma == -1: #matching convention in package
        return Xs
    else: # sigma == 1, apply phase factor for positive B field
        idx = np.arange(Xs.shape[1])
        phase = np.where((idx[:, None] - idx[None, :]) % 2 == 0, 1.0, -1.0)
        phase = phase[:, :, None, None] * phase[None, None, :, :]
        return np.conj(Xs) * phase


__all__ = ["get_exchange_kernels_GaussLegendre"]

"""
Suggestion for changes:
Add a B field direction option. -ve one just requires complex conjugate and sign factor

X^+_{n1,m1,n2,m2}(G) = (X^-_{n1,m1,n2,m2}(G))^* (-1)**(i - j + l - k)

F^+_{n',n}(q) = (F^-_{n',n}(q))^* (-1)**(n' - n)

Allan's code = -ve magnetic field 
This package = +ve magnetic field

Could add the actual wavefunction, hamiltonian and creation and 
annihilation operators in the Readme so that its clear what the convention is for 
either magnetic fields 

"""