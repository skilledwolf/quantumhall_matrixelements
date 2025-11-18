"""Exchange kernels via generalized Gauss–Laguerre quadrature."""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import scipy.special as sps

if TYPE_CHECKING:
    from numpy.typing import NDArray

    ComplexArray = NDArray[np.complex128]
    RealArray = NDArray[np.float64]


def _N_order(n1: int, m1: int, n2: int, m2: int) -> int:
    return (n1 - m1) - (m2 - n2)


def _parity_factor(N: int) -> int:
    """(-1)^((N+|N|)/2) → (-1)^N for N>=0, and 1 for N<0."""
    return (-1) ** ((N + abs(N)) // 2)


@lru_cache(maxsize=None)
def _lag_nodes_weights(nquad: int, alpha: float):
    """Generalized Gauss–Laguerre nodes/weights for ∫_0^∞ e^{-z} z^α f(z) dz."""
    x, w = sps.roots_genlaguerre(nquad, alpha)
    return x, w


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


_L_cache: dict[tuple[int, int, float, int], np.ndarray] = {}


def _laguerre_on_grid(p: int, d: int, alpha: float, nquad: int, z):
    key = (p, d, float(alpha), int(nquad))
    L = _L_cache.get(key)
    if L is None:
        L = sps.eval_genlaguerre(p, d, z)
        _L_cache[key] = L
    return L


def get_exchange_kernels_GaussLag(
    G_magnitudes,
    G_angles,
    nmax: int,
    *,
    potential: str = "coulomb",
    kappa: float = 1.0,
    V_of_q=None,
    nquad: int = 200,
    ell: float = 1.0,
) -> "ComplexArray":
    """Compute X_{n1,m1,n2,m2}(G) using analytic angle and Gauss–Laguerre radial quadrature.

    Parameters
    ----------
    G_magnitudes, G_angles :
        Arrays of the same shape describing |G| and polar angle θ_G.
    nmax :
        Number of Landau levels.
    potential :
        Either ``'coulomb'`` (default) or ``'general'``. In the latter case
        a callable ``V_of_q(q)`` must be provided.
    kappa :
        Interaction strength prefactor. For Coulomb this corresponds to
        :math:`\\kappa = e^2/(\\varepsilon\\ell_B)/\\hbar\\omega_c`.
    V_of_q :
        Callable ``V_of_q(q) -> V(q)`` used when ``potential='general'``.
    nquad :
        Number of Gauss–Laguerre quadrature points.
    ell :
        Magnetic length ℓ_B (default 1.0); |G| is interpreted in 1/ℓ_B units.

    Returns
    -------
    Xs : (nG, nmax, nmax, nmax, nmax) complex array
        Exchange kernels normalized with the chosen kappa.
    """
    G_magnitudes = np.asarray(G_magnitudes, dtype=float)
    G_angles = np.asarray(G_angles, dtype=float)
    if G_magnitudes.shape != G_angles.shape:
        raise ValueError("G_magnitudes and G_angles must have the same shape.")
    nG = G_magnitudes.size

    Gscaled = G_magnitudes * float(ell)
    Xs = np.zeros((nG, nmax, nmax, nmax, nmax), dtype=np.complex128)

    J_cache: dict[tuple[int, float], np.ndarray] = {}

    for n1 in range(nmax):
        for m1 in range(nmax):
            for n2 in range(nmax):
                for m2 in range(nmax):
                    N = _N_order(n1, m1, n2, m2)
                    absN = abs(N)
                    C, p, d1, q, d2 = _C_and_indices(n1, m1, n2, m2)

                    if potential == "coulomb":
                        alpha = 0.5 * (d1 + d2 - 1)
                        if alpha <= -1:
                            raise ValueError(f"Invalid alpha={alpha} for Coulomb case.")
                        z, w = _lag_nodes_weights(nquad, alpha)
                        L1 = _laguerre_on_grid(p, d1, alpha, nquad, z)
                        L2 = _laguerre_on_grid(q, d2, alpha, nquad, z)
                        W = w * L1 * L2
                        key = (absN, float(alpha))
                        J_abs = J_cache.get(key)
                        if J_abs is None:
                            arg = np.sqrt(2.0 * z)[None, :] * Gscaled[:, None]
                            J_abs = sps.jv(absN, arg)
                            J_cache[key] = J_abs
                        signN = _parity_factor(N)
                        radial = (signN * J_abs) @ W
                        phase_factor = (1j) ** (d1 - d2)
                        pref = (kappa * C / np.sqrt(2.0)) * phase_factor
                    else:
                        if not callable(V_of_q):
                            raise ValueError(
                                "For potential='general', provide V_of_q: callable(q)->V(q)."
                            )
                        alpha = 0.5 * (d1 + d2)
                        z, w = _lag_nodes_weights(nquad, alpha)
                        L1 = _laguerre_on_grid(p, d1, alpha, nquad, z)
                        L2 = _laguerre_on_grid(q, d2, alpha, nquad, z)
                        qvals = np.sqrt(2.0 * z) / float(ell)
                        Veff = V_of_q(qvals) / (2.0 * np.pi * float(ell) ** 2)
                        W = w * L1 * L2 * Veff
                        key = (absN, float(alpha))
                        J_abs = J_cache.get(key)
                        if J_abs is None:
                            arg = np.sqrt(2.0 * z)[None, :] * Gscaled[:, None]
                            J_abs = sps.jv(absN, arg)
                            J_cache[key] = J_abs
                        signN = _parity_factor(N)
                        radial = (signN * J_abs) @ W
                        phase_factor = (1j) ** (d1 - d2)
                        pref = C * phase_factor

                    phase = np.exp(-1j * N * G_angles)
                    Xs[:, n1, m1, n2, m2] = (pref * phase) * radial

    return Xs


__all__ = ["get_exchange_kernels_GaussLag"]

