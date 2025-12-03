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

from .diagnostic import get_exchange_kernels_opposite_field


def _parity_factor(N: int) -> int:
    """(-1)^((N-|N|)/2) → (-1)^N for N<0, and 1 for N>=0."""
    return (-1) ** ((N - abs(N)) // 2) 

@lru_cache(maxsize=None)
def _logfact(n: int) -> float:
    return float(sps.gammaln(n + 1))

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
    nquad: int = 8000,
    scale: float = 0.5,
    ell: float = 1.0,
    sign_magneticfield: int = -1,
) -> "ComplexArray":
    """Compute exchange kernels X_{n1,m1,n2,m2}(G) using Gauss-Legendre quadrature.

    This function evaluates the exchange matrix elements for a 2D electron gas
    in a magnetic field, using Gauss-Legendre quadrature with a rational mapping
    from [-1, 1] to [0, ∞). The implementation exploits index-exchange symmetry
    to reduce computation time.

    Parameters
    ----------
    G_magnitudes : array_like of float
        Magnitudes |G| of the reciprocal lattice vectors, convertible to a
        NumPy array of floats. Shape determines the output's leading dimension.
    G_angles : array_like of float
        Polar angles θ_G of the reciprocal lattice vectors in radians,
        convertible to a NumPy array of floats. Must have the same shape as
        ``G_magnitudes``.
    nmax : int
        Number of Landau levels to include in the calculation. Output arrays
        will have dimensions ``(nG, nmax, nmax, nmax, nmax)``.
    potential : str or callable, optional
        Interaction potential. Use ``'coulomb'`` (default) for Coulomb
        interaction, or provide a callable ``V(q)`` that takes momentum
        magnitude (in units of 1/ℓ) and returns the interaction strength.
    kappa : float, optional
        Prefactor for the Coulomb potential. Default is 1.0.
    nquad : int, optional
        Number of quadrature points for the Gauss-Legendre integration.
        Default is 8000. Higher values improve accuracy at computational cost.
    scale : float, optional
        Scale parameter for the rational mapping z = scale * (1+x)/(1-x).
        Default is 0.5. Adjust to optimize convergence for different potentials.
    ell : float, optional
        Magnetic length in the same units as G_magnitudes. Default is 1.0.
    sign_magneticfield : int, optional
        Sign of the charge–field product σ = sgn(q B_z). Must be -1 or +1.
        Default is -1 (internal convention). Use +1 to obtain kernels for
        the opposite magnetic field orientation via conjugation and phase
        factors.

    Returns
    -------
    numpy.ndarray of numpy.complex128
        Exchange kernels with shape ``(nG, nmax, nmax, nmax, nmax)``, where
        ``nG`` is the number of G-vectors. The array element
        ``Xs[g, n1, m1, n2, m2]`` gives the exchange matrix element
        X_{n1,m1,n2,m2}(G_g).

    Raises
    ------
    ValueError
        If ``G_magnitudes`` and ``G_angles`` have different shapes, if
        ``potential`` is not ``'coulomb'`` or a callable, or if a callable
        potential returns an array with shape different from ``(nquad,)``.

    Notes
    -----
    **Optimizations:**

    - Precomputes Laguerre polynomials L_p^d(z) once for all (n, m) pairs.
    - Precomputes z^α for all possible values of d1 + d2.
    - Groups index quadruples by Bessel order N and evaluates integrals in
      batched matrix operations.
    - Exploits index-exchange symmetry (valid for any V(q)):
      ``X[m2, n2, m1, n1] = (-1)^((n1-m1)-(n2-m2)) * X[n1, m1, n2, m2]``
      to fill symmetric partners without additional O(nmax^4) loops.
    """

    # -----------------------------
    # 0. Input handling
    # -----------------------------
    G_magnitudes = np.asarray(G_magnitudes, dtype=float)
    G_angles = np.asarray(G_angles, dtype=float)
    if G_magnitudes.shape != G_angles.shape:
        raise ValueError("G_magnitudes and G_angles must have the same shape.")
    nG = G_magnitudes.size

    # -----------------------------
    # 1. Potential choice
    # -----------------------------
    if callable(potential):
        pot_kind = "callable"
        pot_fn = potential
    else:
        pot_kind = str(potential).strip().lower()
        pot_fn = None

    if pot_kind not in ("coulomb", "callable"):
        raise ValueError("potential must be 'coulomb' or a callable V(q).")
    is_coulomb = (pot_kind == "coulomb")

    # -----------------------------
    # 2. Quadrature grid and z-dependent stuff
    # -----------------------------
    z, w = _legendre_nodes_weights_mapped(nquad, scale)
    z = np.asarray(z, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    exp_minus_z = np.exp(-z)               # (nquad,)
    sqrt2z = np.sqrt(2.0 * z)              # (nquad,)

    # Bessel argument (shared for all orders)
    Gscaled = G_magnitudes * float(ell)    # (nG,)
    arg = Gscaled[:, None] * sqrt2z[None, :]  # (nG, nquad)

    # Callable potential: evaluated once on the quadrature grid
    if is_coulomb:
        Veff = None
    else:
        qvals = sqrt2z / float(ell)        # (nquad,)
        Veff = pot_fn(qvals) / (2.0 * np.pi * float(ell) ** 2)
        Veff = np.asarray(Veff)
        if Veff.shape != z.shape:
            raise ValueError("Callable potential must return array of shape (nquad,)")

    # -----------------------------
    # 3. Per-(n,m) combinatorics
    # -----------------------------
    idx = np.arange(nmax, dtype=int)
    n_idx, m_idx = np.meshgrid(idx, idx, indexing="ij")

    # p = min(n,m), d = |n-m|, D = n-m
    p_nm = np.minimum(n_idx, m_idx)
    d_nm = np.abs(n_idx - m_idx)
    D_nm = n_idx - m_idx

    # (-1)^(n-m) done via parity bits (no pow on negatives)
    extra_sign_nm = 1 - 2 * ((n_idx - m_idx) & 1)  # shape (nmax, nmax)

    # C_nm[n,m] = sqrt(p! / (p + d)!)
    C_nm = np.empty((nmax, nmax), dtype=np.float64)
    for n in range(nmax):
        for m in range(nmax):
            p = int(p_nm[n, m])
            d = int(d_nm[n, m])
            logC = 0.5 * (_logfact(p) - _logfact(p + d))
            C_nm[n, m] = np.exp(logC)

    # -----------------------------
    # 4. Laguerre polynomials L_p^d(z) cached by (p,d)
    # -----------------------------
    laguerre_cache: dict[tuple[int, int], np.ndarray] = {}
    L_nm = np.empty((nmax, nmax), dtype=object)
    for n in range(nmax):
        for m in range(nmax):
            p = int(p_nm[n, m])
            d = int(d_nm[n, m])
            key = (p, d)
            if key not in laguerre_cache:
                laguerre_cache[key] = sps.eval_genlaguerre(p, d, z)
            L_nm[n, m] = laguerre_cache[key]

    # -----------------------------
    # 5. Powers z^alpha for all possible d1 + d2
    # -----------------------------
    max_d_sum = 2 * (nmax - 1)
    z_pows: list[np.ndarray] = []
    if is_coulomb:
        # alpha = (d1 + d2 - 1) / 2
        for ds in range(max_d_sum + 1):
            alpha = 0.5 * (ds - 1)
            z_pows.append(z ** alpha)
    else:
        # alpha = (d1 + d2) / 2
        for ds in range(max_d_sum + 1):
            alpha = 0.5 * ds
            z_pows.append(z ** alpha)

    # -----------------------------
    # 6. N-related: parity, plane-wave phase, Bessel cache
    # -----------------------------
    maxD = 2 * (nmax - 1)
    Ns = np.arange(-maxD, maxD + 1, dtype=int)
    minN = Ns[0]

    parity = np.array([_parity_factor(int(N)) for N in Ns], dtype=int)
    phase_table = np.exp(-1j * Ns[:, None] * G_angles[None, :])  # (2*maxD+1, nG)

    # (1j)^(d1 + d2) for all possible sums
    phase_power = np.array([1j ** k for k in range(max_d_sum + 1)], dtype=np.complex128)

    # Group quadruples by N, but only for canonical pairs (n1,m1) <= (m2,n2)
    buckets: dict[int, list[tuple[int, int, int, int]]] = {int(N): [] for N in Ns}
    for n1 in range(nmax):
        for m1 in range(nmax):
            D1 = D_nm[n1, m1]
            pair1 = n1 * nmax + m1          # "pair" index for (n1,m1)
            for n2 in range(nmax):
                for m2 in range(nmax):
                    pair2 = m2 * nmax + n2  # "pair" index for (m2,n2)
                    if pair1 > pair2:
                        # Non-canonical representative; its partner will be filled by symmetry
                        continue
                    # second physical pair is (m2, n2)
                    D2 = D_nm[m2, n2]       # (m2 - n2)
                    # N = (n1 - m1) + (m2 - n2) = D1 + D2
                    N = int(D1 + D2)
                    buckets[N].append((n1, m1, n2, m2))

    # Bessel J_|N|(arg) cache
    J_cache: dict[int, np.ndarray] = {}

    # Output array
    Xs = np.zeros((nG, nmax, nmax, nmax, nmax), dtype=np.complex128)
    sqrt2 = np.sqrt(2.0)

    # -----------------------------
    # 7. Main loop over N buckets
    # -----------------------------
    for N in Ns:
        quad_list = buckets[int(N)]
        if not quad_list:
            continue

        absN = abs(int(N))
        if absN not in J_cache:
            J_cache[absN] = sps.jv(absN, arg)  # (nG, nquad)
        J_abs = J_cache[absN]

        N_idx = int(N - minN)
        signN = parity[N_idx]
        phase_N = phase_table[N_idx]          # (nG,)

        terms = []        # list of (nquad,) arrays
        coeffs = []       # scalar prefactors
        extra_sgns = []   # (-1)^{n2 - m2}
        quads = []        # store quadruples in this bucket

        for (n1, m1, n2, m2) in quad_list:
            # d1,d2 and Laguerres for first pair (n1,m1) and second phys. pair (m2,n2)
            d1 = int(d_nm[n1, m1])
            d2 = int(d_nm[m2, n2])
            ds = d1 + d2

            L1 = L_nm[n1, m1]
            L2 = L_nm[m2, n2]
            z_alpha = z_pows[ds]

            if is_coulomb:
                term = exp_minus_z * z_alpha * L1 * L2
            else:
                term = exp_minus_z * z_alpha * L1 * L2 * Veff
            terms.append(term)

            C1 = C_nm[n1, m1]
            C2 = C_nm[m2, n2]
            phase_factor = phase_power[ds]

            if is_coulomb:
                pref = (kappa * C1 * C2 / sqrt2) * phase_factor
            else:
                pref = (C1 * C2) * phase_factor

            coeffs.append(pref)
            # extra sign is (-1)^(n2 - m2)
            extra_sgns.append(extra_sign_nm[n2, m2])
            quads.append((n1, m1, n2, m2))

        if not terms:
            continue

        # Stack into a single matrix: T has shape (nquad, nQ_N)
        T = np.stack(terms, axis=1)                 # (nquad, nQ_N)
        # big batched integral: (nG, nquad) @ (nquad, nQ_N) -> (nG, nQ_N)
        radial_all = (J_abs * w[None, :]) @ T       # (nG, nQ_N)

        coeffs = np.asarray(coeffs, dtype=np.complex128)       # (nQ_N,)
        extra_sgns = np.asarray(extra_sgns, dtype=np.int8)     # (nQ_N,)
        scalar_all = coeffs * extra_sgns * signN               # (nQ_N,)

        # tmp[:, q] = phase_N * scalar_all[q] * radial_all[:, q]
        tmp = phase_N[:, None] * radial_all * scalar_all[None, :]  # (nG, nQ_N)

        # Scatter into Xs, and fill symmetric partner on the fly
        for iq, (n1, m1, n2, m2) in enumerate(quads):
            val = tmp[:, iq]
            Xs[:, n1, m1, n2, m2] = val

            # symmetric partner: (m2, n2, m1, n1)
            pair1 = n1 * nmax + m1
            pair2 = m2 * nmax + n2
            if pair1 < pair2:
                # sign = (-1)**((n1 - m1) - (n2 - m2)) via parity bits
                delta = (n1 - m1) - (n2 - m2)
                sign = 1 if (delta & 1) == 0 else -1
                Xs[:, m2, n2, m1, n1] = sign * val

    if sign_magneticfield == 1:
        Xs = get_exchange_kernels_opposite_field(Xs)

    return Xs



__all__ = ["get_exchange_kernels_GaussLegendre"]
