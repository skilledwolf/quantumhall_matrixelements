"""Plane-wave form factors F_{n',n}(G) in a Landau-level basis."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import eval_genlaguerre, gammaln

if TYPE_CHECKING:
    from numpy.typing import NDArray

    ComplexArray = NDArray[np.complex128]
    RealArray = NDArray[np.float64]
    IntArray = NDArray[np.int64]


def _analytic_form_factor(
    n_row: "IntArray",
    n_col: "IntArray",
    q_magnitudes: "RealArray",
    q_angles: "RealArray",
    lB: float,
) -> "ComplexArray":
    """Vectorized Landau level form factor F_{n_row, n_col}(q).

    F_{n',n}(q) = i^{|n-n'|} e^{i(n-n')θ}
                  sqrt(n_min!/n_max!) (|q|ℓ/√2)^{|n'-n|}
                  L_{n_min}^{|n'-n|}(|q|²ℓ²/2) e^{-|q|²ℓ²/4}
    """
    n_min = np.minimum(n_row, n_col)
    n_max = np.maximum(n_row, n_col)
    delta_n_abs = np.abs(n_row - n_col)

    ql = q_magnitudes * lB
    arg_z = 0.5 * (ql**2)

    log_ratio = 0.5 * (gammaln(n_min + 1) - gammaln(n_max + 1))
    ratio = np.exp(log_ratio)

    laguerre_poly = eval_genlaguerre(n_min, delta_n_abs, arg_z)

    # Phase convention: F_{n',n}(q) ∝ i^{|Δn|} e^{i (n - n') θ}
    angles = (n_col - n_row) * q_angles + (np.pi / 2) * delta_n_abs
    angular_phase = np.cos(angles) + 1j * np.sin(angles)

    F = (
        angular_phase
        * ratio
        * np.power(ql / np.sqrt(2.0), delta_n_abs)
        * laguerre_poly
        * np.exp(-0.5 * arg_z)
    )
    return F if F.ndim > 0 else F[()]


def get_form_factors(
    q_magnitudes: "RealArray", q_angles: "RealArray", nmax: int, lB: float = 1.0
) -> "ComplexArray":
    """Precompute F_{n',n}(G) for all G and Landau levels.

    Parameters
    ----------
    q_magnitudes, q_angles :
        Arrays with the same shape, describing |G|ℓ_B and polar angle θ.
    nmax :
        Number of Landau levels (0..nmax-1).
    lB :
        Magnetic length ℓ_B (default 1.0). ``q_magnitudes`` are understood
        to be in units of 1/ℓ_B.

    Returns
    -------
    F : (nG, nmax, nmax) complex array
        Plane-wave form factors F_{n',n}(G).
    """
    n_indices = np.arange(nmax)
    return _analytic_form_factor(
        n_row=n_indices[None, :, None],
        n_col=n_indices[None, None, :],
        q_magnitudes=np.asarray(q_magnitudes)[:, None, None],
        q_angles=np.asarray(q_angles)[:, None, None],
        lB=lB,
    ).astype(np.complex128)


__all__ = ["get_form_factors"]

