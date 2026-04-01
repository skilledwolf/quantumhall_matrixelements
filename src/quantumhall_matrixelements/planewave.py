"""Plane-wave form factors F_{n',n}(G) in a Landau-level basis."""
from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_genlaguerre, gammaln

from .diagnostic import get_form_factors_opposite_field

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

def _analytic_form_factor(
    n_row: IntArray,
    n_col: IntArray,
    q_magnitudes: RealArray,
    q_angles: RealArray,
    lB: float,
    sign_magneticfield: int = -1,
) -> ComplexArray:
    """Vectorized Landau level form factor F_{n_row, n_col}(q).

    F_{n',n}(q) = i^{|n-n'|} e^{i(n-n')θ}
                  sqrt(n_min!/n_max!) (|q|ℓ/√2)^{|n'-n|}
                  L_{n_min}^{|n'-n|}(|q|²ℓ²/2) e^{-|q|²ℓ²/4}
    """

    if sign_magneticfield not in (1, -1):
        raise ValueError("sign_magneticfield must be 1 or -1")
    n_min = np.minimum(n_row, n_col)
    n_max = np.maximum(n_row, n_col)
    delta_n_abs = np.abs(n_row - n_col)

    ql = q_magnitudes * lB
    arg_z = 0.5 * (ql**2)

    log_ratio = 0.5 * (gammaln(n_min + 1) - gammaln(n_max + 1))
    ratio = np.exp(log_ratio)

    laguerre_poly = eval_genlaguerre(n_min, delta_n_abs, arg_z)

    angles = -sign_magneticfield * (n_col - n_row) * q_angles + (np.pi / 2) * delta_n_abs
    angular_phase = np.cos(angles) + 1j * np.sin(angles)

    F = (
        angular_phase
        * ratio
        * np.power(ql / np.sqrt(2.0), delta_n_abs)
        * laguerre_poly
        * np.exp(-0.5 * arg_z)
    )
    return cast(ComplexArray, F)

def get_form_factors(
    q_magnitudes: RealArray,
    q_angles: RealArray,
    nmax: int,
    lB: float = 1.0,
    sign_magneticfield: int = -1,
) -> ComplexArray:
    """Precompute F_{n',n}(G) for all G and Landau levels.

    Parameters
    ----------
    q_magnitudes, q_angles :
        Arrays with the same shape, describing the wavevector magnitude ``|q|``
        and polar angle ``θ``.
    nmax :
        Number of Landau levels (0..nmax-1).
    lB :
        Magnetic length ``ℓ_B`` (default 1.0). The form factors depend on the
        dimensionless combination ``|q|ℓ_B``, so the implementation multiplies
        ``q_magnitudes`` by ``lB`` internally. If you already work with
        dimensionless ``|q|ℓ_B`` values, leave ``lB=1`` and pass those values
        directly as ``q_magnitudes``.
    sign_magneticfield :
        Sign of the charge–field product σ = sgn(q B_z). Use ``-1`` for the
        electron/positive-B convention used internally; ``+1`` returns the
        complex-conjugated form factors with the appropriate phase flip.

    Returns
    -------
    F : (nG, nmax, nmax) complex array
        Plane-wave form factors F_{n',n}(G).
    """
    if sign_magneticfield not in (1, -1):
        raise ValueError("sign_magneticfield must be 1 or -1")
    q_magnitudes = np.asarray(q_magnitudes, dtype=float).ravel()
    q_angles = np.asarray(q_angles, dtype=float).ravel()
    if q_magnitudes.shape != q_angles.shape:
        raise ValueError("q_magnitudes and q_angles must have the same shape.")
    n_indices = np.arange(nmax)
    F = _analytic_form_factor(
        n_row=n_indices[None, :, None],
        n_col=n_indices[None, None, :],
        q_magnitudes=q_magnitudes[:, None, None],
        q_angles=q_angles[:, None, None],
        lB=lB
    ).astype(np.complex128)

    # Just to be explicit, we apply the symmetry transformation explicitly here
    # but we could have also passed sign_magneticfield to _analytic_form_factor
    # --> same result
    if sign_magneticfield == 1:
        F = get_form_factors_opposite_field(F)

    return F


__all__ = ["get_form_factors"]
