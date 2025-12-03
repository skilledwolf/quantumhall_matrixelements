"""Diagnostic helpers for exchange-kernel symmetry checks."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - aliases only
    from numpy.typing import NDArray

    RealArray = NDArray[np.float64]
    ComplexArray = NDArray[np.complex128]

__all__ = [
    "get_form_factors_opposite_field",
    "get_exchange_kernels_opposite_field",
    "verify_exchange_kernel_symmetries",
]


def get_form_factors_opposite_field(F: "ComplexArray") -> "ComplexArray":
    """Transform form factors to the opposite magnetic-field sign (σ→-σ).

    Parameters
    ----------
    F : (nG, nmax, nmax) complex array
        Form factors for ``sign_magneticfield = -1``.

    Returns
    -------
    ComplexArray
        Form factors for ``sign_magneticfield = +1`` obtained via conjugation
        and the standard phase flip.
    """

    nmax = F.shape[1]
    idx = np.arange(nmax)
    phase = np.where((idx[:, None] - idx[None, :]) % 2 == 0, 1.0, -1.0)
    return np.conj(F) * phase


def get_exchange_kernels_opposite_field(Xs: "ComplexArray") -> "ComplexArray":
    """Transform exchange kernels to the opposite magnetic-field sign (σ→-σ).

    Parameters
    ----------
    Xs : (nG, nmax, nmax, nmax, nmax) complex array
        Exchange kernels for ``sign_magneticfield = -1``.

    Returns
    -------
    ComplexArray
        Exchange kernels for ``sign_magneticfield = +1``.
    """

    nmax = Xs.shape[1]
    idx = np.arange(nmax)
    phase = np.where((idx[:, None] - idx[None, :]) % 2 == 0, 1.0, -1.0)
    phase = phase[:, :, None, None] * phase[None, None, :, :]
    return np.conj(Xs) * phase


def verify_exchange_kernel_symmetries(
    G_magnitudes: "RealArray",
    G_angles: "RealArray",
    nmax: int,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> None:
    """Verify the exchange-kernel G-inversion symmetry implied by Σ^F(-G)=Σ^F(G)^†.

    With the convention

      Σ^F_{mn}(G) = - Σ_{r,t} X_{nrtm}(G) ρ_{tr}(G),

    Hermiticity Σ^F(-G) = Σ^F(G)^† for densities obeying ρ(-G)=ρ(G)^† requires

      X_{nrtm}(-G) = X_{m r t n}(G)^*,

    i.e. in array form (Xs[g, n1, m1, n2, m2]):

      Xs[g, n1, m1, n2, m2]_(-G) = Xs[g, n2, m2, n1, m1]_G^*.
    """
    # Lazy import to avoid circular dependency when module is imported from __init__
    from . import get_exchange_kernels

    Xs_G = get_exchange_kernels(G_magnitudes, G_angles, nmax)
    G_angles_minus = (G_angles + np.pi) % (2 * np.pi)
    Xs_minusG = get_exchange_kernels(G_magnitudes, G_angles_minus, nmax)

    # expected_minus[g, n1, m1, n2, m2] = X_G[g, m2, n2, m1, n1]^*
    expected_Xs_minusG = np.transpose(Xs_G, (0, 3, 4, 1, 2)).conj()
    if not np.allclose(Xs_minusG, expected_Xs_minusG, rtol=rtol, atol=atol):
        diff = float(np.max(np.abs(Xs_minusG - expected_Xs_minusG)))
        raise AssertionError(
            f"Exchange kernel G-inversion symmetry failed: max|Δ|={diff:.3e} "
            f"(rtol={rtol}, atol={atol})"
        )
