"""Diagnostic helpers for exchange-kernel symmetry checks."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import get_exchange_kernels

if TYPE_CHECKING:  # pragma: no cover - aliases only
    from numpy.typing import NDArray

    RealArray = NDArray[np.float64]
    ComplexArray = NDArray[np.complex128]

__all__ = [
    "verify_exchange_kernel_symmetries",
]


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

      Xs[g, n1, m1, n2, m2]_(-G) = Xs[g, m2, n2, m1, n1]_G^*.
    """
    Xs_G = get_exchange_kernels(G_magnitudes, G_angles, nmax)
    G_angles_minus = (G_angles + np.pi) % (2 * np.pi)
    Xs_minusG = get_exchange_kernels(G_magnitudes, G_angles_minus, nmax)

    # expected_minus[g, n1, m1, n2, m2] = X_G[g, m2, n2, m1, n1]^*
    expected_Xs_minusG = np.transpose(Xs_G, (0, 4, 3, 2, 1)).conj()
    if not np.allclose(Xs_minusG, expected_Xs_minusG, rtol=rtol, atol=atol):
        diff = float(np.max(np.abs(Xs_minusG - expected_Xs_minusG)))
        raise AssertionError(
            f"Exchange kernel G-inversion symmetry failed: max|Δ|={diff:.3e} "
            f"(rtol={rtol}, atol={atol})"
        )
