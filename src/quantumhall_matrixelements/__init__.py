"""Landau-level plane-wave form factors and exchange kernels.

This package provides reusable numerical kernels for quantum Hall matrix
elements in a Landau-level basis:

- `get_form_factors` for plane-wave form factors :math:`F_{n',n}(G)`.
- `get_exchange_kernels` (and backend-specific variants) for exchange kernels
  :math:`X_{n_1 m_1 n_2 m_2}(G)` built from LL wavefunctions.
- Optional symmetry diagnostics for sanity-checking kernel implementations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .planewave import get_form_factors
from .exchange_gausslag import get_exchange_kernels_GaussLag
from .exchange_hankel import get_exchange_kernels_hankel

if TYPE_CHECKING:
    from numpy.typing import NDArray

    ComplexArray = NDArray[np.complex128]
    RealArray = NDArray[np.float64]


def get_exchange_kernels(
    G_magnitudes: "RealArray",
    G_angles: "RealArray",
    nmax: int,
    *,
    method: str | None = None,
) -> "ComplexArray":
    """Dispatcher for exchange kernels.

    Parameters
    ----------
    G_magnitudes, G_angles :
        Arrays describing the reciprocal vectors :math:`G` in polar form.
        Both must have the same shape; broadcasting is not applied.
    nmax :
        Number of Landau levels (0..nmax-1) to include.
    method :
        Backend selector:

        - ``'gausslag'`` (default): generalized Gauss–Laguerre quadrature
          over the radial integral, using the analytic angular dependence.
        - ``'hankel'``          : Hankel-transform based implementation.

    Notes
    -----
    Both backends return kernels normalized for :math:`\\kappa = 1`. Any
    physical interaction strength should be applied by the caller.
    """
    chosen = (method or "gausslag").strip().lower()
    if chosen in {"gausslag", "gauss-lag", "gausslaguerre", "gauss-laguerre", "gl"}:
        return get_exchange_kernels_GaussLag(G_magnitudes, G_angles, nmax)
    if chosen in {"hankel", "hk"}:
        return get_exchange_kernels_hankel(G_magnitudes, G_angles, nmax)
    raise ValueError(f"Unknown exchange-kernel method: {method!r}. Use 'gausslag' or 'hankel'.")


__all__ = [
    "get_form_factors",
    "get_exchange_kernels",
    "get_exchange_kernels_GaussLag",
    "get_exchange_kernels_hankel",
]

