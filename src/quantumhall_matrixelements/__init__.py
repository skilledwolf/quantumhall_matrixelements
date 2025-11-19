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
from importlib.metadata import PackageNotFoundError, version as _metadata_version

from .planewave import get_form_factors
from .exchange_gausslag import get_exchange_kernels_GaussLag
from .exchange_hankel import get_exchange_kernels_hankel
from .exchange_legendre import get_exchange_kernels_GaussLegendre

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
    **kwargs,
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

        - ``'gausslegendre'`` (default): Gauss-Legendre quadrature with rational mapping.
          Recommended for all nmax.
        - ``'gausslag'``: generalized Gauss–Laguerre quadrature.
          Fast for small nmax (< 10), but unstable for large nmax.
        - ``'hankel'``: Hankel-transform based implementation.

    **kwargs :
        Additional arguments passed to the backend (e.g. ``nquad``, ``scale``).

    Notes
    -----
    Both backends return kernels normalized for :math:`\\kappa = 1`. Any
    physical interaction strength should be applied by the caller.
    """
    chosen = (method or "gausslegendre").strip().lower()
    if chosen in {"gausslag", "gauss-lag", "gausslaguerre", "gauss-laguerre", "gl"}:
        return get_exchange_kernels_GaussLag(G_magnitudes, G_angles, nmax, **kwargs)
    if chosen in {"hankel", "hk"}:
        return get_exchange_kernels_hankel(G_magnitudes, G_angles, nmax, **kwargs)
    if chosen in {"gausslegendre", "gauss-legendre", "legendre", "leg"}:
        return get_exchange_kernels_GaussLegendre(G_magnitudes, G_angles, nmax, **kwargs)
    raise ValueError(f"Unknown exchange-kernel method: {method!r}. Use 'gausslegendre', 'gausslag', or 'hankel'.")


try:
    # Version is managed by setuptools_scm and exposed via package metadata.
    __version__ = _metadata_version("quantumhall_matrixelements")
except PackageNotFoundError:  # pragma: no cover - fallback for local, non-installed usage
    __version__ = "0.0"


__all__ = [
    "get_form_factors",
    "get_exchange_kernels",
    "get_exchange_kernels_GaussLag",
    "get_exchange_kernels_hankel",
    "get_exchange_kernels_GaussLegendre",
    "__version__",
]
