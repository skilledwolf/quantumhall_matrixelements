"""Landau-level plane-wave form factors and exchange kernels.

This package provides reusable numerical kernels for quantum Hall matrix
elements in a Landau-level basis:

- `get_form_factors` for plane-wave form factors :math:`F_{n',n}(G)`.
- `get_guiding_center_form_factors` and
  `get_factorized_density_form_factors` for symmetric-gauge building blocks.
- `get_exchange_kernels` (and backend-specific variants) for exchange kernels
  :math:`X_{n_1 m_1 n_2 m_2}(G)` built from LL wavefunctions.
- `get_central_onebody_matrix_elements_compressed`,
  `get_haldane_pseudopotentials`, and
  `get_twobody_disk_from_pseudopotentials_compressed` for symmetric-gauge
  one- and two-body workflows.
- Optional symmetry diagnostics for sanity-checking kernel implementations.
"""
from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ._materialize import (
    DEFAULT_COMPRESSED_LIMIT_BYTES,
    DEFAULT_FULL_TENSOR_LIMIT_BYTES,
    guard_compressed_values_allocation,
    guard_full_tensor_materialization,
    materialize_full_tensor,
)
from ._select import DEFAULT_CANONICAL_SELECT_MAX_ENTRIES, estimate_canonical_select_size
from .diagnostic import get_exchange_kernels_opposite_field, get_form_factors_opposite_field
from .exchange_hankel import get_exchange_kernels_hankel
from .exchange_laguerre import (
    ExchangeFockPrecompute,
    QuadratureParams,
    build_exchange_fock_precompute,
    get_exchange_kernels_laguerre,
)
from .exchange_ogata import get_exchange_kernels_Ogata
from .fock import build_fockmatrix_apply, get_fockmatrix_constructor, get_fockmatrix_constructor_hf
from .planewave import get_form_factors
from .symmetric import (
    get_central_onebody_matrix_elements_compressed,
    get_factorized_density_form_factors,
    get_guiding_center_form_factors,
    get_haldane_pseudopotentials,
    get_twobody_disk_from_pseudopotentials_compressed,
    materialize_central_onebody_matrix,
    materialize_twobody_disk_tensor,
)

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
Quad = tuple[int, int, int, int]


def get_exchange_kernels(
    G_magnitudes: RealArray,
    G_angles: RealArray,
    nmax: int,
    *,
    method: str | None = None,
    materialize_limit_bytes: float | int | None = DEFAULT_FULL_TENSOR_LIMIT_BYTES,
    canonical_select_max_entries: int | None = DEFAULT_CANONICAL_SELECT_MAX_ENTRIES,
    **kwargs: Any,
) -> ComplexArray:
    """Compute and return the fully materialized 5D exchange tensor.

    Parameters
    ----------
    G_magnitudes, G_angles :
        Arrays describing the reciprocal vectors :math:`G` in polar form.
        Both must have the same shape; broadcasting is not applied.
    nmax :
        Number of Landau levels (0..nmax-1) to include.
    method :
        Backend selector:

        - ``'laguerre'`` (default): Numba-JIT quadrature on [0, qmax] with
          Laguerre three-term recurrence. Stable for all nmax and |G|.
        - ``'ogata'``: Ogata quadrature (Hankel/Ogata) with an automatic small-|G|
          fallback.
        - ``'hankel'``: Hankel-transform based implementation (slow but precise).

    materialize_limit_bytes :
        Soft cap (in bytes) for allocating a full ``(nG, nmax, nmax, nmax, nmax)``
        complex tensor. Pass ``None`` to disable this safety check.

    canonical_select_max_entries :
        Soft cap on the number of canonical select entries constructed when
        ``select`` is omitted. This prevents accidentally building huge Python
        lists with O(nmax^4) entries.

    **kwargs :
        Additional arguments passed to the backend (e.g. ``nquad``, ``scale``).
        Common keywords include ``sign_magneticfield`` (±1) to select the
        magnetic-field orientation convention and, for the Laguerre backend,
        ``workspace_limit_bytes`` to cap dense quadrature-table allocations.

    Notes
    -----
    For the built-in potentials ``'coulomb'`` and ``'constant'``, the ``kappa``
    keyword scales the kernel. For callable potentials, the provided function
    defines the overall energy scale.

    To compute only a small set of entries without allocating the full tensor,
    use :func:`get_exchange_kernels_compressed` with an explicit ``select=...``.
    """
    chosen = (method or "laguerre").strip().lower()
    backend_fn: Any
    if chosen in {"hankel", "hk"}:
        backend_fn = get_exchange_kernels_hankel
    elif chosen in {"ogata", "og"}:
        backend_fn = get_exchange_kernels_Ogata
    elif chosen in {"laguerre", "lag"}:
        backend_fn = get_exchange_kernels_laguerre
    else:
        raise ValueError(
            f"Unknown exchange-kernel method: {method!r}. "
            "Use 'laguerre', 'ogata', or 'hankel'."
        )

    G_magnitudes = np.asarray(G_magnitudes, dtype=float).ravel()
    G_angles = np.asarray(G_angles, dtype=float).ravel()
    if G_magnitudes.shape != G_angles.shape:
        raise ValueError("G_magnitudes and G_angles must have the same shape.")

    # Fast-fail before expensive backend work if we'd materialize a huge tensor.
    guard_full_tensor_materialization(
        select=None,
        nmax=int(nmax),
        nG=int(G_magnitudes.size),
        materialize_full="auto",
        materialize_limit_bytes=materialize_limit_bytes,
        backend_name=f"{chosen} exchange kernels",
    )

    values, select_list = cast(
        tuple[ComplexArray, list[Quad]],
        backend_fn(
            G_magnitudes,
            G_angles,
            nmax,
            select=None,
            canonical_select_max_entries=canonical_select_max_entries,
            **kwargs,
        ),
    )
    return materialize_full_tensor(values, select_list, nmax)


def get_exchange_kernels_compressed(
    G_magnitudes: RealArray,
    G_angles: RealArray,
    nmax: int,
    *,
    method: str | None = None,
    select: Iterable[Quad] | None = None,
    canonical_select_max_entries: int | None = DEFAULT_CANONICAL_SELECT_MAX_ENTRIES,
    compressed_limit_bytes: float | int | None = DEFAULT_COMPRESSED_LIMIT_BYTES,
    **kwargs: Any,
) -> tuple[ComplexArray, list[Quad]]:
    """Return the compressed exchange-kernel representation ``(values, select_list)``.

    Unlike :func:`get_exchange_kernels`, this function never materializes the full
    5D tensor, and always returns the select list used by the backend.

    If ``select`` is omitted, the backend still constructs the canonical
    symmetry-reduced list, so the returned representation remains O(``nmax^4``)
    in the number of stored entries. ``compressed_limit_bytes`` caps the
    resulting ``(nG, n_select)`` complex output array. Pass an explicit
    ``select=...`` to compute only the entries you need.
    """
    chosen = (method or "laguerre").strip().lower()
    backend_fn: Any
    if chosen in {"hankel", "hk"}:
        backend_fn = get_exchange_kernels_hankel
    elif chosen in {"ogata", "og"}:
        backend_fn = get_exchange_kernels_Ogata
    elif chosen in {"laguerre", "lag"}:
        backend_fn = get_exchange_kernels_laguerre
    else:
        raise ValueError(
            f"Unknown exchange-kernel method: {method!r}. "
            "Use 'laguerre', 'ogata', or 'hankel'."
        )

    G_magnitudes = np.asarray(G_magnitudes, dtype=float).ravel()
    G_angles = np.asarray(G_angles, dtype=float).ravel()
    if G_magnitudes.shape != G_angles.shape:
        raise ValueError("G_magnitudes and G_angles must have the same shape.")

    select_list: list[Quad] | None
    if select is None:
        select_list = None
        n_select_est = estimate_canonical_select_size(int(nmax))
    else:
        select_list = [cast(Quad, tuple(int(x) for x in quad)) for quad in select]
        n_select_est = len(select_list)

    guard_compressed_values_allocation(
        nG=int(G_magnitudes.size),
        n_select=int(n_select_est),
        compressed_limit_bytes=compressed_limit_bytes,
        backend_name=f"{chosen} exchange kernels",
    )

    return cast(
        tuple[ComplexArray, list[Quad]],
        backend_fn(
            G_magnitudes,
            G_angles,
            nmax,
            select=select_list,
            canonical_select_max_entries=canonical_select_max_entries,
            **kwargs,
        ),
    )


try:
    # Version is managed by setuptools_scm and exposed via package metadata.
    __version__ = _metadata_version("quantumhall_matrixelements")
except PackageNotFoundError:  # pragma: no cover - fallback for local, non-installed usage
    __version__ = "0.0"


__all__ = [
    "get_form_factors",
    "get_guiding_center_form_factors",
    "get_factorized_density_form_factors",
    "get_form_factors_opposite_field",
    "get_exchange_kernels",
    "get_exchange_kernels_compressed",
    "get_exchange_kernels_opposite_field",
    "get_exchange_kernels_hankel",
    "get_exchange_kernels_Ogata",
    "get_exchange_kernels_laguerre",
    "build_fockmatrix_apply",
    "get_fockmatrix_constructor",
    "get_fockmatrix_constructor_hf",
    "QuadratureParams",
    "ExchangeFockPrecompute",
    "build_exchange_fock_precompute",
    "get_central_onebody_matrix_elements_compressed",
    "materialize_central_onebody_matrix",
    "get_haldane_pseudopotentials",
    "get_twobody_disk_from_pseudopotentials_compressed",
    "materialize_twobody_disk_tensor",
    "__version__",
]
