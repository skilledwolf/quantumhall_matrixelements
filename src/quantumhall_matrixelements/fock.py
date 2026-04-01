"""Convenience helper for constructing exchange Fock operators."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .exchange_hankel import get_exchange_kernels_hankel
from .exchange_laguerre import (
    QuadratureParams,
    build_exchange_fock_precompute,
    get_exchange_kernels_laguerre,
)
from .exchange_ogata import get_exchange_kernels_Ogata

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
Quad = tuple[int, int, int, int]
_FAST_LAGUERRE_PASSTHROUGH_KEYS = {
    "adaptive_nquad",
    "canonical_select_max_entries",
    "kappa",
    "nquad",
    "potential",
    "qmax",
    "sign_magneticfield",
    "workspace_limit_bytes",
}


def _build_compressed_constructor(
    values: ComplexArray,
    select: list[Quad],
    nmax: int,
) -> Callable[[ComplexArray, bool], ComplexArray]:
    """Return a closure that applies exchange kernels (compressed + symmetry) to rho.

    Parameters
    ----------
    values : (nG, n_select) complex array
        Compressed exchange kernel values.
    select : list of (n1, m1, n2, m2)
        Columns corresponding to ``values``.
    nmax : int
        Landau-level cutoff used for validation/shaping.
    """

    sel_n1 = np.fromiter((q[0] for q in select), dtype=int, count=len(select))
    sel_m1 = np.fromiter((q[1] for q in select), dtype=int, count=len(select))
    sel_n2 = np.fromiter((q[2] for q in select), dtype=int, count=len(select))
    sel_m2 = np.fromiter((q[3] for q in select), dtype=int, count=len(select))

    pair1 = sel_n1 * int(nmax) + sel_m1
    pair2 = sel_m2 * int(nmax) + sel_n2
    is_diag = pair1 == pair2

    delta = (sel_n1 - sel_m1) - (sel_n2 - sel_m2)
    sign = np.where((delta & 1) == 0, 1.0, -1.0).astype(np.float64)

    def apply(rho: ComplexArray, include_minus: bool = True) -> ComplexArray:
        rho = cast(ComplexArray, np.asarray(rho, dtype=np.complex128))
        if rho.ndim != 3 or rho.shape[1:] != (nmax, nmax):
            raise ValueError(f"rho must have shape (nG, {nmax}, {nmax})")
        if rho.shape[0] != values.shape[0]:
            raise ValueError(f"rho first dimension must match nG={values.shape[0]}")

        out = np.zeros((rho.shape[0], nmax, nmax), dtype=np.complex128)

        # Treat X as a linear map acting on rho in the same (n,m) index order:
        #   Σ_{n2,m2}(G) = - Σ_{n1,m1} X_{n1,m1,n2,m2}(G) ρ_{n1,m1}(G)
        # where X is returned by get_exchange_kernels with axes (n1,m1,n2,m2).
        # Backends return only symmetry-unique representatives; we add the symmetric
        # partner contribution as well unless the representative lies on the diagonal
        # (n1,m1)==(m2,n2).

        # Representative contribution: out[n2,m2] += X[n1,m1,n2,m2] * rho[n1,m1]
        rho_cols = rho[:, sel_n1, sel_m1]
        contrib = values * rho_cols
        np.add.at(out, (slice(None), sel_n2, sel_m2), contrib)  # type: ignore[arg-type]

        # Partner contribution: X[m2,n2,m1,n1] = sign * X[n1,m1,n2,m2]
        # contributes to out[m1,n1] from rho[m2,n2].
        if not np.all(is_diag):
            mask = ~is_diag
            rho_cols_p = rho[:, sel_m2[mask], sel_n2[mask]]
            contrib_p = (values[:, mask] * sign[mask][None, :]) * rho_cols_p
            np.add.at(out, (slice(None), sel_m1[mask], sel_n1[mask]), contrib_p)  # type: ignore[arg-type]

        return -out if include_minus else out

    return apply


def _build_compressed_constructor_hf(
    values: ComplexArray,
    select: list[Quad],
    nmax: int,
) -> Callable[[ComplexArray, bool], ComplexArray]:
    """Return a closure that applies exchange kernels to rho with the HF convention.

    This matches the exchange convention used in ``quantumhall_hf``:

        Σ^F_{n m}(G) = - Σ_{r,t} X_{m r n t}(G) ρ^*_{t r}(G),

    given compressed kernel representatives ``X_{n1,m1,n2,m2}`` (from
    :func:`get_exchange_kernels`) and a density ``rho`` of shape
    ``(nG, nmax, nmax)``.
    """
    sel_n1 = np.fromiter((q[0] for q in select), dtype=int, count=len(select))
    sel_m1 = np.fromiter((q[1] for q in select), dtype=int, count=len(select))
    sel_n2 = np.fromiter((q[2] for q in select), dtype=int, count=len(select))
    sel_m2 = np.fromiter((q[3] for q in select), dtype=int, count=len(select))

    pair1 = sel_n1 * int(nmax) + sel_m1
    pair2 = sel_m2 * int(nmax) + sel_n2
    is_diag = pair1 == pair2

    delta = (sel_n1 - sel_m1) - (sel_n2 - sel_m2)
    sign = np.where((delta & 1) == 0, 1.0, -1.0).astype(np.float64)

    values = cast(ComplexArray, np.asarray(values, dtype=np.complex128))

    def apply(rho: ComplexArray, include_minus: bool = True) -> ComplexArray:
        rho = cast(ComplexArray, np.asarray(rho, dtype=np.complex128))
        if rho.ndim != 3 or rho.shape[1:] != (nmax, nmax):
            raise ValueError(f"rho must have shape (nG, {nmax}, {nmax})")
        if rho.shape[0] != values.shape[0]:
            raise ValueError(f"rho first dimension must match nG={values.shape[0]}")

        rho_c = rho.conj()
        out = np.zeros((rho.shape[0], nmax, nmax), dtype=np.complex128)

        # Representative contribution:
        #   out[n2,n1] += X[n1,m1,n2,m2] * rho^*[m2,m1]
        rho_cols = rho_c[:, sel_m2, sel_m1]
        contrib = values * rho_cols
        np.add.at(out, (slice(None), sel_n2, sel_n1), contrib)  # type: ignore[arg-type]

        # Partner contribution:
        #   X[m2,n2,m1,n1] = sign * X[n1,m1,n2,m2]
        # contributes to out[m1,m2] from rho^*[n1,n2].
        if not np.all(is_diag):
            mask = ~is_diag
            rho_cols_p = rho_c[:, sel_n1[mask], sel_n2[mask]]
            contrib_p = (values[:, mask] * sign[mask][None, :]) * rho_cols_p
            np.add.at(out, (slice(None), sel_m1[mask], sel_m2[mask]), contrib_p)  # type: ignore[arg-type]

        return -out if include_minus else out

    return apply


def build_fockmatrix_apply(
    values: ComplexArray,
    select: list[Quad],
    nmax: int,
    *,
    convention: str = "standard",
) -> Callable[[ComplexArray, bool], ComplexArray]:
    """Build a callable that applies compressed exchange kernels to a density.

    Parameters
    ----------
    values, select :
        Compressed kernel representation returned by
        :func:`quantumhall_matrixelements.get_exchange_kernels_compressed`.
    nmax :
        Landau-level cutoff.
    convention :
        ``'standard'`` (default) builds a map corresponding to Σ(G) = -X(G)·ρ(G)
        in the natural kernel index order. ``'hf'`` implements the alternate HF
        convention used by ``quantumhall_hf``.

    Returns
    -------
    Callable
        A function ``fock(rho, include_minus=True)``.
    """
    chosen = str(convention).strip().lower()
    if chosen in {"standard", "std"}:
        return _build_compressed_constructor(values, select, nmax)
    if chosen in {"hf", "quantumhall_hf"}:
        return _build_compressed_constructor_hf(values, select, nmax)
    raise ValueError("convention must be 'standard' or 'hf'")


def _maybe_build_fast_laguerre_constructor(
    G_magnitudes: RealArray,
    G_angles: RealArray,
    nmax: int,
    *,
    select: Iterable[Quad] | None,
    kwargs: dict[str, Any],
) -> Callable[[ComplexArray, bool], ComplexArray] | None:
    """Use the dedicated Laguerre Fock path when the request maps exactly onto it."""
    if select is not None:
        return None
    if any(key not in _FAST_LAGUERRE_PASSTHROUGH_KEYS for key in kwargs):
        return None

    adaptive_nquad = bool(kwargs.get("adaptive_nquad", True))
    qmax = float(kwargs.get("qmax", 35.0))
    nquad = int(kwargs.get("nquad", 800))

    G_magnitudes_arr = cast(RealArray, np.asarray(G_magnitudes, dtype=np.float64).ravel())
    G_angles_arr = cast(RealArray, np.asarray(G_angles, dtype=np.float64).ravel())
    if G_magnitudes_arr.shape != G_angles_arr.shape:
        raise ValueError("G_magnitudes and G_angles must have the same shape.")

    if adaptive_nquad and G_magnitudes_arr.size > 0:
        G_max = float(np.max(np.abs(G_magnitudes_arr)))
        nquad_bessel = int(np.ceil(8.0 * G_max * qmax / (2.0 * np.pi)))
        nquad = max(nquad, nquad_bessel)

    sign_magneticfield = int(kwargs.get("sign_magneticfield", -1))
    if sign_magneticfield not in (1, -1):
        raise ValueError("sign_magneticfield must be 1 or -1")

    kappa = float(kwargs.get("kappa", 1.0))
    potential = kwargs.get("potential", "coulomb")
    fast_potential: Callable[[RealArray], RealArray] | None
    if potential == "coulomb":
        fast_potential = None
    elif potential == "constant":
        def _constant_potential(q: RealArray, value: float = kappa) -> RealArray:
            return cast(RealArray, np.full_like(q, value, dtype=np.float64))

        fast_potential = _constant_potential
    elif callable(potential):
        fast_potential = potential
    else:
        return None

    precompute_kwargs: dict[str, Any] = {}
    if "workspace_limit_bytes" in kwargs:
        precompute_kwargs["workspace_limit_bytes"] = kwargs["workspace_limit_bytes"]

    precompute = build_exchange_fock_precompute(
        nmax,
        G_magnitudes_arr,
        G_angles_arr,
        QuadratureParams(qmax=qmax, N=nquad),
        sigma=float(sign_magneticfield),
        kappa=kappa,
        potential=fast_potential,
        include_minus=False,
        **precompute_kwargs,
    )

    def apply(rho: ComplexArray, include_minus: bool = True) -> ComplexArray:
        out = precompute.exchange_fock(rho)
        return -out if include_minus else out

    return apply


def get_fockmatrix_constructor(
    G_magnitudes: RealArray,
    G_angles: RealArray,
    nmax: int,
    *,
    method: str | None = None,
    materialize_full: bool = False,
    select: Iterable[Quad] | None = None,
    **kwargs: Any,
) -> Callable[[ComplexArray, bool], ComplexArray]:
    """Precompute an exchange Fock-matrix operator for repeated rho applications.

    Parameters
    ----------
    G_magnitudes, G_angles : array-like
        Reciprocal vectors in polar coordinates (same shape, no broadcasting).
    nmax : int
        Landau-level cutoff.
    method : str, optional
        Exchange-kernel backend name (``'laguerre'``, ``'ogata'``, or ``'hankel'``).
    materialize_full : bool, optional
        Deprecated/ignored; the constructor always uses the compressed format to avoid
        ``nmax^4`` allocations.
    select : iterable of index quadruples, optional
        If provided, compute only these symmetry representatives; otherwise uses the
        canonical symmetry-reduced list.
    **kwargs :
        Forwarded to the chosen backend (e.g. ``potential``, ``kappa``, ``nquad``).

    Returns
    -------
    Callable
        A function ``fock(rho, include_minus=True)`` that returns the exchange
        Fock matrix for each G given a density matrix ``rho`` of shape
        ``(nG, nmax, nmax)``. By default the returned Fock includes the leading
        minus sign (Σ = -X·ρ); pass ``include_minus=False`` to disable it.
    """

    chosen = (method or "laguerre").strip().lower()
    backend: Any
    if chosen in {"hankel", "hk"}:
        backend = get_exchange_kernels_hankel
    elif chosen in {"ogata", "og"}:
        backend = get_exchange_kernels_Ogata
    elif chosen in {"laguerre", "lag"}:
        fast = _maybe_build_fast_laguerre_constructor(
            G_magnitudes,
            G_angles,
            nmax,
            select=select,
            kwargs=dict(kwargs),
        )
        if fast is not None:
            return fast
        backend = get_exchange_kernels_laguerre
    else:
        raise ValueError(
            f"Unknown exchange-kernel method: {method!r}. "
            "Use 'laguerre', 'ogata', or 'hankel'."
        )

    values, select_list = cast(
        tuple[ComplexArray, list[Quad]],
        backend(
            G_magnitudes,
            G_angles,
            nmax,
            select=select,
            **kwargs,
        ),
    )

    return _build_compressed_constructor(values, select_list, nmax)


def get_fockmatrix_constructor_hf(
    G_magnitudes: RealArray,
    G_angles: RealArray,
    nmax: int,
    *,
    method: str | None = None,
    materialize_full: bool = False,
    select: Iterable[Quad] | None = None,
    **kwargs: Any,
) -> Callable[[ComplexArray, bool], ComplexArray]:
    """Precompute an exchange Fock-matrix operator using the HF convention.

    This mirrors :func:`get_fockmatrix_constructor`, but returns a callable that
    applies kernels in the ``quantumhall_hf`` exchange convention:

        Σ^F_{n m}(G) = - Σ_{r,t} X_{m r n t}(G) ρ^*_{t r}(G).
    """
    chosen = (method or "laguerre").strip().lower()
    backend: Any
    if chosen in {"hankel", "hk"}:
        backend = get_exchange_kernels_hankel
    elif chosen in {"ogata", "og"}:
        backend = get_exchange_kernels_Ogata
    elif chosen in {"laguerre", "lag"}:
        backend = get_exchange_kernels_laguerre
    else:
        raise ValueError(
            f"Unknown exchange-kernel method: {method!r}. "
            "Use 'laguerre', 'ogata', or 'hankel'."
        )

    values, select_list = cast(
        tuple[ComplexArray, list[Quad]],
        backend(
            G_magnitudes,
            G_angles,
            nmax,
            select=select,
            **kwargs,
        ),
    )
    return _build_compressed_constructor_hf(values, select_list, nmax)


__all__ = ["build_fockmatrix_apply", "get_fockmatrix_constructor", "get_fockmatrix_constructor_hf"]
