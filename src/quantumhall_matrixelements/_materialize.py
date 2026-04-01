"""Utilities for guarding against accidental full-tensor materialization."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

# Default soft cap for allocating a full (nG, nmax, nmax, nmax, nmax) complex tensor.
DEFAULT_FULL_TENSOR_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MiB
# Default soft cap for allocating compressed (nG, n_select) complex arrays.
DEFAULT_COMPRESSED_LIMIT_BYTES = DEFAULT_FULL_TENSOR_LIMIT_BYTES
# Default soft cap for dense backend work tables such as quadrature precomputes.
DEFAULT_WORKSPACE_LIMIT_BYTES = DEFAULT_FULL_TENSOR_LIMIT_BYTES

if TYPE_CHECKING:
    from numpy.typing import NDArray

    ComplexArray = NDArray[np.complex128]
    Quad = tuple[int, int, int, int]


def format_bytes(nbytes: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(nbytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def _normalize_materialize_flag(materialize_full: Any) -> str:
    if materialize_full is True:
        return "auto"
    if materialize_full is False:
        return "never"
    if materialize_full is None:
        return "auto"
    if isinstance(materialize_full, str):
        flag = materialize_full.strip().lower()
        if flag in {"auto", "always", "never"}:
            return flag
    raise ValueError(
        "materialize_full must be one of 'auto', 'always', 'never', True, False, or None"
    )


def guard_full_tensor_materialization(
    select: Any,
    nmax: int,
    nG: int,
    *,
    materialize_full: Any = "auto",
    materialize_limit_bytes: float | int | None = DEFAULT_FULL_TENSOR_LIMIT_BYTES,
    backend_name: str = "exchange kernels",
    dtype: Any = np.complex128,
) -> None:
    """
    Raise proactively if a full exchange tensor would exceed a soft memory limit.

    If ``select`` is provided, the guard is a no-op (the caller is not requesting
    a full tensor).

    For ``materialize_full='auto'`` (the default), the estimated allocation is
    compared to ``materialize_limit_bytes``. Use ``materialize_full='always'``
    to bypass the guard.

    For ``materialize_full='never'`` / ``False`` the caller is not materializing
    a full tensor, so the guard is a no-op.
    """
    if select is not None:
        return

    mode = _normalize_materialize_flag(materialize_full)
    if mode in {"always", "never"}:
        return

    # mode == "auto"
    est = float(nG) * float(nmax) ** 4 * np.dtype(dtype).itemsize
    if materialize_limit_bytes is None:
        return

    if est > float(materialize_limit_bytes):
        human_est = format_bytes(est)
        human_lim = format_bytes(float(materialize_limit_bytes))
        raise MemoryError(
            f"Refusing to materialize full ({nG}, {nmax}, {nmax}, {nmax}, {nmax}) tensor "
            f"(~{human_est} > {human_lim}) for {backend_name}; "
            "use get_exchange_kernels_compressed(select=...) to compute only required "
            "elements, increase materialize_limit_bytes, or pass "
            "materialize_limit_bytes=None to disable this guard."
        )


def guard_compressed_values_allocation(
    *,
    nG: int,
    n_select: int,
    compressed_limit_bytes: float | int | None = DEFAULT_COMPRESSED_LIMIT_BYTES,
    backend_name: str = "exchange kernels",
    dtype: Any = np.complex128,
) -> None:
    """Raise proactively if a compressed ``(nG, n_select)`` array would be too large."""
    if compressed_limit_bytes is None:
        return

    est = float(nG) * float(n_select) * np.dtype(dtype).itemsize
    if est > float(compressed_limit_bytes):
        human_est = format_bytes(est)
        human_lim = format_bytes(float(compressed_limit_bytes))
        raise MemoryError(
            f"Refusing to allocate compressed ({nG}, {n_select}) values array "
            f"(~{human_est} > {human_lim}) for {backend_name}; pass a smaller "
            "explicit select=..., split the G grid, increase compressed_limit_bytes, "
            "or pass compressed_limit_bytes=None to disable this guard."
        )


# --------------------------------------------------------------------------- #
# Helpers for controlled materialization / reconstruction                     #
# --------------------------------------------------------------------------- #
def build_canonical_select(nmax: int) -> list[tuple[int, int, int, int]]:
    """
    Canonical list of index quadruples (n1,m1,n2,m2) with the symmetry
    representative (n1,m1) <= (m2,n2) in lexicographic order. Mirrors the
    canonical branch used in the original full-tensor implementation.
    """
    quads: list[tuple[int, int, int, int]] = []
    for n1 in range(nmax):
        for m1 in range(nmax):
            pair1 = n1 * nmax + m1
            for n2 in range(nmax):
                for m2 in range(nmax):
                    pair2 = m2 * nmax + n2
                    if pair1 > pair2:
                        continue
                    quads.append((n1, m1, n2, m2))
    return quads


def materialize_full_tensor(
    values: ComplexArray,
    select: list[Quad],
    nmax: int,
) -> ComplexArray:
    """
    Expand a compressed exchange-kernel representation back to the full
    (nG, nmax, nmax, nmax, nmax) tensor using the internal symmetry
    X[m2,n2,m1,n1] = (-1)^((n1-m1)-(n2-m2)) X[n1,m1,n2,m2].
    """
    if values.shape[1] != len(select):
        raise ValueError("values second dimension must match length of select list")
    nG = values.shape[0]
    X_full = np.zeros((nG, nmax, nmax, nmax, nmax), dtype=np.complex128)
    for idx, (n1, m1, n2, m2) in enumerate(select):
        val = values[:, idx]
        X_full[:, n1, m1, n2, m2] = val

        pair1 = n1 * nmax + m1
        pair2 = m2 * nmax + n2
        if pair1 == pair2:
            continue
        delta = (n1 - m1) - (n2 - m2)
        sign = 1.0 if (delta & 1) == 0 else -1.0
        X_full[:, m2, n2, m1, n1] = sign * val
    return X_full


__all__ = [
    "guard_full_tensor_materialization",
    "guard_compressed_values_allocation",
    "DEFAULT_COMPRESSED_LIMIT_BYTES",
    "DEFAULT_FULL_TENSOR_LIMIT_BYTES",
    "DEFAULT_WORKSPACE_LIMIT_BYTES",
    "build_canonical_select",
    "format_bytes",
    "materialize_full_tensor",
]
