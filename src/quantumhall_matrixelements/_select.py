"""Shared helpers for select-list normalization across backends."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from ._materialize import build_canonical_select

if TYPE_CHECKING:
    from numpy.typing import NDArray

    IntArray = NDArray[np.int64]

Quad = tuple[int, int, int, int]

# Soft guard against accidentally building enormous canonical select lists.
# Canonical select size is ~ (nmax^4)/2 entries; each entry is a Python tuple
# so memory use grows quickly even before any numeric work starts.
DEFAULT_CANONICAL_SELECT_MAX_ENTRIES = 2_000_000


def _coerce_quad(nmax: int, quad: Sequence[int]) -> Quad:
    if len(quad) != 4:
        raise ValueError("select entries must be (n1, m1, n2, m2) tuples.")
    n1, m1, n2, m2 = (int(x) for x in quad)
    if any(q < 0 or q >= nmax for q in (n1, m1, n2, m2)):
        raise ValueError("select indices must satisfy 0 <= index < nmax.")
    return (n1, m1, n2, m2)


def normalize_select(
    nmax: int,
    select: Iterable[Sequence[int]] | None,
    *,
    canonical_select_max_entries: int | None = DEFAULT_CANONICAL_SELECT_MAX_ENTRIES,
) -> tuple[list[Quad], IntArray, IntArray, IntArray, IntArray]:
    """Return (select_list, sel_n1, sel_m1, sel_n2, sel_m2).

    If ``select`` is None, use the canonical symmetry-reduced list.
    Validates bounds and shape.
    """
    if select is None:
        if canonical_select_max_entries is not None:
            # number of (n,m) pairs is nmax^2; canonical selection is the upper
            # triangle in that pair space.
            n_pairs = int(nmax) * int(nmax)
            n_select = (n_pairs * (n_pairs + 1)) // 2
            if n_select > int(canonical_select_max_entries):
                raise MemoryError(
                    f"Refusing to build canonical select list for nmax={nmax}: "
                    f"{n_select:,} entries exceeds canonical_select_max_entries="
                    f"{int(canonical_select_max_entries):,}. "
                    "Pass an explicit select=... (e.g. via get_exchange_kernels_compressed) "
                    "to compute only required elements, "
                    "or raise/disable the guard via canonical_select_max_entries=None."
                )
        select_list = build_canonical_select(nmax)
    else:
        select_list = [_coerce_quad(nmax, quad) for quad in select]
        if not select_list:
            raise ValueError("select must contain at least one index quadruple.")

    sel_n1 = np.fromiter((q[0] for q in select_list), dtype=int, count=len(select_list))
    sel_m1 = np.fromiter((q[1] for q in select_list), dtype=int, count=len(select_list))
    sel_n2 = np.fromiter((q[2] for q in select_list), dtype=int, count=len(select_list))
    sel_m2 = np.fromiter((q[3] for q in select_list), dtype=int, count=len(select_list))
    return select_list, sel_n1, sel_m1, sel_n2, sel_m2


__all__ = [
    "DEFAULT_CANONICAL_SELECT_MAX_ENTRIES",
    "normalize_select",
]
