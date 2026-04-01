"""Central one-body matrix elements in symmetric gauge."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .._ho import logfact_table, precompute_radial_table
from .._quadrature import build_radial_potential_weights, legendre_q_nodes_weights
from .._select import DEFAULT_CANONICAL_SELECT_MAX_ENTRIES

RealArray = NDArray[np.float64]
OneBodyQuad = tuple[int, int, int, int]
_ONEBODY_BLOCK_TARGET_BYTES = 128 * 1024 * 1024


def _coerce_select_entry(nmax: int, mmax: int, quad: Sequence[int]) -> OneBodyQuad:
    if len(quad) != 4:
        raise ValueError("select entries must be (n_row, m_row, n_col, m_col) tuples.")
    n_row, m_row, n_col, m_col = (int(x) for x in quad)
    if not (0 <= n_row < nmax and 0 <= n_col < nmax):
        raise ValueError("Landau-level indices must satisfy 0 <= n < nmax.")
    if not (0 <= m_row < mmax and 0 <= m_col < mmax):
        raise ValueError("Guiding-center indices must satisfy 0 <= m < mmax.")
    return (n_row, m_row, n_col, m_col)


def estimate_canonical_central_onebody_size(nmax: int, mmax: int) -> int:
    """Return the number of symmetry-unique entries in the canonical select list."""
    counts: dict[int, int] = {}
    for n in range(int(nmax)):
        for m in range(int(mmax)):
            ell = m - n
            counts[ell] = counts.get(ell, 0) + 1
    return sum(count * (count + 1) // 2 for count in counts.values())


def _build_canonical_select(nmax: int, mmax: int) -> list[OneBodyQuad]:
    blocks: dict[int, list[tuple[int, int]]] = {}
    for n in range(int(nmax)):
        for m in range(int(mmax)):
            blocks.setdefault(m - n, []).append((n, m))

    select_list: list[OneBodyQuad] = []
    for ell in sorted(blocks):
        states = blocks[ell]
        for i, (n_row, m_row) in enumerate(states):
            for n_col, m_col in states[i:]:
                select_list.append((n_row, m_row, n_col, m_col))
    return select_list


def _normalize_select(
    nmax: int,
    mmax: int,
    select: Iterable[Sequence[int]] | None,
    *,
    canonical_select_max_entries: int | None,
) -> list[OneBodyQuad]:
    if select is None:
        n_select = estimate_canonical_central_onebody_size(nmax, mmax)
        if (
            canonical_select_max_entries is not None
            and n_select > int(canonical_select_max_entries)
        ):
            raise MemoryError(
                "Refusing to build canonical central one-body select list: "
                f"{n_select:,} entries exceeds canonical_select_max_entries="
                f"{int(canonical_select_max_entries):,}. Pass an explicit select=... "
                "or raise/disable the guard."
            )
        return _build_canonical_select(nmax, mmax)

    select_list = [_coerce_select_entry(nmax, mmax, quad) for quad in select]
    if not select_list:
        raise ValueError("select must contain at least one index quadruple.")
    return select_list


def _prepare_pair_indices(
    select_list: list[OneBodyQuad],
) -> tuple[RealArray, NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    pair_to_idx: dict[tuple[int, int], int] = {}
    pair_rows: list[int] = []
    pair_cols: list[int] = []
    n_pair_idx = np.zeros(len(select_list), dtype=np.int64)
    m_pair_idx = np.zeros(len(select_list), dtype=np.int64)
    signs = np.zeros(len(select_list), dtype=np.float64)

    for idx, (n_row, m_row, n_col, m_col) in enumerate(select_list):
        if (m_row - n_row) != (m_col - n_col):
            continue
        signs[idx] = -1.0 if (n_row - n_col) % 2 else 1.0

        pair_n = (n_row, n_col)
        pair_idx = pair_to_idx.get(pair_n)
        if pair_idx is None:
            pair_idx = len(pair_rows)
            pair_to_idx[pair_n] = pair_idx
            pair_rows.append(n_row)
            pair_cols.append(n_col)
        n_pair_idx[idx] = pair_idx

        pair_m = (m_row, m_col)
        pair_idx = pair_to_idx.get(pair_m)
        if pair_idx is None:
            pair_idx = len(pair_rows)
            pair_to_idx[pair_m] = pair_idx
            pair_rows.append(m_row)
            pair_cols.append(m_col)
        m_pair_idx[idx] = pair_idx

    return (
        signs,
        n_pair_idx,
        m_pair_idx,
        np.asarray(pair_rows, dtype=np.int64),
        np.asarray(pair_cols, dtype=np.int64),
    )


def get_central_onebody_matrix_elements_compressed(
    nmax: int,
    mmax: int,
    *,
    potential: str | Callable[[RealArray], RealArray] = "coulomb",
    kappa: float = 1.0,
    qmax: float = 35.0,
    nquad: int = 800,
    select: Iterable[Sequence[int]] | None = None,
    canonical_select_max_entries: int | None = DEFAULT_CANONICAL_SELECT_MAX_ENTRIES,
) -> tuple[RealArray, list[OneBodyQuad]]:
    """Return compressed matrix elements of an origin-centered radial potential."""
    nmax = int(nmax)
    mmax = int(mmax)
    if nmax <= 0 or mmax <= 0:
        raise ValueError("nmax and mmax must be positive.")

    select_list = _normalize_select(
        nmax,
        mmax,
        select,
        canonical_select_max_entries=canonical_select_max_entries,
    )

    q_nodes, wq = legendre_q_nodes_weights(int(nquad), float(qmax))
    w_eff = build_radial_potential_weights(
        q_nodes,
        wq,
        potential=potential,
        kappa=float(kappa),
    )

    max_index = max(nmax, mmax)
    logfact = logfact_table(max_index).astype(np.float64, copy=False)
    full_radial_bytes = q_nodes.size * max_index * max_index * np.dtype(np.float64).itemsize
    if full_radial_bytes <= _ONEBODY_BLOCK_TARGET_BYTES:
        radial = precompute_radial_table(q_nodes, logfact)

        values = np.zeros(len(select_list), dtype=np.float64)
        for idx, (n_row, m_row, n_col, m_col) in enumerate(select_list):
            if (m_row - n_row) != (m_col - n_col):
                continue
            delta = n_row - n_col
            sign = -1.0 if delta % 2 else 1.0
            values[idx] = sign * np.dot(
                w_eff,
                radial[:, n_row, n_col] * radial[:, m_row, m_col],
            )
        return values, select_list

    signs, n_pair_idx, m_pair_idx, pair_rows, pair_cols = _prepare_pair_indices(select_list)
    if pair_rows.size == 0:
        return np.zeros(len(select_list), dtype=np.float64), select_list

    bytes_per_q = np.dtype(np.float64).itemsize * (
        max_index * max_index + pair_rows.size + 3 * len(select_list)
    )
    q_block = max(1, min(q_nodes.size, int(_ONEBODY_BLOCK_TARGET_BYTES // max(bytes_per_q, 1))))

    values = np.zeros(len(select_list), dtype=np.float64)
    for start in range(0, q_nodes.size, q_block):
        stop = min(start + q_block, q_nodes.size)
        radial_block = precompute_radial_table(q_nodes[start:stop], logfact)
        pair_block = radial_block[:, pair_rows, pair_cols]
        lhs = pair_block[:, n_pair_idx]
        rhs = pair_block[:, m_pair_idx]
        values += signs * np.einsum(
            "q,qj,qj->j",
            w_eff[start:stop],
            lhs,
            rhs,
            optimize=True,
        )

    return values, select_list


def materialize_central_onebody_matrix(
    values: RealArray,
    select: list[OneBodyQuad],
    nmax: int,
    mmax: int,
) -> RealArray:
    """Materialize explicit-index form ``(n_row, m_row, n_col, m_col)``."""
    dense = np.zeros((int(nmax), int(mmax), int(nmax), int(mmax)), dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64).ravel()
    if values_arr.size != len(select):
        raise ValueError("values length must match the length of select.")

    for value, (n_row, m_row, n_col, m_col) in zip(values_arr, select, strict=True):
        dense[n_row, m_row, n_col, m_col] = value
        if (n_row, m_row) != (n_col, m_col):
            dense[n_col, m_col, n_row, m_row] = value
    return dense


__all__ = [
    "estimate_canonical_central_onebody_size",
    "get_central_onebody_matrix_elements_compressed",
    "materialize_central_onebody_matrix",
]
