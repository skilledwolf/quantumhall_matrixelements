"""LLL disk two-body matrix elements reconstructed from pseudopotentials."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._ho import logfact_table
from .._select import DEFAULT_CANONICAL_SELECT_MAX_ENTRIES

RealArray = NDArray[np.float64]
OrbitalQuad = tuple[int, int, int, int]


@dataclass(frozen=True)
class _DiskBlock:
    pairs: list[tuple[int, int]]
    coeffs: RealArray
    pair_to_index: dict[tuple[int, int], int]


def _coerce_select_entry(mmax: int, quad: Sequence[int]) -> OrbitalQuad:
    if len(quad) != 4:
        raise ValueError("select entries must be (m1, m2, m3, m4) tuples.")
    m1, m2, m3, m4 = (int(x) for x in quad)
    if any(m < 0 or m >= mmax for m in (m1, m2, m3, m4)):
        raise ValueError("Orbital indices must satisfy 0 <= m < mmax.")
    return (m1, m2, m3, m4)


def estimate_canonical_twobody_disk_size(mmax: int) -> int:
    """Return the size of the angular-momentum conserving canonical select list."""
    total = 0
    for total_m in range(2 * int(mmax) - 1):
        count = min(total_m + 1, 2 * int(mmax) - 1 - total_m)
        total += count * (count + 1) // 2
    return total


def _pairs_for_total_angular_momentum(mmax: int, total_m: int) -> list[tuple[int, int]]:
    start = max(0, total_m - (int(mmax) - 1))
    stop = min(int(mmax) - 1, total_m)
    return [(m1, total_m - m1) for m1 in range(start, stop + 1)]


def _build_canonical_select(mmax: int) -> list[OrbitalQuad]:
    select_list: list[OrbitalQuad] = []
    for total_m in range(2 * int(mmax) - 1):
        pairs = _pairs_for_total_angular_momentum(mmax, total_m)
        for i, (m1, m2) in enumerate(pairs):
            for m3, m4 in pairs[i:]:
                select_list.append((m1, m2, m3, m4))
    return select_list


def _normalize_select(
    mmax: int,
    select: Iterable[Sequence[int]] | None,
    *,
    canonical_select_max_entries: int | None,
) -> list[OrbitalQuad]:
    if select is None:
        n_select = estimate_canonical_twobody_disk_size(mmax)
        if (
            canonical_select_max_entries is not None
            and n_select > int(canonical_select_max_entries)
        ):
            raise MemoryError(
                "Refusing to build canonical disk two-body select list: "
                f"{n_select:,} entries exceeds canonical_select_max_entries="
                f"{int(canonical_select_max_entries):,}. Pass an explicit select=... "
                "or raise/disable the guard."
            )
        return _build_canonical_select(mmax)

    select_list = [_coerce_select_entry(mmax, quad) for quad in select]
    if not select_list:
        raise ValueError("select must contain at least one index quadruple.")
    return select_list


def _log_binom(logfact: RealArray, n: int, k: int) -> float:
    return float(logfact[n] - logfact[k] - logfact[n - k])


def _cm_relative_coefficients(m1: int, m2: int, logfact: RealArray) -> RealArray:
    total_m = m1 + m2
    coeffs = np.zeros(total_m + 1, dtype=np.float64)
    for cm_m in range(total_m + 1):
        pref = np.exp(
            0.5
            * (
                logfact[cm_m]
                + logfact[total_m - cm_m]
                - logfact[m1]
                - logfact[m2]
                - total_m * np.log(2.0)
            )
        )
        acc = 0.0
        k_min = max(0, cm_m - m2)
        k_max = min(m1, cm_m)
        for k in range(k_min, k_max + 1):
            term = np.exp(_log_binom(logfact, m1, k) + _log_binom(logfact, m2, cm_m - k))
            if (m2 - cm_m + k) % 2:
                acc -= term
            else:
                acc += term
        coeffs[cm_m] = pref * acc
    return coeffs


def _build_blocks(mmax: int, logfact: RealArray) -> dict[int, _DiskBlock]:
    blocks: dict[int, _DiskBlock] = {}
    for total_m in range(2 * int(mmax) - 1):
        pairs = _pairs_for_total_angular_momentum(mmax, total_m)
        coeffs = np.empty((len(pairs), total_m + 1), dtype=np.float64)
        pair_to_index: dict[tuple[int, int], int] = {}
        for idx, pair in enumerate(pairs):
            coeffs[idx, :] = _cm_relative_coefficients(pair[0], pair[1], logfact)
            pair_to_index[pair] = idx
        blocks[total_m] = _DiskBlock(
            pairs=pairs,
            coeffs=coeffs,
            pair_to_index=pair_to_index,
        )
    return blocks


def get_twobody_disk_from_pseudopotentials_compressed(
    V_m: RealArray,
    mmax: int,
    *,
    select: Iterable[Sequence[int]] | None = None,
    canonical_select_max_entries: int | None = DEFAULT_CANONICAL_SELECT_MAX_ENTRIES,
    antisymmetrize: bool = False,
) -> tuple[RealArray, list[OrbitalQuad]]:
    """Return compressed LLL disk two-body matrix elements.

    Channels above ``len(V_m)-1`` are treated as zero.
    """
    mmax = int(mmax)
    if mmax <= 0:
        raise ValueError("mmax must be positive.")

    select_list = _normalize_select(
        mmax,
        select,
        canonical_select_max_entries=canonical_select_max_entries,
    )

    v_in = np.asarray(V_m, dtype=np.float64).ravel()
    if v_in.size == 0:
        raise ValueError("V_m must contain at least one pseudopotential channel.")
    max_rel = max(v_in.size - 1, 2 * mmax - 2)
    v_pad = np.zeros(max_rel + 1, dtype=np.float64)
    v_pad[: v_in.size] = v_in

    logfact = logfact_table(2 * mmax).astype(np.float64, copy=False)
    blocks = _build_blocks(mmax, logfact)

    values = np.zeros(len(select_list), dtype=np.float64)
    for idx, (m1, m2, m3, m4) in enumerate(select_list):
        if (m1 + m2) != (m3 + m4):
            continue
        total_m = m1 + m2
        block = blocks[total_m]
        coeff_bra = block.coeffs[block.pair_to_index[(m1, m2)], :]
        coeff_ket = block.coeffs[block.pair_to_index[(m3, m4)], :]
        rel_channels = v_pad[total_m - np.arange(total_m + 1)]
        if antisymmetrize:
            coeff_swap = block.coeffs[block.pair_to_index[(m4, m3)], :]
            values[idx] = np.dot(coeff_bra * (coeff_ket - coeff_swap), rel_channels)
        else:
            values[idx] = np.dot(coeff_bra * coeff_ket, rel_channels)

    return values, select_list


def materialize_twobody_disk_tensor(
    values: RealArray,
    select: list[OrbitalQuad],
    mmax: int,
) -> RealArray:
    """Materialize explicit-index form ``(m1, m2, m3, m4)``."""
    dense = np.zeros((int(mmax), int(mmax), int(mmax), int(mmax)), dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64).ravel()
    if values_arr.size != len(select):
        raise ValueError("values length must match the length of select.")

    for value, (m1, m2, m3, m4) in zip(values_arr, select, strict=True):
        dense[m1, m2, m3, m4] = value
        if (m1, m2) != (m3, m4):
            dense[m3, m4, m1, m2] = value
    return dense


__all__ = [
    "estimate_canonical_twobody_disk_size",
    "get_twobody_disk_from_pseudopotentials_compressed",
    "materialize_twobody_disk_tensor",
]
