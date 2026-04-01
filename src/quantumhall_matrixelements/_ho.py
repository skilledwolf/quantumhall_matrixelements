"""Shared harmonic-oscillator matrix-element helpers."""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy import special

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]

_NumbaFunc = TypeVar("_NumbaFunc", bound=Callable[..., object])


def _typed_njit(*, fastmath: bool = False) -> Callable[[_NumbaFunc], _NumbaFunc]:
    """Type-preserving wrapper around ``numba.njit`` for strict mypy."""
    return cast("Callable[[_NumbaFunc], _NumbaFunc]", njit(fastmath=fastmath))


def logfact_table(nmax: int) -> RealArray:
    """Return ``log(n!)`` for ``n = 0..nmax``."""
    n = np.arange(int(nmax) + 1, dtype=np.float64)
    return cast("RealArray", special.gammaln(n + 1.0))


@_typed_njit(fastmath=False)
def precompute_radial_table(q_nodes: RealArray, logfact: RealArray) -> RealArray:
    """Compute real HO displacement radial pieces ``R[iq, a, b]``."""
    nq = q_nodes.size
    nmax = logfact.size - 1
    radial = np.empty((nq, nmax, nmax), dtype=np.float64)
    radial.fill(0.0)

    for iq in range(nq):
        q = q_nodes[iq]
        x = 0.5 * q * q

        logx = np.log(x) if x > 0.0 else 0.0

        for alpha in range(nmax):
            if x > 0.0:
                # Start from the normalized k=0 amplitude so large factorial and
                # power factors are combined before exponentiation.
                r_prev = np.exp(0.5 * (alpha * logx - x - logfact[alpha]))
            else:
                r_prev = 1.0 if alpha == 0 else 0.0

            radial[iq, alpha, 0] = r_prev
            radial[iq, 0, alpha] = r_prev

            kmax = nmax - alpha
            if kmax <= 1:
                continue

            r_curr = ((1.0 + alpha - x) / np.sqrt(1.0 + alpha)) * r_prev
            radial[iq, alpha + 1, 1] = r_curr
            radial[iq, 1, alpha + 1] = r_curr

            for k in range(1, kmax - 1):
                coeff1 = (2.0 * k + 1.0 + alpha - x) / np.sqrt((k + 1.0) * (k + 1.0 + alpha))
                coeff2 = np.sqrt((k * (k + alpha)) / ((k + 1.0) * (k + 1.0 + alpha)))
                r_next = coeff1 * r_curr - coeff2 * r_prev
                radial[iq, k + 1 + alpha, k + 1] = r_next
                radial[iq, k + 1, k + 1 + alpha] = r_next
                r_prev = r_curr
                r_curr = r_next

    return radial


def phase_out_table(angles: RealArray, max_d: int, sigma: float) -> ComplexArray:
    """Return ``i^{|d|} exp(i sigma d theta)`` for ``d in [-max_d, max_d]``."""
    angles = np.asarray(angles, dtype=np.float64).ravel()
    d_vals = np.arange(-int(max_d), int(max_d) + 1, dtype=np.int32)
    phase_base = ((1j) ** np.abs(d_vals)).astype(np.complex128)

    ang = float(sigma) * angles[:, None] * d_vals[None, :]
    epos = np.cos(ang) + 1j * np.sin(ang)
    return phase_base[None, :] * epos


def build_exchange_phase_tables(
    angles: RealArray, max_d: int, sigma: float
) -> tuple[ComplexArray, ComplexArray]:
    """Return the input/output phase tables used by the exchange backend."""
    phase_out = phase_out_table(angles, max_d, sigma)
    phase_in = np.conjugate(phase_out)
    return phase_in, phase_out


def assemble_ho_form_factors(
    q_magnitudes: RealArray,
    q_angles: RealArray,
    max_index: int,
    *,
    lB: float = 1.0,
    sigma: float = -1.0,
) -> ComplexArray:
    """Assemble HO displacement matrix elements from radial and phase tables."""
    if int(max_index) <= 0:
        raise ValueError("max_index must be positive")

    q_magnitudes = np.asarray(q_magnitudes, dtype=np.float64).ravel()
    q_angles = np.asarray(q_angles, dtype=np.float64).ravel()
    if q_magnitudes.shape != q_angles.shape:
        raise ValueError("q_magnitudes and q_angles must have the same shape.")

    nmax = int(max_index)
    q_scaled = (q_magnitudes * float(lB)).astype(np.float64, copy=False)
    logfact = logfact_table(nmax).astype(np.float64, copy=False)
    radial = precompute_radial_table(q_scaled, logfact)

    max_d = nmax - 1
    phase = phase_out_table(q_angles, max_d, float(sigma))

    indices = np.arange(nmax, dtype=np.int64)
    d_index = indices[:, None] - indices[None, :] + max_d
    phase_tensor = phase[:, d_index]
    return radial.astype(np.complex128) * phase_tensor


__all__ = [
    "assemble_ho_form_factors",
    "build_exchange_phase_tables",
    "logfact_table",
    "phase_out_table",
    "precompute_radial_table",
]
