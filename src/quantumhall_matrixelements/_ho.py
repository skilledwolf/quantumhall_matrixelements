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

    laguerre = np.empty((nmax, nmax), dtype=np.float64)
    pow_x = np.empty(nmax, dtype=np.float64)

    for iq in range(nq):
        q = q_nodes[iq]
        x = 0.5 * q * q

        for alpha in range(nmax):
            laguerre[alpha, 0] = 1.0
            if nmax > 1:
                laguerre[alpha, 1] = 1.0 + alpha - x
            for k in range(1, nmax - 1):
                laguerre[alpha, k + 1] = (
                    (2 * k + 1 + alpha - x) * laguerre[alpha, k]
                    - (k + alpha) * laguerre[alpha, k - 1]
                ) / (k + 1)

        if x > 0.0:
            logx = np.log(x)
            for alpha in range(nmax):
                if alpha == 0:
                    pow_x[alpha] = 1.0
                else:
                    pow_x[alpha] = np.exp(0.5 * alpha * logx)
        else:
            pow_x[0] = 1.0
            for alpha in range(1, nmax):
                pow_x[alpha] = 0.0

        ex = np.exp(-0.5 * x)
        for row in range(nmax):
            for col in range(nmax):
                alpha = row - col
                if alpha < 0:
                    alpha = -alpha
                kmin = row if row < col else col
                ratio = np.exp(0.5 * (logfact[kmin] - logfact[kmin + alpha]))
                radial[iq, row, col] = ratio * pow_x[alpha] * laguerre[alpha, kmin] * ex

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
