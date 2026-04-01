"""Plane Haldane pseudopotentials."""
from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .._quadrature import build_radial_potential_weights, legendre_q_nodes_weights

RealArray = NDArray[np.float64]


def _laguerre_table(max_n: int, x: RealArray) -> RealArray:
    """Return ``L_n(x)`` for ``n = 0..max_n`` using the three-term recurrence."""
    x = np.asarray(x, dtype=np.float64)
    table = np.empty((int(max_n) + 1, x.size), dtype=np.float64)
    table[0, :] = 1.0
    if max_n >= 1:
        table[1, :] = 1.0 - x
    for n in range(1, int(max_n)):
        table[n + 1, :] = ((2 * n + 1 - x) * table[n, :] - n * table[n - 1, :]) / (n + 1)
    return table


def get_haldane_pseudopotentials(
    mmax: int,
    *,
    n_ll: int = 0,
    potential: str | Callable[[RealArray], RealArray] = "coulomb",
    kappa: float = 1.0,
    qmax: float = 25.0,
    nquad: int = 2000,
) -> RealArray:
    """Return plane pseudopotentials ``V_m^(n_ll)`` for ``m = 0..mmax-1``."""
    mmax = int(mmax)
    n_ll = int(n_ll)
    if mmax <= 0:
        raise ValueError("mmax must be positive.")
    if n_ll < 0:
        raise ValueError("n_ll must be non-negative.")

    q_nodes, wq = legendre_q_nodes_weights(int(nquad), float(qmax))
    w_eff = build_radial_potential_weights(
        q_nodes,
        wq,
        potential=potential,
        kappa=float(kappa),
    )

    x = 0.5 * q_nodes * q_nodes
    t = q_nodes * q_nodes
    laguerre_x = _laguerre_table(n_ll, x)
    laguerre_t = _laguerre_table(mmax - 1, t)

    base = w_eff * (laguerre_x[n_ll, :] ** 2) * np.exp(-t)
    return cast("RealArray", np.dot(laguerre_t, base).astype(np.float64, copy=False))


__all__ = ["get_haldane_pseudopotentials"]
