"""Shared radial quadrature helpers."""
from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

RealArray = NDArray[np.float64]


def legendre_q_nodes_weights(nquad: int, qmax: float) -> tuple[RealArray, RealArray]:
    """Return Gauss-Legendre nodes and weights on ``[0, qmax]``."""
    t, w = leggauss(int(nquad))
    q = 0.5 * float(qmax) * (t + 1.0)
    wq = 0.5 * float(qmax) * w
    return q.astype(np.float64), wq.astype(np.float64)


def build_radial_potential_weights(
    q_nodes: RealArray,
    wq: RealArray,
    *,
    potential: str | Callable[[RealArray], RealArray] = "coulomb",
    kappa: float = 1.0,
) -> RealArray:
    """Return radial weights for ``∫ (q dq / 2π) V(q) f(q)``."""
    q_nodes = np.asarray(q_nodes, dtype=np.float64)
    wq = np.asarray(wq, dtype=np.float64)
    if q_nodes.shape != wq.shape:
        raise ValueError("q_nodes and wq must have the same shape.")

    if callable(potential):
        v_raw = np.asarray(potential(q_nodes))
        if np.iscomplexobj(v_raw):
            raise ValueError("Callable potential must be real-valued.")
        try:
            v = np.broadcast_to(v_raw.astype(np.float64, copy=False), q_nodes.shape)
        except ValueError as exc:
            raise ValueError(
                "Callable potential must return a real scalar or array broadcastable "
                "to the q-node grid."
            ) from exc
        return cast("RealArray", (wq * (q_nodes / (2.0 * np.pi)) * v).astype(np.float64))

    pot_kind = str(potential).strip().lower()
    if pot_kind == "coulomb":
        return cast("RealArray", (float(kappa) * wq).astype(np.float64))
    if pot_kind == "constant":
        return cast(
            "RealArray",
            (wq * (q_nodes / (2.0 * np.pi)) * float(kappa)).astype(np.float64),
        )
    raise ValueError("potential must be 'coulomb', 'constant', or a callable V(q)")


__all__ = ["build_radial_potential_weights", "legendre_q_nodes_weights"]
