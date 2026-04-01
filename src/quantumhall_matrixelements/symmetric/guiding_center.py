"""Guiding-center form factors in symmetric gauge."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._ho import assemble_ho_form_factors

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def get_guiding_center_form_factors(
    q_magnitudes: RealArray,
    q_angles: RealArray,
    mmax: int,
    *,
    lB: float = 1.0,
    sign_magneticfield: int = -1,
) -> ComplexArray:
    """Return guiding-center matrix elements ``<m'|exp(i q.R)|m>``.

    The guiding-center oscillator has the opposite chirality to the cyclotron
    sector, so this routine internally uses ``sigma_gc = -sign_magneticfield``.
    """
    if sign_magneticfield not in (1, -1):
        raise ValueError("sign_magneticfield must be 1 or -1")
    return assemble_ho_form_factors(
        q_magnitudes,
        q_angles,
        int(mmax),
        lB=float(lB),
        sigma=float(-sign_magneticfield),
    )


__all__ = ["get_guiding_center_form_factors"]
