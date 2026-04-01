"""Factorized density form factors in the symmetric-gauge ``|n,m>`` basis."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..planewave import get_form_factors
from .guiding_center import get_guiding_center_form_factors

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def get_factorized_density_form_factors(
    q_magnitudes: RealArray,
    q_angles: RealArray,
    nmax: int,
    mmax: int,
    *,
    lB: float = 1.0,
    sign_magneticfield: int = -1,
) -> tuple[ComplexArray, ComplexArray]:
    """Return the cyclotron and guiding-center factors separately."""
    f_cyc = get_form_factors(
        q_magnitudes,
        q_angles,
        int(nmax),
        lB=float(lB),
        sign_magneticfield=int(sign_magneticfield),
    )
    g_gc = get_guiding_center_form_factors(
        q_magnitudes,
        q_angles,
        int(mmax),
        lB=float(lB),
        sign_magneticfield=int(sign_magneticfield),
    )
    return f_cyc, g_gc


__all__ = ["get_factorized_density_form_factors"]
