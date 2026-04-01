"""Plane-wave form factors F_{n',n}(G) in a Landau-level basis."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._ho import assemble_ho_form_factors

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def get_form_factors(
    q_magnitudes: RealArray,
    q_angles: RealArray,
    nmax: int,
    lB: float = 1.0,
    sign_magneticfield: int = -1,
) -> ComplexArray:
    """Precompute F_{n',n}(G) for all G and Landau levels.

    Parameters
    ----------
    q_magnitudes, q_angles :
        Arrays with the same shape, describing the wavevector magnitude ``|q|``
        and polar angle ``θ``.
    nmax :
        Number of Landau levels (0..nmax-1).
    lB :
        Magnetic length ``ℓ_B`` (default 1.0). The form factors depend on the
        dimensionless combination ``|q|ℓ_B``, so the implementation multiplies
        ``q_magnitudes`` by ``lB`` internally. If you already work with
        dimensionless ``|q|ℓ_B`` values, leave ``lB=1`` and pass those values
        directly as ``q_magnitudes``.
    sign_magneticfield :
        Sign of the charge–field product σ = sgn(q B_z). Use ``-1`` for the
        electron/positive-B convention used internally; ``+1`` returns the
        complex-conjugated form factors with the appropriate phase flip.

    Returns
    -------
    F : (nG, nmax, nmax) complex array
        Plane-wave form factors F_{n',n}(G).
    """
    if sign_magneticfield not in (1, -1):
        raise ValueError("sign_magneticfield must be 1 or -1")
    return assemble_ho_form_factors(
        q_magnitudes,
        q_angles,
        int(nmax),
        lB=float(lB),
        sigma=float(sign_magneticfield),
    )


__all__ = ["get_form_factors"]
