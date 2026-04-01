r"""
MacDonald-Ritchie Table II from package matrix elements
=======================================================

This example reproduces the high-field coefficients in Table II of
MacDonald and Ritchie, Phys. Rev. B 33, 8336 (1986), using the public
``quantumhall_matrixelements`` symmetric-gauge one-body API.

The paper's matrix elements ``V^m_{N',N}`` are not printed in the same
normalization and phase convention as the package's
``<N', M' | V | N, M>`` blocks. For the sectors used in Table II
(``m = N - M = N' - M'`` with ``M >= N``), the conversion is

.. math::

   V^m_{N',N}
   =
   -\frac{2}{\sqrt{\pi}}
   (-1)^{N' - N}
   \langle N', M' | V_{\mathrm{coul}} | N, M \rangle_{\mathrm{pkg}}.

The pieces of this map have different origins:

- the leading minus sign is the attractive impurity potential ``-e^2/r``,
- ``(-1)^{N'-N}`` comes from the Landau-level phase convention,
- ``2 / sqrt(pi)`` is the normalization difference between the package's
  standard Fourier-space Coulomb convention and the matrix-element
  normalization quoted by MacDonald-Ritchie Eq. (13), which in turn uses
  the Glasser-Horing integral convention.

The paper also defines the strong-field expansion parameter as
``x = (pi / 2 gamma)^(1/2)`` in Eq. (15); the coefficients printed below
are the corresponding ``alpha^(i)`` values. The cutoff and quadrature
choices below are intentionally modest so the example stays lightweight;
for release-grade validation, use the dedicated validation and oracle
workflows instead.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import numpy.linalg as la

from quantumhall_matrixelements import get_central_onebody_matrix_elements_compressed


def build_macdonald_ritchie_matrix(
    m: int,
    nmax: int,
    *,
    qmax: float = 35.0,
    nquad: int = 1200,
) -> np.ndarray:
    """Build the fixed-``m`` impurity block in the paper's convention."""
    ell = -int(m)
    if ell < 0:
        raise ValueError("This example expects Table-II sectors with m <= 0 and M >= N.")

    select = [
        (n_row, n_row + ell, n_col, n_col + ell)
        for n_row in range(nmax)
        for n_col in range(nmax)
    ]
    values, select_list = get_central_onebody_matrix_elements_compressed(
        nmax,
        nmax + ell,
        potential="coulomb",
        qmax=qmax,
        nquad=nquad,
        select=select,
    )

    matrix = np.empty((nmax, nmax), dtype=float)
    scale = -2.0 / math.sqrt(math.pi)
    for value, (n_row, _m_row, n_col, _m_col) in zip(values, select_list, strict=True):
        sign = 1.0 if (n_row - n_col) % 2 == 0 else -1.0
        matrix[n_row, n_col] = scale * sign * value
    return matrix


def alphas_at_cutoff(matrix: np.ndarray, n_level: int) -> np.ndarray:
    """Evaluate Eqs. (14a-d) for one truncation."""
    nmax = matrix.shape[0]
    if not (0 <= n_level < nmax):
        raise ValueError("n_level out of range for matrix.")

    indices = np.arange(nmax)
    denom = indices - n_level
    mask = denom != 0

    inv1 = np.zeros(nmax, dtype=float)
    inv2 = np.zeros(nmax, dtype=float)
    inv3 = np.zeros(nmax, dtype=float)
    inv1[mask] = 1.0 / denom[mask]
    inv2[mask] = 1.0 / (denom[mask] ** 2)
    inv3[mask] = 1.0 / (denom[mask] ** 3)

    row_n = matrix[n_level, :]
    col_n = matrix[:, n_level]

    c1 = matrix[n_level, n_level]
    c2 = -np.sum((row_n[mask] ** 2) * inv1[mask])

    a = row_n * inv1
    b = col_n * inv1
    a[n_level] = 0.0
    b[n_level] = 0.0
    term_a = float(a @ (matrix @ b))
    s2 = float(np.sum((row_n[mask] ** 2) * inv2[mask]))
    c3 = term_a - c1 * s2

    step1 = col_n * inv1
    step1[n_level] = 0.0
    step2 = matrix @ step1
    step3 = step2 * inv1
    step3[n_level] = 0.0
    step4 = matrix @ step3
    step5 = step4 * inv1
    step5[n_level] = 0.0
    term1 = -float(row_n @ step5)

    s3 = float(np.sum((row_n[mask] ** 2) * inv3[mask]))
    term2 = -(c1**2) * s3

    b2 = col_n * inv2
    b2[n_level] = 0.0
    part1 = 2.0 * c1 * float(a @ (matrix @ b2))
    s1 = float(np.sum((row_n[mask] ** 2) * inv1[mask]))
    part2 = s1 * s2
    c4 = term1 + term2 + part1 + part2

    return np.array(
        [
            c1 / math.sqrt(2.0),
            c2 / 2.0,
            c3 / (2.0 ** 1.5),
            c4 / 4.0,
        ],
        dtype=float,
    )


def extrapolate_infinite_limit(values: np.ndarray, n_list: list[int], *, order: int = 2) -> float:
    """Fit ``value(n) = L + A/n + B/n^2 + ...`` and return ``L``."""
    n_arr = np.asarray(n_list, dtype=float)
    y_arr = np.asarray(values, dtype=float)
    design = np.vstack([n_arr**0] + [n_arr ** (-power) for power in range(1, order + 1)]).T
    coeff, *_ = la.lstsq(design, y_arr, rcond=None)
    return float(coeff[0])


def alphas_infinite(
    matrix_max: np.ndarray,
    n_level: int,
    n_list: list[int],
    *,
    order: int = 2,
) -> np.ndarray:
    """Extrapolate the perturbative coefficients to infinite cutoff."""
    values = np.asarray(
        [alphas_at_cutoff(matrix_max[:ncut, :ncut], n_level) for ncut in n_list],
        dtype=float,
    )
    return np.asarray(
        [extrapolate_infinite_limit(values[:, idx], n_list, order=order) for idx in range(4)],
        dtype=float,
    )


def frac_str(x: float, max_denominator: int = 4096) -> str:
    """Format exact-looking rationals in the same style as the paper."""
    fraction = Fraction(x).limit_denominator(max_denominator)
    if abs(float(fraction) - x) < 1e-12:
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator}/{fraction.denominator}"
    return f"{x:.12g}"


rows = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (2, 3),
    (3, 3),
]
n_list = [120, 160, 200, 240]
order = 2

alpha_map: dict[tuple[int, int], np.ndarray] = {}
for m_value in sorted({n_level - m_level for (n_level, m_level) in rows}):
    matrix_max = build_macdonald_ritchie_matrix(m_value, max(n_list))
    for n_level in range(4):
        alpha_map[(m_value, n_level)] = alphas_infinite(matrix_max, n_level, n_list, order=order)

print("Package-based reproduction of MacDonald-Ritchie Table II")
print()
header = (
    f"{'N':>2} {'M':>2} "
    f"{'alpha^(1)':>12} {'alpha^(2)':>14} {'alpha^(3)':>14} {'alpha^(4)':>14}"
)
print(header)
print("-" * len(header))
for n_level, m_level in rows:
    alphas = alpha_map[(n_level - m_level, n_level)]
    print(
        f"{n_level:2d} {m_level:2d} {frac_str(alphas[0]):>12} "
        f"{alphas[1]:14.8e} {alphas[2]:14.8e} {alphas[3]:14.8e}"
    )
