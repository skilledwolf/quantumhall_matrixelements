from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from quantumhall_matrixelements import (
    get_guiding_center_form_factors,
    get_haldane_pseudopotentials,
    get_twobody_disk_from_pseudopotentials_compressed,
    materialize_twobody_disk_tensor,
)


def _pairs_for_total_angular_momentum(mmax: int, total_m: int) -> list[tuple[int, int]]:
    start = max(0, total_m - (mmax - 1))
    stop = min(mmax - 1, total_m)
    return [(m1, total_m - m1) for m1 in range(start, stop + 1)]


def _cm_relative_coefficients_small(m1: int, m2: int) -> np.ndarray:
    total_m = m1 + m2
    coeffs = np.zeros(total_m + 1, dtype=float)
    for cm_m in range(total_m + 1):
        prefactor = math.sqrt(
            math.factorial(cm_m)
            * math.factorial(total_m - cm_m)
            / (math.factorial(m1) * math.factorial(m2) * (2.0**total_m))
        )
        accum = 0.0
        for k in range(max(0, cm_m - m2), min(m1, cm_m) + 1):
            term = math.comb(m1, k) * math.comb(m2, cm_m - k)
            accum += -term if (m2 - cm_m + k) % 2 else term
        coeffs[cm_m] = prefactor * accum
    return coeffs


def _direct_lll_coulomb_matrix_elements(
    select: list[tuple[int, int, int, int]],
    *,
    mmax: int,
    sign_magneticfield: int,
    qmax: float = 18.0,
    nquad_q: int = 120,
    nquad_theta: int = 96,
) -> np.ndarray:
    q_ref, wq_ref = leggauss(nquad_q)
    q_nodes = 0.5 * qmax * (q_ref + 1.0)
    wq = 0.5 * qmax * wq_ref

    theta_ref, wtheta_ref = leggauss(nquad_theta)
    theta_nodes = np.pi * (theta_ref + 1.0)
    wtheta = np.pi * wtheta_ref

    q_grid = np.repeat(q_nodes, theta_nodes.size)
    theta_grid = np.tile(theta_nodes, q_nodes.size)

    weight = np.repeat(wq, theta_nodes.size) * np.tile(wtheta, q_nodes.size) / (2.0 * np.pi)
    cyclotron_factor = np.exp(-0.5 * q_grid * q_grid)

    g_q = get_guiding_center_form_factors(
        q_grid,
        theta_grid,
        mmax,
        sign_magneticfield=sign_magneticfield,
    )
    g_minus_q = get_guiding_center_form_factors(
        q_grid,
        theta_grid + np.pi,
        mmax,
        sign_magneticfield=sign_magneticfield,
    )

    values = np.empty(len(select), dtype=np.complex128)
    for idx, (m1, m2, m3, m4) in enumerate(select):
        values[idx] = np.sum(
            weight * cyclotron_factor * g_q[:, m1, m3] * g_minus_q[:, m2, m4]
        )
    return values


def _occupation_basis(mmax: int, n_particles: int, total_lz: int) -> list[int]:
    basis: list[int] = []
    for orbitals in itertools.combinations(range(mmax), n_particles):
        if sum(orbitals) != total_lz:
            continue
        state = 0
        for orbital in orbitals:
            state |= 1 << orbital
        basis.append(state)
    return basis


def _parity_before(state: int, orbital: int) -> int:
    return (state & ((1 << orbital) - 1)).bit_count() & 1


def _annihilate(state: int, orbital: int) -> tuple[int, int] | None:
    if not (state & (1 << orbital)):
        return None
    sign = -1 if _parity_before(state, orbital) else 1
    return sign, state ^ (1 << orbital)


def _create(state: int, orbital: int) -> tuple[int, int] | None:
    if state & (1 << orbital):
        return None
    sign = -1 if _parity_before(state, orbital) else 1
    return sign, state | (1 << orbital)


def _fermionic_hamiltonian_from_antisymmetrized_tensor(
    v_antisym: np.ndarray,
    *,
    mmax: int,
    n_particles: int,
    total_lz: int,
) -> np.ndarray:
    basis = _occupation_basis(mmax, n_particles, total_lz)
    basis_index = {state: idx for idx, state in enumerate(basis)}
    hamiltonian = np.zeros((len(basis), len(basis)), dtype=float)

    nonzero_indices = np.argwhere(np.abs(v_antisym) > 1e-14)
    for ket_idx, ket_state in enumerate(basis):
        for i, j, k, ell in nonzero_indices:
            coeff = 0.25 * float(v_antisym[i, j, k, ell])

            out = _annihilate(ket_state, int(k))
            if out is None:
                continue
            sign_k, state_k = out

            out = _annihilate(state_k, int(ell))
            if out is None:
                continue
            sign_l, state_l = out

            out = _create(state_l, int(j))
            if out is None:
                continue
            sign_j, state_j = out

            out = _create(state_j, int(i))
            if out is None:
                continue
            sign_i, bra_state = out

            bra_idx = basis_index.get(bra_state)
            if bra_idx is None:
                continue
            hamiltonian[bra_idx, ket_idx] += coeff * sign_k * sign_l * sign_j * sign_i

    return hamiltonian


def test_disk_two_body_relative_basis_reconstruction_recovers_pseudopotentials():
    mmax = 4
    pseudopotentials = np.array([1.3, 0.7, 0.25, 0.1, 0.0, 0.05], dtype=float)

    values, select = get_twobody_disk_from_pseudopotentials_compressed(pseudopotentials, mmax)
    dense = materialize_twobody_disk_tensor(values, select, mmax)

    for total_m in range(mmax):
        pairs = _pairs_for_total_angular_momentum(mmax, total_m)
        block = np.array(
            [
                [dense[m1, m2, m3, m4] for (m3, m4) in pairs]
                for (m1, m2) in pairs
            ],
            dtype=float,
        )
        coeffs = np.array(
            [_cm_relative_coefficients_small(m1, m2) for (m1, m2) in pairs],
            dtype=float,
        )
        relative_block = coeffs.T @ block @ coeffs
        expected = np.array(
            [pseudopotentials[total_m - cm_m] for cm_m in range(total_m + 1)],
            dtype=float,
        )

        assert np.allclose(np.diag(relative_block), expected, rtol=1e-12, atol=1e-12)
        off_diagonal = relative_block - np.diag(np.diag(relative_block))
        assert np.allclose(off_diagonal, 0.0, rtol=0.0, atol=1e-12)


@pytest.mark.slow
def test_disk_two_body_coulomb_matches_direct_qspace_oracle():
    mmax = 4
    select = [
        (0, 0, 0, 0),
        (0, 1, 0, 1),
        (0, 1, 1, 0),
        (1, 1, 1, 1),
        (0, 2, 1, 1),
        (0, 0, 0, 1),
    ]

    pseudopotentials = get_haldane_pseudopotentials(2 * mmax + 2, n_ll=0, qmax=35.0, nquad=2500)
    values, select_list = get_twobody_disk_from_pseudopotentials_compressed(
        pseudopotentials,
        mmax,
        select=select,
    )

    for sign in (-1, +1):
        refs = _direct_lll_coulomb_matrix_elements(
            select_list,
            mmax=mmax,
            sign_magneticfield=sign,
        )
        assert np.allclose(refs.imag, 0.0, rtol=0.0, atol=1e-12)
        assert np.allclose(refs.real, values, rtol=1e-11, atol=1e-11)


@pytest.mark.release
@pytest.mark.parametrize(
    ("mmax", "n_particles", "total_lz", "expected_dim"),
    [
        (8, 3, 9, 6),
        (10, 4, 18, 18),
    ],
)
def test_v1_pseudopotential_has_unique_fermionic_laughlin_zero_mode(
    mmax: int,
    n_particles: int,
    total_lz: int,
    expected_dim: int,
) -> None:
    pseudopotentials = np.zeros(2 * mmax, dtype=float)
    pseudopotentials[1] = 1.0

    values, select = get_twobody_disk_from_pseudopotentials_compressed(
        pseudopotentials,
        mmax,
        antisymmetrize=True,
    )
    v_antisym = materialize_twobody_disk_tensor(values, select, mmax)

    hamiltonian = _fermionic_hamiltonian_from_antisymmetrized_tensor(
        v_antisym,
        mmax=mmax,
        n_particles=n_particles,
        total_lz=total_lz,
    )
    eigvals = np.linalg.eigvalsh(hamiltonian)

    assert hamiltonian.shape == (expected_dim, expected_dim)
    assert np.all(eigvals >= -1e-12)
    assert np.count_nonzero(np.abs(eigvals) < 1e-12) == 1
    assert abs(eigvals[0]) < 1e-12
    assert eigvals[1] > 1e-2
