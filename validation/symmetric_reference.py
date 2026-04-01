"""Frozen symmetric-gauge reference data helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

REFERENCE_PATH = Path(__file__).with_name("symmetric_reference_data.json")
DEFAULT_RTOL = 1e-10
DEFAULT_ATOL = 1e-12


def _segmented_quad_mpmath(integrand, *, qmax: float, segments: int) -> mp.mpf:
    total = mp.mpf("0")
    step = mp.mpf(qmax) / int(segments)
    for idx in range(int(segments)):
        total += mp.quad(integrand, [idx * step, (idx + 1) * step])
    return total


def _ho_radial_mpmath(row: int, col: int, q: mp.mpf) -> mp.mpf:
    x = mp.mpf("0.5") * q * q
    amin = min(row, col)
    aabs = abs(row - col)
    return (
        mp.sqrt(mp.factorial(amin) / mp.factorial(amin + aabs))
        * (q / mp.sqrt(2)) ** aabs
        * mp.laguerre(amin, aabs, x)
        * mp.e ** (-x / 2)
    )


def _guiding_center_form_factor_mpmath(
    m_row: int,
    m_col: int,
    q: float,
    theta: float,
    *,
    sign_magneticfield: int,
    dps: int = 80,
) -> complex:
    mp.mp.dps = int(dps)
    q_mp = mp.mpf(q)
    theta_mp = mp.mpf(theta)
    diff = int(m_row - m_col)
    sigma_gc = -int(sign_magneticfield)
    angular = complex(mp.cos(sigma_gc * diff * theta_mp), mp.sin(sigma_gc * diff * theta_mp))
    return ((1j) ** abs(diff)) * angular * complex(_ho_radial_mpmath(m_row, m_col, q_mp))


def _central_onebody_mpmath(
    n_row: int,
    m_row: int,
    n_col: int,
    m_col: int,
    *,
    potential: str,
    kappa: float,
    dps: int = 80,
    qmax: float = 35.0,
    segments: int = 8,
) -> float:
    if (m_row - n_row) != (m_col - n_col):
        return 0.0

    mp.mp.dps = int(dps)
    kappa_mp = mp.mpf(kappa)

    def integrand(q: mp.mpf) -> mp.mpf:
        radial = _ho_radial_mpmath(n_row, n_col, q) * _ho_radial_mpmath(m_row, m_col, q)
        if potential == "coulomb":
            weight = kappa_mp
        elif potential == "constant":
            weight = kappa_mp * q / (2 * mp.pi)
        else:
            raise ValueError(f"Unsupported potential for reference generation: {potential!r}")
        return weight * radial

    value = _segmented_quad_mpmath(integrand, qmax=qmax, segments=segments)
    if (n_row - n_col) % 2:
        value = -value
    return float(value)


def _pseudopotential_coulomb_mpmath(
    m: int,
    *,
    n_ll: int,
    kappa: float,
    dps: int = 80,
    qmax: float = 35.0,
    segments: int = 8,
) -> float:
    mp.mp.dps = int(dps)
    half = mp.mpf("0.5")
    kappa_mp = mp.mpf(kappa)

    def integrand(q: mp.mpf) -> mp.mpf:
        x = half * q * q
        t = q * q
        return kappa_mp * mp.laguerre(n_ll, 0, x) ** 2 * mp.laguerre(m, 0, t) * mp.e ** (-t)

    return float(_segmented_quad_mpmath(integrand, qmax=qmax, segments=segments))


def _disk_cm_relative_coefficient_mpmath(m1: int, m2: int, cm_m: int, *, dps: int = 80) -> mp.mpf:
    mp.mp.dps = int(dps)
    total_m = m1 + m2
    pref = mp.sqrt(
        mp.factorial(cm_m)
        * mp.factorial(total_m - cm_m)
        / (mp.factorial(m1) * mp.factorial(m2) * (mp.mpf("2") ** total_m))
    )

    acc = mp.mpf("0")
    k_min = max(0, cm_m - m2)
    k_max = min(m1, cm_m)
    for k in range(k_min, k_max + 1):
        term = mp.binomial(m1, k) * mp.binomial(m2, cm_m - k)
        if (m2 - cm_m + k) % 2:
            acc -= term
        else:
            acc += term
    return pref * acc


def _disk_two_body_mpmath(
    m1: int,
    m2: int,
    m3: int,
    m4: int,
    *,
    pseudopotentials: list[float],
    antisymmetrize: bool = False,
    dps: int = 80,
) -> float:
    if (m1 + m2) != (m3 + m4):
        return 0.0

    mp.mp.dps = int(dps)
    total_m = m1 + m2
    rel_channels = [mp.mpf(v) for v in pseudopotentials]
    while len(rel_channels) <= total_m:
        rel_channels.append(mp.mpf("0"))

    value = mp.mpf("0")
    for cm_m in range(total_m + 1):
        coeff_bra = _disk_cm_relative_coefficient_mpmath(m1, m2, cm_m, dps=dps)
        coeff_ket = _disk_cm_relative_coefficient_mpmath(m3, m4, cm_m, dps=dps)
        if antisymmetrize:
            coeff_swap = _disk_cm_relative_coefficient_mpmath(m4, m3, cm_m, dps=dps)
            coeff_ket -= coeff_swap
        value += coeff_bra * coeff_ket * rel_channels[total_m - cm_m]
    return float(value)


def _complex_record(value: complex) -> dict[str, float]:
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def _complex_from_record(record: dict[str, float]) -> complex:
    return complex(float(record["real"]), float(record["imag"]))


def build_reference_dataset() -> dict[str, Any]:
    guiding_entries = [
        [0, 0, 0],
        [0, 1, 0],
        [1, 2, 1],
        [1, 4, 2],
        [2, 5, 5],
    ]
    guiding_cases: list[dict[str, Any]] = []
    for sign in (-1, +1):
        q_magnitudes = [0.25, 1.1, 2.8]
        q_angles = [0.1, -0.7, 1.2]
        guiding_cases.append(
            {
                "label": f"sign_{sign:+d}",
                "mmax": 6,
                "sign_magneticfield": sign,
                "q_magnitudes": q_magnitudes,
                "q_angles": q_angles,
                "entries": guiding_entries,
                "values": [
                    _complex_record(
                        _guiding_center_form_factor_mpmath(
                            m_row,
                            m_col,
                            q_magnitudes[iq],
                            q_angles[iq],
                            sign_magneticfield=sign,
                        )
                    )
                    for iq, m_row, m_col in guiding_entries
                ],
            }
        )

    central_cases = [
        {
            "label": "coulomb",
            "nmax": 4,
            "mmax": 4,
            "potential": "coulomb",
            "kappa": 1.0,
            "qmax": 35.0,
            "nquad": 2000,
            "select": [
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [2, 1, 1, 0],
                [3, 2, 2, 1],
                [3, 3, 2, 2],
                [0, 1, 1, 1],
            ],
        },
        {
            "label": "constant",
            "nmax": 4,
            "mmax": 4,
            "potential": "constant",
            "kappa": 1.5,
            "qmax": 35.0,
            "nquad": 2000,
            "select": [
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [2, 1, 1, 0],
                [3, 2, 2, 1],
                [3, 3, 2, 2],
                [0, 1, 1, 1],
            ],
        },
    ]
    for case in central_cases:
        case["values"] = [
            _central_onebody_mpmath(
                n_row,
                m_row,
                n_col,
                m_col,
                potential=str(case["potential"]),
                kappa=float(case["kappa"]),
                qmax=float(case["qmax"]),
            )
            for n_row, m_row, n_col, m_col in case["select"]
        ]

    pseudopotential_cases = [
        {
            "label": f"coulomb_n{n_ll}",
            "mmax": 6,
            "n_ll": n_ll,
            "potential": "coulomb",
            "kappa": 1.0,
            "qmax": 35.0,
            "nquad": 2500,
            "values": [
                _pseudopotential_coulomb_mpmath(
                    m,
                    n_ll=n_ll,
                    kappa=1.0,
                    qmax=35.0,
                )
                for m in range(6)
            ],
        }
        for n_ll in (0, 1, 2)
    ]

    disk_cases = [
        {
            "label": "model_symmetrized",
            "mmax": 4,
            "pseudopotentials": [1.5, 0.75, 0.125, 0.0, 0.05],
            "antisymmetrize": False,
            "select": [
                [0, 0, 0, 0],
                [0, 1, 0, 1],
                [0, 1, 1, 0],
                [0, 2, 1, 1],
                [1, 2, 0, 3],
                [0, 0, 0, 1],
            ],
        },
        {
            "label": "model_antisymmetrized",
            "mmax": 4,
            "pseudopotentials": [1.5, 0.75, 0.125, 0.0, 0.05],
            "antisymmetrize": True,
            "select": [
                [0, 1, 0, 1],
                [0, 1, 1, 0],
                [0, 2, 1, 1],
                [1, 2, 0, 3],
            ],
        },
    ]
    for case in disk_cases:
        case["values"] = [
            _disk_two_body_mpmath(
                m1,
                m2,
                m3,
                m4,
                pseudopotentials=list(case["pseudopotentials"]),
                antisymmetrize=bool(case["antisymmetrize"]),
            )
            for m1, m2, m3, m4 in case["select"]
        ]

    return {
        "format_version": 1,
        "tolerances": {"rtol": DEFAULT_RTOL, "atol": DEFAULT_ATOL},
        "guiding_center": guiding_cases,
        "central_onebody": central_cases,
        "pseudopotentials": pseudopotential_cases,
        "disk_two_body": disk_cases,
    }


def write_reference_data(path: Path = REFERENCE_PATH) -> Path:
    path = Path(path)
    path.write_text(json.dumps(build_reference_dataset(), indent=2, sort_keys=True) + "\n")
    return path


def load_reference_data(path: Path = REFERENCE_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _max_abs_rel_err(values: np.ndarray, refs: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(values) - np.asarray(refs)
    abs_err = float(np.max(np.abs(diff)))
    rel_err = float(np.max(np.abs(diff) / np.maximum(np.abs(refs), DEFAULT_ATOL)))
    return abs_err, rel_err


def validate_reference_data(path: Path = REFERENCE_PATH) -> list[dict[str, Any]]:
    from quantumhall_matrixelements import (
        get_central_onebody_matrix_elements_compressed,
        get_guiding_center_form_factors,
        get_haldane_pseudopotentials,
        get_twobody_disk_from_pseudopotentials_compressed,
    )

    data = load_reference_data(path)
    rtol = float(data["tolerances"]["rtol"])
    atol = float(data["tolerances"]["atol"])

    results: list[dict[str, Any]] = []

    for case in data["guiding_center"]:
        values = get_guiding_center_form_factors(
            np.asarray(case["q_magnitudes"], dtype=float),
            np.asarray(case["q_angles"], dtype=float),
            int(case["mmax"]),
            sign_magneticfield=int(case["sign_magneticfield"]),
        )
        refs = np.asarray([_complex_from_record(v) for v in case["values"]], dtype=np.complex128)
        numeric = np.asarray(
            [values[iq, m_row, m_col] for iq, m_row, m_col in case["entries"]],
            dtype=np.complex128,
        )
        abs_err, rel_err = _max_abs_rel_err(numeric, refs)
        results.append(
            {
                "section": "guiding_center",
                "label": str(case["label"]),
                "abs_err": abs_err,
                "rel_err": rel_err,
                "ok": bool(np.allclose(numeric, refs, rtol=rtol, atol=atol)),
            }
        )

    for case in data["central_onebody"]:
        numeric, select_list = get_central_onebody_matrix_elements_compressed(
            int(case["nmax"]),
            int(case["mmax"]),
            potential=str(case["potential"]),
            kappa=float(case["kappa"]),
            qmax=float(case["qmax"]),
            nquad=int(case["nquad"]),
            select=case["select"],
        )
        refs = np.asarray(case["values"], dtype=float)
        abs_err, rel_err = _max_abs_rel_err(numeric, refs)
        results.append(
            {
                "section": "central_onebody",
                "label": str(case["label"]),
                "abs_err": abs_err,
                "rel_err": rel_err,
                "ok": bool(select_list == [tuple(item) for item in case["select"]])
                and bool(np.allclose(numeric, refs, rtol=rtol, atol=atol)),
            }
        )

    for case in data["pseudopotentials"]:
        numeric = get_haldane_pseudopotentials(
            int(case["mmax"]),
            n_ll=int(case["n_ll"]),
            potential=str(case["potential"]),
            kappa=float(case["kappa"]),
            qmax=float(case["qmax"]),
            nquad=int(case["nquad"]),
        )
        refs = np.asarray(case["values"], dtype=float)
        abs_err, rel_err = _max_abs_rel_err(numeric, refs)
        results.append(
            {
                "section": "pseudopotentials",
                "label": str(case["label"]),
                "abs_err": abs_err,
                "rel_err": rel_err,
                "ok": bool(np.allclose(numeric, refs, rtol=rtol, atol=atol)),
            }
        )

    for case in data["disk_two_body"]:
        numeric, select_list = get_twobody_disk_from_pseudopotentials_compressed(
            np.asarray(case["pseudopotentials"], dtype=float),
            int(case["mmax"]),
            select=case["select"],
            antisymmetrize=bool(case["antisymmetrize"]),
        )
        refs = np.asarray(case["values"], dtype=float)
        abs_err, rel_err = _max_abs_rel_err(numeric, refs)
        results.append(
            {
                "section": "disk_two_body",
                "label": str(case["label"]),
                "abs_err": abs_err,
                "rel_err": rel_err,
                "ok": bool(select_list == [tuple(item) for item in case["select"]])
                and bool(np.allclose(numeric, refs, rtol=rtol, atol=atol)),
            }
        )

    return results


__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "REFERENCE_PATH",
    "build_reference_dataset",
    "load_reference_data",
    "validate_reference_data",
    "write_reference_data",
]
