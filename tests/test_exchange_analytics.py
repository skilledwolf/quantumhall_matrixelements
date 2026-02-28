from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import scipy.special as sps

from quantumhall_matrixelements import get_exchange_kernels, get_exchange_kernels_compressed


def _x_0000_g_closed(G: np.ndarray) -> np.ndarray:
    x = 0.25 * G * G
    return math.sqrt(math.pi / 2.0) * np.exp(-x) * sps.i0(x)


def _x_nnnn_g_quad_mpmath(
    n: int,
    G: float,
    *,
    dps: int = 90,
    qmax: float = 35.0,
    segments: int = 8,
) -> float:
    mp.mp.dps = int(dps)
    G_mp = mp.mpf(G)
    half = mp.mpf("0.5")

    def integrand(q):
        x = q * q * half
        L = mp.laguerre(n, 0, x)
        val = mp.e ** (-x) * L * L
        if G_mp != 0:
            val *= mp.besselj(0, q * G_mp)
        return val

    segments = max(1, int(segments))
    if segments == 1:
        total = mp.quad(integrand, [0, qmax])
    else:
        total = mp.mpf("0")
        step = qmax / segments
        for i in range(segments):
            a = i * step
            b = (i + 1) * step
            total += mp.quad(integrand, [a, b])
    return float(total)


def test_lll_closed_form_select():
    Gs = np.array([0.0, 0.5, 2.0], dtype=float)
    angles = np.zeros_like(Gs)
    select = [(0, 0, 0, 0)]

    values, select_list = get_exchange_kernels_compressed(
        Gs,
        angles,
        1,
        method="fock_fast",
        select=select,
    )
    assert select_list == select
    vals = np.real(values[:, 0])
    refs = _x_0000_g_closed(Gs)
    assert np.allclose(vals, refs, rtol=5e-4, atol=5e-6)


def test_select_matches_full_small():
    Gs = np.array([0.0, 2.0], dtype=float)
    angles = np.array([0.0, 0.3], dtype=float)
    select = [(0, 0, 0, 0), (1, 1, 1, 1), (1, 0, 1, 0)]

    for method, kwargs in [
        ("fock_fast", {}),
        ("ogata", {"nquad": 400}),
        ("hankel", {}),
    ]:
        X_full = get_exchange_kernels(Gs, angles, 2, method=method, **kwargs)
        values_sel, select_list = get_exchange_kernels_compressed(
            Gs,
            angles,
            2,
            method=method,
            select=select,
            **kwargs,
        )
        for idx, (n1, m1, n2, m2) in enumerate(select_list):
            assert np.allclose(
                values_sel[:, idx],
                X_full[:, n1, m1, n2, m2],
                rtol=1e-6,
                atol=1e-8,
            )


def test_n100_backend_sweep_fock_fast_ogata():
    n = 100
    Gs = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0], dtype=float)
    angles = np.zeros_like(Gs)
    select = [(n, n, n, n)]

    refs = np.array(
        [_x_nnnn_g_quad_mpmath(n, float(G), dps=80, qmax=35.0, segments=8) for G in Gs],
        dtype=float,
    )

    values_ff, select_ff = get_exchange_kernels_compressed(
        Gs,
        angles,
        n + 1,
        method="fock_fast",
        select=select,
    )
    values_og, select_og = get_exchange_kernels_compressed(
        Gs,
        angles,
        n + 1,
        method="ogata",
        nquad=800,
        scale=0.04,
        kmin_ogata=5.0,
        ogata_auto=True,
        select=select,
    )

    assert select_ff == select
    assert select_og == select
    vals_ff = np.real(values_ff[:, 0])
    vals_og = np.real(values_og[:, 0])

    assert np.allclose(vals_ff, refs, rtol=1.5e-2, atol=1e-3)
    assert np.allclose(vals_og, refs, rtol=1.5e-2, atol=1e-3)
