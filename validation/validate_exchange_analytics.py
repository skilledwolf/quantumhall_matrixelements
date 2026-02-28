#!/usr/bin/env python3
"""Validate analytic closed-form references for exchange kernels.

Run from the repo root:
  PYTHONPATH=src python validation/validate_exchange_analytics.py

The checks cover:
  1) LLL closed form for X_0000(G).
  2) G=0 closed form for X_nnnn(0) via a terminating 3F2 series.
  3) G=0 angular selection rule: X_{n1 m1 n2 m2}(0) = 0 unless (m1-n1)=(m2-n2).
  4) Optional finite double-sum reference for X_nnnn(G) at nonzero G.
  5) Optional targeted single-n quadrature (mpmath), suitable for n=100.

Notes:
  - Keep nmax modest; the backend computes full (nmax^4) kernels.
  - Requires mpmath; no fallback paths are provided.
  - The targeted single-n check avoids backend tensors and is independent.
"""
from __future__ import annotations

import argparse
import math

import mpmath as mp
import numpy as np
import scipy.special as sps

from quantumhall_matrixelements import get_exchange_kernels, get_exchange_kernels_compressed


def _parse_csv_floats(text: str) -> np.ndarray:
    items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected a comma-separated list of floats.")
    return np.asarray([float(item) for item in items], dtype=float)


def x_nnnn_g_quad_mpmath(
    n: int,
    G: float,
    kappa: float,
    *,
    dps: int,
    qmax: float,
    segments: int,
) -> float:
    """Direct 1D quadrature of X_nnnn(G) using mpmath (independent reference)."""
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

    qmax = float(qmax)
    segments = max(1, int(segments))
    if segments == 1:
        total = mp.quad(integrand, [0, qmax])
    else:
        step = qmax / segments
        total = mp.mpf("0")
        for i in range(segments):
            a = i * step
            b = (i + 1) * step
            total += mp.quad(integrand, [a, b])

    return float(kappa * total)


def x_nnnn_g_series_mpmath(n: int, G: float, kappa: float, *, dps: int) -> float:
    """Finite double-sum reference using mpmath (stable for large n)."""
    mp.mp.dps = int(dps)
    total = mp.mpf("0")
    G2 = mp.mpf(G) ** 2
    half = mp.mpf("0.5")

    for k in range(n + 1):
        binom_n_k = mp.binomial(n, k)
        fact_k = mp.factorial(k)
        sign_k = -1 if (k % 2) else 1
        for ell in range(n + 1):
            s = k + ell + half
            coeff = binom_n_k * mp.binomial(n, ell) * sign_k * (-1 if (ell % 2) else 1)
            coeff /= fact_k * mp.factorial(ell)
            term = coeff * mp.gamma(s) * mp.hyp1f1(s, 1, -G2 / 2)
            total += term

    return float(kappa * total / mp.sqrt(2))


def threef2_terminating(n: int, *, dps: int | None = None) -> float:
    """Compute 3F2(-n,1/2,1/2;1,1;1) with mpmath."""
    if dps is None:
        dps = max(50, int(0.7 * n) + 30)
    mp.mp.dps = dps

    total = mp.mpf("0")
    half = mp.mpf("0.5")
    for k in range(n + 1):
        term = mp.rf(-n, k) * mp.rf(half, k) ** 2 / (mp.factorial(k) ** 3)
        total += term
    return float(total)


def x_nnnn_g0_closed(n: int, kappa: float) -> float:
    return kappa * math.sqrt(math.pi / 2.0) * threef2_terminating(n)


def x_0000_g_closed(G: np.ndarray, kappa: float) -> np.ndarray:
    G = np.asarray(G, dtype=float)
    x = 0.25 * G * G
    return kappa * math.sqrt(math.pi / 2.0) * np.exp(-x) * sps.i0(x)


def x_nnnn_g_series(n: int, G: float, kappa: float) -> float:
    """Finite double-sum reference for X_nnnn(G) (Coulomb, ell=1)."""
    log_n_fact = sps.gammaln(n + 1)
    ks = np.arange(n + 1)
    log_fact = sps.gammaln(ks + 1)
    log_factor = log_n_fact - sps.gammaln(n - ks + 1) - 2.0 * log_fact
    signs = 1.0 - 2.0 * (ks & 1)

    total = 0.0
    G2 = float(G) * float(G)
    for k in range(n + 1):
        log_fk = log_factor[k]
        sign_k = signs[k]
        for ell in range(n + 1):
            s = k + ell + 0.5
            log_coeff = log_fk + log_factor[ell] + sps.gammaln(s)
            coeff = sign_k * signs[ell] * math.exp(log_coeff)
            total += coeff * sps.hyp1f1(s, 1.0, -0.5 * G2)
    return kappa * total / math.sqrt(2.0)


def _max_abs_and_rel_err(values: np.ndarray, refs: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(values) - np.asarray(refs)
    abs_err = np.max(np.abs(diff))
    denom = np.maximum(np.abs(refs), 1e-300)
    rel_err = np.max(np.abs(diff) / denom)
    return float(abs_err), float(rel_err)


def _format_pass(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _print_table(Gs: np.ndarray, values: np.ndarray, refs: np.ndarray, rtol: float, atol: float):
    abs_err = np.abs(values - refs)
    rel_err = abs_err / np.maximum(np.abs(refs), 1e-300)
    ok = np.isclose(values, refs, rtol=rtol, atol=atol)
    header = "    G        numeric                 analytic                abs_err       rel_err"
    print(header)
    for G, v, r, ae, re, ok_i in zip(Gs, values, refs, abs_err, rel_err, ok):
        print(
            f"  {G:7.3f}  {v: .16e}  {r: .16e}  {ae: .3e}  {re: .3e}  {_format_pass(bool(ok_i))}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="laguerre", help="Backend method.")
    parser.add_argument("--nmax", type=int, default=8, help="nmax for G=0 checks and selection rule.")
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--sign", type=int, default=-1, dest="sign_magneticfield")
    parser.add_argument(
        "--g-values",
        default="0,0.5,1,2,3",
        help="Comma-separated |G| values for LLL check.",
    )
    parser.add_argument(
        "--diag-nmax",
        type=int,
        default=10,
        help="Max n for X_nnnn(G=0) closed-form comparison.",
    )
    parser.add_argument(
        "--series-n",
        type=int,
        default=None,
        help="If set, run finite double-sum reference for this n.",
    )
    parser.add_argument(
        "--series-g-values",
        default="0.5,1.5,3.0",
        help="Comma-separated |G| values for the double-sum check.",
    )
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--atol", type=float, default=5e-6)
    parser.add_argument("--zero-atol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-n",
        type=int,
        default=None,
        help="Run targeted single-n check via mpmath quadrature (avoids nmax^4 tensors).",
    )
    parser.add_argument(
        "--target-g-values",
        default="0,2",
        help="Comma-separated |G| values for targeted single-n check.",
    )
    parser.add_argument("--quad-qmax", type=float, default=35.0)
    parser.add_argument("--quad-segments", type=int, default=8)
    parser.add_argument("--quad-dps", type=int, default=None)
    parser.add_argument("--target-series", action="store_true")
    parser.add_argument("--series-dps", type=int, default=None)

    # Backend knobs (optional)
    parser.add_argument("--nquad", type=int, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--ogata-h", type=float, default=None)
    parser.add_argument("--ogata-N", type=int, default=None)
    parser.add_argument("--kmin-ogata", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)

    return parser


def run_validation(args: argparse.Namespace) -> int:

    rng = np.random.default_rng(int(args.seed))

    backend_kwargs: dict[str, object] = {
        "kappa": float(args.kappa),
        "sign_magneticfield": int(args.sign_magneticfield),
    }
    if args.nquad is not None:
        backend_kwargs["nquad"] = int(args.nquad)
    if args.scale is not None:
        backend_kwargs["scale"] = float(args.scale)
    if args.ogata_h is not None:
        backend_kwargs["ogata_h"] = float(args.ogata_h)
    if args.ogata_N is not None:
        backend_kwargs["ogata_N"] = int(args.ogata_N)
    if args.kmin_ogata is not None:
        backend_kwargs["kmin_ogata"] = float(args.kmin_ogata)
    if args.chunk_size is not None:
        backend_kwargs["chunk_size"] = int(args.chunk_size)

    ok_all = True

    # ------------------------------------------------------------------
    # 1) LLL closed form for X_0000(G)
    # ------------------------------------------------------------------
    Gs = _parse_csv_floats(args.g_values)
    angles = rng.uniform(0.0, 2.0 * math.pi, size=Gs.size)
    angles = np.where(Gs == 0.0, 0.0, angles)

    X_lll = get_exchange_kernels(Gs, angles, 1, method=args.method, **backend_kwargs)
    vals_lll = X_lll[:, 0, 0, 0, 0]
    imag_lll = float(np.max(np.abs(np.imag(vals_lll))))
    vals_lll = np.real(vals_lll)
    refs_lll = x_0000_g_closed(Gs, float(args.kappa))

    print("LLL closed form X_0000(G):")
    _print_table(Gs, vals_lll, refs_lll, args.rtol, args.atol)
    abs_err, rel_err = _max_abs_and_rel_err(vals_lll, refs_lll)
    ok_lll = np.allclose(vals_lll, refs_lll, rtol=args.rtol, atol=args.atol)
    ok_all &= bool(ok_lll)
    print(
        f"  max abs err: {abs_err:.3e}, max rel err: {rel_err:.3e}, "
        f"max imag: {imag_lll:.3e} -> {_format_pass(bool(ok_lll))}\n"
    )

    # ------------------------------------------------------------------
    # 2) G=0 closed form for X_nnnn(0)
    # ------------------------------------------------------------------
    nmax = int(args.nmax)
    if nmax < 1:
        raise ValueError("nmax must be >= 1")
    diag_nmax = min(nmax - 1, int(args.diag_nmax))

    X0 = get_exchange_kernels([0.0], [0.0], nmax, method=args.method, **backend_kwargs)[0]
    diag_vals = np.array([X0[n, n, n, n] for n in range(diag_nmax + 1)], dtype=complex)
    diag_imag = float(np.max(np.abs(np.imag(diag_vals))))
    diag_vals = np.real(diag_vals)

    diag_refs = np.array(
        [x_nnnn_g0_closed(n, float(args.kappa)) for n in range(diag_nmax + 1)],
        dtype=float,
    )

    print("G=0 diagonal closed form X_nnnn(0):")
    for n, v, r in zip(range(diag_nmax + 1), diag_vals, diag_refs):
        abs_err = abs(v - r)
        rel_err = abs_err / (abs(r) if r != 0 else 1.0)
        ok = math.isclose(v, r, rel_tol=args.rtol, abs_tol=args.atol)
        print(
            f"  n={n:3d}  {v: .16e}  {r: .16e}  abs {abs_err: .3e}  rel {rel_err: .3e}  "
            f"{_format_pass(ok)}"
        )
    abs_err, rel_err = _max_abs_and_rel_err(diag_vals, diag_refs)
    ok_diag = np.allclose(diag_vals, diag_refs, rtol=args.rtol, atol=args.atol)
    ok_all &= bool(ok_diag)
    print(
        f"  max abs err: {abs_err:.3e}, max rel err: {rel_err:.3e}, "
        f"max imag: {diag_imag:.3e} -> {_format_pass(bool(ok_diag))}\n"
    )

    # ------------------------------------------------------------------
    # 3) G=0 selection rule check
    # ------------------------------------------------------------------
    idx = np.arange(nmax)
    D = idx[:, None] - idx[None, :]
    mask = D[:, :, None, None] != D[None, None, :, :]
    viol = np.abs(X0[mask])
    max_viol = float(np.max(viol)) if viol.size else 0.0
    count_viol = int(np.sum(viol > float(args.zero_atol)))
    ok_sel = max_viol <= float(args.zero_atol)
    ok_all &= bool(ok_sel)
    print("G=0 selection rule (m1-n1 == m2-n2):")
    print(
        f"  max violation: {max_viol:.3e}, count > {float(args.zero_atol):.1e}: {count_viol} "
        f"-> {_format_pass(bool(ok_sel))}\n"
    )

    # ------------------------------------------------------------------
    # 4) Optional finite double-sum check for X_nnnn(G)
    # ------------------------------------------------------------------
    if args.series_n is not None:
        n_series = int(args.series_n)
        Gs_series = _parse_csv_floats(args.series_g_values)
        angles_series = rng.uniform(0.0, 2.0 * math.pi, size=Gs_series.size)
        angles_series = np.where(Gs_series == 0.0, 0.0, angles_series)

        X_series = get_exchange_kernels(
            Gs_series,
            angles_series,
            n_series + 1,
            method=args.method,
            **backend_kwargs,
        )
        vals_series = X_series[:, n_series, n_series, n_series, n_series]
        imag_series = float(np.max(np.abs(np.imag(vals_series))))
        vals_series = np.real(vals_series)

        refs_series = np.array(
            [x_nnnn_g_series(n_series, float(G), float(args.kappa)) for G in Gs_series],
            dtype=float,
        )

        print(f"Double-sum reference for X_{n_series}{n_series}{n_series}{n_series}(G):")
        _print_table(Gs_series, vals_series, refs_series, args.rtol, args.atol)
        abs_err, rel_err = _max_abs_and_rel_err(vals_series, refs_series)
        ok_series = np.allclose(vals_series, refs_series, rtol=args.rtol, atol=args.atol)
        ok_all &= bool(ok_series)
        print(
            f"  max abs err: {abs_err:.3e}, max rel err: {rel_err:.3e}, "
            f"max imag: {imag_series:.3e} -> {_format_pass(bool(ok_series))}\n"
        )

    # ------------------------------------------------------------------
    # 4b) Targeted single-n check via mpmath quadrature (no backend tensor)
    # ------------------------------------------------------------------
    if args.target_n is not None:
        n_target = int(args.target_n)
        Gs_target = _parse_csv_floats(args.target_g_values)
        quad_dps = (
            int(args.quad_dps)
            if args.quad_dps is not None
            else max(70, int(0.6 * n_target) + 40)
        )
        series_dps = (
            int(args.series_dps)
            if args.series_dps is not None
            else max(200, int(1.5 * n_target) + 50)
        )
        print(
            f"Targeted mpmath quad check for n={n_target} "
            f"(dps={quad_dps}, qmax={float(args.quad_qmax):.1f}):"
        )
        quad_vals: list[float] = []
        ref_vals: list[float] = []
        for G in Gs_target:
            quad_val = x_nnnn_g_quad_mpmath(
                n_target,
                float(G),
                float(args.kappa),
                dps=quad_dps,
                qmax=float(args.quad_qmax),
                segments=int(args.quad_segments),
            )
            quad_vals.append(quad_val)
            if G == 0.0:
                ref_vals.append(x_nnnn_g0_closed(n_target, float(args.kappa)))
            elif args.target_series:
                ref_vals.append(
                    x_nnnn_g_series_mpmath(n_target, float(G), float(args.kappa), dps=series_dps)
                )
            else:
                ref_vals.append(float("nan"))

        quad_vals_arr = np.asarray(quad_vals, dtype=float)
        ref_vals_arr = np.asarray(ref_vals, dtype=float)
        finite = np.isfinite(ref_vals_arr)
        if np.any(finite):
            _print_table(
                Gs_target[finite], quad_vals_arr[finite], ref_vals_arr[finite], args.rtol, args.atol
            )
            abs_err, rel_err = _max_abs_and_rel_err(quad_vals_arr[finite], ref_vals_arr[finite])
            ok_target = np.allclose(
                quad_vals_arr[finite], ref_vals_arr[finite], rtol=args.rtol, atol=args.atol
            )
            ok_all &= bool(ok_target)
            print(
                f"  max abs err: {abs_err:.3e}, max rel err: {rel_err:.3e} -> "
                f"{_format_pass(bool(ok_target))}"
            )
        if np.any(~finite):
            print("  quad-only values (no analytic reference requested):")
            for G, val in zip(Gs_target[~finite], quad_vals_arr[~finite]):
                print(f"    G={G:7.3f}  quad {val: .16e}")
        print()

        # Backend check using package implementation (use compressed API to avoid nmax^4 tensor).
        select = [(n_target, n_target, n_target, n_target)]
        angles_backend = np.zeros_like(Gs_target, dtype=float)
        values_backend, select_backend = get_exchange_kernels_compressed(
            Gs_target,
            angles_backend,
            n_target + 1,
            method=args.method,
            select=select,
            **backend_kwargs,
        )
        assert select_backend == select
        backend_vals = values_backend[:, 0]
        backend_imag = float(np.max(np.abs(np.imag(backend_vals))))
        backend_vals = np.real(backend_vals)
        print(f"Backend ({args.method}) vs mpmath quad for n={n_target}:")
        _print_table(Gs_target, backend_vals, quad_vals_arr, args.rtol, args.atol)
        abs_err, rel_err = _max_abs_and_rel_err(backend_vals, quad_vals_arr)
        ok_backend = np.allclose(backend_vals, quad_vals_arr, rtol=args.rtol, atol=args.atol)
        ok_all &= bool(ok_backend)
        print(
            f"  max abs err: {abs_err:.3e}, max rel err: {rel_err:.3e}, "
            f"max imag: {backend_imag:.3e} -> {_format_pass(bool(ok_backend))}\n"
        )

    # ------------------------------------------------------------------
    # 5) n=100 scalar reference (analytic only)
    # ------------------------------------------------------------------
    n_ref = 100
    ref_val = x_nnnn_g0_closed(n_ref, float(args.kappa))
    target = 0.2098403349794916687 * float(args.kappa)
    abs_err = abs(ref_val - target)
    print("n=100 scalar reference (analytic only):")
    print(f"  X_100100100100(0) = {ref_val:.19f}")
    print(f"  target value       = {target:.19f}")
    print(f"  abs diff           = {abs_err:.3e}\n")

    return 0 if ok_all else 1


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_validation(args)


if __name__ == "__main__":
    raise SystemExit(main())
