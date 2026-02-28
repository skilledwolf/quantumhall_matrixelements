#!/usr/bin/env python3
"""Benchmark accuracy and speed of exchange-kernel backends.

Run from the repo root:

  PYTHONPATH=src python validation/benchmark_exchange_backends.py

For more stable timing results, pin BLAS threading, e.g.:

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
    python validation/benchmark_exchange_backends.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from quantumhall_matrixelements import get_exchange_kernels


@dataclass(frozen=True)
class Timing:
    t_min: float
    t_mean: float


def _max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _per_g_max_abs_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return max abs error per leading (G) index."""
    return np.max(np.abs(a - b), axis=(1, 2, 3, 4))


def _time_call(fn, repeats: int, warmup: int) -> Timing:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, dtype=float)
    return Timing(t_min=float(arr.min()), t_mean=float(arr.mean()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmax", type=int, default=10)
    parser.add_argument("--nG", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--nquad", type=int, default=800, help="fock_fast nquad for perf runs")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    nmax = int(args.nmax)
    nG = int(args.nG)

    # -----------------------------
    # Accuracy vs Hankel (reference)
    # -----------------------------
    kvals = np.array([0.0, 0.5, 1.0, 2.0, 4.0], dtype=float)
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=kvals.size)

    X_hk = get_exchange_kernels(kvals, thetas, nmax, method="hankel")
    X_og = get_exchange_kernels(kvals, thetas, nmax, method="ogata")
    X_ff = get_exchange_kernels(kvals, thetas, nmax, method="fock_fast")

    print("Accuracy vs hankel (max abs error per |G|ℓ):")
    print("  k:", kvals)
    print("  ogata:", _per_g_max_abs_err(X_og, X_hk))
    print("  fock_fast:", _per_g_max_abs_err(X_ff, X_hk))

    # -----------------------------
    # Performance comparison
    # -----------------------------
    scenarios = {
        "A k in [2,6] (ogata-only)": (2.0, 6.0),
        "B k in [0,6] (mixed)": (0.0, 6.0),
        "C k in [0,1] (fallback-only)": (0.0, 1.0),
    }

    print("\nPerformance (nmax={}, nG={}, repeats={}):".format(nmax, nG, args.repeats))
    for name, (k_lo, k_hi) in scenarios.items():
        G_mags = rng.uniform(k_lo, k_hi, size=nG)
        G_angles = rng.uniform(0.0, 2.0 * np.pi, size=nG)

        t_ff = _time_call(
            lambda: get_exchange_kernels(
                G_mags, G_angles, nmax, method="fock_fast", nquad=int(args.nquad)
            ),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
        )
        t_og = _time_call(
            lambda: get_exchange_kernels(G_mags, G_angles, nmax, method="ogata"),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
        )

        print(f"  {name}")
        print(f"    fock_fast nquad={int(args.nquad)}: {t_ff.t_min:.3f}s min  {t_ff.t_mean:.3f}s mean")
        print(f"    ogata (defaults):          {t_og.t_min:.3f}s min  {t_og.t_mean:.3f}s mean")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
