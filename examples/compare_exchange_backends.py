"""Compare exchange-kernel backends: accuracy and timing.

This script computes exchange kernels with all three public backends
(laguerre, ogata, hankel) and plots:
  1. Relative error of each backend vs the Hankel reference.
  2. Wall-clock timing comparison at large nmax.

It demonstrates laguerre's numerical stability at large nmax
and the Ogata-in-q-space path for large |G|.
"""
from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt

from quantumhall_matrixelements import get_exchange_kernels_compressed


# ── helpers ──────────────────────────────────────────────────────────────
def _rel_error(vals: np.ndarray, ref: np.ndarray) -> float:
    """Max relative error over all entries."""
    diff = np.abs(vals - ref)
    scale = np.max(np.abs(ref))
    if scale == 0:
        return float(np.max(diff))
    return float(np.max(diff) / scale)


def _timed_compressed(G, theta, nmax, select, **kw):
    """Call get_exchange_kernels_compressed and return (values, elapsed_seconds)."""
    t0 = time.perf_counter()
    vals, sel = get_exchange_kernels_compressed(G, theta, nmax, select=select, **kw)
    elapsed = time.perf_counter() - t0
    return vals, sel, elapsed


# ── main ─────────────────────────────────────────────────────────────────
def main() -> None:
    # --- Panel 1: moderate nmax, sweep over |G| ---
    nmax_mod = 4
    Gs_mod = np.linspace(0.2, 25.0, 50)
    thetas_mod = np.zeros_like(Gs_mod)
    select_mod = [(0, 0, 0, 0), (1, 1, 1, 1), (2, 1, 2, 1), (3, 0, 3, 0)]

    print(f"Panel 1: nmax={nmax_mod}, {len(Gs_mod)} G-points, {len(select_mod)} entries")
    print("  Computing reference (hankel)…")
    v_hk, _, t_hk = _timed_compressed(
        Gs_mod, thetas_mod, nmax_mod, select_mod, method="hankel",
    )
    print(f"    hankel:        {t_hk:.2f}s")

    backends_mod = {}
    for label, kw in [
        ("ogata", dict(method="ogata")),
        ("laguerre (GL)", dict(method="laguerre", use_ogata=False)),
        ("laguerre (hybrid)", dict(method="laguerre", use_ogata=True, kmin_ogata=10.0)),
    ]:
        v, _, t = _timed_compressed(
            Gs_mod, thetas_mod, nmax_mod, select_mod, **kw,
        )
        # per-G relative error for entry 0
        err_per_G = np.abs(v - v_hk).max(axis=1) / np.maximum(np.abs(v_hk).max(axis=1), 1e-30)
        backends_mod[label] = (err_per_G, t)
        print(f"    {label:25s}: {t:.2f}s  max_rel_err={_rel_error(v, v_hk):.2e}")

    # --- Panel 2: large nmax, fixed G set ---
    nmax_large = 50
    Gs_large = np.array([0.0, 5.0, 15.0, 30.0])
    thetas_large = np.zeros_like(Gs_large)
    select_large = [
        (0, 0, 0, 0),
        (10, 10, 10, 10),
        (nmax_large - 1, nmax_large - 1, nmax_large - 1, nmax_large - 1),
        (nmax_large - 1, 0, nmax_large - 1, 0),
    ]

    print(f"\nPanel 2: nmax={nmax_large}, G={list(Gs_large)}, {len(select_large)} entries")
    print("  Computing reference (hankel)…")
    v_hk_lg, _, t_hk_lg = _timed_compressed(
        Gs_large, thetas_large, nmax_large, select_large, method="hankel",
    )
    print(f"    hankel:        {t_hk_lg:.2f}s")

    backends_large: dict[str, tuple[float, float]] = {}
    for label, kw in [
        ("laguerre (GL)", dict(method="laguerre", use_ogata=False)),
        ("laguerre (hybrid)", dict(method="laguerre", use_ogata=True, kmin_ogata=10.0)),
    ]:
        v, _, t = _timed_compressed(
            Gs_large, thetas_large, nmax_large, select_large, **kw,
        )
        err = _rel_error(v, v_hk_lg)
        backends_large[label] = (err, t)
        print(f"    {label:25s}: {t:.2f}s  max_rel_err={err:.2e}")

    # ── plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: per-G error
    ax = axes[0]
    for label, (err_g, t) in backends_mod.items():
        ax.semilogy(Gs_mod, err_g, label=f"{label} ({t:.1f}s)")
    ax.set_xlabel(r"$|G|\,\ell_B$")
    ax.set_ylabel("max relative error vs Hankel")
    ax.set_title(f"Backend accuracy (nmax={nmax_mod})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=1e-16)

    # Panel 2: bar chart for large nmax
    ax2 = axes[1]
    labels_lg = list(backends_large.keys())
    errs_lg = [backends_large[l][0] for l in labels_lg]
    times_lg = [backends_large[l][1] for l in labels_lg]
    x = np.arange(len(labels_lg))
    colors = ["#2ca02c", "#1f77b4"]
    bars = ax2.bar(x, errs_lg, color=colors[:len(x)])
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_lg, fontsize=8, rotation=15, ha="right")
    ax2.set_ylabel("max relative error vs Hankel")
    ax2.set_title(f"Large nmax={nmax_large}, G up to {Gs_large.max():.0f}")
    for i, (e, t) in enumerate(zip(errs_lg, times_lg)):
        ax2.text(i, e * 2, f"{t:.1f}s", ha="center", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
