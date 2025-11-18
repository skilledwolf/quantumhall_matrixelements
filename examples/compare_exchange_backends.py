"""Compare Gauss–Laguerre and Hankel exchange-kernel backends.

For a small |G|ℓ_B grid and nmax=2, this script computes the exchange
kernels using both the Gauss–Laguerre and Hankel backends and plots the
relative difference of a representative diagonal element X_{0000}(G).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from quantumhall_matrixelements import get_exchange_kernels


def main() -> None:
    nmax = 2
    q = np.linspace(0.2, 3.0, 60)
    theta = np.zeros_like(q)

    X_gl = get_exchange_kernels(q, theta, nmax, method="gausslag")
    X_hk = get_exchange_kernels(q, theta, nmax, method="hankel")

    # Focus on X_{0000}(G) as a simple representative component
    X_gl_diag = X_gl[:, 0, 0, 0, 0]
    X_hk_diag = X_hk[:, 0, 0, 0, 0]

    abs_diff = np.abs(X_gl_diag - X_hk_diag)
    denom = np.maximum(np.abs(X_gl_diag), np.abs(X_hk_diag))
    rel_diff = np.where(denom > 0, abs_diff / denom, 0.0)

    fig, ax = plt.subplots()
    ax.plot(q, rel_diff, marker="o", linestyle="-")
    ax.set_xlabel(r"$|G| \ell_B$")
    ax.set_ylabel(r"relative difference")
    ax.set_title(r"Relative difference of $X_{0000}(G)$: Gauss–Laguerre vs Hankel")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

