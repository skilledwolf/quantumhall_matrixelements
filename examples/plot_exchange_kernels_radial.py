"""Exchange kernel diagonal elements X_{nnnn}(G) vs |G|ℓ_B.

This example computes selected diagonal components of the exchange kernel
using the Gauss–Laguerre backend and plots their real parts as a function
of |G|ℓ_B.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from quantumhall_matrixelements import get_exchange_kernels


def main() -> None:
    nmax = 3
    # Avoid G=0 to dodge the Coulomb singularity; start from small q
    q = np.linspace(0.2, 4.0, 80)
    theta = np.zeros_like(q)

    X = get_exchange_kernels(q, theta, nmax, method="gausslag")

    fig, ax = plt.subplots()
    for n in range(nmax):
        vals = X[:, n, n, n, n]
        ax.plot(q, vals.real, label=fr"$\mathrm{{Re}}\,X_{{{n}{n}{n}{n}}}(G)$")

    for n in range(1,nmax):
        vals = X[:, n, n-1, n-1, n]
        ax.plot(q, vals.real, label=fr"$\mathrm{{Re}}\,X_{{{n}{n-1}{n-1}{n}}}(G)$")

    ax.set_xlabel(r"$|G| \ell_B$")
    ax.set_ylabel(r"$\mathrm{Re}\,X_{nnnn}(G)$  (κ=1)")
    ax.set_title("Diagonal exchange kernels (Gauss–Laguerre backend)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

