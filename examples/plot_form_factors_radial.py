"""Radial Landau-level form factors F_{n',n}(q) vs qℓ_B.

This example plots the lowest diagonal LL form factors as a function of
dimensionless momentum qℓ_B, illustrating the Gaussian envelope and
Laguerre oscillations.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from quantumhall_matrixelements import get_form_factors


def main() -> None:
    # Dimensionless |G|ℓ_B grid
    q = np.linspace(0.0, 5.0, 400)
    theta = np.zeros_like(q)
    nmax = 3

    F = get_form_factors(q, theta, nmax)  # shape (nq, nmax, nmax)

    fig, ax = plt.subplots()
    for n in range(nmax):
        ax.plot(q, F[:, n, n].real, label=fr"$\mathrm{{Re}}\,F_{{{n}{n}}}(q)$")

    ax.set_xlabel(r"$q \ell_B$")
    ax.set_ylabel(r"$\mathrm{Re}\,F_{nn}(q)$")
    ax.set_title("Diagonal Landau-level form factors")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

