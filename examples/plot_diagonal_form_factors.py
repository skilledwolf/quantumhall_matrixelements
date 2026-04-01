r"""
Diagonal Landau-level form factors
==================================

This example plots the diagonal plane-wave form factors
$F_{nn}(q)$ as a function of $q \ell_B$.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from quantumhall_matrixelements import get_form_factors

q = np.linspace(0.0, 5.0, 400)
theta = np.zeros_like(q)
form_factors = get_form_factors(q, theta, nmax=3)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for n in range(3):
    ax.plot(q, form_factors[:, n, n].real, lw=2.0, label=fr"$F_{{{n}{n}}}(q)$")

ax.set_xlabel(r"$q \ell_B$")
ax.set_ylabel(r"$\mathrm{Re}\,F_{nn}(q)$")
ax.set_title("Diagonal Landau-level form factors")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
