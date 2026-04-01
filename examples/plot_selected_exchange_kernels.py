r"""
Selected diagonal exchange kernels
==================================

This example computes a few diagonal exchange-kernel channels with the
compressed API and plots their real parts as a function of $|G| \ell_B$.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from quantumhall_matrixelements import get_exchange_kernels_compressed

g = np.linspace(0.2, 4.0, 80)
theta = np.zeros_like(g)
select = [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)]
values, used_select = get_exchange_kernels_compressed(g, theta, nmax=3, select=select)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for idx, (n1, _, n2, _) in enumerate(used_select):
    ax.plot(g, values[:, idx].real, lw=2.0, label=fr"$X_{{{n1}{n1}{n2}{n2}}}(G)$")

ax.set_xlabel(r"$|G| \ell_B$")
ax.set_ylabel(r"$\mathrm{Re}\,X(G)$")
ax.set_title("Selected diagonal exchange-kernel channels")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
