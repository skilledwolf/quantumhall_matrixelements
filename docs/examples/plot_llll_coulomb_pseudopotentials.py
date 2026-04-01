r"""
Lowest-Landau-level Coulomb pseudopotentials
============================================

This example plots the plane Coulomb pseudopotentials $V_m^{(0)}$
in the lowest Landau level.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from quantumhall_matrixelements import get_haldane_pseudopotentials

m = np.arange(10)
v_m = get_haldane_pseudopotentials(m.size, n_ll=0)

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(m, v_m, "o-", lw=2.0, ms=5.0)
ax.set_xlabel(r"relative angular momentum $m$")
ax.set_ylabel(r"$V_m^{(0)} / E_C$")
ax.set_title("Lowest-Landau-level Coulomb pseudopotentials")
ax.grid(alpha=0.3)
fig.tight_layout()
