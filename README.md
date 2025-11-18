# quantumhall_matrixelements

Landau-level plane-wave form factors and exchange kernels for quantum Hall systems.

This library factors out the continuum matrix-element kernels used in Hartree–Fock and
related calculations into a small, reusable package. It provides:

- Analytic Landau-level plane-wave form factors $F_{n',n}(\mathbf{q})$.
- Exchange kernels $X_{n_1 m_1 n_2 m_2}(\mathbf{G})$ computed via:
  - Generalized Gauss–Laguerre quadrature (`gausslag` backend).
  - Hankel-transform based integration (`hankel` backend).
- Symmetry diagnostics for verifying kernel implementations on a given G-grid.

## Mathematical Definitions

### Plane-Wave Form Factors

The form factors are the matrix elements of the plane-wave operator $e^{i \mathbf{q} \cdot \mathbf{R}}$ in the Landau level basis $|n\rangle$:

$$ F_{n',n}(\mathbf{q}) = \langle n' | e^{i \mathbf{q} \cdot \mathbf{R}} | n \rangle $$

Analytically, these are given by:

$$ F_{n',n}(\mathbf{q}) = i^{|n-n'|} e^{i(n-n')\theta_\mathbf{q}} \sqrt{\frac{n_<!}{n_>!}} \left( \frac{|\mathbf{q}|\ell_B}{\sqrt{2}} \right)^{|n-n'|} L_{n_<}^{|n-n'|}\left( \frac{|\mathbf{q}|^2\ell_B^2}{2} \right) e^{-|\mathbf{q}|^2\ell_B^2/4} $$

where $n_< = \min(n, n')$, $n_> = \max(n, n')$, and $L_n^\alpha$ are the generalized Laguerre polynomials, and $\ell_B$ is the magnetic length.

### Exchange Kernels

The exchange kernels $X_{n_1 m_1 n_2 m_2}(\mathbf{G})$ are defined as the Fourier transform of the interaction potential weighted by the form factors:

$$ X_{n_1 m_1 n_2 m_2}(\mathbf{G}) = \int \frac{d^2 q}{(2\pi)^2} V(q) F_{n_1, m_1}(\mathbf{q}) F_{m_2, n_2}(-\mathbf{q}) e^{-i \mathbf{q} \cdot \mathbf{G} \ell_B^2} $$

where $V(q)$ is the interaction potential. For the Coulomb interaction, $V(q) = \frac{2\pi e^2}{\epsilon q}$.

### Units and Interaction Strength

The package performs calculations in dimensionless units where lengths are scaled by $\ell_B$. The interaction strength is parameterized by a dimensionless prefactor $\kappa$.

- **Coulomb Interaction**: The code assumes a potential of the form $V(q) = \kappa \frac{2\pi \ell_B}{q \ell_B}$ (in effective dimensionless form).
    - If you set `kappa = 1.0`, the resulting exchange kernels will be in units of the **Coulomb energy scale** $E_C = e^2 / (\epsilon \ell_B)$.
    - If you want the results in units of the cyclotron energy $\hbar \omega_c$, you should set $\kappa = E_C / (\hbar \omega_c) = (e^2/\epsilon \ell_B) / (\hbar \omega_c)$.

- **General Potential**: For a general $V(q)$, the function `V_of_q` should return values in your desired energy units. The integration measure $d^2q/(2\pi)^2$ introduces a factor of $1/\ell_B^2$, so ensure your potential scaling is consistent.

## Installation

From a local checkout:

```bash
pip install -e .[dev]
```

## Basic usage

```python
import numpy as np
from quantumhall_matrixelements import (
    get_form_factors,
    get_exchange_kernels,
)

# Simple G set: G0=(0,0), G+=(1,0), G-=(-1,0)
Gs_dimless = np.array([0.0, 1.0, 1.0])
thetas = np.array([0.0, 0.0, np.pi])
nmax = 2

F = get_form_factors(Gs_dimless, thetas, nmax)          # shape (nG, nmax, nmax)
X = get_exchange_kernels(Gs_dimless, thetas, nmax)      # default 'gausslag' backend

print("F shape:", F.shape)
print("X shape:", X.shape)
```

For more detailed examples, see the tests under `tests/`.

## Development

- Run tests and coverage:

  ```bash
  pytest
  ```

- Lint and type-check:

  ```bash
  ruff check .
  mypy .
  ```

