# quantumhall-matrixelements: Quantum Hall Landau-Level Matrix Elements

[![Docs](https://img.shields.io/badge/docs-published-0A66C2)](http://tobiaswolf.net/quantumhall_matrixelements/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17807688.svg)](https://doi.org/10.5281/zenodo.17807688)

Landau-level plane-wave form factors, exchange kernels, and symmetric-gauge helper
objects for quantum Hall systems in a small, reusable package (useful for
Hartree-Fock, impurity, and pseudopotential calculations). It provides:

- Analytic Landau-level plane-wave form factors $F_{n',n}^\sigma(\mathbf{q})$.
- Exchange kernels $X_{n_1 m_1 n_2 m_2}^\sigma(\mathbf{G})$.
- Symmetric-gauge guiding-center and factorized density form factors.
- Central one-body matrix elements, plane Haldane pseudopotentials, and LLL disk two-body reconstruction helpers.
- Symmetry diagnostics for verifying kernel implementations.

### Plane-wave Landau-level form factors

For $\sigma = \mathrm{sgn}(qB_z)$, where $q$ is the charge of the carrier and
$B_z$ is the magnetic field direction, the plane-wave matrix element
$F^\sigma_{n',n}(\mathbf{q}) = \langle n' | e^{i \mathbf{q} \cdot \mathbf{R}_\sigma} | n \rangle$
can be written as

$$
F_{n',n}^\sigma(\mathbf{q}) =
i^{|n-n'|}
e^{i\sigma(n'-n)\theta_{\mathbf{q}}}
\sqrt{\frac{n_{<}!}{n_{>}!}}
\left( \frac{|\mathbf{q}|\ell_{B}}{\sqrt{2}} \right)^{|n-n'|}
L_{n_<}^{|n-n'|}\left( \frac{|\mathbf{q}|^2 \ell_{B}^2}{2} \right)
e^{-|\mathbf{q}|^2 \ell_{B}^2/4}
$$

where $n_< = \min(n, n')$, $n_> = \max(n, n')$, $L_n^\alpha$ are the generalized
Laguerre polynomials, and $\ell_B$ is the magnetic length.

### Exchange kernels

$$
X_{n_1 m_1 n_2 m_2}^\sigma(\mathbf{G}) = \int \frac{d^2 q}{(2\pi)^2} V(q)\, F_{m_1, n_1}^\sigma(\mathbf{q})\, F_{n_2, m_2}^\sigma(-\mathbf{q})\, e^{i\sigma (\mathbf{q} \times \mathbf{G})_z \ell_B^2}
$$

where $V(q)$ is the interaction potential. For the Coulomb interaction,
$V(q) = 2\pi e^2 / (\epsilon q)$. Units, $\kappa$ scaling, and the
magnetic-field-sign convention are documented under Conventions.

## Installation

From PyPI:

```bash
pip install quantumhall-matrixelements
```

From a local checkout (development install):

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

Gs_dimless = np.array([0.0, 1.0, 1.0])
thetas = np.array([0.0, 0.0, np.pi])
nmax = 2

F = get_form_factors(Gs_dimless, thetas, nmax)          # shape (nG, nmax, nmax)
X = get_exchange_kernels(Gs_dimless, thetas, nmax)      # built-in Coulomb
```

A custom interaction potential can be supplied as a callable of `|q|` in
`1/ℓ_B` units:

```python
def V_screened(q, kappa=1.0, q_TF=0.5):
    return kappa * 2.0 * np.pi / (q + q_TF)

X_screened = get_exchange_kernels(
    Gs_dimless,
    thetas,
    nmax,
    potential=V_screened,
)
```

For more detailed examples, see the example scripts under `examples/` and the
tests under `tests/`. The documentation covers:

- Plane-wave workflows — form factors, exchange kernels, Fock-matrix construction, memory guards, and backend choice.
- Symmetric-gauge workflows — factorized density form factors, Haldane pseudopotentials, central one-body matrix elements, and LLL disk two-body reconstruction.

## Citation

If you use the package `quantumhall-matrixelements` in academic work, you must cite:

> Sparsh Mishra and Tobias Wolf, *quantumhall-matrixelements: Quantum Hall Landau-Level Matrix Elements*, version 0.1.0, 2025.
> DOI: https://doi.org/10.5281/zenodo.17807688

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17807688.svg)](https://doi.org/10.5281/zenodo.17807688)

A machine-readable `CITATION.cff` file is included in the repository and can be used with tools that support it (for example, GitHub's "Cite this repository" button).

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

## Authors and license

- Authors: Dr. Tobias Wolf, Sparsh Mishra
- License: MIT (see `LICENSE`).
