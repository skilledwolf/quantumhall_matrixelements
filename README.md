# quantumhall-matrixelements: Quantum Hall Landau-Level Matrix Elements

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17807688.svg)](https://doi.org/10.5281/zenodo.17807688)

Landau-level plane-wave form factors and exchange kernels for quantum Hall systems in a small, reusable package (useful for Hartree-Fock and related calculations). It provides:

- Analytic Landau-level plane-wave form factors $F_{n',n}^\sigma(\mathbf{q})$.
- Exchange kernels $X_{n_1 m_1 n_2 m_2}^\sigma(\mathbf{G})$. 
- Symmetry diagnostics for verifying kernel implementations. 

### Plane-Wave Landau-level Form Factors 
For $\sigma = \mathrm{sgn}(qB_z)$, where $q$ is the charge of the carrier and $B_z$ is the magnetic field direction,
The plane-wave matrix element $F^\sigma_{n',n}(\mathbf{q}) = \langle n' | e^{i \mathbf{q} \cdot \mathbf{R}_\sigma} | n \rangle$ can be written as

$$
F_{n',n}^\sigma(\mathbf{q}) =
i^{|n-n'|}
e^{i\sigma(n'-n)\theta_{\mathbf{q}}}
\sqrt{\frac{n_{<}!}{n_{>}!}}
\left( \frac{|\mathbf{q}|\ell_{B}}{\sqrt{2}} \right)^{|n-n'|}
L_{n_<}^{|n-n'|}\left( \frac{|\mathbf{q}|^2 \ell_{B}^2}{2} \right)
e^{-|\mathbf{q}|^2 \ell_{B}^2/4}
$$

where $n_< = \min(n, n')$, $n_> = \max(n, n')$, and $L_n^\alpha$ are the generalized Laguerre polynomials, and $\ell_B$ is the magnetic length.

### Exchange Kernels


$$ X_{n_1 m_1 n_2 m_2}^\sigma(\mathbf{G}) = \int \frac{d^2 q}{(2\pi)^2} V(q) F_{m_1, n_1}^\sigma(\mathbf{q}) F_{n_2, m_2}^\sigma(-\mathbf{q}) e^{i\sigma (\mathbf{q} \times \mathbf{G})_z \ell_B^2} $$

where $V(q)$ is the interaction potential. For the Coulomb interaction, $V(q) = \frac{2\pi e^2}{\epsilon q}$.

### Units and Interaction Strength

The package performs calculations in dimensionless units where lengths are scaled by $\ell_B$. The interaction strength is parameterized by a dimensionless prefactor $\kappa$.

- **Coulomb interaction**: The code assumes a potential of the form $V(q) = \kappa \frac{2\pi e^2}{q \ell_B}$ (in effective dimensionless form).
  - If you set `kappa = 1.0`, the resulting exchange kernels are in units of the Coulomb energy scale $E_C = e^2 / (\epsilon \ell_B)$.
  - To express results in units of the cyclotron energy $\hbar \omega_c$, set $\kappa = E_C / (\hbar \omega_c) = (e^2/\epsilon \ell_B) / (\hbar \omega_c)$.
- **Custom potential**: Provide a callable `potential(q)` that returns values in your desired energy units. The integration measure $d^2q/(2\pi)^2$ introduces a factor of $1/\ell_B^2$, so ensure your potential scaling is consistent.

## Installation

From PyPI (once published):

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

# Simple G set: G0=(0,0), G+=(1,0), G-=(-1,0)
Gs_dimless = np.array([0.0, 1.0, 1.0])
thetas = np.array([0.0, 0.0, np.pi])
nmax = 2

F = get_form_factors(Gs_dimless, thetas, nmax)          # shape (nG, nmax, nmax)
X = get_exchange_kernels(Gs_dimless, thetas, nmax)      # default 'laguerre' backend

print("F shape:", F.shape)
print("X shape:", X.shape)
```

### Avoiding huge allocations

The exchange kernel scales as ``nmax^4`` per ``G``. Low-level backends return a
compressed representation ``(values, select_list)`` by default. The public
``get_exchange_kernels`` API always materializes the full 5D tensor, but includes
a safety guard that prevents accidental large allocations:

- By default, materialization is refused if the estimated tensor size exceeds
  ``materialize_limit_bytes`` (default 512 MiB).
- To opt out, pass ``materialize_limit_bytes=None``.

To avoid full ``nmax^4`` scaling, use ``get_exchange_kernels_compressed`` and
provide an explicit ``select=...`` to compute only the entries you need.

To use a user-provided interaction, pass a callable directly as `potential`:

```python
def V_coulomb(q, kappa=1.0):
    # q is in 1/ℓ_B units; this returns V(q) in Coulomb units
    return kappa * 2.0 * np.pi / q

X_coulomb = get_exchange_kernels(
    Gs_dimless,
    thetas,
    nmax,
    potential=lambda q: V_coulomb(q, kappa=1.0),
)
```

For more detailed examples, see the example scripts under `examples/` and the tests under `tests/`.

### Fock-matrix construction

For iterative Hartree-Fock workflows, pre-compute exchange kernels once and apply
them to a density matrix `rho` on every iteration without materializing the
full `nmax^4` tensor:

```python
from quantumhall_matrixelements import get_fockmatrix_constructor

fock = get_fockmatrix_constructor(Gs_dimless, thetas, nmax)
Sigma = fock(rho)          # Sigma(G) = -X(G) . rho(G),  shape (nG, nmax, nmax)
```

An alternate HF convention used by `quantumhall_hf` is available via
`get_fockmatrix_constructor_hf`.

## Magnetic-field sign

The public APIs expose a `sign_magneticfield` keyword that represents
$\sigma = \mathrm{sgn}(q B_z)$, the sign of the charge–field product.
The default `sign_magneticfield=-1` matches the package's internal convention
(electrons in a positive $B_z$). Passing `sign_magneticfield=+1` returns the
appropriate complex-conjugated form factors or exchange kernels for the
opposite field direction without requiring any manual phase adjustments:

```python
F_plusB = get_form_factors(Gs_dimless, thetas, nmax, sign_magneticfield=+1)
X_plusB = get_exchange_kernels(Gs_dimless, thetas, nmax, method="hankel", sign_magneticfield=+1)
```

## Citation

If you use the package `quantumhall-matrixelements` in academic work, you must cite:

> Sparsh Mishra and Tobias Wolf, *quantumhall-matrixelements: Quantum Hall Landau-Level Matrix Elements*, version 0.1.0, 2025.  
> DOI: https://doi.org/10.5281/zenodo.17807688

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17807688.svg)](https://doi.org/10.5281/zenodo.17807688)

A machine-readable `CITATION.cff` file is included in the repository and can be used with tools that support it (for example, GitHub’s “Cite this repository” button).

## Backends and Reliability

The package provides three backends for computing exchange kernels:

1. **`laguerre` (Default)**
   - **Method**: Gauss-Legendre quadrature on the finite interval $[0, q_\mathrm{max}]$ with Numba-JIT form-factor tables computed via the Laguerre three-term recurrence. For large $|G|$, an optional Ogata-in-$q$-space path provides exponential convergence with $\sim\!200$ nodes.
   - **Pros**: Numerically stable for arbitrarily large $n_\mathrm{max}$ (no intermediate overflow), adaptive node count, and optional Ogata mode for large $|G|$. Also provides a fast Fock-contraction path $\Sigma(G) = -X(G)\cdot\rho(G)$ without materializing the full kernel tensor.
   - **Recommended for**: General usage, large $n_\mathrm{max}$ ($\gtrsim 50$), large $|G|$ ($\gtrsim 30$), and iterative Hartree–Fock workflows.

2. **`hankel`**
   - **Method**: Discrete Hankel transform.
   - **Pros**: High precision and stability, no `numba` dependency.
   - **Cons**: Significantly slower than quadrature methods.
   - **Recommended for**: Reference calculations and cross-checking.

3. **`ogata`**
   - **Method**: Ogata quadrature for Hankel-type integrals with an automatic small-|G| fallback.
   - **Pros**: Typically much faster than the discrete Hankel backend while retaining good accuracy at moderate/large |G|.
   - **Cons**: May require tuning `ogata_h` / `kmin_ogata` for edge cases.
   - **Recommended for**: Faster cross-checks against `hankel`, and workloads dominated by larger |G|.

## Notes
The following wavefunction is used to find all matrix elements:

$$
\Psi_{nX}^\sigma(x,y)
= \frac{e^{i\sigma X y \ell_B^{-2}}}{\sqrt{L_y}}i^n\,
\phi_{n}(x -X),
\qquad
X = \sigma k_y \ell_B^{2}.
$$

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
- Copyright © 2025 Tobias Wolf
- License: MIT (see `LICENSE`).
