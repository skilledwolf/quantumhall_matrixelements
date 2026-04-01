# Symmetric-gauge workflows

The symmetric-gauge helpers are designed for calculations in the single-particle basis `|n, m>` without forcing large tensors unless they are actually needed.

## Factorized density form factors

The density matrix element factorizes into cyclotron and guiding-center pieces:

$$
\langle n', m' | e^{i \mathbf{q}\cdot \mathbf{r}} | n, m \rangle
= F_{n',n}^\sigma(\mathbf{q}) G_{m',m}^{-\sigma}(\mathbf{q}).
$$

```python
import numpy as np
from quantumhall_matrixelements import get_factorized_density_form_factors

q = np.array([0.5, 1.0, 1.5])
theta = np.zeros_like(q)

F_cyc, G_gc = get_factorized_density_form_factors(
    q,
    theta,
    nmax=3,
    mmax=5,
)
```

This is usually the right object to keep around in code. The helper already
handles the opposite-chirality guiding-center convention internally.

## Haldane pseudopotentials

Use `get_haldane_pseudopotentials(...)` for plane pseudopotentials `V_m^{(n_\mathrm{LL})}`:

```python
from quantumhall_matrixelements import get_haldane_pseudopotentials

V_m = get_haldane_pseudopotentials(10, n_ll=0)
```

See the worked example {doc}`generated/gallery/plot_llll_coulomb_pseudopotentials`.

## Reconstruct disk two-body matrix elements

If your workflow starts from pseudopotentials but needs explicit LLL disk matrix elements:

```python
from quantumhall_matrixelements import (
    get_haldane_pseudopotentials,
    get_twobody_disk_from_pseudopotentials_compressed,
)

V_m = get_haldane_pseudopotentials(12, n_ll=0)
values, select = get_twobody_disk_from_pseudopotentials_compressed(V_m, mmax=6)
```

The returned `select` entries are orbital quadruples `(m1, m2, m3, m4)`.

## Central one-body matrix elements

For origin-centered radial potentials in the symmetric-gauge basis:

```python
from quantumhall_matrixelements import (
    get_central_onebody_matrix_elements_compressed,
    materialize_central_onebody_matrix,
)

values, select = get_central_onebody_matrix_elements_compressed(
    nmax=3,
    mmax=5,
    potential="coulomb",
)

V_dense = materialize_central_onebody_matrix(values, select, nmax=3, mmax=5)
```

The compressed representation is usually the better storage format unless downstream code explicitly requires a dense tensor.
