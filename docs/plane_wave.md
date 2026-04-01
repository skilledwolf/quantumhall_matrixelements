# Plane-wave workflows

This page covers the common reciprocal-space tasks: form factors, exchange kernels, and repeated Fock applications.

## Form factors

Use `get_form_factors(...)` when you need

$$
F_{n',n}^\sigma(\mathbf{q}) = \langle n' | e^{i \mathbf{q}\cdot \mathbf{R}_\sigma} | n \rangle.
$$

```python
import numpy as np
from quantumhall_matrixelements import get_form_factors

q = np.linspace(0.0, 5.0, 400)
theta = np.zeros_like(q)

F = get_form_factors(q, theta, nmax=3)
F00 = F[:, 0, 0]
F11 = F[:, 1, 1]
F22 = F[:, 2, 2]
```

See the worked example {doc}`generated/gallery/plot_diagonal_form_factors`.

## Exchange kernels

For small problems, the dense API is straightforward:

```python
import numpy as np
from quantumhall_matrixelements import get_exchange_kernels

G = np.linspace(0.2, 4.0, 80)
theta = np.zeros_like(G)
X = get_exchange_kernels(G, theta, nmax=3)
```

For anything larger, ask only for the entries you need:

```python
import numpy as np
from quantumhall_matrixelements import get_exchange_kernels_compressed

G = np.linspace(0.2, 4.0, 80)
theta = np.zeros_like(G)
select = [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)]

values, used_select = get_exchange_kernels_compressed(
    G,
    theta,
    nmax=3,
    select=select,
)
```

See the worked example {doc}`generated/gallery/plot_selected_exchange_kernels`.

## Repeated Fock applications

If you will apply the exchange operator to many density matrices, precompute the reusable contraction:

```python
import numpy as np
from quantumhall_matrixelements import get_fockmatrix_constructor

G = np.array([0.0, 1.0, 2.0])
theta = np.zeros_like(G)
fock = get_fockmatrix_constructor(G, theta, nmax=4)

rho = np.zeros((len(G), 4, 4), dtype=np.complex128)
rho[:, 0, 0] = 1.0

Sigma = fock(rho)
```

This is the right entry point for iterative Hartree-Fock workflows.

## Backend choice

- `method="laguerre"` is the default and the right starting point for most work.
- `method="hankel"` is slower but useful as a reference calculation.
- `method="ogata"` is useful for faster cross-checks and larger `|G|`.

For many practical workloads, the important first choice is not the backend. It is whether you can stay in the compressed representation.
