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

## Memory guards

The exchange kernel scales as `O(nmax^4)` per `G`. Low-level backends return a
compressed representation `(values, select_list)` by default. The public
`get_exchange_kernels` API always materializes the full 5D tensor, but includes
a safety guard that prevents accidental large allocations:

- By default, materialization is refused if the estimated tensor size exceeds
  `materialize_limit_bytes` (default 512 MiB).
- To opt out, pass `materialize_limit_bytes=None`.

To avoid full `nmax^4` scaling, use `get_exchange_kernels_compressed` and
provide an explicit `select=...` to compute only the entries you need. Calling
`get_exchange_kernels_compressed(select=None)` still builds the canonical
symmetry-reduced list, so it avoids 5D materialization but not the underlying
`O(nmax^4)` output scaling.

The public compressed API also guards the numeric `(nG, n_select)` values
array via `compressed_limit_bytes` (default 512 MiB). If you omit `select`,
`canonical_select_max_entries` separately guards the number of canonical
representatives before any numeric backend work begins.

For the `'laguerre'` backend, dense Gauss-Legendre work tables are guarded by
`workspace_limit_bytes` (default 512 MiB). Pass `workspace_limit_bytes=None`
to disable that backend-level guard.

## Backend choice

The package provides three backends for computing exchange kernels.

### `laguerre` (default)

- **Method**: Gauss-Legendre quadrature on the finite interval $[0, q_\mathrm{max}]$ with Numba-JIT form-factor tables computed via the Laguerre three-term recurrence. For large $|G|$, an optional Ogata-in-$q$-space path provides exponential convergence with $\sim\!200$ nodes.
- **Pros**: Numerically stable for arbitrarily large $n_\mathrm{max}$ (no intermediate overflow), adaptive node count, and optional Ogata mode for large $|G|$. Also provides a fast Fock-contraction path $\Sigma(G) = -X(G)\cdot\rho(G)$ without materializing the full kernel tensor.
- **Recommended for**: General usage, large $n_\mathrm{max}$ ($\gtrsim 50$), large $|G|$ ($\gtrsim 30$), and iterative Hartree-Fock workflows.

### `hankel`

- **Method**: Discrete Hankel transform.
- **Pros**: High precision and stability, no `numba` dependency.
- **Cons**: Significantly slower than quadrature methods.
- **Recommended for**: Reference calculations and cross-checking.

### `ogata`

- **Method**: Ogata quadrature for Hankel-type integrals with an automatic small-$|G|$ fallback.
- **Pros**: Typically much faster than the discrete Hankel backend while retaining good accuracy at moderate/large $|G|$.
- **Cons**: May require tuning `ogata_h` / `kmin_ogata` for edge cases.
- **Recommended for**: Faster cross-checks against `hankel`, and workloads dominated by larger $|G|$.

For many practical workloads, the important first choice is not the backend. It is whether you can stay in the compressed representation.
