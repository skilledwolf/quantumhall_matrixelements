# quantumhall-matrixelements

`quantumhall-matrixelements` provides a compact set of kernels and helper objects for common quantum Hall workflows:

- plane-wave Landau-level form factors
- exchange kernels in a Landau-level basis
- symmetric-gauge density, one-body, and pseudopotential building blocks

The documentation is written for users who want correct conventions, equations, and working examples without reading backend internals.

## Start from the right page

```{toctree}
:maxdepth: 2

conventions
plane_wave
symmetric_gauge
api
generated/gallery/index
```

## Installation

```bash
pip install quantumhall_matrixelements
```

For a local checkout with tests and docs tooling:

```bash
pip install -e .[dev,docs]
```

## Minimal example

```python
import numpy as np
from quantumhall_matrixelements import get_exchange_kernels, get_form_factors

q = np.array([0.0, 1.0, 2.0])
theta = np.zeros_like(q)
nmax = 3

F = get_form_factors(q, theta, nmax)
X = get_exchange_kernels(q[1:], theta[1:], nmax)

print(F.shape)  # (3, 3, 3)
print(X.shape)  # (2, 3, 3, 3, 3)
```

## What to expect from the package

- Physical conventions are explicit. The `lB` scaling, the charge-field sign, and the tensor index order are part of the public contract.
- Large tensors are not hidden behind friendly names. The exchange kernel scales like `O(nmax^4)` per `G`, so the compressed APIs matter.
- The default path is meant to be reliable for production calculations, not just toy examples.
