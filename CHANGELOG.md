# Changelog

All notable changes to `quantumhall-matrixelements` will be documented in
this file. Version numbers follow [SemVer](https://semver.org/) as published
via `setuptools_scm` from git tags (`vMAJOR.MINOR.PATCH`).

## 1.3.0 — Fast HF-convention Fock-apply path

### Added

- **`get_fockmatrix_constructor_hf` now has a fast path** for the
  default `method="laguerre"` case. The previous implementation built
  the compressed `(values, select)` representation and applied it via a
  vectorized `np.add.at` loop — fast for very small `nmax` but bottlenecked
  by Python-level binning at `nmax >= 3`.

  The new path materializes the full kernel tensor once at construction
  time and applies via BLAS-backed `np.einsum`. Costs:

  - Build: `O(nG * nmax^4 * Nq)` (single Numba kernel build) plus
    `O(nG * nmax^4)` storage.
  - Apply (per call): `O(nG * nmax^4)` via einsum, with a precomputed
    contraction path so `np.einsum_path` overhead is paid once.

  Measured speedups vs. the old compressed apply (Coulomb, μs/call):

  | nG  | nmax | old apply | new apply | speedup |
  |----:|----:|--------:|--------:|-----:|
  | 137 | 1 |     5    |    8    | ~1×  |
  | 137 | 2 |    41    |   51    | ~1×  |
  | 137 | 3 |   116    |   50    | 2.3× |
  | 137 | 4 |   330    |   96    | 3.4× |
  | 137 | 6 |  1870    |  280    | 6.7× |
  | 137 | 8 |  6700    |  658    | 10×  |

  At `nmax = 1, 2` (the typical `quantumhall_hf` workload) the new path
  is roughly tied with the old one — it neither helps nor hurts. At
  `nmax >= 3` (multi-LL HF, LL-mixing) it's a clear win. If
  materializing the full tensor would exceed the configured memory
  limit, the constructor transparently falls back to the compressed
  apply.

- **`ExchangeFockPrecompute.exchange_fock_hf`**: a Numba-JIT method on
  the precompute object that applies the kernel in the HF convention
  via on-the-fly q-quadrature. This is the parallel of `exchange_fock`
  for the HF index pattern. Note that, unlike `exchange_fock`, the HF
  reduction has no d-collapse factorization and the per-G cost is
  intrinsically `O(nmax^4 * Nq)` — making this method useful as a
  reference / for memory-constrained workloads, but slower than the
  einsum-on-materialized path used by the constructor.

### Internal

- `_maybe_build_fast_laguerre_constructor` in `fock.py` gained a
  `convention` parameter and now routes the HF case through the
  materialize+einsum apply.
- New Numba kernel `_exchange_fock_hf_numba` in `exchange_laguerre.py`
  with full-coverage tests against the existing compressed HF apply for
  `nmax in {1, 2, 3, 4}` and `sign_magneticfield in {-1, +1}`.

## 1.2.0 — Magnetic-field sign convention bug fix

### Fixed — **breaking numerical change**

- **Exchange kernels now match the documented analytic magnetic-field
  sign convention.** Versions `<= 1.1.x` silently inverted the
  `sign_magneticfield` argument: calling
  `get_exchange_kernels(..., sign_magneticfield=X)` produced values
  corresponding to `sigma = -X` in the formulas in
  `docs/conventions.md`. Starting in 1.2.0, `sign_magneticfield = X`
  produces `sigma = X` as advertised, for **all** backends
  (`laguerre`, `ogata`, `hankel`, `legendre`) and for the fast
  Fock-apply path (`get_fockmatrix_constructor` /
  `build_exchange_fock_precompute`).

  Concretely:

  - `_evaluate_exchange_kernels_laguerre_numba` and
    `_ogata_q_evaluate_numba` (Numba kernels): the `d_in`/`d_out`
    labels used to access `phase_in` / `phase_out` were swapped.
    Fixed so that `phase_in` is applied at `d_in = m2 - n2` and
    `phase_out` at `d_out = m1 - n1`, matching the analytic phase
    `i^{a_tot} \cdot (-1)^{d_in} \cdot \exp(i \sigma (d_out - d_in) \theta)`.
  - `exchange_ogata`, `exchange_hankel`, `exchange_legendre`: the
    post-hoc sign-flip branch was gated on the wrong sign; the
    resulting kernels had `sigma` flipped relative to the user's
    `sign_magneticfield` value. Fixed.
  - `_exchange_fock_numba`: after the kernel path was corrected, the
    fast Fock-apply path needed to match the canonical convention
    that the **first** index pair `(n1, m1)` of
    `X_{n1, m1, n2, m2}` is the input (density) pair and the
    **second** pair `(n2, m2)` is the output (Fock) pair, matching
    `_build_compressed_constructor` and the documented relation
    `\Sigma_{n2, m2}(G) = -\sum_{n1, m1} X_{n1, m1, n2, m2}(G)\,
    \rho_{n1, m1}(G)`. Previously the fast path applied `X` as a map
    `(n2, m2) -> (n1, m1)`; it now applies it as `(n1, m1) -> (n2, m2)`.
    For **Hermitian** densities and single-Landau-level calculations,
    the hidden identity
    `X^{\sigma}[a, b, p, q] = X^{-\sigma}[p, q, a, b]` meant
    most scalar observables were unaffected by the compensating bugs;
    multi-Landau-level and complex-density workflows were not.

### Impact and mitigation

- Scalar Fock energies for Hermitian single-Landau-level densities
  (including many textbook HF-at-half-filling calculations)
  are **numerically unchanged** to floating-point precision.
- Anything that reads the literal kernel values, mixes Landau
  levels, or uses non-Hermitian densities (including the
  multi-LL hex Wigner-crystal benchmark in
  `harfok_macdonald` / `quantumhall_hf`) can change at the
  ~1% level.
- If you must reproduce pre-1.2.0 numerics, pin
  `quantumhall-matrixelements < 1.2`.

### Added

- New analytic reference tests in
  `tests/test_coulomb_analytic_general.py` that compare every
  backend against an `mpmath`-based closed-form Coulomb
  integral for both `sign_magneticfield = +1` and `-1`, plus
  cross-backend consistency and form-factor ↔ exchange-kernel
  consistency checks.

## 1.1.0 and earlier

See the git log (`git log v1.1.0`) for earlier history. The magnetic-field
sign convention bug described above was present throughout this range.
