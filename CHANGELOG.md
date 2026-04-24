# Changelog

All notable changes to `quantumhall-matrixelements` will be documented in
this file. Version numbers follow [SemVer](https://semver.org/) as published
via `setuptools_scm` from git tags (`vMAJOR.MINOR.PATCH`).

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
