# AGENTS.md

## Repo Context

- This repository implements quantum Hall Landau-level form factors, exchange kernels, and Fock-operator helpers.
- Scientific correctness, convention consistency, and numerical stability matter as much as API design and raw speed.
- Main package code lives in `src/quantumhall_matrixelements/`.
- Regression tests live in `tests/`.
- Validation and benchmark scripts live in `validation/`.

## Working Rules

- Start each work session by reading `Roadmap.local.md`.
- `Roadmap.local.md` is a local planning file. Keep it untracked, do not stage it, and do not include it in commits.
- Track work in `Roadmap.local.md` using only two sections:
  - `## Active` for anything in progress or newly discovered.
  - `## Complete` for finished work.
- Do not create a `Pending` section. New work items should be added to `## Active` as soon as they are discovered.
- Move tasks from `## Active` to `## Complete` as soon as the work is actually done.
- When a meaningful amount of work is complete, create a commit before moving on.
- A "meaningful amount of work" means one coherent fix, one well-bounded refactor, or one validated roadmap item.
- Do not bundle unrelated fixes into the same commit.
- Use imperative commit messages with a clear scope, for example:
  - `Preserve explicit select in Ogata fallback`
  - `Align magnetic field defaults in fast Fock path`
- Before committing numerical or API changes, run the most relevant tests or validation commands for the affected area.
- If a change touches scientific conventions, document the convention clearly in code and tests.
- Treat runtime warnings in validation paths as scientific reliability issues, not cosmetic issues.
- For performance work, capture a small before/after measurement whenever practical.
- Never commit generated build artifacts or local planning files.

## Current Review-Driven Focus

- Preserve explicit `select` behavior in `ogata_auto` fallback paths.
- Align magnetic-field sign defaults across public and low-level APIs.
- Eliminate overflow and invalid-value warnings in large-`nmax` Ogata and Legendre paths.
- Resolve the `get_form_factors(..., lB=...)` units contract.
- Close the performance gap between the generic Laguerre Fock constructor and the dedicated fast apply path.
- Add guardrails for compressed calls that still allocate or scale like `O(nmax^4)`.
- Stabilize the `hankel` backend dependency contract.
- Fix the documented `mypy .` workflow for normal development checkouts.
