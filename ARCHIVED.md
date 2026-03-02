# Archived Repo Policy

Active development lives in `../trix-z/`.

This repository exists as a reproducible, runnable historical record:

- Keep the tree installable and testable on CPU (`trix doctor`, `python -m pytest`, `trix bench`).
- Keep docs navigable and claims falsifiable (`scripts/repro/` with saved expected outputs).
- Accept only hygiene + reproducibility changes (docs, tests, harnesses, link fixes).

Not in scope for this repo:

- New research features or architectural refactors.
- Performance work that changes semantics without a strict reference + falsification harness.

If you are trying to build new features, open changes in `../trix-z/` instead.
