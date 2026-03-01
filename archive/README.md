# Archive

This directory contains historical snapshots of the project. They are **not installable packages** and should not be used directly.

## Contents

- `TriXO/` — Snapshot of the project at v0.7.3. Contains the `guardian/` module (Mesa 12 / HALO) and Mesa 11-13 documentation that was later pruned from the root package.
- `TriXOR/` — Snapshot of the project at v0.7.3. Structurally identical to TriXO.

## Why They're Here

Both directories declare `name = "trix"` in their `pyproject.toml`. Running `pip install -e .` from inside either would shadow the real root package. They were moved here to prevent accidental namespace collisions.

## If You Need Something From Here

- Mesa 11/12/13 docs have been copied into `docs/archive/` in the root project.
- The `guardian/` module has been restored to `src/trix/guardian/` in the root package.
- For anything else, check git history.
