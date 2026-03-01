# Plan: Repo Hygiene (2026-03-01)

Audit-driven plan to address six issues found during a full repo review.
Guiding principle: **archive, don't delete.**

---

## 1. Resolve TriXO/TriXOR Namespace Hazard (HIGH)

**Problem**: `TriXO/` and `TriXOR/` are archival snapshots frozen at v0.7.3.
Both declare `name = "trix"` in their `pyproject.toml`. A stray `pip install -e .`
from inside either directory silently overwrites the real v0.12.0 install.

**Fix**:
- Move `TriXO/` -> `archive/TriXO/` and `TriXOR/` -> `archive/TriXOR/`.
- Add `archive/README.md` explaining these are historical snapshots, not installable packages.
- This removes them from the repo's active working tree without deleting history.

---

## 2. Fix Broken Doc References (HIGH)

**Problem**: Three classes of broken links across CHANGELOG, notes, and docs:

| Class | Count | Example |
|-------|-------|---------|
| Archived docs referenced at old paths | ~15 | `docs/TUTORIAL.md` -> actually `docs/archive/TUTORIAL.md` |
| Mesa 11/12/13 docs only in TriXO/TriXOR | ~5 | `MESA13_XOR_SUPERPOSITION.md`, `MESA12_HALO.md` |
| Truly missing | 2 | `docs/EMERGENT_ROUTING.md`, `experiments/mesa11/rigorous/README.md` |

**Fix**:
- Copy Mesa 11/12/13 docs from TriXO/TriXOR into root `docs/archive/`.
- Update `docs/QUICKSTART.md` references to point to `docs/archive/` where applicable.
- For the 2 truly missing files, remove the dangling reference or note `[archived]`.
- Do NOT rewrite CHANGELOG history; it's a historical record. Add a note at the top that
  doc paths referenced in older entries may have moved to `docs/archive/`.

---

## 3. Formalize Tier Boundaries in Tests (HIGH)

**Problem**: `test_rigorous.py` and `test_butterfly_matmul.py` use `sys.path.insert`
to import from `experiments/`. If experiment files are absent (e.g. wheel install),
these tests crash at import time with no graceful skip.

**Fix**:
- Wrap experiment imports in each file with `try/except ImportError` + module-level
  `pytest.skip("requires experiment code under experiments/")`.
- Register an `experiment` marker in `pyproject.toml` so these can be filtered
  with `-m "not experiment"`.

---

## 4. Add Golden-Output Repro Scripts for Top Claims (MEDIUM)

**Problem**: Big claims in the CHANGELOG (129x compression, 0.00 DFT error,
compiled dispatch agreement) have test coverage but no single-command
"reproduce this number and diff against expected output" workflow.

**Fix**: Create `scripts/repro/` with 2-3 small scripts that:
- Run on CPU in < 60 seconds.
- Print key metrics to stdout.
- Compare against a saved `.expected.json`.
- Exit 0 on match, 1 on mismatch.

Cartridges:
1. `repro_xor_compression.py` — compress/decompress signatures, report ratio + losslessness.
2. `repro_compiled_dispatch.py` — train tiny model, compile, verify agreement.
3. `repro_dft_compilation.py` — compile DFT N=8, measure error vs numpy.

---

## 5. Restore Missing `guardian/` Module (MEDIUM)

**Problem**: CHANGELOG documents Mesa 12 (HALO/Guardian) as shipped (v0.11.0),
but the guardian code only exists in TriXO/TriXOR. It was removed from the root
package at some point without a changelog note.

**Fix**:
- Copy `TriXO/src/trix/guardian/` into `src/trix/guardian/`.
- Copy `TriXO/tests/test_guardian.py` into `tests/`.
- Verify the module imports cleanly and tests pass.
- Do NOT add guardian to `src/trix/__init__.py` top-level exports (it self-describes
  as experimental). It stays importable as `from trix.guardian import ...`.

---

## 6. Add Optional Extras for Heavy Experiment Deps (LOW)

**Problem**: `gmpy2` is the only heavy optional dep; it's guarded in tests.
Having an explicit extra makes the intent clear and sets a pattern for future deps.

**Fix**: Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
experiments = ["gmpy2>=2.1.0"]
```

Mention in README: `pip install -e ".[experiments]"` for the full experiment suite.

---

## Execution Order

1 -> 2 -> 3 -> 5 -> 4 -> 6

After all changes: run `python -m pytest` and confirm 0 failures.
