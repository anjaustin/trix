# Lincoln Manifold (2026-03-01) - Thoughts On This Repo

Goal: apply the Lincoln Manifold Method to my current read of `trix` as a software artifact (not as a manifesto).

## Phase 1: RAW (First Chop)

My gut reaction is that this repo has two distinct personalities that keep stepping on each other.

Personality A is a real engineering library: there are well-defined modules, meaningful tests, and a clear attempt to make “routing” inspectable, debuggable, and deployable. The moment I ran the suite and got it green (after fixing platform assumptions), my confidence went up a lot: the code is not purely narrative; it has invariants.

Personality B is a research epic: sweeping claims, big metaphors, big numbers, and a lot of “we proved X” language. That can be inspiring, but it also creates a trust gap because the reader can’t easily map claim -> artifact -> reproduction.

The most interesting idea is not “conditional compute” itself (that’s well-trodden); it’s the decision to treat routing as an operable, versionable object: profile it, edit it, compile it, and ship it. That’s a step toward making sparsity systems maintainable.

What worries me: version hygiene was inconsistent (package metadata vs runtime `__version__` vs changelog), and portability assumptions were brittle (native lib naming, optional deps). Those are “adult supervision” problems: easy to fix, but they also predict where future bugs hide.

What feels underrated: the compiler subpackage. Even if you disagree with the framing, the pipeline structure is coherent and the idea of exact/verified atoms is a concrete direction.

If I had to bet, the highest-leverage future work is narrowing the contract of what’s a supported library surface vs what’s an experiment, then making the reproducibility story match the strength of the prose.

## Phase 2: NODES (Identify The Grain)

Node 1: Two modes of communication (library vs manifesto)
- Why it matters: people trust code they can reproduce; they distrust vibes.

Node 2: Routing is treated as an artifact (inspect/edit/compile)
- Why it matters: this is the differentiator that could survive contact with production.

Node 3: Portability is the hidden constraint
- Why it matters: native kernel + optional deps can turn “works here” into CI/UX pain.

Node 4: Versioning was inconsistent across metadata/runtime/docs
- Why it matters: makes everything downstream (bugs, wheels, citations, API promises) less credible.

Node 5: Test suite is broad and mostly library-oriented
- Why it matters: suggests there is a real intended surface area; also indicates where refactors are safe.

Node 6: Number theory / big-claim experiments are coupled into tests
- Tension: “keep experiments in-tree” vs “core tests should be dependency-light.”

Node 7: The compiler subsystem is an independent pillar
- Why it matters: it can become a clean product story even if the rest evolves.

Node 8: XOR compression is a real systems idea, but needs crisp boundaries
- Why it matters: a good optimization story dies if it’s not framed with constraints + proofs.

## Phase 3: REFLECT (Sharpen The Axe)

The underlying structure is “operable sparsity.” The repo is strongest when it treats sparsity/routing like an engineering system with observability and lifecycle (profile -> edit -> compile -> deploy -> monitor). That’s the grain.

The manifesto voice is not inherently bad, but it needs containment. The mistake is mixing “core library guarantees” with “research narrative claims” in the same channel. The reader can’t tell which statements are backed by tests, which are backed by a single experiment script, and which are aspirational.

The portability failures were a gift: they revealed the exact boundary where the repo’s self-image (“lightweight portable”) diverged from its reality (“native lib + optional heavy deps”). Fixing them improved trust because now the repo fails gracefully: either it uses the native path, or it falls back / skips optional test sections.

Versioning hygiene is not cosmetics. It’s a proxy for whether the project can be depended on. A project can be experimental and still be disciplined about versions.

If I had to resolve the major tension: keep the epic narrative, but move it into explicitly labeled research artifacts, while tightening the library to a small set of guarantees that are always true on supported platforms.

## Phase 4: SYNTHESIZE (The Clean Cut)

Working thesis:
- TriX is most compelling as an “operable routing + conditional compute toolkit,” not as a sweeping proof machine.

Concrete decisions (if we keep working on this repo):
1) Define three tiers of code and documentation:
   - `core`: importable, dependency-light, tested in CI
   - `native`: optional acceleration with clear build instructions and fallbacks
   - `experiments`: research narratives and heavy deps, explicitly optional

2) Make versioning single-source-of-truth:
   - `pyproject.toml` version == `trix.__version__`
   - changelog entries reference released tags, not just “Mesa” milestones

3) Make reproducibility match rhetoric:
   - For each big claim in README/CHANGELOG, link to a minimal script + expected output + hardware notes.

4) Preserve the differentiator:
   - Double down on routing observability/surgery/compilation as the “product surface,” and keep it well-tested.

Success criteria for future work:
- A new reader can run `pip install -e ".[dev]" && pytest` and get a clean pass on supported platforms.
- The README points to 2-3 “canonical” demos that are deterministic and match the stated claims.
- Routing lifecycle (profile/edit/compile) remains stable across refactors and has explicit invariants.
