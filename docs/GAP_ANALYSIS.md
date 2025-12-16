# Gap Analysis: What Needs Closing

**A systematic review of assumptions, missing pieces, and potential failure modes.**

---

## 1. Verification Gaps

### 1.1 FP4 Atoms

**What we claim:** 10 atoms at 100% accuracy.

**What we verified:** Exhaustive truth table checking for 2-3 input atoms.

**Gap:** No verification that atoms compose correctly when wired together.

**Risk:** Individual atoms correct, but composition introduces errors.

**Closing action:** Add composition verification tests (atom A feeds atom B, verify all input combinations).

---

### 1.2 Adder Circuits

**What we claim:** 8-bit adder is 100% exact.

**What we verified:** Random sampling of inputs.

**Gap:** Not exhaustive for 8-bit (2^16 = 65,536 combinations). We tested ~1000.

**Risk:** Edge cases missed (overflow, carry propagation chains).

**Closing action:** Exhaustive test for 8-bit adder (feasible), or prove correctness by construction.

---

### 1.3 Transform Compilation

**What we claim:** WHT and DFT are exact.

**What we verified:** 
- WHT: 100 random tests per N
- DFT: 20 random tests per N

**Gap:** Not exhaustive. Float comparison uses tolerance.

**Risk:** Specific inputs could fail.

**Closing action:** 
- WHT: Test all integer inputs for small N (N=8: 8^8 = 16M, too large; sample systematically)
- DFT: Define acceptable tolerance explicitly, test boundary cases

---

### 1.4 Twiddle Opcodes

**What we claim:** No runtime trig.

**What we verified:** Source code inspection via `verify_no_runtime_trig()`.

**Gap:** Only checks `execute()` method, not helper functions or initialization.

**Risk:** Trig could sneak in elsewhere.

**Closing action:** Audit entire call graph from `execute()`.

---

## 2. Documentation Gaps

### 2.1 Installation & Setup

**Gap:** No clear installation instructions. Assumes reader has environment set up.

**Closing action:** Create `INSTALL.md` with step-by-step setup.

---

### 2.2 API Reference

**Gap:** Functions documented in docstrings but no unified API reference.

**Closing action:** Generate or write API documentation for key modules.

---

### 2.3 Theory Background

**Gap:** Documentation assumes familiarity with FFT, WHT, threshold circuits.

**Closing action:** Create `BACKGROUND.md` with prerequisite concepts.

---

### 2.4 Glossary

**Gap:** Terms like "atom," "tile," "routing," "microcode" used without definition.

**Closing action:** Create `GLOSSARY.md` with precise definitions.

---

## 3. Code Gaps

### 3.1 Error Handling

**Gap:** Most functions assume valid input. No graceful error messages.

**Risk:** Confusing failures for users.

**Closing action:** Add input validation with clear error messages.

---

### 3.2 Edge Cases

**Gap:** What happens with N=1? N not power of 2? Empty input?

**Risk:** Silent failures or crashes.

**Closing action:** Define behavior for edge cases, add tests.

---

### 3.3 Numerical Stability

**Gap:** Float comparisons use ad-hoc tolerances.

**Risk:** False positives/negatives in verification.

**Closing action:** Define tolerance policy, use consistent epsilon throughout.

---

### 3.4 Performance

**Gap:** No benchmarks. Unknown scaling behavior.

**Risk:** Claims of efficiency without evidence.

**Closing action:** Add benchmarks for N=8 through N=1024.

---

## 4. Test Gaps

### 4.1 No CI Integration

**Gap:** Tests exist but aren't run automatically.

**Closing action:** Add pytest configuration, CI workflow.

---

### 4.2 No Regression Tests

**Gap:** No way to detect if changes break existing functionality.

**Closing action:** Create regression test suite with known-good outputs.

---

### 4.3 No Property-Based Tests

**Gap:** Only example-based tests. Don't explore input space systematically.

**Closing action:** Add hypothesis-based property tests.

---

### 4.4 No Adversarial Tests

**Gap:** Tests use random or simple inputs. No adversarial cases.

**Closing action:** Design inputs that stress edge cases (all zeros, all ones, alternating, max values).

---

## 5. Accessibility Gaps

### 5.1 No Tutorial

**Gap:** No guided introduction for beginners.

**Closing action:** Create `TUTORIAL.md` with progressive examples.

---

### 5.2 No Visual Explanations

**Gap:** All documentation is text. Complex structures hard to visualize.

**Closing action:** Add ASCII diagrams, consider generating SVGs.

---

### 5.3 No "Why" Documentation

**Gap:** Documentation explains what and how, not why.

**Closing action:** Add motivation sections explaining design decisions.

---

### 5.4 Assumed Knowledge

**Gap:** Assumes knowledge of PyTorch, numpy, signal processing.

**Closing action:** Create prerequisite checklist, link to learning resources.

---

## 6. Reproducibility Gaps

### 6.1 No Version Pinning

**Gap:** `requirements.txt` may not pin exact versions.

**Closing action:** Pin all dependencies with exact versions.

---

### 6.2 No Seed Control

**Gap:** Random tests may not be reproducible.

**Closing action:** Add seed parameter to all random tests.

---

### 6.3 No Environment Specification

**Gap:** Python version, OS requirements not documented.

**Closing action:** Document tested environments.

---

## Priority Matrix

| Gap | Impact | Effort | Priority |
|-----|--------|--------|----------|
| Exhaustive 8-bit adder test | High | Low | **P0** |
| Composition verification | High | Medium | **P0** |
| Tutorial for beginners | High | Medium | **P0** |
| Glossary | Medium | Low | **P1** |
| Edge case handling | Medium | Medium | **P1** |
| CI integration | Medium | Low | **P1** |
| Regression tests | Medium | Medium | **P1** |
| API reference | Medium | High | **P2** |
| Performance benchmarks | Low | Medium | **P2** |
| Visual explanations | Low | High | **P3** |

---

## Closing Plan

### Phase 1: Critical (Today)
1. Exhaustive 8-bit adder test
2. Composition verification tests
3. Beginner tutorial
4. Glossary

### Phase 2: Important (This Week)
5. Edge case tests
6. CI configuration
7. Regression test suite
8. Background document

### Phase 3: Nice-to-Have (Later)
9. API reference
10. Benchmarks
11. Visual diagrams

---

*Gaps identified. Now close them.*
