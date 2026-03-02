# Install

Note: this repo is archived. Active development lives in `../trix-z/`.

This is the canonical setup guide for the root `trix` package.

## Python Package

1) Install PyTorch for your platform.
2) From the repo root:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

## Optional: Native Kernel (CPU)

TriX includes an optional native library used by `trix.kernel`.

```bash
cmake -S src/trix/kernel -B src/trix/kernel/build
cmake --build src/trix/kernel/build -j
```

Notes:
- The Python code has a reference fallback path; tests pass without the native library.

## Optional: Native Routing Tools (C++)

There is a standalone C++ benchmark and test harness under `native/`.

```bash
cmake -S native -B native/build
cmake --build native/build -j
ctest --test-dir native/build
```

## Optional Dependencies

- `gmpy2` is optional; tests that require it are skipped when it is not installed.
