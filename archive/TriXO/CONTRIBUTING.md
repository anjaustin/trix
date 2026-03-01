# Contributing to TriX

Thank you for your interest in contributing to TriX! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Submitting Changes](#submitting-changes)
7. [Style Guide](#style-guide)
8. [Areas of Interest](#areas-of-interest)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and constructive
- Focus on the technical merits
- Welcome newcomers
- Acknowledge different perspectives

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- Git

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/trix.git
cd trix
git remote add upstream https://github.com/trix-org/trix.git
```

---

## Development Setup

### Install in Development Mode

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Verify Setup

```bash
# Run tests
pytest tests/ -v

# Check imports
python -c "from trix import HierarchicalTriXFFN; print('OK')"
```

---

## Making Changes

### Branch Naming

Create a descriptive branch name:

```bash
git checkout -b feature/temporal-routing-improvement
git checkout -b fix/signature-gradient-flow
git checkout -b docs/api-reference-update
```

### Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement

Examples:
```
feat(nn): add XOR-based routing layer
fix(kernel): correct 2-bit unpacking for edge case
docs(api): add CompiledDispatch examples
test(temporal): add state persistence tests
```

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Tests

```bash
# Single file
pytest tests/test_hierarchical.py -v

# Single test
pytest tests/test_hierarchical.py::TestHierarchicalFFN::test_forward_shape -v

# With coverage
pytest tests/ -v --cov=src/trix --cov-report=html
```

### Writing Tests

Place tests in `tests/` with the naming convention `test_*.py`:

```python
# tests/test_my_feature.py
import pytest
import torch
from trix.nn import MyNewLayer

class TestMyNewLayer:
    def test_forward_shape(self):
        layer = MyNewLayer(d_model=64)
        x = torch.randn(2, 16, 64)
        out, info, aux = layer(x)
        assert out.shape == x.shape
    
    def test_gradient_flow(self):
        layer = MyNewLayer(d_model=64)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out, _, _ = layer(x)
        out.sum().backward()
        assert x.grad is not None
```

### Test Requirements

All contributions must:

1. Pass all existing tests
2. Include tests for new functionality
3. Maintain or improve coverage

---

## Submitting Changes

### Pull Request Process

1. **Update your branch:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push changes:**
   ```bash
   git push origin feature/your-feature
   ```

3. **Create Pull Request:**
   - Use a descriptive title
   - Reference any related issues
   - Describe what changes you made and why

### PR Checklist

- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Documentation updated if needed
- [ ] Commit messages follow convention
- [ ] Branch is up to date with main

### Review Process

1. Maintainers will review your PR
2. Address any feedback
3. Once approved, maintainers will merge

---

## Style Guide

### Python Style

We follow PEP 8 with these specifics:

```python
# Imports: standard library, third-party, local
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from trix.kernel import TriXLinear

# Type hints for public APIs
def forward(
    self,
    x: torch.Tensor,
    state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Forward pass.
    
    Args:
        x: Input tensor of shape (batch, seq, d_model)
        state: Optional state tensor
        
    Returns:
        output: Transformed tensor
        info: Routing information dictionary
    """
    pass

# Constants in UPPER_CASE
DEFAULT_NUM_TILES = 64
MAX_SIGNATURE_DIM = 4096

# Classes in PascalCase
class HierarchicalTriXFFN(nn.Module):
    pass

# Functions and variables in snake_case
def compute_signature(weights):
    pass
```

### Documentation Style

Use Google-style docstrings:

```python
def route_input(
    input: torch.Tensor,
    signatures: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Route input to best-matching tile.
    
    Computes alignment scores between input and tile signatures,
    then returns the winning tile index.
    
    Args:
        input: Input tensor of shape (batch, seq, d_model)
        signatures: Tile signatures of shape (num_tiles, d_model)
        temperature: Softmax temperature for soft routing
        
    Returns:
        indices: Winning tile indices of shape (batch, seq)
        scores: Alignment scores of shape (batch, seq, num_tiles)
        
    Raises:
        ValueError: If input and signatures have mismatched dimensions
        
    Example:
        >>> signatures = torch.randn(16, 64).sign()
        >>> x = torch.randn(4, 32, 64)
        >>> indices, scores = route_input(x, signatures)
        >>> indices.shape
        torch.Size([4, 32])
    """
```

---

## Areas of Interest

We welcome contributions in these areas:

### High Priority

1. **Hardware Backends**
   - CUDA kernels for 2-bit operations
   - Metal shaders for Apple Silicon
   - TPU support

2. **Benchmark Reproductions**
   - Independent validation of results
   - New benchmark tasks

3. **Documentation**
   - Tutorials
   - Examples
   - API documentation

### Medium Priority

4. **New Architectures**
   - Novel routing strategies
   - Hybrid architectures
   - Integration with other frameworks

5. **Training Improvements**
   - Better auxiliary losses
   - Curriculum strategies
   - Progressive quantization

### Research Directions

6. **Theoretical Analysis**
   - Generalization bounds
   - Capacity analysis
   - Routing dynamics

7. **Applications**
   - Domain-specific adaptations
   - Edge deployment
   - Real-time systems

---

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Email maintainers for sensitive matters

Thank you for contributing to TriX!
