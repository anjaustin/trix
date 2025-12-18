# TriX: Mathematical Foundations

This document presents the theoretical foundations of the TriX architecture.

## Table of Contents

1. [Ternary Weight Spaces](#ternary-weight-spaces)
2. [Signature Theory](#signature-theory)
3. [Emergent Routing Analysis](#emergent-routing-analysis)
4. [Capacity and Expressiveness](#capacity-and-expressiveness)
5. [Training Dynamics](#training-dynamics)
6. [Connections to Related Work](#connections-to-related-work)

---

## Ternary Weight Spaces

### Definition

A ternary weight matrix W ∈ T^(m×n) has entries from the ternary set:

$$\mathcal{T} = \{-1, 0, +1\}$$

### Cardinality

The number of distinct ternary matrices of size m × n is:

$$|\mathcal{T}^{m \times n}| = 3^{mn}$$

For a typical tile with m=512, n=2048:

$$3^{1,048,576} \approx 10^{500,000}$$

This vast space enables diverse specialization without parameter sharing conflicts.

### Linear Transformation

For input x ∈ ℝ^n, the ternary linear transformation is:

$$y = Wx = \sum_{j=1}^{n} w_{\cdot j} x_j$$

where each column w_{\cdot j} contributes:
- +x_j if w_{\cdot j} = +1 (include feature)
- -x_j if w_{\cdot j} = -1 (include negated feature)
- 0 if w_{\cdot j} = 0 (ignore feature)

### Information Content

Each ternary weight encodes log₂(3) ≈ 1.585 bits of information. With 2-bit storage:

$$\text{Efficiency} = \frac{\log_2(3)}{2} \approx 79.2\%$$

The remaining 20.8% is overhead for the unused encoding (binary 11).

---

## Signature Theory

### Definition

The **signature** of a weight matrix W ∈ T^(m×n) is:

$$\sigma(W) = \text{sign}\left(\sum_{i=1}^{m} W_{i\cdot}\right) \in \mathcal{T}^n$$

This aggregates column-wise preferences into a single ternary vector.

### Interpretation

For each feature dimension j:

$$\sigma(W)_j = \begin{cases}
+1 & \text{if } \sum_i W_{ij} > 0 \text{ (tile prefers high values)} \\
-1 & \text{if } \sum_i W_{ij} < 0 \text{ (tile prefers low values)} \\
0 & \text{if } \sum_i W_{ij} = 0 \text{ (tile is indifferent)}
\end{cases}$$

### Signature Space

The signature space Σ = T^n has:
- 3^n possible signatures
- Each signature defines a region of input space

For d_model = 512:
$$|\Sigma| = 3^{512} \approx 10^{244}$$

### Signature Diversity

For k tiles with signatures σ₁, ..., σₖ, diversity is measured by:

$$D = \frac{1}{k(k-1)} \sum_{i \neq j} d_H(\sigma_i, \sigma_j)$$

where d_H is Hamming distance. Maximum diversity occurs when signatures are maximally separated.

---

## Emergent Routing Analysis

### Routing Function

Given input x ∈ ℝ^n and tile signatures Σ = {σ₁, ..., σₖ}, the routing function is:

$$r(x) = \arg\max_{i \in [k]} \langle x, \sigma_i \rangle$$

### Voronoi Decomposition

The routing function partitions input space into Voronoi cells:

$$V_i = \{x \in \mathbb{R}^n : \langle x, \sigma_i \rangle \geq \langle x, \sigma_j \rangle, \forall j\}$$

Each cell V_i contains inputs routed to tile i.

### Decision Boundaries

The boundary between tiles i and j is the hyperplane:

$$H_{ij} = \{x : \langle x, \sigma_i - \sigma_j \rangle = 0\}$$

Since σᵢ - σⱼ ∈ {-2, -1, 0, 1, 2}^n, boundaries have discrete normal vectors.

### Routing Stability

**Theorem (Routing Stability):** For input x with margin:

$$\gamma = \max_i \langle x, \sigma_i \rangle - \max_{j \neq i^*} \langle x, \sigma_j \rangle > 0$$

where i* is the winning tile, the routing is stable under perturbations ‖δ‖ < γ/2.

*Proof:* The score difference satisfies:
$$\langle x + \delta, \sigma_{i^*} \rangle - \langle x + \delta, \sigma_j \rangle = \gamma + \langle \delta, \sigma_{i^*} - \sigma_j \rangle$$

Since ‖σᵢ* - σⱼ‖ ≤ 2√n, we have |⟨δ, σᵢ* - σⱼ⟩| ≤ 2√n ‖δ‖.

For ‖δ‖ < γ/(4√n), the winner remains i*. □

---

## Capacity and Expressiveness

### Tile Capacity

Each tile implements a function f_i: ℝ^n → ℝ^n via:

$$f_i(x) = W_2 \cdot \phi(W_1 x)$$

where W₁, W₂ are ternary and φ is the activation.

**Theorem (Ternary Universal Approximation):** For any continuous function g: [0,1]^n → ℝ and ε > 0, there exists a ternary network with sufficiently many hidden units that ε-approximates g.

*Sketch:* Ternary networks can implement arbitrary step functions, which can approximate continuous functions by discretization.

### Mixture Capacity

With k tiles, the overall function is:

$$F(x) = f_{r(x)}(x)$$

where r(x) selects the tile. This is a **piecewise function** with k pieces.

**Theorem (Piecewise Expressiveness):** The TriX architecture with k tiles and d-dimensional hidden layers can represent any function that is:
1. Piecewise continuous with at most k pieces
2. Each piece approximable by a ternary network of width d

### Comparison to Dense Networks

| Architecture | Parameters | Functions |
|--------------|------------|-----------|
| Dense (d×d) | d² | Smooth maps |
| TriX (k tiles, d/k hidden) | kd²/k² = d²/k | k-piecewise maps |

TriX achieves similar expressiveness with 1/k parameters by specialization.

---

## Training Dynamics

### Straight-Through Estimator

The sign function is non-differentiable:

$$\text{sign}(w) = \begin{cases} +1 & w > 0 \\ 0 & w = 0 \\ -1 & w < 0 \end{cases}$$

The STE provides surrogate gradients:

$$\frac{\partial \mathcal{L}}{\partial w} \approx \frac{\partial \mathcal{L}}{\partial \text{sign}(w)}$$

### Gradient Flow Through Routing

For differentiable routing with temperature τ:

$$p_i = \frac{\exp(\langle x, \sigma_i \rangle / \tau)}{\sum_j \exp(\langle x, \sigma_j \rangle / \tau)}$$

The output becomes:
$$y = \sum_i p_i f_i(x)$$

As τ → 0, this recovers hard routing.

### Signature Evolution

**Proposition (Signature Dynamics):** Under gradient descent on a balanced routing objective, signatures evolve to maximize coverage of the input distribution.

*Intuition:* If tile i receives too many inputs, its loss contribution is high, pushing its signature away from the input cluster center. Conversely, underutilized tiles attract inputs.

### Equilibrium Conditions

At equilibrium, the routing satisfies:

$$\mathbb{E}_{x \sim p(x)}[\mathbf{1}_{r(x) = i}] = \frac{1}{k} \quad \forall i$$

This is the **load balancing** condition enforced by auxiliary losses.

---

## Connections to Related Work

### Mixture of Experts (MoE)

| Aspect | MoE | TriX |
|--------|-----|------|
| Routing | Learned network | Emergent from weights |
| Router params | O(n·k) | 0 |
| Routing cost | O(n·k) | O(n·k) or O(n·√k) |
| Specialization | Explicit | Emergent |

### Vector Quantization (VQ)

TriX signatures are related to VQ codebooks:
- Codebook vectors ↔ Signatures
- Quantization ↔ Routing
- Commitment loss ↔ Load balancing

Key difference: VQ codebooks are learned separately; TriX signatures emerge from task learning.

### Ternary Neural Networks

Prior ternary work focuses on compression:
- Binary/Ternary Weight Networks (Li et al., 2016)
- XNOR-Net (Rastegari et al., 2016)

TriX uses ternarization for **routing**, not just compression.

### Content-Addressable Memory

Hopfield networks retrieve patterns via:
$$x_{t+1} = \text{sign}(Wx_t)$$

TriX routing is a single-step content-addressable lookup:
$$\text{tile} = \arg\max_i \langle x, \sigma_i \rangle$$

### Kolmogorov-Arnold Representation

The KA theorem states any continuous function can be represented as:

$$f(x_1, ..., x_n) = \sum_{q=0}^{2n} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)$$

TriX splines connect to this via univariate basis functions composed through routing.

---

## Unified Addressing Theory (Mesa 11)

### The Addressing Mode Framework

**Key Insight**: All computation can be viewed as addressed access to transformations. The addressing mode determines how computation is accessed.

Three fundamental addressing modes:

| Mode | Definition | Access Pattern |
|------|------------|----------------|
| Temporal | f(position) → computation | Sequential stages |
| Spatial | f(topology) → computation | Graph neighbors |
| Content | f(similarity) → computation | Signature matching |

### The Unification

These modes are not alternatives - they are projections of a unified address space:

$$\text{Address} = [\text{position}_\text{dims} \mid \text{topology}_\text{dims} \mid \text{feature}_\text{dims}]$$

- Temporal = [pos, 0, 0]
- Spatial = [0, top, 0]
- Content = [0, 0, feat]
- Mixed = [pos, top, feat] (learned)

### Formal Result

**Theorem (Temporal ⊂ Content):** Any temporal addressing scheme can be exactly embedded in content addressing.

*Proof sketch:* Map position i to one-hot vector e_i. Stage signatures {e_1, ..., e_n} are orthogonal. Content routing recovers exact sequential execution via argmax matching.

**Experimental validation:** 4-stage pipeline emulated with 0.00 error. See `experiments/mesa11/01_pipeline_emulation.py`.

### Implications

1. Content-addressing is computationally universal for pipelines
2. TriX's domain equivalences (FFT, CUDA, etc.) follow from this universality
3. Architectures can blend addressing modes via mixed signatures

For complete treatment, see [Mesa 11 Documentation](MESA11_UAT.md).

---

## Open Questions

1. **Optimal Signature Distribution:** What signature distribution maximizes approximation capacity for a given input distribution?

2. **Hierarchical Depth:** Is two-level hierarchy optimal, or do deeper hierarchies improve scaling?

3. **Dynamic Signatures:** Can signatures adapt at inference time for distribution shift?

4. **Theoretical Generalization:** What are the generalization bounds for ternary sparse mixtures?

---

## References

1. Shazeer, N., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer."
2. Li, F., Zhang, B., & Liu, B. (2016). "Ternary Weight Networks."
3. Rastegari, M., et al. (2016). "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks."
4. Kolmogorov, A. N. (1957). "On the representation of continuous functions of many variables."
5. Hopfield, J. J. (1982). "Neural networks and physical systems with emergent collective computational abilities."
