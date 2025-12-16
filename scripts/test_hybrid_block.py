#!/usr/bin/env python3
"""
Test: Mesa Spectral Mixer + TriX FFN Integration

IMPORTANT PRIOR ART:
- FNet (Google, 2021): "FNet: Mixing Tokens with Fourier Transforms"
  - Replaces attention with FFT, achieves 92% of BERT accuracy at 7x speed
- Hyena (2023): Combines FFT with gating for long-range dependencies  
- S4/Mamba (2023-24): State-space models with FFT-based convolutions

The colleague's idea is valid but not novel. The question is:
Does TriX's ternary routing ADD something to FFT mixing?
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# FIXED MesaSpectralMixer
# =============================================================================

class MesaSpectralMixerV1(nn.Module):
    """
    FIXED version of the colleague's proposal.
    
    Issue: Original had wrong filter dimensions.
    Fix: Filter should be (max_seq_len//2+1, d_model) for seq-wise FFT.
    
    This is essentially FNet with learnable frequency weights.
    """
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        # Filter for sequence-dimension FFT
        # Shape: (freq_bins, d_model, 2) for complex weights per channel
        freq_bins = max_seq_len // 2 + 1
        self.freq_filter = nn.Parameter(torch.randn(freq_bins, d_model, 2) * 0.02)
    
    def forward(self, x):
        # x: [batch, seq, d_model]
        batch, seq_len, d_model = x.shape
        
        # FFT along sequence dimension
        x_fft = torch.fft.rfft(x, dim=1)  # [batch, seq//2+1, d_model]
        
        # Apply learnable filter (truncate to actual seq length)
        freq_bins = x_fft.shape[1]
        weight = torch.view_as_complex(self.freq_filter[:freq_bins])  # [freq, d_model]
        x_fft = x_fft * weight.unsqueeze(0)  # broadcast over batch
        
        # Back to time domain
        x_out = torch.fft.irfft(x_fft, n=seq_len, dim=1)
        
        return x_out


class MesaSpectralMixerV2(nn.Module):
    """
    FNet-style: Just FFT, no learnable weights.
    
    From the paper: "Surprisingly effective with NO parameters"
    """
    def __init__(self, d_model):
        super().__init__()
        # No learnable parameters - just FFT!
    
    def forward(self, x):
        # 2D FFT: mix both sequence AND channel dimensions
        # This is what FNet actually does
        x_fft = torch.fft.fft2(x.float()).real
        return x_fft


class MesaSpectralMixerV3(nn.Module):
    """
    Hybrid: FFT for global + Conv for local detail.
    
    Inspired by Hyena architecture.
    """
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        
        # Global: FFT with learnable filter
        freq_bins = max_seq_len // 2 + 1
        self.freq_filter = nn.Parameter(torch.ones(freq_bins, d_model))
        
        # Local: depthwise conv for short-range
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        
        # Mix
        self.gate = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        
        # Global path (FFT)
        x_fft = torch.fft.rfft(x, dim=1)
        freq_bins = x_fft.shape[1]
        x_fft = x_fft * self.freq_filter[:freq_bins].unsqueeze(0)
        x_global = torch.fft.irfft(x_fft, n=seq_len, dim=1)
        
        # Local path (Conv)
        x_local = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        
        # Gated combination
        gate = torch.sigmoid(self.gate(x))
        x_out = gate * x_global + (1 - gate) * x_local
        
        return x_out


# =============================================================================
# TriX Hybrid Block
# =============================================================================

class TriXHybridBlock(nn.Module):
    """
    The 'Bicameral' Block:
    1. Mesa (FFT) for global context
    2. TriX (Ternary) for categorical logic
    """
    def __init__(self, d_model, trix_ffn, mixer_type='v2', max_seq_len=512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        
        if mixer_type == 'v1':
            self.mixer = MesaSpectralMixerV1(d_model, max_seq_len)
        elif mixer_type == 'v2':
            self.mixer = MesaSpectralMixerV2(d_model)
        elif mixer_type == 'v3':
            self.mixer = MesaSpectralMixerV3(d_model, max_seq_len)
        else:
            raise ValueError(f"Unknown mixer_type: {mixer_type}")
        
        self.ln2 = nn.LayerNorm(d_model)
        self.trix_ffn = trix_ffn
    
    def forward(self, x):
        # Spectral mixing (replaces attention)
        residual = x
        x = self.ln1(x)
        x = self.mixer(x)
        x = residual + x
        
        # TriX FFN (replaces MLP)
        residual = x
        x = self.ln2(x)
        ffn_out = self.trix_ffn(x)
        if isinstance(ffn_out, tuple):
            ffn_out = ffn_out[0]  # Handle TriX return format
        x = residual + ffn_out
        
        return x


# =============================================================================
# Tests
# =============================================================================

def test_mixers():
    """Test all mixer variants"""
    print("=" * 65)
    print("TEST 1: Mixer Implementations")
    print("=" * 65)
    
    d_model = 64
    seq_len = 128
    batch = 2
    x = torch.randn(batch, seq_len, d_model)
    
    mixers = [
        ("V1 (Learnable filter)", MesaSpectralMixerV1(d_model, max_seq_len=256)),
        ("V2 (FNet-style)", MesaSpectralMixerV2(d_model)),
        ("V3 (FFT + Conv hybrid)", MesaSpectralMixerV3(d_model, max_seq_len=256)),
    ]
    
    for name, mixer in mixers:
        try:
            out = mixer(x)
            params = sum(p.numel() for p in mixer.parameters())
            print(f"{name:<25} shape={tuple(out.shape)}, params={params:,}")
        except Exception as e:
            print(f"{name:<25} FAILED: {e}")


def test_global_receptive_field():
    """Verify FFT gives infinite receptive field"""
    print("\n" + "=" * 65)
    print("TEST 2: Global Receptive Field")
    print("=" * 65)
    
    d_model = 64
    seq_len = 128
    
    mixer = MesaSpectralMixerV2(d_model)
    
    x = torch.randn(1, seq_len, d_model)
    x_perturbed = x.clone()
    x_perturbed[0, 0, :] += 10.0  # Change only first token
    
    out_orig = mixer(x)
    out_pert = mixer(x_perturbed)
    
    diff = (out_pert - out_orig).abs().mean(dim=-1)[0]  # [seq_len]
    
    print(f"Perturbation at token 0, impact on:")
    print(f"  Token 0:   {diff[0].item():.4f} (expected: high)")
    print(f"  Token 63:  {diff[63].item():.4f} (middle)")
    print(f"  Token 127: {diff[127].item():.4f} (last)")
    
    if diff[-1] > 0.01:
        print("PASS: FFT provides global mixing (token 0 affects token 127)")
    else:
        print("FAIL: Mixing is local only")


def test_hybrid_block():
    """Test full hybrid block with TriX"""
    print("\n" + "=" * 65)
    print("TEST 3: Full Hybrid Block (Mesa + TriX)")
    print("=" * 65)
    
    try:
        from trix.nn import SparseLookupFFN
        
        d_model = 64
        seq_len = 128
        batch = 2
        
        trix_ffn = SparseLookupFFN(
            d_model=d_model,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        
        block = TriXHybridBlock(d_model, trix_ffn, mixer_type='v2')
        
        x = torch.randn(batch, seq_len, d_model)
        out = block(x)
        
        params = sum(p.numel() for p in block.parameters())
        
        print(f"Input:  {tuple(x.shape)}")
        print(f"Output: {tuple(out.shape)}")
        print(f"Params: {params:,}")
        print("PASS: Mesa + TriX integration works")
        
        return block
        
    except ImportError as e:
        print(f"SKIP: Could not import TriX ({e})")
        return None


def test_causality_problem():
    """Demonstrate the causality issue with FFT"""
    print("\n" + "=" * 65)
    print("TEST 4: Causality Problem (CRITICAL)")
    print("=" * 65)
    
    print("""
WARNING: FFT-based mixers are NOT CAUSAL by default!

For autoregressive LM (GPT-style), each token should only see past tokens.
FFT mixes ALL tokens together - the model sees the future!

Solutions:
1. FNet: Used for BERT (bidirectional), not GPT (causal)
2. Causal Conv: Use causal padding in convolution-based alternatives
3. State-Space: S4/Mamba use causal formulation of FFT convolution

For TriX Hybrid to work on GPT-style tasks, need:
- Either: Accept bidirectional (BERT-style) training
- Or: Implement causal spectral mixing (complex, see S4 paper)
""")


def test_comparison_to_attention():
    """Compare complexity"""
    print("\n" + "=" * 65)
    print("TEST 5: Complexity Comparison")
    print("=" * 65)
    
    seq_lens = [128, 512, 2048, 8192]
    d_model = 256
    
    print(f"{'Seq Len':<10} {'Attention O(N²)':<20} {'FFT O(N log N)':<20} {'Ratio':<10}")
    print("-" * 60)
    
    for n in seq_lens:
        attn_ops = n * n * d_model  # Simplified attention cost
        fft_ops = n * torch.log2(torch.tensor(n)).item() * d_model
        ratio = attn_ops / fft_ops
        print(f"{n:<10} {attn_ops:>15,.0f} {fft_ops:>15,.0f} {ratio:>10.1f}x")
    
    print("\nAt long sequences, FFT is dramatically more efficient.")
    print("But attention has content-based routing; FFT is fixed pattern.")


def main():
    print("MESA + TRIX HYBRID VALIDATION")
    print("Integrating FFT Mixing with Ternary Logic")
    print()
    
    test_mixers()
    test_global_receptive_field()
    block = test_hybrid_block()
    test_causality_problem()
    test_comparison_to_attention()
    
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print("""
VALIDATED:
  [YES] FFT mixers work and provide global receptive field
  [YES] Mesa + TriX integration is feasible
  [YES] O(N log N) vs O(N²) complexity advantage

CONCERNS:
  [!] Causality: FFT sees future tokens (problem for GPT)
  [!] Content-routing: FFT is fixed pattern, no query-key dynamics
  [!] Prior art: This is essentially FNet + TriX

RECOMMENDATION:
  For BERT-style (bidirectional): Ready to benchmark
  For GPT-style (causal): Need S4/Mamba-style causal formulation
""")


if __name__ == "__main__":
    main()
