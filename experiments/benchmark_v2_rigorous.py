#!/usr/bin/env python3
"""
Rigorous Benchmark: SparseLookupFFNv2

Comprehensive testing with logging:
  1. Surgery API validation
  2. Regularizer effectiveness
  3. Training dynamics
  4. Island quality metrics
  5. Comparison to baseline

All results logged to JSON for documentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List
import sys

sys.path.insert(0, '/workspace/trix_latest/src')

from trix.nn import SparseLookupFFNv2, SparseLookupBlockV2


# =============================================================================
# Logging
# =============================================================================

class BenchmarkLogger:
    """Logger for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = datetime.now()
        self.results = {
            'name': name,
            'timestamp': self.start_time.isoformat(),
            'tests': {},
        }
    
    def log_test(self, test_name: str, result: Dict):
        """Log a test result."""
        self.results['tests'][test_name] = result
        print(f"  [{test_name}] {'PASS' if result.get('passed', False) else 'FAIL'}")
        for key, value in result.items():
            if key != 'passed':
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
    
    def save(self, filepath: str):
        """Save results to JSON."""
        self.results['duration_seconds'] = (datetime.now() - self.start_time).total_seconds()
        self.results['all_passed'] = all(
            t.get('passed', False) for t in self.results['tests'].values()
        )
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")
    
    def summary(self):
        """Print summary."""
        total = len(self.results['tests'])
        passed = sum(1 for t in self.results['tests'].values() if t.get('passed', False))
        
        print("\n" + "=" * 70)
        print(f"SUMMARY: {passed}/{total} tests passed")
        print("=" * 70)


# =============================================================================
# Test Functions
# =============================================================================

def test_basic_functionality(logger: BenchmarkLogger):
    """Test basic forward/backward."""
    print("\n[1] Basic Functionality")
    print("-" * 70)
    
    ffn = SparseLookupFFNv2(d_model=64, num_tiles=16, tiles_per_cluster=4)
    x = torch.randn(4, 32, 64)
    
    # Forward
    output, routing_info, aux_losses = ffn(x)
    
    shape_ok = output.shape == x.shape
    has_routing = 'tile_idx' in routing_info
    has_aux = 'total_aux' in aux_losses
    
    # Backward
    loss = output.sum() + aux_losses['total_aux']
    loss.backward()
    
    grad_ok = all(p.grad is not None for p in [ffn.signatures_raw, ffn.directions])
    
    logger.log_test('basic_forward', {
        'passed': shape_ok and has_routing and has_aux and grad_ok,
        'output_shape': list(output.shape),
        'has_routing_info': has_routing,
        'has_aux_losses': has_aux,
        'gradients_flow': grad_ok,
    })


def test_surgery_api(logger: BenchmarkLogger):
    """Test signature surgery API."""
    print("\n[2] Surgery API")
    print("-" * 70)
    
    ffn = SparseLookupFFNv2(d_model=64, num_tiles=16, tiles_per_cluster=4)
    
    # Design signature
    sig = torch.zeros(64)
    sig[:16] = 1.0
    
    # Test insert
    ffn.insert_signature(0, sig, freeze=True, tag="test")
    
    insert_ok = torch.allclose(ffn.signatures[0, :16], torch.ones(16))
    freeze_ok = ffn.is_frozen(0)
    
    # Test unfreeze
    ffn.unfreeze_signature(0)
    unfreeze_ok = not ffn.is_frozen(0)
    
    # Test analysis
    ffn.freeze_signature(0)
    analysis = ffn.get_signature_analysis(0)
    analysis_ok = len(analysis['positive_dims']) == 16
    
    # Test history
    history = ffn.get_surgery_history()
    history_ok = len(history) == 1 and history[0]['tag'] == "test"
    
    logger.log_test('surgery_insert', {
        'passed': insert_ok,
        'signature_set_correctly': insert_ok,
    })
    
    logger.log_test('surgery_freeze_unfreeze', {
        'passed': freeze_ok and unfreeze_ok,
        'freeze_works': freeze_ok,
        'unfreeze_works': unfreeze_ok,
    })
    
    logger.log_test('surgery_analysis', {
        'passed': analysis_ok and history_ok,
        'analysis_correct': analysis_ok,
        'history_tracked': history_ok,
        'positive_dims_count': len(analysis['positive_dims']),
    })


def test_frozen_signature_stability(logger: BenchmarkLogger):
    """Test that frozen signatures don't change during training."""
    print("\n[3] Frozen Signature Stability")
    print("-" * 70)
    
    ffn = SparseLookupFFNv2(d_model=64, num_tiles=16, tiles_per_cluster=4)
    
    # Insert and freeze
    sig = torch.zeros(64)
    sig[:16] = 1.0
    ffn.insert_signature(0, sig, freeze=True)
    
    initial_sig = ffn.signatures[0].clone()
    
    # Train
    optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
    
    for _ in range(50):
        x = torch.randn(4, 16, 64)
        output, _, aux = ffn(x)
        loss = output.sum() + aux['total_aux']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    final_sig = ffn.signatures[0]
    
    # Check stability
    sig_unchanged = torch.allclose(initial_sig, final_sig)
    max_diff = (final_sig - initial_sig).abs().max().item()
    
    logger.log_test('frozen_stability', {
        'passed': sig_unchanged,
        'signature_unchanged': sig_unchanged,
        'max_difference': max_diff,
    })


def test_regularizers(logger: BenchmarkLogger):
    """Test island regularizers."""
    print("\n[4] Island Regularizers")
    print("-" * 70)
    
    # Test ternary regularizer
    ffn = SparseLookupFFNv2(d_model=64, num_tiles=16, tiles_per_cluster=4, ternary_weight=0.01)
    
    # Set near-ternary
    with torch.no_grad():
        ffn.signatures_raw.zero_()
        ffn.signatures_raw[:, :16] = 1.0
    loss_ternary = ffn.compute_ternary_loss().item()
    
    # Set non-ternary
    with torch.no_grad():
        ffn.signatures_raw.fill_(0.5)
    loss_non_ternary = ffn.compute_ternary_loss().item()
    
    ternary_prefers = loss_ternary < loss_non_ternary
    
    logger.log_test('ternary_regularizer', {
        'passed': ternary_prefers,
        'loss_ternary': loss_ternary,
        'loss_non_ternary': loss_non_ternary,
        'prefers_ternary': ternary_prefers,
    })
    
    # Test sparsity regularizer
    ffn = SparseLookupFFNv2(d_model=64, num_tiles=16, tiles_per_cluster=4, sparsity_weight=0.01)
    
    # Sparse
    with torch.no_grad():
        ffn.signatures_raw.zero_()
        ffn.signatures_raw[:, :4] = 1.0
    loss_sparse = ffn.compute_sparsity_loss().item()
    
    # Dense
    with torch.no_grad():
        ffn.signatures_raw.fill_(1.0)
    loss_dense = ffn.compute_sparsity_loss().item()
    
    sparsity_prefers = loss_sparse < loss_dense
    
    logger.log_test('sparsity_regularizer', {
        'passed': sparsity_prefers,
        'loss_sparse': loss_sparse,
        'loss_dense': loss_dense,
        'prefers_sparse': sparsity_prefers,
    })
    
    # Test diversity regularizer
    ffn = SparseLookupFFNv2(d_model=64, num_tiles=16, tiles_per_cluster=4, diversity_weight=0.01)
    
    # All same
    with torch.no_grad():
        ffn.signatures_raw.fill_(0.0)
        ffn.signatures_raw[:, :8] = 1.0
    loss_similar = ffn.compute_diversity_loss().item()
    
    # All different
    with torch.no_grad():
        for i in range(16):
            ffn.signatures_raw[i].zero_()
            ffn.signatures_raw[i, i*4:(i+1)*4] = 1.0
    loss_diverse = ffn.compute_diversity_loss().item()
    
    diversity_prefers = loss_diverse < loss_similar
    
    logger.log_test('diversity_regularizer', {
        'passed': diversity_prefers,
        'loss_similar': loss_similar,
        'loss_diverse': loss_diverse,
        'prefers_diverse': diversity_prefers,
    })


def test_training_dynamics(logger: BenchmarkLogger):
    """Test training improves island metrics."""
    print("\n[5] Training Dynamics")
    print("-" * 70)
    
    ffn = SparseLookupFFNv2(
        d_model=64, 
        num_tiles=16, 
        tiles_per_cluster=4,
        ternary_weight=0.01,
        sparsity_weight=0.01,
        diversity_weight=0.01,
    )
    
    # Initial stats
    initial_stats = ffn.get_island_stats()
    
    # Train
    optimizer = torch.optim.Adam(ffn.parameters(), lr=0.001)
    
    losses = []
    for step in range(200):
        x = torch.randn(8, 32, 64)
        output, _, aux = ffn(x)
        
        # Simulate task loss
        task_loss = output.pow(2).mean()
        total_loss = task_loss + aux['total_aux']
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
    
    # Final stats
    final_stats = ffn.get_island_stats()
    
    # Check improvements
    ternary_improved = final_stats['ternary_fraction'] >= initial_stats['ternary_fraction']
    loss_decreased = np.mean(losses[-20:]) < np.mean(losses[:20])
    
    logger.log_test('training_dynamics', {
        'passed': ternary_improved and loss_decreased,
        'initial_ternary': initial_stats['ternary_fraction'],
        'final_ternary': final_stats['ternary_fraction'],
        'initial_sparsity': initial_stats['sparsity'],
        'final_sparsity': final_stats['sparsity'],
        'initial_diversity': initial_stats['diversity'],
        'final_diversity': final_stats['diversity'],
        'loss_decreased': loss_decreased,
        'initial_loss': np.mean(losses[:20]),
        'final_loss': np.mean(losses[-20:]),
    })


def test_claim_tracking(logger: BenchmarkLogger):
    """Test claim tracking with designed signatures."""
    print("\n[6] Claim Tracking")
    print("-" * 70)
    
    ffn = SparseLookupFFNv2(d_model=64, num_tiles=16, tiles_per_cluster=4)
    
    # Design signature for "class 0" inputs
    sig = torch.zeros(64)
    sig[:16] = 1.0
    ffn.insert_signature(0, sig, freeze=True)
    
    ffn.reset_claim_tracking()
    
    # Create inputs that should match signature
    for _ in range(100):
        x = torch.randn(1, 1, 64)
        x[0, 0, :16] = 2.0  # Strong positive on matching dims
        labels = torch.zeros(1, 1, dtype=torch.long)
        
        ffn(x, labels=labels)
    
    # Check claim rate
    claim_rate = ffn.get_claim_rate(0, 0)
    dominant_class, purity = ffn.get_tile_purity(0)
    
    claim_ok = claim_rate > 0.3  # Should have reasonable claim
    
    logger.log_test('claim_tracking', {
        'passed': claim_ok,
        'claim_rate': claim_rate,
        'dominant_class': dominant_class,
        'purity': purity,
    })


def test_block_integration(logger: BenchmarkLogger):
    """Test SparseLookupBlockV2 integration."""
    print("\n[7] Block Integration")
    print("-" * 70)
    
    block = SparseLookupBlockV2(
        d_model=64,
        n_heads=4,
        num_tiles=16,
        tiles_per_cluster=4,
        ternary_weight=0.01,
    )
    
    x = torch.randn(2, 32, 64)
    
    # Forward
    output, routing_info, aux_losses = block(x)
    
    shape_ok = output.shape == x.shape
    
    # Gradient
    loss = output.sum() + aux_losses['total_aux']
    loss.backward()
    
    grad_ok = block.ffn.signatures_raw.grad is not None
    
    logger.log_test('block_integration', {
        'passed': shape_ok and grad_ok,
        'output_shape_correct': shape_ok,
        'gradients_flow': grad_ok,
    })


def test_edge_cases(logger: BenchmarkLogger):
    """Test edge cases."""
    print("\n[8] Edge Cases")
    print("-" * 70)
    
    ffn = SparseLookupFFNv2(d_model=64, num_tiles=16, tiles_per_cluster=4)
    
    # Single sample
    x1 = torch.randn(1, 1, 64)
    out1, _, _ = ffn(x1)
    single_ok = out1.shape == x1.shape and not torch.isnan(out1).any()
    
    # Zero input
    x2 = torch.zeros(2, 16, 64)
    out2, _, _ = ffn(x2)
    zero_ok = not torch.isnan(out2).any() and not torch.isinf(out2).any()
    
    # Large input
    x3 = torch.randn(2, 16, 64) * 100
    out3, _, _ = ffn(x3)
    large_ok = not torch.isnan(out3).any() and not torch.isinf(out3).any()
    
    logger.log_test('edge_cases', {
        'passed': single_ok and zero_ok and large_ok,
        'single_sample': single_ok,
        'zero_input': zero_ok,
        'large_input': large_ok,
    })


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("RIGOROUS BENCHMARK: SparseLookupFFNv2")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    logger = BenchmarkLogger("SparseLookupFFNv2_Rigorous")
    
    # Run all tests
    test_basic_functionality(logger)
    test_surgery_api(logger)
    test_frozen_signature_stability(logger)
    test_regularizers(logger)
    test_training_dynamics(logger)
    test_claim_tracking(logger)
    test_block_integration(logger)
    test_edge_cases(logger)
    
    # Save and summarize
    output_dir = "/workspace/trix_latest/experiments/results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"benchmark_v2_{timestamp}.json")
    
    logger.save(filepath)
    logger.summary()
    
    return logger.results


if __name__ == "__main__":
    results = main()
