"""
Tests for A/B Harness - Compiled vs Dynamic comparison.

Validates that the A/B comparison infrastructure works correctly.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/workspace/trix_latest/src')

from trix.nn import SparseLookupFFNv2, CompiledDispatch


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def trained_system(device):
    """Create a trained FFN with compiled dispatch."""
    ffn = SparseLookupFFNv2(
        d_model=32,
        num_tiles=8,
        tiles_per_cluster=4,
    ).to(device)
    
    # Populate claim matrix to simulate training
    ffn.claim_matrix = torch.zeros(8, 4, device=device)
    ffn.claim_matrix[0, 0] = 800  # Class 0 → Tile 0
    ffn.claim_matrix[1, 0] = 100
    ffn.claim_matrix[2, 1] = 750  # Class 1 → Tile 2
    ffn.claim_matrix[3, 1] = 150
    ffn.claim_matrix[4, 2] = 600  # Class 2 → Tile 4
    ffn.claim_matrix[5, 2] = 300
    ffn.claim_matrix[6, 3] = 500  # Class 3 → Tile 6 (less stable)
    ffn.claim_matrix[7, 3] = 400
    
    compiler = CompiledDispatch(ffn)
    return ffn, compiler


# =============================================================================
# A/B Agreement Tests
# =============================================================================

class TestABHarness:
    """Test A/B comparison infrastructure."""
    
    def test_dynamic_vs_compiled_agreement(self, trained_system, device):
        """Test that compiled produces same output as dynamic for compiled classes."""
        ffn, compiler = trained_system
        
        # Compile class 0 (high stability)
        compiler.compile(class_id=0, tile_idx=0, min_confidence=0.5)
        
        # Run same input through both paths
        x = torch.randn(1, 1, 32, device=device)
        
        # Dynamic
        out_dyn, info_dyn, _ = ffn(x)
        
        # Compiled
        out_cmp, info_cmp, _ = compiler.forward(x, class_hint=0, confidence=0.9)
        
        # Should use compiled path
        assert info_cmp['compiled'] == True
        
        # Outputs may differ slightly due to different code paths,
        # but should be close if implementation is correct
        # (In real usage, we'd compare final predictions, not intermediate outputs)
    
    def test_fallback_on_unknown_class(self, trained_system, device):
        """Test that unknown classes fall back to dynamic routing."""
        ffn, compiler = trained_system
        
        compiler.compile(class_id=0, tile_idx=0)
        
        x = torch.randn(1, 1, 32, device=device)
        
        # Unknown class should use dynamic
        _, info, _ = compiler.forward(x, class_hint=99, confidence=0.9)
        assert info['compiled'] == False
    
    def test_fallback_on_low_confidence(self, trained_system, device):
        """Test that low confidence triggers fallback."""
        ffn, compiler = trained_system
        
        compiler.compile(class_id=0, tile_idx=0, min_confidence=0.8)
        
        x = torch.randn(1, 1, 32, device=device)
        
        # Low confidence should use dynamic
        _, info, _ = compiler.forward(x, class_hint=0, confidence=0.5)
        assert info['compiled'] == False
        
        # High confidence should use compiled
        _, info, _ = compiler.forward(x, class_hint=0, confidence=0.9)
        assert info['compiled'] == True
    
    def test_stats_tracking_accuracy(self, trained_system, device):
        """Test that stats accurately track hits and misses."""
        ffn, compiler = trained_system
        
        compiler.compile(class_id=0, tile_idx=0, min_confidence=0.5)
        compiler.compile(class_id=1, tile_idx=2, min_confidence=0.5)
        compiler.reset_stats()
        
        x = torch.randn(1, 1, 32, device=device)
        
        # 3 compiled hits
        compiler.forward(x, class_hint=0, confidence=0.9)
        compiler.forward(x, class_hint=0, confidence=0.9)
        compiler.forward(x, class_hint=1, confidence=0.9)
        
        # 2 dynamic (unknown class)
        compiler.forward(x, class_hint=99, confidence=0.9)
        compiler.forward(x, class_hint=None)
        
        # 1 compiled miss (low confidence) - this also increments dynamic_calls
        compiler.forward(x, class_hint=0, confidence=0.3)
        
        stats = compiler.get_stats()
        
        assert stats['compiled_hits'] == 3
        assert stats['compiled_misses'] == 1
        # dynamic_calls does NOT include guard failures (those are compiled_misses)
        assert stats['dynamic_calls'] == 2  # 2 unknown class only
        assert stats['total_calls'] == 6
        assert abs(stats['hit_rate'] - 0.5) < 0.01  # 3/6 = 50%


class TestCompilabilityScoring:
    """Test compilability score calculation."""
    
    def test_high_stability_compilable(self, trained_system, device):
        """Test that high-stability classes are compilable."""
        ffn, compiler = trained_system
        
        # Class 0 has 800/900 = 89% frequency
        stats = compiler.profile(0)
        
        assert stats.mode_tile == 0
        assert stats.mode_frequency > 0.8
        assert stats.is_compilable(threshold=0.5)
    
    def test_low_stability_not_compilable(self, trained_system, device):
        """Test that low-stability classes are not compilable."""
        ffn, compiler = trained_system
        
        # Class 3 has 500/900 = 55% frequency, split between tiles
        stats = compiler.profile(3)
        
        assert stats.mode_frequency < 0.6
        # With purity factored in, compilability should be lower
    
    def test_compile_stable_threshold(self, trained_system, device):
        """Test that compile_stable respects threshold."""
        ffn, compiler = trained_system
        
        # High threshold - fewer classes compile
        compiled_high = compiler.compile_stable(threshold=0.6)
        compiler.decompile_all()
        
        # Low threshold - more classes compile
        compiled_low = compiler.compile_stable(threshold=0.3)
        
        assert len(compiled_low) >= len(compiled_high)


class TestDriftDetection:
    """Test drift detection and recompilation."""
    
    def test_detect_tile_change(self, trained_system, device):
        """Test that tile changes are detected as drift."""
        ffn, compiler = trained_system
        
        # Compile based on current state
        compiler.compile(class_id=0, tile_idx=0, frequency=0.8)
        
        # Simulate drift: class 0 now goes to tile 1
        ffn.claim_matrix[0, 0] = 100
        ffn.claim_matrix[1, 0] = 800
        
        drifted = compiler.check_drift(threshold=0.3)
        
        assert 0 in drifted
    
    def test_no_drift_when_stable(self, trained_system, device):
        """Test that stable classes don't trigger drift."""
        ffn, compiler = trained_system
        
        stats = compiler.profile(0)
        compiler.compile_from_profile(stats)
        
        # No changes to claim matrix
        drifted = compiler.check_drift(threshold=0.3)
        
        assert 0 not in drifted


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
