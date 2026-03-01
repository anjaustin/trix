"""
Tests for CompiledDispatch - Path Compilation for TriX v2.

Tests the full lifecycle:
    Train → Profile → Compile → Execute → Monitor
"""

import pytest
import torch
import numpy as np
from collections import defaultdict

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

from trix.nn import SparseLookupFFNv2
from trix.nn.compiled_dispatch import CompiledDispatch, CompiledEntry, ProfileStats


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def trained_ffn(device):
    """Create and train an FFN with clear class→tile specialization."""
    ffn = SparseLookupFFNv2(
        d_model=64,
        num_tiles=8,
        tiles_per_cluster=4,
    ).to(device)
    
    # Simulate training that creates class→tile specialization
    # By manually populating the claim matrix
    num_classes = 5
    
    # Clear and resize claim matrix
    ffn.claim_matrix = torch.zeros(8, num_classes, device=device)
    
    # Class 0 → Tile 2 (90% of samples)
    ffn.claim_matrix[2, 0] = 900
    ffn.claim_matrix[3, 0] = 50
    ffn.claim_matrix[4, 0] = 50
    
    # Class 1 → Tile 5 (85% of samples)
    ffn.claim_matrix[5, 1] = 850
    ffn.claim_matrix[6, 1] = 100
    ffn.claim_matrix[7, 1] = 50
    
    # Class 2 → Tile 0 (70% of samples)
    ffn.claim_matrix[0, 2] = 700
    ffn.claim_matrix[1, 2] = 200
    ffn.claim_matrix[2, 2] = 100
    
    # Class 3 → scattered (no clear winner)
    ffn.claim_matrix[0, 3] = 250
    ffn.claim_matrix[1, 3] = 250
    ffn.claim_matrix[2, 3] = 250
    ffn.claim_matrix[3, 3] = 250
    
    # Class 4 → Tile 7 (95% of samples, high purity)
    ffn.claim_matrix[7, 4] = 950
    ffn.claim_matrix[6, 4] = 50
    
    return ffn


@pytest.fixture
def compiler(trained_ffn):
    """Create a CompiledDispatch wrapper."""
    return CompiledDispatch(trained_ffn)


# =============================================================================
# Profiling Tests
# =============================================================================

class TestProfiling:
    """Test the profiling phase."""
    
    def test_profile_single_class(self, compiler):
        """Test profiling a single class."""
        stats = compiler.profile(0)
        
        assert isinstance(stats, ProfileStats)
        assert stats.class_id == 0
        assert stats.total_samples == 1000
        assert stats.mode_tile == 2
        assert abs(stats.mode_frequency - 0.9) < 0.01
    
    def test_profile_compilability_score(self, compiler):
        """Test compilability score calculation."""
        # Class 0: mode_freq=0.9, high purity
        stats0 = compiler.profile(0)
        assert stats0.compilability > 0.5
        
        # Class 3: scattered, low compilability
        stats3 = compiler.profile(3)
        assert stats3.compilability < 0.3
        
        # Class 4: very focused
        stats4 = compiler.profile(4)
        assert stats4.compilability > 0.8
    
    def test_profile_all_classes(self, compiler):
        """Test profiling all classes at once."""
        profiles = compiler.profile_all(num_classes=5)
        
        assert len(profiles) == 5
        assert all(isinstance(p, ProfileStats) for p in profiles.values())
    
    def test_is_compilable(self, compiler):
        """Test compilability threshold."""
        stats0 = compiler.profile(0)
        stats3 = compiler.profile(3)
        
        assert stats0.is_compilable(threshold=0.5)
        assert not stats3.is_compilable(threshold=0.5)


# =============================================================================
# Compilation Tests
# =============================================================================

class TestCompilation:
    """Test the compilation phase."""
    
    def test_compile_manual(self, compiler):
        """Test manual compilation."""
        entry = compiler.compile(class_id=0, tile_idx=2, frequency=0.9, purity=0.8)
        
        assert isinstance(entry, CompiledEntry)
        assert entry.tile_idx == 2
        assert 0 in compiler.dispatch
    
    def test_compile_from_profile(self, compiler):
        """Test compilation from profile stats."""
        stats = compiler.profile(0)
        entry = compiler.compile_from_profile(stats)
        
        assert entry is not None
        assert entry.tile_idx == stats.mode_tile
        assert 0 in compiler.dispatch
    
    def test_compile_stable_classes(self, compiler):
        """Test automatic compilation of stable classes."""
        compiled = compiler.compile_stable(threshold=0.5)
        
        # Classes 0, 1, 2, 4 should be compiled (above threshold)
        # Class 3 should NOT be compiled (scattered)
        assert 0 in compiled
        assert 1 in compiled
        assert 4 in compiled
        assert 3 not in compiled
    
    def test_decompile(self, compiler):
        """Test removing a compiled entry."""
        compiler.compile(class_id=0, tile_idx=2)
        assert 0 in compiler.dispatch
        
        compiler.decompile(0)
        assert 0 not in compiler.dispatch
    
    def test_version_increment(self, compiler):
        """Test version increments on compilation."""
        v0 = compiler.version
        
        compiler.compile_stable()
        v1 = compiler.version
        
        assert v1 == v0 + 1


# =============================================================================
# Execution Tests
# =============================================================================

class TestExecution:
    """Test the execution phase."""
    
    def test_compiled_execution(self, compiler, trained_ffn, device):
        """Test that compiled execution bypasses routing."""
        compiler.compile(class_id=0, tile_idx=2)
        
        x = torch.randn(2, 16, 64, device=device)
        output, info, aux = compiler.forward(x, class_hint=0, confidence=1.0)
        
        assert output.shape == x.shape
        assert info['compiled'] == True
        assert info['compiled_class'] == 0
    
    def test_dynamic_execution_no_hint(self, compiler, trained_ffn, device):
        """Test dynamic execution when no class hint provided."""
        compiler.compile(class_id=0, tile_idx=2)
        
        x = torch.randn(2, 16, 64, device=device)
        output, info, aux = compiler.forward(x, class_hint=None)
        
        assert output.shape == x.shape
        assert info['compiled'] == False
    
    def test_dynamic_execution_unknown_class(self, compiler, trained_ffn, device):
        """Test dynamic execution for unknown class."""
        compiler.compile(class_id=0, tile_idx=2)
        
        x = torch.randn(2, 16, 64, device=device)
        output, info, aux = compiler.forward(x, class_hint=99)  # Unknown class
        
        assert output.shape == x.shape
        assert info['compiled'] == False
    
    def test_guard_threshold(self, compiler, trained_ffn, device):
        """Test that low confidence triggers dynamic routing."""
        compiler.compile(class_id=0, tile_idx=2, min_confidence=0.8)
        
        x = torch.randn(2, 16, 64, device=device)
        
        # High confidence → compiled
        output1, info1, _ = compiler.forward(x, class_hint=0, confidence=0.9)
        assert info1['compiled'] == True
        
        # Low confidence → dynamic
        output2, info2, _ = compiler.forward(x, class_hint=0, confidence=0.5)
        assert info2['compiled'] == False


# =============================================================================
# Monitoring Tests
# =============================================================================

class TestMonitoring:
    """Test the monitoring phase."""
    
    def test_stats_tracking(self, compiler, trained_ffn, device):
        """Test that execution stats are tracked."""
        compiler.compile(class_id=0, tile_idx=2)
        compiler.reset_stats()
        
        x = torch.randn(2, 16, 64, device=device)
        
        # Compiled hits
        compiler.forward(x, class_hint=0, confidence=1.0)
        compiler.forward(x, class_hint=0, confidence=1.0)
        
        # Dynamic calls
        compiler.forward(x, class_hint=None)
        
        stats = compiler.get_stats()
        
        assert stats['compiled_hits'] == 2
        assert stats['dynamic_calls'] == 1
        assert stats['hit_rate'] == 2/3
    
    def test_class_stats(self, compiler, trained_ffn, device):
        """Test per-class statistics."""
        compiler.compile(class_id=0, tile_idx=2)
        
        x = torch.randn(2, 16, 64, device=device)
        compiler.forward(x, class_hint=0, confidence=1.0)
        
        class_stats = compiler.get_class_stats(0)
        
        assert class_stats['compiled'] == True
        assert class_stats['tile_idx'] == 2
        assert class_stats['hits'] == 1
    
    def test_drift_detection(self, compiler, trained_ffn, device):
        """Test drift detection."""
        # Compile based on current state
        compiler.compile(class_id=0, tile_idx=2, frequency=0.9)
        
        # Simulate drift: change the claim matrix
        trained_ffn.claim_matrix[2, 0] = 100  # Was 900
        trained_ffn.claim_matrix[3, 0] = 800  # Now tile 3 dominates
        
        drifted = compiler.check_drift(threshold=0.3)
        
        assert 0 in drifted


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Test export/import of dispatch tables."""
    
    def test_export_dispatch_table(self, compiler):
        """Test exporting dispatch table."""
        compiler.compile(class_id=0, tile_idx=2, frequency=0.9, purity=0.8)
        compiler.compile(class_id=1, tile_idx=5, frequency=0.85, purity=0.7)
        
        exported = compiler.export_dispatch_table()
        
        assert 'version' in exported
        assert 'entries' in exported
        assert '0' in exported['entries']
        assert '1' in exported['entries']
        assert exported['entries']['0']['tile_idx'] == 2
    
    def test_import_dispatch_table(self, compiler):
        """Test importing dispatch table."""
        data = {
            'version': 5,
            'confidence_threshold': 0.6,
            'entries': {
                '0': {'tile_idx': 2, 'frequency': 0.9, 'purity': 0.8, 'min_confidence': 0.5},
                '1': {'tile_idx': 5, 'frequency': 0.85, 'purity': 0.7, 'min_confidence': 0.5},
            }
        }
        
        compiler.import_dispatch_table(data)
        
        assert compiler.version == 5
        assert 0 in compiler.dispatch
        assert 1 in compiler.dispatch
        assert compiler.dispatch[0].tile_idx == 2
    
    def test_roundtrip(self, compiler):
        """Test export → import roundtrip."""
        compiler.compile(class_id=0, tile_idx=2)
        compiler.compile(class_id=1, tile_idx=5)
        
        exported = compiler.export_dispatch_table()
        
        # Create new compiler and import
        new_compiler = CompiledDispatch(compiler.ffn)
        new_compiler.import_dispatch_table(exported)
        
        assert new_compiler.dispatch[0].tile_idx == compiler.dispatch[0].tile_idx
        assert new_compiler.dispatch[1].tile_idx == compiler.dispatch[1].tile_idx


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full workflow."""
    
    def test_full_workflow(self, trained_ffn, device):
        """Test complete Train → Profile → Compile → Execute → Monitor."""
        # 1. Create compiler
        compiler = CompiledDispatch(trained_ffn)
        
        # 2. Profile all classes
        profiles = compiler.profile_all(num_classes=5)
        
        assert len(profiles) == 5
        
        # 3. Compile stable classes
        compiled = compiler.compile_stable(threshold=0.4)
        
        assert len(compiled) >= 3  # At least classes 0, 1, 4
        
        # 4. Execute with compiled paths
        x = torch.randn(4, 8, 64, device=device)
        
        # Compiled execution
        out1, info1, _ = compiler.forward(x, class_hint=0, confidence=1.0)
        assert info1['compiled'] == True
        
        # Dynamic execution
        out2, info2, _ = compiler.forward(x, class_hint=None)
        assert info2['compiled'] == False
        
        # 5. Check monitoring
        stats = compiler.get_stats()
        assert stats['compiled_hits'] >= 1
        assert stats['dynamic_calls'] >= 1
    
    def test_repr(self, compiler):
        """Test string representation."""
        compiler.compile(class_id=0, tile_idx=2)
        
        repr_str = repr(compiler)
        
        assert 'CompiledDispatch' in repr_str
        assert 'compiled_classes=1' in repr_str


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
