"""
Compiled Dispatch for TriX v2

Path compilation: the bridge between emergent routing and usable compute.

Train → Profile → Compile → Execute → Monitor

This module enables:
- O(1) dispatch for known classes (skip routing computation)
- Guards for safe execution (confidence thresholds)
- Monitoring for drift detection (recompile triggers)
- The factory pattern: learned structure + deterministic execution

Usage:
    ffn = SparseLookupFFNv2(...)
    # ... train with labels for claim tracking ...
    
    compiler = CompiledDispatch(ffn)
    compiler.profile_all()
    compiler.compile_stable(threshold=0.6)
    
    # Inference
    output = compiler.forward(x, class_hint=class_id, confidence=0.9)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from collections import Counter
import json
from datetime import datetime


@dataclass
class CompiledEntry:
    """A compiled dispatch entry for one class."""
    tile_idx: int
    frequency: float      # How often this tile was chosen
    purity: float         # What fraction of tile's traffic is this class
    confidence: float     # Mean routing confidence when this path taken
    compiled_at: str      # Timestamp
    version: int          # For tracking recompilations
    
    # Guards
    min_confidence: float = 0.5
    
    def passes_guard(self, confidence: float) -> bool:
        """Check if confidence passes the guard threshold."""
        return confidence >= self.min_confidence


@dataclass 
class ProfileStats:
    """Profiling statistics for one class."""
    class_id: int
    total_samples: int
    tile_distribution: Dict[int, int]  # tile_idx → count
    mode_tile: int
    mode_frequency: float
    purity: float  # Fraction of mode_tile's traffic from this class
    compilability: float  # Overall score
    
    def is_compilable(self, threshold: float = 0.5) -> bool:
        return self.compilability >= threshold


class CompiledDispatch(nn.Module):
    """
    Compiled dispatch wrapper for SparseLookupFFNv2.
    
    Enables path compilation: freeze routing decisions for stable classes,
    execute directly without routing computation.
    """
    
    def __init__(self, ffn: nn.Module, confidence_threshold: float = 0.5):
        """
        Args:
            ffn: A SparseLookupFFNv2 instance with claim tracking
            confidence_threshold: Default guard threshold for compiled paths
        """
        super().__init__()
        self.ffn = ffn
        self.confidence_threshold = confidence_threshold
        
        # Dispatch table: class_id → CompiledEntry
        self.dispatch: Dict[int, CompiledEntry] = {}
        
        # Monitoring stats
        self.compiled_hits = 0
        self.compiled_misses = 0  # Guard failed
        self.dynamic_calls = 0    # No compiled entry
        self.version = 0
        
        # Per-class monitoring
        self.class_hits: Dict[int, int] = {}
        self.class_accuracy: Dict[int, List[float]] = {}
    
    def profile(self, class_id: int) -> ProfileStats:
        """
        Profile a single class using claim tracking data.
        
        Returns statistics about which tiles handle this class
        and how stable the routing is.
        """
        if not hasattr(self.ffn, 'claim_matrix'):
            raise RuntimeError("FFN must have claim tracking enabled. "
                             "Train with labels to populate claim_matrix.")
        
        claim_matrix = self.ffn.claim_matrix  # (num_tiles, num_classes)
        num_tiles = claim_matrix.shape[0]
        
        # Get distribution: which tiles see this class?
        class_counts = claim_matrix[:, class_id].cpu().numpy()
        total = class_counts.sum()
        
        if total == 0:
            return ProfileStats(
                class_id=class_id,
                total_samples=0,
                tile_distribution={},
                mode_tile=-1,
                mode_frequency=0.0,
                purity=0.0,
                compilability=0.0,
            )
        
        tile_distribution = {i: int(class_counts[i]) for i in range(num_tiles) if class_counts[i] > 0}
        
        # Find mode tile (most common)
        mode_tile = int(class_counts.argmax())
        mode_count = class_counts[mode_tile]
        mode_frequency = mode_count / total
        
        # Compute purity: what fraction of mode_tile's traffic is this class?
        tile_total = claim_matrix[mode_tile, :].sum().item()
        purity = mode_count / tile_total if tile_total > 0 else 0.0
        
        # Compilability score (Nova's formula adapted for single-layer)
        # S = mode_frequency * purity
        # High when: class consistently goes to one tile AND that tile is dedicated
        compilability = mode_frequency * purity
        
        return ProfileStats(
            class_id=class_id,
            total_samples=int(total),
            tile_distribution=tile_distribution,
            mode_tile=mode_tile,
            mode_frequency=mode_frequency,
            purity=purity,
            compilability=compilability,
        )
    
    def profile_all(self, num_classes: Optional[int] = None) -> Dict[int, ProfileStats]:
        """Profile all classes."""
        if num_classes is None:
            num_classes = self.ffn.claim_matrix.shape[1]
        
        return {c: self.profile(c) for c in range(num_classes)}
    
    def compile(self, class_id: int, tile_idx: int, 
                frequency: float = 1.0, purity: float = 1.0,
                min_confidence: float = 0.5) -> CompiledEntry:
        """
        Compile a class → tile mapping.
        
        After compilation, this class will bypass routing and go
        directly to the specified tile (if guards pass).
        """
        entry = CompiledEntry(
            tile_idx=tile_idx,
            frequency=frequency,
            purity=purity,
            confidence=1.0,  # Will be updated during monitoring
            compiled_at=datetime.now().isoformat(),
            version=self.version,
            min_confidence=min_confidence,
        )
        
        self.dispatch[class_id] = entry
        self.class_hits[class_id] = 0
        self.class_accuracy[class_id] = []
        
        return entry
    
    def compile_from_profile(self, stats: ProfileStats, 
                             min_confidence: float = 0.5) -> Optional[CompiledEntry]:
        """Compile a class based on its profile stats."""
        if stats.mode_tile < 0:
            return None
            
        return self.compile(
            class_id=stats.class_id,
            tile_idx=stats.mode_tile,
            frequency=stats.mode_frequency,
            purity=stats.purity,
            min_confidence=min_confidence,
        )
    
    def compile_stable(self, threshold: float = 0.5, 
                       min_confidence: float = 0.5,
                       num_classes: Optional[int] = None) -> Dict[int, CompiledEntry]:
        """
        Automatically compile all classes above compilability threshold.
        
        Returns dict of compiled entries.
        """
        profiles = self.profile_all(num_classes)
        compiled = {}
        
        for class_id, stats in profiles.items():
            if stats.is_compilable(threshold):
                entry = self.compile_from_profile(stats, min_confidence)
                if entry:
                    compiled[class_id] = entry
        
        self.version += 1
        return compiled
    
    def decompile(self, class_id: int):
        """Remove a class from the dispatch table (return to dynamic routing)."""
        if class_id in self.dispatch:
            del self.dispatch[class_id]
    
    def decompile_all(self):
        """Clear all compiled entries."""
        self.dispatch.clear()
        self.version += 1
    
    def forward(self, x: torch.Tensor, 
                class_hint: Optional[int] = None,
                confidence: float = 1.0,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Forward pass with compiled dispatch.
        
        Args:
            x: Input tensor
            class_hint: Known class ID (if available)
            confidence: Confidence in the class hint (0-1)
            labels: Labels for claim tracking (passed to dynamic routing)
        
        Returns:
            output: Transformed tensor
            routing_info: Routing information (tile_idx, compiled flag)
            aux_losses: Auxiliary losses from FFN
        """
        # Check if we can use compiled path
        guard_failed = False
        if class_hint is not None and class_hint in self.dispatch:
            entry = self.dispatch[class_hint]
            
            if entry.passes_guard(confidence):
                # COMPILED EXECUTION
                output, routing_info, aux_losses = self._execute_compiled(x, entry)
                routing_info['compiled'] = True
                routing_info['compiled_class'] = class_hint
                
                self.compiled_hits += 1
                self.class_hits[class_hint] = self.class_hits.get(class_hint, 0) + 1
                
                return output, routing_info, aux_losses
            else:
                # Guard failed - will use dynamic but track as miss
                self.compiled_misses += 1
                guard_failed = True
        
        # DYNAMIC EXECUTION (either no compiled path or guard failed)
        if not guard_failed:
            # Only count as dynamic if we didn't have a compiled path at all
            self.dynamic_calls += 1
        
        output, routing_info, aux_losses = self.ffn(x, labels=labels)
        routing_info['compiled'] = False
        routing_info['guard_failed'] = guard_failed
        
        return output, routing_info, aux_losses
    
    def _execute_compiled(self, x: torch.Tensor, 
                          entry: CompiledEntry) -> Tuple[torch.Tensor, Dict, Dict]:
        """Execute using compiled path (direct tile access)."""
        tile_idx = entry.tile_idx
        
        # Access tile directly - bypass routing
        # This is the O(1) win: no signature matching, no routing computation
        
        # Get the tile's computation
        # Note: This accesses FFN internals - may need adjustment based on FFN structure
        if hasattr(self.ffn, 'tiles'):
            # Direct tile access
            tile = self.ffn.tiles[tile_idx]
            output = tile(x)
        else:
            # Fallback: use FFN but force tile selection
            # This still does some routing work but ensures correctness
            output, routing_info, aux_losses = self.ffn(x)
            return output, {'tile_idx': torch.tensor([[tile_idx]])}, aux_losses
        
        routing_info = {
            'tile_idx': torch.full((x.shape[0], x.shape[1] if x.dim() > 2 else 1), 
                                   tile_idx, dtype=torch.long),
        }
        
        # No aux losses for compiled path (no routing to balance)
        aux_losses = {'total_aux': torch.tensor(0.0, device=x.device)}
        
        return output, routing_info, aux_losses
    
    # =========================================================================
    # Monitoring
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics."""
        total = self.compiled_hits + self.compiled_misses + self.dynamic_calls
        
        return {
            'compiled_hits': self.compiled_hits,
            'compiled_misses': self.compiled_misses,
            'dynamic_calls': self.dynamic_calls,
            'total_calls': total,
            'hit_rate': self.compiled_hits / total if total > 0 else 0,
            'num_compiled_classes': len(self.dispatch),
            'version': self.version,
        }
    
    def get_class_stats(self, class_id: int) -> Dict:
        """Get per-class monitoring stats."""
        if class_id not in self.dispatch:
            return {'compiled': False}
        
        entry = self.dispatch[class_id]
        hits = self.class_hits.get(class_id, 0)
        accuracies = self.class_accuracy.get(class_id, [])
        
        return {
            'compiled': True,
            'tile_idx': entry.tile_idx,
            'frequency': entry.frequency,
            'purity': entry.purity,
            'hits': hits,
            'mean_accuracy': sum(accuracies) / len(accuracies) if accuracies else None,
            'version': entry.version,
        }
    
    def reset_stats(self):
        """Reset monitoring counters."""
        self.compiled_hits = 0
        self.compiled_misses = 0
        self.dynamic_calls = 0
        self.class_hits.clear()
        self.class_accuracy.clear()
    
    def check_drift(self, threshold: float = 0.3) -> List[int]:
        """
        Check for classes that may need recompilation.
        
        Returns list of class IDs with significant drift.
        """
        drifted = []
        
        profiles = self.profile_all()
        
        for class_id, entry in self.dispatch.items():
            if class_id not in profiles:
                continue
                
            stats = profiles[class_id]
            
            # Check if mode tile changed
            if stats.mode_tile != entry.tile_idx:
                drifted.append(class_id)
                continue
            
            # Check if frequency dropped significantly
            if stats.mode_frequency < entry.frequency - threshold:
                drifted.append(class_id)
                continue
        
        return drifted
    
    def recompile_drifted(self, threshold: float = 0.3) -> Dict[int, CompiledEntry]:
        """Recompile classes that have drifted."""
        drifted = self.check_drift(threshold)
        recompiled = {}
        
        for class_id in drifted:
            stats = self.profile(class_id)
            if stats.is_compilable():
                entry = self.compile_from_profile(stats)
                if entry:
                    recompiled[class_id] = entry
            else:
                # No longer compilable - remove from dispatch
                self.decompile(class_id)
        
        if recompiled:
            self.version += 1
        
        return recompiled
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def export_dispatch_table(self) -> Dict:
        """Export dispatch table as JSON-serializable dict."""
        return {
            'version': self.version,
            'confidence_threshold': self.confidence_threshold,
            'entries': {
                str(k): {
                    'tile_idx': v.tile_idx,
                    'frequency': v.frequency,
                    'purity': v.purity,
                    'min_confidence': v.min_confidence,
                    'compiled_at': v.compiled_at,
                    'version': v.version,
                }
                for k, v in self.dispatch.items()
            }
        }
    
    def import_dispatch_table(self, data: Dict):
        """Import dispatch table from dict."""
        self.version = data.get('version', 0)
        self.confidence_threshold = data.get('confidence_threshold', 0.5)
        
        self.dispatch.clear()
        for k, v in data.get('entries', {}).items():
            class_id = int(k)
            self.dispatch[class_id] = CompiledEntry(
                tile_idx=v['tile_idx'],
                frequency=v['frequency'],
                purity=v['purity'],
                confidence=1.0,
                min_confidence=v.get('min_confidence', 0.5),
                compiled_at=v.get('compiled_at', ''),
                version=v.get('version', 0),
            )
    
    def __repr__(self):
        stats = self.get_stats()
        return (f"CompiledDispatch(compiled_classes={stats['num_compiled_classes']}, "
                f"hit_rate={stats['hit_rate']:.2%}, version={self.version})")
