#!/usr/bin/env python3
"""
Hunt Logger - Async logging for Riemann Zero Hunter
=====================================================

Logs everything without blocking computation.
Every zero found, every batch processed, every anomaly detected.

"We need to log everything or we are going to be laughed out of reality."
"""

import os
import json
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import hashlib


# =============================================================================
# LOG ENTRY TYPES
# =============================================================================

@dataclass
class BatchEntry:
    """Log entry for a processed batch."""
    timestamp: str
    batch_id: int
    t_start: float
    t_end: float
    num_points: int
    zeros_found: int
    expected_zeros: float
    accuracy_pct: float
    elapsed_sec: float
    zeros_per_sec: float
    M_terms: int
    checksum: str  # Hash of Z values for reproducibility


@dataclass
class ZeroEntry:
    """Log entry for a detected zero."""
    timestamp: str
    batch_id: int
    zero_index: int
    t_lower: float
    t_upper: float
    t_estimate: float
    Z_lower: float
    Z_upper: float
    gram_point: Optional[int]


@dataclass 
class CheckpointEntry:
    """Log entry for verification checkpoint."""
    timestamp: str
    t_checkpoint: float
    total_zeros_found: int
    expected_by_rvm: float  # Riemann-von Mangoldt
    deviation_pct: float
    elapsed_total_sec: float
    projected_completion_sec: float


@dataclass
class AnomalyEntry:
    """Log entry for anomalies (Gram violations, etc.)."""
    timestamp: str
    anomaly_type: str
    t_location: float
    description: str
    data: Dict[str, Any]


# =============================================================================
# ASYNC LOGGER
# =============================================================================

class HuntLogger:
    """
    Asynchronous logger for the Riemann Zero Hunt.
    
    - Non-blocking writes (async queue)
    - Structured JSON logs
    - Automatic rotation
    - Checksums for verification
    """
    
    def __init__(self, log_dir: str = "hunt_logs", run_name: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Run identifier
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name
        self.run_dir = self.log_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.batch_file = open(self.run_dir / "batches.jsonl", "a")
        self.zeros_file = open(self.run_dir / "zeros.jsonl", "a")
        self.checkpoints_file = open(self.run_dir / "checkpoints.jsonl", "a")
        self.anomalies_file = open(self.run_dir / "anomalies.jsonl", "a")
        
        # Async queue
        self.queue = queue.Queue()
        self.running = True
        
        # Writer thread
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        
        # Stats
        self.total_zeros = 0
        self.total_batches = 0
        self.start_time = time.time()
        
        # Write run metadata
        self._write_metadata()
    
    def _write_metadata(self):
        """Write run metadata."""
        metadata = {
            "run_name": self.run_name,
            "start_time": datetime.now().isoformat(),
            "version": "1.0.0",
            "target": "10^16 zeros",
        }
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _writer_loop(self):
        """Background thread that writes logs."""
        while self.running or not self.queue.empty():
            try:
                entry_type, entry = self.queue.get(timeout=0.1)
                
                if entry_type == "batch":
                    self.batch_file.write(json.dumps(asdict(entry)) + "\n")
                    self.batch_file.flush()
                elif entry_type == "zero":
                    self.zeros_file.write(json.dumps(asdict(entry)) + "\n")
                elif entry_type == "checkpoint":
                    self.checkpoints_file.write(json.dumps(asdict(entry)) + "\n")
                    self.checkpoints_file.flush()
                elif entry_type == "anomaly":
                    self.anomalies_file.write(json.dumps(asdict(entry)) + "\n")
                    self.anomalies_file.flush()
                
                self.queue.task_done()
            except queue.Empty:
                continue
    
    def log_batch(self, t_start: float, t_end: float, num_points: int,
                  zeros_found: int, elapsed: float, M_terms: int,
                  Z_values=None):
        """Log a processed batch (non-blocking)."""
        self.total_batches += 1
        self.total_zeros += zeros_found
        
        expected = (t_end - t_start) * (2.2 + 0.1 * (t_start / 1e6))  # Rough estimate
        
        # Checksum of Z values for reproducibility
        checksum = ""
        if Z_values is not None:
            try:
                import torch
                if hasattr(Z_values, 'cpu'):
                    Z_bytes = Z_values.cpu().numpy().tobytes()
                else:
                    Z_bytes = Z_values.tobytes()
                checksum = hashlib.md5(Z_bytes).hexdigest()[:16]
            except:
                checksum = "unavailable"
        
        entry = BatchEntry(
            timestamp=datetime.now().isoformat(),
            batch_id=self.total_batches,
            t_start=t_start,
            t_end=t_end,
            num_points=num_points,
            zeros_found=zeros_found,
            expected_zeros=expected,
            accuracy_pct=zeros_found / max(expected, 1) * 100,
            elapsed_sec=elapsed,
            zeros_per_sec=zeros_found / max(elapsed, 0.001),
            M_terms=M_terms,
            checksum=checksum,
        )
        
        self.queue.put(("batch", entry))
    
    def log_zero(self, batch_id: int, zero_index: int, 
                 t_lower: float, t_upper: float,
                 Z_lower: float, Z_upper: float,
                 gram_point: Optional[int] = None):
        """Log a detected zero (non-blocking)."""
        entry = ZeroEntry(
            timestamp=datetime.now().isoformat(),
            batch_id=batch_id,
            zero_index=zero_index,
            t_lower=t_lower,
            t_upper=t_upper,
            t_estimate=(t_lower + t_upper) / 2,
            Z_lower=Z_lower,
            Z_upper=Z_upper,
            gram_point=gram_point,
        )
        
        self.queue.put(("zero", entry))
    
    def log_checkpoint(self, t_checkpoint: float, target_zeros: float):
        """Log a verification checkpoint."""
        elapsed = time.time() - self.start_time
        
        # Riemann-von Mangoldt expected count
        import math
        T = t_checkpoint
        expected_rvm = T / (2 * math.pi) * math.log(T / (2 * math.pi)) - T / (2 * math.pi)
        
        deviation = abs(self.total_zeros - expected_rvm) / max(expected_rvm, 1) * 100
        
        # Project completion
        if self.total_zeros > 0:
            rate = self.total_zeros / elapsed
            remaining = target_zeros - self.total_zeros
            projected = elapsed + remaining / rate
        else:
            projected = float('inf')
        
        entry = CheckpointEntry(
            timestamp=datetime.now().isoformat(),
            t_checkpoint=t_checkpoint,
            total_zeros_found=self.total_zeros,
            expected_by_rvm=expected_rvm,
            deviation_pct=deviation,
            elapsed_total_sec=elapsed,
            projected_completion_sec=projected,
        )
        
        self.queue.put(("checkpoint", entry))
        
        # Also print to console
        print(f"\n{'='*60}")
        print(f"CHECKPOINT @ t = {t_checkpoint:.2e}")
        print(f"  Zeros found: {self.total_zeros:,}")
        print(f"  Expected (RvM): {expected_rvm:,.0f}")
        print(f"  Deviation: {deviation:.2f}%")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Rate: {self.total_zeros/elapsed:,.0f} zeros/sec")
        if projected < float('inf'):
            print(f"  Projected completion: {projected/3600:.1f} hours")
        print(f"{'='*60}\n")
    
    def log_anomaly(self, anomaly_type: str, t_location: float,
                    description: str, data: Dict[str, Any] = None):
        """Log an anomaly (non-blocking)."""
        entry = AnomalyEntry(
            timestamp=datetime.now().isoformat(),
            anomaly_type=anomaly_type,
            t_location=t_location,
            description=description,
            data=data or {},
        )
        
        self.queue.put(("anomaly", entry))
        
        # Anomalies also print immediately
        print(f"[ANOMALY] {anomaly_type} @ t={t_location:.6f}: {description}")
    
    def close(self):
        """Close logger and flush all pending writes."""
        self.running = False
        self.writer_thread.join(timeout=5.0)
        
        # Flush remaining
        while not self.queue.empty():
            try:
                entry_type, entry = self.queue.get_nowait()
                if entry_type == "batch":
                    self.batch_file.write(json.dumps(asdict(entry)) + "\n")
                elif entry_type == "zero":
                    self.zeros_file.write(json.dumps(asdict(entry)) + "\n")
                elif entry_type == "checkpoint":
                    self.checkpoints_file.write(json.dumps(asdict(entry)) + "\n")
                elif entry_type == "anomaly":
                    self.anomalies_file.write(json.dumps(asdict(entry)) + "\n")
            except queue.Empty:
                break
        
        # Close files
        self.batch_file.close()
        self.zeros_file.close()
        self.checkpoints_file.close()
        self.anomalies_file.close()
        
        # Write final summary
        self._write_summary()
    
    def _write_summary(self):
        """Write final run summary."""
        elapsed = time.time() - self.start_time
        
        summary = {
            "run_name": self.run_name,
            "end_time": datetime.now().isoformat(),
            "total_zeros": self.total_zeros,
            "total_batches": self.total_batches,
            "elapsed_seconds": elapsed,
            "zeros_per_second": self.total_zeros / max(elapsed, 1),
        }
        
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("HUNT COMPLETE")
        print(f"  Total zeros: {self.total_zeros:,}")
        print(f"  Total time: {elapsed:.1f}s ({elapsed/3600:.2f} hours)")
        print(f"  Average rate: {self.total_zeros/max(elapsed,1):,.0f} zeros/sec")
        print(f"  Logs saved to: {self.run_dir}")
        print(f"{'='*60}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current stats."""
        elapsed = time.time() - self.start_time
        return {
            "total_zeros": self.total_zeros,
            "total_batches": self.total_batches,
            "elapsed_sec": elapsed,
            "zeros_per_sec": self.total_zeros / max(elapsed, 1),
            "pending_logs": self.queue.qsize(),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_global_logger: Optional[HuntLogger] = None

def init_logger(log_dir: str = "hunt_logs", run_name: Optional[str] = None) -> HuntLogger:
    """Initialize global logger."""
    global _global_logger
    _global_logger = HuntLogger(log_dir, run_name)
    return _global_logger

def get_logger() -> Optional[HuntLogger]:
    """Get global logger."""
    return _global_logger

def close_logger():
    """Close global logger."""
    global _global_logger
    if _global_logger:
        _global_logger.close()
        _global_logger = None


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Hunt Logger...")
    
    logger = init_logger(run_name="test_run")
    
    # Simulate some batches
    for i in range(10):
        t_start = 1e6 + i * 1000
        logger.log_batch(
            t_start=t_start,
            t_end=t_start + 1000,
            num_points=10000,
            zeros_found=850 + i * 10,
            elapsed=0.1,
            M_terms=400,
        )
        time.sleep(0.05)
    
    # Log a checkpoint
    logger.log_checkpoint(t_checkpoint=1e6 + 10000, target_zeros=1e9)
    
    # Log an anomaly
    logger.log_anomaly(
        anomaly_type="gram_violation",
        t_location=1000500.123,
        description="Gram's law violated: 2 zeros between consecutive Gram points",
        data={"gram_n": 12345, "zeros_in_interval": 2}
    )
    
    # Close
    close_logger()
    
    print("\nLog files created in hunt_logs/test_run/")
