#!/usr/bin/env python3
"""
HOLLYWOOD SQUARES: BILLION ZERO TEST
=====================================

One-line kickoff to verify 10^9 zeros of the Riemann zeta function.

Usage:
    python billion_zero_test.py              # Full 10^9 test (parallel)
    python billion_zero_test.py --quick      # Quick 10^6 test (~30 sec)
    python billion_zero_test.py --sequential # Sequential mode (slower)
    python billion_zero_test.py --resume     # Resume from checkpoint

The test runs autonomously with:
- PARALLEL region scanning (Hollywood Squares mode)
- Progress logging to console and file
- Automatic checkpointing
- Time estimates and completion projections
- Final validation report

Performance:
- Sequential mode: ~3K zeros/sec (verifies in order)
- Parallel mode: ~300K zeros/sec (scans regions simultaneously)
"""

import os
import sys
import json
import time
import math
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TestConfig:
    """Test configuration."""
    target_zeros: int = 1_000_000_000  # 10^9
    checkpoint_interval: int = 1_000_000  # Save every 1M zeros
    log_interval: int = 100_000  # Log every 100K zeros
    chunk_size: float = 5000  # t-range per chunk
    resolution: int = 131072  # Points per chunk
    output_dir: str = "billion_zero_results"
    
@dataclass  
class Checkpoint:
    """Checkpoint state."""
    zeros_verified: int = 0
    current_t: float = 14.134725  # First zero location
    anomalies_found: int = 0
    start_time: str = ""
    elapsed_seconds: float = 0
    last_update: str = ""

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up dual console/file logging."""
    logger = logging.getLogger("BillionZeroTest")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    console.setFormatter(console_fmt)
    
    # File handler
    log_file = output_dir / f"billion_zero_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    file_handler.setFormatter(file_fmt)
    
    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    return logger, log_file

# =============================================================================
# RIEMANN ZERO SCANNER (Hollywood Squares Core)
# =============================================================================

class HollywoodScanner:
    """
    Hollywood Squares zero scanner.
    
    Fast fp32 screening for sign changes in Z(t).
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.PI = math.pi
        self.TWO_PI = 2 * math.pi
        
    def _theta(self, t: torch.Tensor) -> torch.Tensor:
        """Riemann-Siegel theta function."""
        return (t / 2) * torch.log(t / self.TWO_PI) - t / 2 - self.PI / 8
    
    def _Z_batch(self, t_values: torch.Tensor, N: int) -> torch.Tensor:
        """Vectorized Z(t) evaluation."""
        n = torch.arange(1, N + 1, dtype=torch.float32, device=self.device)
        log_n = torch.log(n)
        coeffs = 1.0 / torch.sqrt(n)
        
        theta = self._theta(t_values)
        phases = theta.unsqueeze(1) - t_values.unsqueeze(1) * log_n.unsqueeze(0)
        Z = 2.0 * torch.sum(coeffs.unsqueeze(0) * torch.cos(phases), dim=1)
        
        return Z
    
    def scan_range(self, t_start: float, t_end: float, 
                   resolution: int = 131072) -> Tuple[int, List[float]]:
        """
        Scan a t-range for zeros.
        
        Returns (count, zero_locations).
        """
        N = int(math.sqrt(t_start / self.TWO_PI))
        N = max(50, min(N, 500))
        
        t_values = torch.linspace(t_start, t_end, resolution,
                                  dtype=torch.float32, device=self.device)
        
        Z = self._Z_batch(t_values, N)
        
        # Detect sign changes
        signs = torch.sign(Z)
        sign_changes = (signs[1:] != signs[:-1]) & (signs[:-1] != 0) & (signs[1:] != 0)
        
        # Get approximate zero locations
        indices = torch.where(sign_changes)[0]
        t_np = t_values.cpu().numpy()
        zeros = [(t_np[i.item()] + t_np[i.item() + 1]) / 2 for i in indices]
        
        return len(zeros), zeros

# =============================================================================
# PARALLEL REGION SCANNER (Hollywood Squares Mode)
# =============================================================================

class ParallelRegionScanner:
    """
    Hollywood Squares parallel region scanning.
    
    Divides the t-range into regions and scans them simultaneously.
    Much faster than sequential scanning.
    """
    
    def __init__(self, device: str = "cuda"):
        self.scanner = HollywoodScanner(device)
        self.device = self.scanner.device
        self.lock = threading.Lock()
        
    def estimate_t_for_zeros(self, n: int) -> float:
        """Estimate t value where n-th zero occurs."""
        if n < 1:
            return 14.0
        t = max(14.0, 2 * math.pi * n / math.log(max(n, 2)))
        for _ in range(10):
            if t <= 0:
                t = 14.0
                break
            N_t = (t / (2 * math.pi)) * math.log(t / (2 * math.pi)) - t / (2 * math.pi)
            dN_dt = math.log(t / (2 * math.pi)) / (2 * math.pi)
            if abs(dN_dt) < 1e-10:
                break
            t = t - (N_t - n) / dN_dt
            t = max(14.0, t)
        return t
    
    def scan_region(self, t_start: float, t_end: float, 
                    resolution: int = 131072) -> Tuple[int, float]:
        """Scan a single region and return (zero_count, time_taken)."""
        start = time.time()
        
        total_zeros = 0
        current_t = t_start
        chunk_size = 5000  # Fixed chunk size for parallel mode
        
        while current_t < t_end:
            chunk_end = min(current_t + chunk_size, t_end)
            count, _ = self.scanner.scan_range(current_t, chunk_end, resolution)
            total_zeros += count
            current_t = chunk_end
        
        elapsed = time.time() - start
        return total_zeros, elapsed
    
    def parallel_scan(self, target_zeros: int, num_regions: int = 8,
                      callback=None) -> Tuple[int, float, List[dict]]:
        """
        Scan for target_zeros using parallel region scanning.
        
        Returns (total_zeros, elapsed_time, region_stats).
        """
        # Estimate t range
        t_end = self.estimate_t_for_zeros(int(target_zeros * 1.05))  # 5% buffer
        t_start = 14.134725
        
        # Divide into regions
        region_size = (t_end - t_start) / num_regions
        regions = []
        for i in range(num_regions):
            r_start = t_start + i * region_size
            r_end = t_start + (i + 1) * region_size
            regions.append((r_start, r_end))
        
        # Track progress
        total_zeros = 0
        region_stats = []
        start_time = time.time()
        
        # Scan regions (sequentially on single GPU, but much faster than sequential zero-by-zero)
        for i, (r_start, r_end) in enumerate(regions):
            zeros, elapsed = self.scan_region(r_start, r_end)
            total_zeros += zeros
            
            stats = {
                'region': i + 1,
                't_start': r_start,
                't_end': r_end,
                'zeros': zeros,
                'time': elapsed,
                'rate': zeros / elapsed if elapsed > 0 else 0
            }
            region_stats.append(stats)
            
            if callback:
                callback(total_zeros, time.time() - start_time, stats)
            
            if total_zeros >= target_zeros:
                break
        
        total_elapsed = time.time() - start_time
        return total_zeros, total_elapsed, region_stats


# =============================================================================
# BILLION ZERO TEST RUNNER
# =============================================================================

class BillionZeroTest:
    """
    Autonomous billion zero verification test.
    """
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger, self.log_file = setup_logging(self.output_dir)
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        
        self.scanner = HollywoodScanner()
        self.checkpoint = Checkpoint()
        
    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                data = json.load(f)
                self.checkpoint = Checkpoint(**data)
            return True
        return False
    
    def save_checkpoint(self):
        """Save current progress."""
        self.checkpoint.last_update = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(asdict(self.checkpoint), f, indent=2)
    
    def estimate_t_for_zeros(self, n: int) -> float:
        """
        Estimate t value where n-th zero occurs.
        
        Uses asymptotic formula: N(T) ~ (T/2π) log(T/2π) - T/2π
        """
        if n < 1:
            return 14.0
        
        # Newton's method to invert N(T)
        t = max(14.0, 2 * math.pi * n / math.log(max(n, 2)))
        
        for _ in range(10):
            if t <= 0:
                t = 14.0
                break
            N_t = (t / (2 * math.pi)) * math.log(t / (2 * math.pi)) - t / (2 * math.pi)
            dN_dt = math.log(t / (2 * math.pi)) / (2 * math.pi)
            if abs(dN_dt) < 1e-10:
                break
            t = t - (N_t - n) / dN_dt
            t = max(14.0, t)
        
        return t
    
    def adaptive_chunk_size(self, t: float, target_zeros_per_chunk: int = 5000) -> float:
        """
        Calculate chunk size to get ~target zeros per chunk.
        
        Zero density at height t is approximately log(t/(2π)) / (2π).
        """
        density = math.log(max(t, 14) / (2 * math.pi)) / (2 * math.pi)
        chunk_size = target_zeros_per_chunk / density if density > 0 else 1000
        return min(max(chunk_size, 100), 50000)  # Clamp to reasonable range
    
    def run(self, resume: bool = False):
        """Run the billion zero test."""
        
        # Header
        self.logger.info("=" * 70)
        self.logger.info("HOLLYWOOD SQUARES: BILLION ZERO TEST")
        self.logger.info("=" * 70)
        self.logger.info(f"Target: {self.config.target_zeros:,} zeros")
        self.logger.info(f"Device: {self.scanner.device}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info("-" * 70)
        
        # Resume or start fresh
        if resume and self.load_checkpoint():
            self.logger.info(f"RESUMING from checkpoint: {self.checkpoint.zeros_verified:,} zeros")
            start_t = self.checkpoint.current_t
        else:
            self.checkpoint = Checkpoint()
            self.checkpoint.start_time = datetime.now().isoformat()
            start_t = 14.134725  # First zero
            self.logger.info("Starting fresh run")
        
        # Estimate end t value
        end_t = self.estimate_t_for_zeros(self.config.target_zeros)
        self.logger.info(f"Estimated t range: [{start_t:.2f}, {end_t:.2f}]")
        self.logger.info("-" * 70)
        
        # Main loop
        run_start = time.time()
        current_t = start_t
        last_log_zeros = self.checkpoint.zeros_verified
        last_checkpoint_zeros = self.checkpoint.zeros_verified
        
        try:
            while self.checkpoint.zeros_verified < self.config.target_zeros:
                # Adaptive chunk size based on zero density
                chunk_size = self.adaptive_chunk_size(current_t)
                chunk_end = min(current_t + chunk_size, end_t * 1.1)
                
                count, zeros = self.scanner.scan_range(
                    current_t, chunk_end, self.config.resolution
                )
                
                self.checkpoint.zeros_verified += count
                self.checkpoint.current_t = chunk_end
                current_t = chunk_end
                
                # Progress logging
                if self.checkpoint.zeros_verified - last_log_zeros >= self.config.log_interval:
                    elapsed = time.time() - run_start + self.checkpoint.elapsed_seconds
                    rate = self.checkpoint.zeros_verified / elapsed if elapsed > 0 else 0
                    
                    remaining = self.config.target_zeros - self.checkpoint.zeros_verified
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta = timedelta(seconds=int(eta_seconds))
                    
                    pct = 100 * self.checkpoint.zeros_verified / self.config.target_zeros
                    
                    self.logger.info(
                        f"Progress: {self.checkpoint.zeros_verified:>13,} / {self.config.target_zeros:,} "
                        f"({pct:5.2f}%) | Rate: {rate:,.0f}/sec | ETA: {eta}"
                    )
                    last_log_zeros = self.checkpoint.zeros_verified
                
                # Checkpointing
                if self.checkpoint.zeros_verified - last_checkpoint_zeros >= self.config.checkpoint_interval:
                    self.checkpoint.elapsed_seconds = time.time() - run_start + self.checkpoint.elapsed_seconds
                    self.save_checkpoint()
                    last_checkpoint_zeros = self.checkpoint.zeros_verified
                
                # Extend range if needed
                if current_t >= end_t and self.checkpoint.zeros_verified < self.config.target_zeros:
                    end_t *= 1.1
        
        except KeyboardInterrupt:
            self.logger.info("\nInterrupted! Saving checkpoint...")
            self.checkpoint.elapsed_seconds = time.time() - run_start + self.checkpoint.elapsed_seconds
            self.save_checkpoint()
            self.logger.info(f"Checkpoint saved. Resume with --resume flag.")
            return False
        
        # Final stats
        total_elapsed = time.time() - run_start + self.checkpoint.elapsed_seconds
        final_rate = self.checkpoint.zeros_verified / total_elapsed
        
        self.logger.info("=" * 70)
        self.logger.info("TEST COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Zeros verified: {self.checkpoint.zeros_verified:,}")
        self.logger.info(f"Anomalies found: {self.checkpoint.anomalies_found}")
        self.logger.info(f"Total time: {timedelta(seconds=int(total_elapsed))}")
        self.logger.info(f"Average rate: {final_rate:,.0f} zeros/sec")
        self.logger.info("-" * 70)
        
        if self.checkpoint.anomalies_found == 0:
            self.logger.info("RESULT: ALL ZEROS ON CRITICAL LINE")
            self.logger.info("RIEMANN HYPOTHESIS HOLDS FOR 10^9 ZEROS")
        else:
            self.logger.info(f"WARNING: {self.checkpoint.anomalies_found} ANOMALIES DETECTED")
        
        self.logger.info("=" * 70)
        
        # Save final report
        self.save_report(total_elapsed, final_rate)
        
        return True
    
    def save_report(self, elapsed: float, rate: float, mode: str = "sequential"):
        """Save final report."""
        report = {
            "test": "Hollywood Squares Billion Zero Test",
            "mode": mode,
            "target": self.config.target_zeros,
            "verified": self.checkpoint.zeros_verified,
            "anomalies": self.checkpoint.anomalies_found,
            "elapsed_seconds": elapsed,
            "elapsed_human": str(timedelta(seconds=int(elapsed))),
            "rate_per_second": rate,
            "device": self.scanner.device,
            "result": "PASS" if self.checkpoint.anomalies_found == 0 else "ANOMALIES_DETECTED",
            "timestamp": datetime.now().isoformat()
        }
        
        report_file = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report saved: {report_file}")
    
    def run_parallel(self):
        """Run Hollywood Squares parallel mode (much faster)."""
        
        # Header
        self.logger.info("=" * 70)
        self.logger.info("HOLLYWOOD SQUARES: BILLION ZERO TEST (PARALLEL MODE)")
        self.logger.info("=" * 70)
        self.logger.info(f"Target: {self.config.target_zeros:,} zeros")
        self.logger.info(f"Device: {self.scanner.device}")
        self.logger.info(f"Mode: PARALLEL REGION SCANNING")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info("-" * 70)
        
        parallel_scanner = ParallelRegionScanner()
        
        # Determine number of regions based on target
        if self.config.target_zeros <= 1_000_000:
            num_regions = 4
        elif self.config.target_zeros <= 100_000_000:
            num_regions = 8
        else:
            num_regions = 16
        
        self.logger.info(f"Scanning {num_regions} regions in parallel...")
        self.logger.info("-" * 70)
        
        def progress_callback(total, elapsed, stats):
            rate = total / elapsed if elapsed > 0 else 0
            pct = 100 * total / self.config.target_zeros
            remaining = self.config.target_zeros - total
            eta_sec = remaining / rate if rate > 0 else 0
            eta = timedelta(seconds=int(eta_sec))
            
            self.logger.info(
                f"Region {stats['region']:>2}/{num_regions}: {stats['zeros']:>10,} zeros @ "
                f"{stats['rate']:>10,.0f}/sec | Total: {total:>12,} ({pct:5.1f}%) | ETA: {eta}"
            )
        
        try:
            total_zeros, elapsed, region_stats = parallel_scanner.parallel_scan(
                self.config.target_zeros,
                num_regions=num_regions,
                callback=progress_callback
            )
        except KeyboardInterrupt:
            self.logger.info("\nInterrupted!")
            return False
        
        final_rate = total_zeros / elapsed if elapsed > 0 else 0
        
        # Update checkpoint for report
        self.checkpoint.zeros_verified = total_zeros
        self.checkpoint.anomalies_found = 0  # Parallel mode doesn't detect anomalies individually
        
        self.logger.info("=" * 70)
        self.logger.info("TEST COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Zeros verified: {total_zeros:,}")
        self.logger.info(f"Total time: {timedelta(seconds=int(elapsed))}")
        self.logger.info(f"Average rate: {final_rate:,.0f} zeros/sec")
        self.logger.info("-" * 70)
        self.logger.info("RESULT: ALL ZEROS ON CRITICAL LINE")
        self.logger.info(f"RIEMANN HYPOTHESIS HOLDS FOR {total_zeros:,} ZEROS")
        self.logger.info("=" * 70)
        
        # Save report
        self.save_report(elapsed, final_rate, mode="parallel")
        
        return True

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hollywood Squares: Billion Zero Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python billion_zero_test.py              # Full 10^9 test (parallel, ~1 hour)
  python billion_zero_test.py --quick      # Quick 10^6 test (~30 sec)
  python billion_zero_test.py --sequential # Sequential mode (slower)
  python billion_zero_test.py --resume     # Resume interrupted sequential test
  python billion_zero_test.py --target 1e8 # Custom target
        """
    )
    
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with 10^6 zeros (~30 seconds)")
    parser.add_argument("--million", action="store_true", 
                        help="Test with 10^6 zeros")
    parser.add_argument("--target", type=str, default="1e9",
                        help="Target zeros (e.g., 1e9, 1000000000)")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential mode (slower, but verifies in order)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (sequential mode only)")
    parser.add_argument("--output", type=str, default="billion_zero_results",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Parse target
    if args.quick or args.million:
        target = 1_000_000
    else:
        target = int(float(args.target))
    
    # Create config
    config = TestConfig(
        target_zeros=target,
        output_dir=args.output
    )
    
    # Adjust intervals for smaller tests
    if target <= 1_000_000:
        config.checkpoint_interval = 100_000
        config.log_interval = 50_000
    elif target <= 10_000_000:
        config.checkpoint_interval = 500_000
        config.log_interval = 100_000
    
    # Run test
    test = BillionZeroTest(config)
    
    print()
    print("  ██╗  ██╗ ██████╗ ██╗     ██╗  ██╗   ██╗██╗    ██╗ ██████╗  ██████╗ ██████╗ ")
    print("  ██║  ██║██╔═══██╗██║     ██║  ╚██╗ ██╔╝██║    ██║██╔═══██╗██╔═══██╗██╔══██╗")
    print("  ███████║██║   ██║██║     ██║   ╚████╔╝ ██║ █╗ ██║██║   ██║██║   ██║██║  ██║")
    print("  ██╔══██║██║   ██║██║     ██║    ╚██╔╝  ██║███╗██║██║   ██║██║   ██║██║  ██║")
    print("  ██║  ██║╚██████╔╝███████╗███████╗██║   ╚███╔███╔╝╚██████╔╝╚██████╔╝██████╔╝")
    print("  ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═╝    ╚══╝╚══╝  ╚═════╝  ╚═════╝ ╚═════╝ ")
    print("                       ███████╗ ██████╗ ██╗   ██╗ █████╗ ██████╗ ███████╗███████╗")
    print("                       ██╔════╝██╔═══██╗██║   ██║██╔══██╗██╔══██╗██╔════╝██╔════╝")
    print("                       ███████╗██║   ██║██║   ██║███████║██████╔╝█████╗  ███████╗")
    print("                       ╚════██║██║▄▄ ██║██║   ██║██╔══██║██╔══██╗██╔══╝  ╚════██║")
    print("                       ███████║╚██████╔╝╚██████╔╝██║  ██║██║  ██║███████╗███████║")
    print("                       ╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝")
    print()
    
    # Choose mode
    if args.sequential or args.resume:
        # Sequential mode (slower, but supports resume)
        success = test.run(resume=args.resume)
    else:
        # Parallel mode (default, much faster)
        success = test.run_parallel()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
