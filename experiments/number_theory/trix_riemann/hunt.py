#!/usr/bin/env python3
"""
RIEMANN ZERO HUNT - Fire and Forget
====================================

Launch and walk away. It handles everything:
- Auto-saves state every checkpoint
- Resumes from interruption
- Logs everything to disk
- Writes final report when done

Usage:
    python hunt.py                    # Default: 10^6 to 10^7
    python hunt.py --target 1e9       # Hunt to 10^9 zeros
    python hunt.py --resume           # Resume interrupted hunt
    nohup python hunt.py &            # Background (fire and forget)

"Launch it and go eat."
"""

import os
import sys
import json
import math
import time
import signal
import pickle
import argparse
import traceback
from datetime import datetime
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from hunt_logger import HuntLogger

# Constants
PI = math.pi
TWO_PI = 2.0 * PI


class AutonomousHunter:
    """
    Fire-and-forget Riemann zero hunter.
    
    - Checkpoints state automatically
    - Resumes from interruption
    - Handles errors gracefully
    - Logs everything
    """
    
    def __init__(self, 
                 target_zeros: float = 1e7,
                 t_start: float = 14.0,
                 pts_per_zero: int = 10,
                 checkpoint_zeros: int = 100000,
                 run_dir: str = "hunt_runs",
                 run_name: str = None):
        
        self.target_zeros = target_zeros
        self.pts_per_zero = pts_per_zero
        self.checkpoint_zeros = checkpoint_zeros
        
        # Compute t_end from target zeros using Riemann-von Mangoldt
        # N(T) ≈ T/(2π) * ln(T/(2π)) - T/(2π)
        # Rough inverse: T ≈ 2π * N / ln(N) for large N
        if target_zeros > 1000:
            self.t_end = self._estimate_t_for_zeros(target_zeros)
        else:
            self.t_end = 1000.0
        self.t_start = t_start
        
        # Run directory
        if run_name is None:
            run_name = f"hunt_{target_zeros:.0e}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_name = run_name
        self.run_dir = Path(run_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # State file for resume
        self.state_file = self.run_dir / "state.pkl"
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # State
        self.t_current = t_start
        self.total_zeros = 0
        self.total_points = 0
        self.start_time = None
        self.running = True
        
        # Logger
        self.logger = HuntLogger(
            log_dir=str(self.run_dir / "logs"),
            run_name="hunt"
        )
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _estimate_t_for_zeros(self, N: float) -> float:
        """Estimate t value needed to find N zeros."""
        # N(T) ≈ T/(2π) * ln(T/(2π))
        # Iterative solution
        T = N * 2 * PI / max(1, math.log(N))
        for _ in range(10):
            N_est = T / TWO_PI * math.log(T / TWO_PI) - T / TWO_PI
            if N_est > 0:
                T = T * (N / N_est) ** 0.5
        return T * 1.1  # 10% margin
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n[SIGNAL {signum}] Graceful shutdown initiated...")
        self.running = False
    
    def save_state(self):
        """Save current state for resume."""
        state = {
            't_current': self.t_current,
            'total_zeros': self.total_zeros,
            'total_points': self.total_points,
            't_start': self.t_start,
            't_end': self.t_end,
            'target_zeros': self.target_zeros,
            'pts_per_zero': self.pts_per_zero,
            'checkpoint_zeros': self.checkpoint_zeros,
            'elapsed_before': time.time() - self.start_time if self.start_time else 0,
        }
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
        
        # Also save human-readable
        with open(self.run_dir / "state.json", 'w') as f:
            json.dump({k: float(v) if isinstance(v, (int, float)) else v 
                      for k, v in state.items()}, f, indent=2)
    
    def load_state(self) -> bool:
        """Load state from checkpoint. Returns True if state existed."""
        if not self.state_file.exists():
            return False
        
        try:
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)
            
            self.t_current = state['t_current']
            self.total_zeros = state['total_zeros']
            self.total_points = state['total_points']
            self.t_start = state.get('t_start', self.t_start)
            self.t_end = state.get('t_end', self.t_end)
            self.target_zeros = state.get('target_zeros', self.target_zeros)
            
            print(f"[RESUME] Loaded state: t={self.t_current:.2e}, zeros={self.total_zeros:,}")
            return True
        except Exception as e:
            print(f"[WARN] Could not load state: {e}")
            return False
    
    def theta(self, t: torch.Tensor) -> torch.Tensor:
        """Riemann-Siegel theta function."""
        t_half = t / 2
        result = t_half * torch.log(t / TWO_PI) - t_half - PI / 8
        result = result + 1.0 / (48.0 * t)
        result = result + 7.0 / (5760.0 * t ** 3)
        return result
    
    def process_chunk(self, t_start: float, t_end: float) -> tuple:
        """Process a chunk and return (zeros_found, num_points)."""
        # Number of terms
        M = int(math.sqrt(t_end / TWO_PI)) + 10
        M = min(M, 100000)
        
        # Sampling
        density = math.log((t_start + t_end) / 2) / TWO_PI
        num_points = int((t_end - t_start) * density * self.pts_per_zero)
        num_points = max(num_points, 1000)
        
        # Generate points
        t_vals = torch.linspace(t_start, t_end, num_points,
                               dtype=torch.float64, device=self.device)
        
        # Compute Z values
        n = torch.arange(1, M + 1, dtype=torch.float64, device=self.device)
        log_n = torch.log(n)
        rsqrt_n = torch.rsqrt(n)
        theta = self.theta(t_vals)
        
        # Batch to avoid OOM
        batch_size = 50000
        Z_all = []
        
        for i in range(0, num_points, batch_size):
            end = min(i + batch_size, num_points)
            t_batch = t_vals[i:end]
            theta_batch = theta[i:end]
            
            phases = theta_batch.unsqueeze(-1) - t_batch.unsqueeze(-1) * log_n.unsqueeze(0)
            Z_batch = 2.0 * (rsqrt_n.unsqueeze(0) * torch.cos(phases)).sum(dim=-1)
            Z_all.append(Z_batch)
        
        Z_vals = torch.cat(Z_all)
        
        # Count sign changes
        signs = torch.sign(Z_vals)
        zeros_found = ((signs[:-1] * signs[1:]) < 0).sum().item()
        
        return zeros_found, num_points, M
    
    def hunt(self, resume: bool = False):
        """
        Execute the hunt. Fire and forget.
        """
        # Resume if requested and state exists
        if resume:
            self.load_state()
        
        self.start_time = time.time()
        
        # Write run info
        run_info = {
            'target_zeros': self.target_zeros,
            't_start': self.t_start,
            't_end': self.t_end,
            'pts_per_zero': self.pts_per_zero,
            'device': self.device,
            'start_time': datetime.now().isoformat(),
            'resumed': resume,
        }
        with open(self.run_dir / "run_info.json", 'w') as f:
            json.dump(run_info, f, indent=2)
        
        print("="*70)
        print("RIEMANN ZERO HUNT - AUTONOMOUS MODE")
        print("="*70)
        print(f"Target: {self.target_zeros:.0e} zeros")
        print(f"Range: [{self.t_start:.2e}, {self.t_end:.2e}]")
        print(f"Run dir: {self.run_dir}")
        print(f"Device: {self.device}")
        print("="*70)
        print("Hunt started. Logs will be saved automatically.")
        print("Safe to disconnect. Use --resume to continue if interrupted.")
        print("="*70 + "\n")
        
        # Chunk size based on checkpoint interval
        # Adjust chunk to yield ~checkpoint_zeros zeros
        avg_density = math.log(self.t_end / 2) / TWO_PI
        chunk_t_size = self.checkpoint_zeros / avg_density
        chunk_t_size = max(chunk_t_size, 1000)  # Minimum chunk
        
        last_checkpoint_zeros = self.total_zeros
        last_log_time = time.time()
        
        try:
            while self.t_current < self.t_end and self.running:
                chunk_end = min(self.t_current + chunk_t_size, self.t_end)
                
                # Process chunk
                chunk_start_time = time.time()
                zeros_found, num_points, M = self.process_chunk(self.t_current, chunk_end)
                chunk_elapsed = time.time() - chunk_start_time
                
                self.total_zeros += zeros_found
                self.total_points += num_points
                
                # Log batch
                self.logger.log_batch(
                    t_start=self.t_current,
                    t_end=chunk_end,
                    num_points=num_points,
                    zeros_found=zeros_found,
                    elapsed=chunk_elapsed,
                    M_terms=M,
                )
                
                # Progress (throttled to every 10 seconds)
                if time.time() - last_log_time > 10:
                    progress = (self.t_current - self.t_start) / (self.t_end - self.t_start) * 100
                    elapsed = time.time() - self.start_time
                    rate = self.total_zeros / elapsed
                    remaining = (self.target_zeros - self.total_zeros) / rate if rate > 0 else float('inf')
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"t={self.t_current:.2e} ({progress:.1f}%) | "
                          f"Zeros: {self.total_zeros:,} | "
                          f"Rate: {rate:,.0f}/s | "
                          f"ETA: {remaining/3600:.1f}h")
                    last_log_time = time.time()
                
                # Checkpoint
                if self.total_zeros - last_checkpoint_zeros >= self.checkpoint_zeros:
                    self.save_state()
                    self.logger.log_checkpoint(self.t_current, self.target_zeros)
                    last_checkpoint_zeros = self.total_zeros
                
                # Advance
                self.t_current = chunk_end
                
                # Check if we've found enough zeros
                if self.total_zeros >= self.target_zeros:
                    print(f"\n[COMPLETE] Target reached: {self.total_zeros:,} zeros")
                    break
        
        except Exception as e:
            print(f"\n[ERROR] {e}")
            traceback.print_exc()
            self.save_state()
            raise
        
        finally:
            # Final save
            self.save_state()
            
            # Final report
            elapsed = time.time() - self.start_time
            
            report = {
                'status': 'complete' if self.total_zeros >= self.target_zeros else 'interrupted',
                'total_zeros': self.total_zeros,
                'target_zeros': self.target_zeros,
                't_reached': self.t_current,
                't_end': self.t_end,
                'elapsed_seconds': elapsed,
                'zeros_per_second': self.total_zeros / elapsed if elapsed > 0 else 0,
                'end_time': datetime.now().isoformat(),
            }
            
            with open(self.run_dir / "report.json", 'w') as f:
                json.dump(report, f, indent=2)
            
            # Close logger
            self.logger.close()
            
            # Print summary
            print("\n" + "="*70)
            print("HUNT SUMMARY")
            print("="*70)
            print(f"Status: {report['status'].upper()}")
            print(f"Zeros found: {self.total_zeros:,}")
            print(f"Time: {elapsed:.1f}s ({elapsed/3600:.2f} hours)")
            print(f"Rate: {report['zeros_per_second']:,.0f} zeros/sec")
            print(f"Logs: {self.run_dir}")
            print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Riemann Zero Hunt - Fire and Forget",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hunt.py                      # Default hunt (10^7 zeros)
  python hunt.py --target 1e9         # Hunt for 10^9 zeros
  python hunt.py --resume             # Resume interrupted hunt
  nohup python hunt.py --target 1e12 > hunt.log 2>&1 &   # Background
        """
    )
    
    parser.add_argument("--target", type=float, default=1e7,
                        help="Target number of zeros (default: 1e7)")
    parser.add_argument("--start", type=float, default=14.0,
                        help="Starting t value (default: 14.0)")
    parser.add_argument("--pts-per-zero", type=int, default=10,
                        help="Sampling density (default: 10)")
    parser.add_argument("--checkpoint", type=int, default=100000,
                        help="Checkpoint every N zeros (default: 100000)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom run name")
    parser.add_argument("--run-dir", type=str, default="hunt_runs",
                        help="Directory for run data")
    
    args = parser.parse_args()
    
    hunter = AutonomousHunter(
        target_zeros=args.target,
        t_start=args.start,
        pts_per_zero=args.pts_per_zero,
        checkpoint_zeros=args.checkpoint,
        run_dir=args.run_dir,
        run_name=args.run_name,
    )
    
    hunter.hunt(resume=args.resume)


if __name__ == "__main__":
    main()
