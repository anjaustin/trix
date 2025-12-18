#!/usr/bin/env python3
"""
RIEMANN ZERO HUNTER
===================

Full-range hunt from 10^9 to 10^13 zeros.

Features:
- Checkpoints at each 10^N milestone
- Proof files with verification data
- Non-blocking log (tail -f hunt.log)
- Pause button (touch PAUSE to pause, rm PAUSE to resume)
- Resume from last checkpoint on restart

Usage:
    python riemann_hunt.py              # Start/resume hunt
    touch PAUSE                         # Pause gracefully
    rm PAUSE                            # Resume
    tail -f hunt.log                    # Monitor progress
    
Output:
    hunt.log                            # Live progress log
    checkpoints/checkpoint_10^N.pt      # State at each milestone
    proofs/proof_10^N.json              # Verification proof at each milestone
"""

import torch
import math
import time
import json
import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

PI = math.pi
TWO_PI = 2 * PI
DEVICE = 'cuda'

# Directories
SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
PROOF_DIR = SCRIPT_DIR / 'proofs'
LOG_FILE = SCRIPT_DIR / 'hunt.log'
PAUSE_FILE = SCRIPT_DIR / 'PAUSE'

CHECKPOINT_DIR.mkdir(exist_ok=True)
PROOF_DIR.mkdir(exist_ok=True)

# Targets
TARGETS = [10**9, 10**10, 10**11, 10**12, 10**13]


class HuntLogger:
    """Non-blocking file logger."""
    
    def __init__(self, path):
        self.path = path
        self.start_time = time.time()
    
    def log(self, msg, also_print=True):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed = time.time() - self.start_time
        hours, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        elapsed_str = f'{int(hours):02d}:{int(mins):02d}:{int(secs):02d}'
        
        line = f'[{timestamp}] [{elapsed_str}] {msg}'
        
        with open(self.path, 'a') as f:
            f.write(line + '\n')
            f.flush()
        
        if also_print:
            print(line)


class HollywoodEngine:
    """Hollywood pre-computed topology engine."""
    
    def __init__(self, t0_center, num_evals, device='cuda'):
        self.device = device
        self.num_evals = num_evals
        self.oversampling = 4
        
        self.M = int(math.sqrt(t0_center / TWO_PI)) + 10
        self.num_grid = num_evals * self.oversampling
        
        density = math.log(t0_center) / TWO_PI
        self.delta = 1.0 / (density * 10)
        
        n = torch.arange(1, self.M + 1, device=device, dtype=torch.float64)
        self.ln_n = torch.log(n)
        self.rsqrt_n = torch.rsqrt(n)
        
        self._build_topology()
    
    def _build_topology(self):
        delta_omega = TWO_PI / self.num_grid
        grid_idx = self.delta * self.ln_n / delta_omega
        idx_floor = grid_idx.long()
        frac = grid_idx - idx_floor.float()
        
        offsets = torch.tensor([-1, 0, 1, 2], device=self.device)
        self.scatter_indices = ((idx_floor.unsqueeze(1) + offsets.unsqueeze(0)) % self.num_grid)
        self.scatter_indices = self.scatter_indices.flatten().long()
        
        t = frac
        w0 = (1 - t)**3 / 6
        w1 = (3*t**3 - 6*t**2 + 4) / 6
        w2 = (-3*t**3 + 3*t**2 + 3*t + 1) / 6
        w3 = t**3 / 6
        self.weights = torch.stack([w0, w1, w2, w3], dim=1).to(torch.complex128)
    
    def evaluate(self, t0):
        phase = t0 * self.ln_n
        a_n = (self.rsqrt_n * torch.exp(1j * phase)).to(torch.complex128)
        
        contributions = (a_n.unsqueeze(1) * self.weights).flatten()
        
        grid = torch.zeros(self.num_grid, device=self.device, dtype=torch.complex128)
        grid.scatter_add_(0, self.scatter_indices, contributions)
        
        G = torch.fft.ifft(grid) * self.num_grid
        S = G[:self.num_evals]
        
        t_vals = t0 + self.delta * torch.arange(self.num_evals, device=self.device, dtype=torch.float64)
        th = t_vals/2 * torch.log(t_vals/TWO_PI) - t_vals/2 - PI/8 + 1/(48*t_vals)
        Z = 2.0 * (S * torch.exp(-1j * th)).real
        
        return t_vals, Z.float()


class RiemannHunter:
    """Full-range Riemann zero hunter with checkpointing."""
    
    def __init__(self):
        self.logger = HuntLogger(LOG_FILE)
        self.zeros_found = 0
        self.t_current = 0.0
        self.zeros_at_target = {}
        self.start_time = time.time()
        self.batch_zeros = []  # Store recent zeros for proofs
        
        # Engine (rebuilt as t grows)
        self.engine = None
        self.engine_t_center = 0
    
    def log(self, msg, also_print=True):
        self.logger.log(msg, also_print)
    
    def save_checkpoint(self, target):
        """Save state at milestone."""
        path = CHECKPOINT_DIR / f'checkpoint_{target:.0e}.pt'
        state = {
            'zeros_found': self.zeros_found,
            't_current': self.t_current,
            'zeros_at_target': self.zeros_at_target,
            'elapsed': time.time() - self.start_time,
            'timestamp': datetime.now().isoformat(),
        }
        torch.save(state, path)
        self.log(f'Checkpoint saved: {path}')
    
    def load_checkpoint(self):
        """Load most recent checkpoint."""
        checkpoints = sorted(CHECKPOINT_DIR.glob('checkpoint_*.pt'))
        if not checkpoints:
            return False
        
        latest = checkpoints[-1]
        state = torch.load(latest)
        
        self.zeros_found = state['zeros_found']
        self.t_current = state['t_current']
        self.zeros_at_target = state.get('zeros_at_target', {})
        
        self.log(f'Resumed from {latest}')
        self.log(f'  Zeros: {self.zeros_found:,}, t: {self.t_current:.6e}')
        
        return True
    
    def generate_proof(self, target):
        """Generate verification proof at milestone."""
        # Compute expected count via Riemann-von Mangoldt
        t = self.t_current
        expected = (t / TWO_PI) * (math.log(t / TWO_PI) - 1) + 7/8
        
        # Sample zeros for verification
        sample_zeros = self.batch_zeros[-100:] if self.batch_zeros else []
        
        proof = {
            'target': target,
            'zeros_found': self.zeros_found,
            't_reached': self.t_current,
            'expected_count': expected,
            'deviation': self.zeros_found - expected,
            'deviation_pct': (self.zeros_found - expected) / expected * 100,
            'sample_zeros': [float(z) for z in sample_zeros],
            'elapsed_seconds': time.time() - self.start_time,
            'timestamp': datetime.now().isoformat(),
            'rate_zeros_per_sec': self.zeros_found / (time.time() - self.start_time),
        }
        
        # Compute checksum
        proof_str = json.dumps(proof, sort_keys=True)
        proof['checksum'] = hashlib.sha256(proof_str.encode()).hexdigest()[:16]
        
        path = PROOF_DIR / f'proof_{target:.0e}.json'
        with open(path, 'w') as f:
            json.dump(proof, f, indent=2)
        
        self.log(f'Proof generated: {path}')
        self.log(f'  Found: {self.zeros_found:,}, Expected: {expected:,.0f}, Dev: {proof["deviation_pct"]:.2f}%')
        
        return proof
    
    def rebuild_engine(self, t_center):
        """Rebuild engine for new t range."""
        M = int(math.sqrt(t_center / TWO_PI)) + 10
        num_evals = M * 8
        
        self.engine = HollywoodEngine(t_center, num_evals, DEVICE)
        self.engine_t_center = t_center
        
        return num_evals
    
    def check_pause(self):
        """Check if PAUSE file exists."""
        return PAUSE_FILE.exists()
    
    def wait_for_resume(self):
        """Wait until PAUSE file is removed."""
        self.log('PAUSED - remove PAUSE file to resume')
        while PAUSE_FILE.exists():
            time.sleep(1)
        self.log('RESUMED')
    
    def hunt(self):
        """Main hunt loop."""
        self.log('='*60)
        self.log('RIEMANN ZERO HUNTER - Starting')
        self.log(f'Targets: {[f"10^{int(math.log10(t))}" for t in TARGETS]}')
        self.log('='*60)
        
        # Try to resume
        if not self.load_checkpoint():
            self.log('Starting fresh hunt')
            self.t_current = 14.134725  # First zero
        
        # Find next target
        current_target_idx = 0
        for i, target in enumerate(TARGETS):
            if self.zeros_found < target:
                current_target_idx = i
                break
        else:
            self.log('All targets completed!')
            return
        
        # Main loop
        last_log_time = time.time()
        last_log_zeros = self.zeros_found
        
        while current_target_idx < len(TARGETS):
            target = TARGETS[current_target_idx]
            
            # Check pause
            if self.check_pause():
                self.save_checkpoint(target)
                self.wait_for_resume()
            
            # Rebuild engine if t has grown significantly
            if self.engine is None or self.t_current > self.engine_t_center * 2:
                self.log(f'Rebuilding engine for t ~ {self.t_current:.2e}')
                num_evals = self.rebuild_engine(self.t_current)
                self.log(f'  M={self.engine.M}, evals={num_evals}')
            
            # Evaluate batch
            t_vals, Z_vals = self.engine.evaluate(self.t_current)
            
            # Count zeros (sign changes)
            signs = torch.sign(Z_vals)
            sign_changes = (signs[:-1] * signs[1:]) < 0
            new_zeros = sign_changes.sum().item()
            
            # Record zero locations
            zero_indices = torch.where(sign_changes)[0]
            for idx in zero_indices[:10]:  # Keep first 10 per batch
                self.batch_zeros.append(t_vals[idx].item())
            self.batch_zeros = self.batch_zeros[-1000:]  # Keep last 1000
            
            self.zeros_found += new_zeros
            self.t_current = t_vals[-1].item()
            
            # Progress logging (every 10 seconds)
            now = time.time()
            if now - last_log_time > 10:
                zeros_delta = self.zeros_found - last_log_zeros
                rate = zeros_delta / (now - last_log_time)
                
                remaining = target - self.zeros_found
                eta_sec = remaining / rate if rate > 0 else float('inf')
                
                if eta_sec < 3600:
                    eta_str = f'{eta_sec/60:.1f}m'
                elif eta_sec < 86400:
                    eta_str = f'{eta_sec/3600:.1f}h'
                else:
                    eta_str = f'{eta_sec/86400:.1f}d'
                
                pct = self.zeros_found / target * 100
                self.log(f'Progress: {self.zeros_found:,}/{target:,} ({pct:.2f}%) | '
                        f't={self.t_current:.4e} | {rate/1e6:.2f}M/s | ETA: {eta_str}')
                
                last_log_time = now
                last_log_zeros = self.zeros_found
            
            # Check if target reached
            if self.zeros_found >= target:
                self.log(f'TARGET REACHED: 10^{int(math.log10(target))} zeros!')
                self.zeros_at_target[target] = {
                    'zeros': self.zeros_found,
                    't': self.t_current,
                    'elapsed': time.time() - self.start_time
                }
                self.save_checkpoint(target)
                self.generate_proof(target)
                current_target_idx += 1
                self.log('='*60)
        
        # Done!
        total_time = time.time() - self.start_time
        self.log('='*60)
        self.log('HUNT COMPLETE!')
        self.log(f'Total zeros: {self.zeros_found:,}')
        self.log(f'Total time: {timedelta(seconds=int(total_time))}')
        self.log('='*60)


def main():
    # Clear log on fresh start (not resume)
    if not list(CHECKPOINT_DIR.glob('checkpoint_*.pt')):
        LOG_FILE.unlink(missing_ok=True)
    
    hunter = RiemannHunter()
    
    try:
        hunter.hunt()
    except KeyboardInterrupt:
        hunter.log('Interrupted by user')
        # Save emergency checkpoint
        hunter.save_checkpoint(TARGETS[0])
    except Exception as e:
        hunter.log(f'ERROR: {e}')
        hunter.save_checkpoint(TARGETS[0])
        raise


if __name__ == '__main__':
    main()
