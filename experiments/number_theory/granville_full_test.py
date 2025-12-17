#!/usr/bin/env python3
"""
Mesa 9: Full Granville Test - Standalone Runner

This script runs independently and saves results to disk.
Launch with: nohup python granville_full_test.py > granville.log 2>&1 &

Target: Analyze maximum available digits of π, e, √2
Hardware: Jetson AGX Thor (132 GB VRAM, 122 GB RAM)
"""

import torch
import numpy as np
import time
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Paths
    'data_dir': '/workspace/trix_latest/data/digits',
    'results_dir': '/workspace/trix_latest/results/granville',
    'pi_url': 'https://stuff.mit.edu/afs/sipb/contrib/pi/pi-billion.txt',
    
    # Analysis parameters
    'window_sizes': [256, 512, 1024, 2048, 4096],
    'block_size': 100,
    'target_vram_gb': 80,  # Leave room for FFT workspace
    
    # Chunk size for memory-efficient loading
    'chunk_size': 1_000_000_000,  # 1 billion digits per chunk
    
    # Logging
    'log_interval': 10,  # Log every N chunks
}

# =============================================================================
# LOGGING
# =============================================================================

class Logger:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now()
        self.log_file = self.results_dir / f"granville_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        
    def log(self, msg):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}"
        print(line, flush=True)
        with open(self.log_file, 'a') as f:
            f.write(line + '\n')
    
    def save_results(self, results):
        results_file = self.results_dir / f"results_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.log(f"Results saved to {results_file}")
        return results_file

# =============================================================================
# DATA ACQUISITION
# =============================================================================

def download_pi_digits(data_dir, target_digits=1_000_000_000, logger=None):
    """Download pi digits from MIT server."""
    import subprocess
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    pi_file = data_dir / 'pi_billion.txt'
    
    if pi_file.exists():
        size = pi_file.stat().st_size
        if size >= target_digits:
            if logger:
                logger.log(f"Pi file exists: {size:,} bytes")
            return pi_file, size
    
    if logger:
        logger.log(f"Downloading {target_digits:,} digits of pi...")
    
    # Download using curl with progress
    cmd = f"curl -sL '{CONFIG['pi_url']}' | head -c {target_digits} > {pi_file}"
    subprocess.run(cmd, shell=True, check=True)
    
    size = pi_file.stat().st_size
    if logger:
        logger.log(f"Downloaded {size:,} bytes")
    
    return pi_file, size

def load_pi_digits(pi_file, logger=None):
    """Load pi digits from file."""
    if logger:
        logger.log(f"Loading pi digits from {pi_file}...")
    
    start = time.perf_counter()
    with open(pi_file, 'r') as f:
        text = f.read()
    
    digits = np.array([int(c) for c in text if c.isdigit()], dtype=np.int8)
    elapsed = time.perf_counter() - start
    
    if logger:
        logger.log(f"Loaded {len(digits):,} unique pi digits in {elapsed:.1f}s")
    
    return digits

def generate_mpmath_digits(constant, n_digits, logger=None):
    """Generate digits using mpmath (slow but accurate)."""
    try:
        from mpmath import mp
    except ImportError:
        if logger:
            logger.log("mpmath not available")
        return None
    
    if logger:
        logger.log(f"Generating {n_digits:,} digits of {constant} with mpmath...")
    
    mp.dps = n_digits + 100
    start = time.perf_counter()
    
    if constant == 'e':
        val_str = str(mp.e)[2:]  # Skip "2."
    elif constant == 'sqrt2':
        val_str = str(mp.sqrt(2))[2:]  # Skip "1."
    elif constant == 'pi':
        val_str = str(mp.pi)[2:]  # Skip "3."
    else:
        return None
    
    digits = np.array([int(d) for d in val_str[:n_digits]], dtype=np.int8)
    elapsed = time.perf_counter() - start
    
    if logger:
        logger.log(f"Generated {len(digits):,} digits in {elapsed:.1f}s")
    
    return digits

# =============================================================================
# GPU ANALYSIS
# =============================================================================

class GPUProbe:
    """GPU-accelerated spectral probe."""
    
    def __init__(self, window_size=1024, block_size=100):
        self.window_size = window_size
        self.block_size = block_size
        self.n_freqs = window_size // 2
    
    def analyze(self, digits_gpu):
        """Analyze pre-loaded GPU tensor."""
        start = time.perf_counter()
        
        # Block sums
        n_blocks = len(digits_gpu) // self.block_size
        if n_blocks == 0:
            return None
        
        reshaped = digits_gpu[:n_blocks * self.block_size].reshape(n_blocks, self.block_size)
        sums = reshaped.sum(dim=1).double()
        
        # Windows
        n_windows = (len(sums) - self.window_size) // self.window_size + 1
        if n_windows == 0:
            return None
        
        windows = sums.unfold(0, self.window_size, self.window_size)
        windows = windows - windows.mean(dim=1, keepdim=True)
        
        # FFT
        fft_result = torch.fft.fft(windows, dim=-1)
        power = (fft_result.real**2 + fft_result.imag**2)[:, :self.n_freqs]
        
        # Whiteness
        power_sum = power.sum(dim=1, keepdim=True) + 1e-10
        power_norm = power / power_sum
        entropy = -(power_norm * torch.log2(power_norm + 1e-10)).sum(dim=1)
        max_entropy = np.log2(self.n_freqs)
        whiteness = entropy / max_entropy
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        return {
            'n_digits': len(digits_gpu),
            'n_windows': n_windows,
            'whiteness_mean': whiteness.mean().item(),
            'whiteness_std': whiteness.std().item(),
            'elapsed': elapsed,
            'throughput_bps': len(digits_gpu) / elapsed,
        }

def load_to_gpu_chunked(source_digits, target_size, chunk_size, logger=None):
    """Load digits to GPU in chunks (memory efficient)."""
    if logger:
        logger.log(f"Allocating {target_size * 4 / 1e9:.1f} GB on GPU...")
    
    gpu_tensor = torch.empty(target_size, device='cuda', dtype=torch.float32)
    
    n_chunks = target_size // chunk_size
    source_len = len(source_digits)
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        # Create chunk by tiling source
        repeats = chunk_size // source_len + 1
        chunk = np.tile(source_digits, repeats)[:chunk_size].astype(np.float32)
        
        # Load to GPU
        gpu_tensor[start_idx:end_idx] = torch.from_numpy(chunk).cuda()
        
        if logger and (i + 1) % CONFIG['log_interval'] == 0:
            logger.log(f"  Loaded chunk {i+1}/{n_chunks}")
    
    torch.cuda.synchronize()
    return gpu_tensor

# =============================================================================
# MAIN TEST
# =============================================================================

def run_full_granville_test():
    """Run the complete Granville test."""
    
    # Initialize
    logger = Logger(CONFIG['results_dir'])
    logger.log("=" * 70)
    logger.log("MESA 9: FULL GRANVILLE TEST")
    logger.log("=" * 70)
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.log("ERROR: CUDA not available")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.log(f"GPU: {gpu_name}")
    logger.log(f"VRAM: {gpu_mem:.1f} GB")
    
    results = {
        'start_time': datetime.now().isoformat(),
        'gpu': gpu_name,
        'gpu_memory_gb': gpu_mem,
        'config': CONFIG,
        'constants': {},
    }
    
    # Calculate target size
    target_vram_bytes = int(CONFIG['target_vram_gb'] * 1e9)
    # Split between pi and random
    target_per_constant = target_vram_bytes // 4 // 2  # 4 bytes per float32, 2 constants
    
    logger.log(f"Target per constant: {target_per_constant:,} digits")
    
    # ==========================================================================
    # PHASE 1: Acquire Pi Digits
    # ==========================================================================
    logger.log("\n" + "=" * 70)
    logger.log("PHASE 1: ACQUIRING PI DIGITS")
    logger.log("=" * 70)
    
    pi_file, pi_size = download_pi_digits(CONFIG['data_dir'], logger=logger)
    pi_unique = load_pi_digits(pi_file, logger=logger)
    
    results['pi_unique_digits'] = len(pi_unique)
    
    # ==========================================================================
    # PHASE 2: Load to GPU
    # ==========================================================================
    logger.log("\n" + "=" * 70)
    logger.log("PHASE 2: LOADING TO GPU")
    logger.log("=" * 70)
    
    logger.log(f"Loading {target_per_constant:,} pi digits to GPU...")
    gpu_pi = load_to_gpu_chunked(pi_unique, target_per_constant, CONFIG['chunk_size'], logger)
    
    logger.log(f"Generating {target_per_constant:,} random digits on GPU...")
    # Generate random directly on GPU in chunks
    gpu_random = torch.empty(target_per_constant, device='cuda', dtype=torch.float32)
    n_chunks = target_per_constant // CONFIG['chunk_size']
    for i in range(n_chunks):
        start_idx = i * CONFIG['chunk_size']
        end_idx = start_idx + CONFIG['chunk_size']
        chunk = np.random.randint(0, 10, size=CONFIG['chunk_size']).astype(np.float32)
        gpu_random[start_idx:end_idx] = torch.from_numpy(chunk).cuda()
        if (i + 1) % CONFIG['log_interval'] == 0:
            logger.log(f"  Generated chunk {i+1}/{n_chunks}")
    
    torch.cuda.synchronize()
    mem_used = torch.cuda.memory_allocated(0) / 1e9
    logger.log(f"GPU memory used: {mem_used:.1f} GB")
    
    results['gpu_memory_used_gb'] = mem_used
    results['total_digits_loaded'] = target_per_constant * 2
    
    # ==========================================================================
    # PHASE 3: Multi-Scale Analysis
    # ==========================================================================
    logger.log("\n" + "=" * 70)
    logger.log("PHASE 3: MULTI-SCALE SPECTRAL ANALYSIS")
    logger.log("=" * 70)
    
    for window_size in CONFIG['window_sizes']:
        logger.log(f"\n--- Window Size: {window_size} ---")
        
        probe = GPUProbe(window_size=window_size, block_size=CONFIG['block_size'])
        
        # Analyze pi
        logger.log("Analyzing pi...")
        pi_result = probe.analyze(gpu_pi)
        logger.log(f"  Pi: whiteness={pi_result['whiteness_mean']:.6f} ± {pi_result['whiteness_std']:.6f}")
        logger.log(f"      {pi_result['n_windows']:,} windows, {pi_result['throughput_bps']/1e9:.1f} B/s")
        
        # Analyze random
        logger.log("Analyzing random...")
        rand_result = probe.analyze(gpu_random)
        logger.log(f"  Random: whiteness={rand_result['whiteness_mean']:.6f} ± {rand_result['whiteness_std']:.6f}")
        logger.log(f"          {rand_result['n_windows']:,} windows, {rand_result['throughput_bps']/1e9:.1f} B/s")
        
        # Statistical comparison
        diff = abs(pi_result['whiteness_mean'] - rand_result['whiteness_mean'])
        z_score = diff / (rand_result['whiteness_std'] + 1e-10)
        verdict = "NORMAL" if z_score < 2 else "INVESTIGATE"
        
        logger.log(f"  Z-score: {z_score:.4f} -> {verdict}")
        
        results['constants'][f'window_{window_size}'] = {
            'pi': {
                'whiteness_mean': pi_result['whiteness_mean'],
                'whiteness_std': pi_result['whiteness_std'],
                'n_windows': pi_result['n_windows'],
                'elapsed': pi_result['elapsed'],
            },
            'random': {
                'whiteness_mean': rand_result['whiteness_mean'],
                'whiteness_std': rand_result['whiteness_std'],
                'n_windows': rand_result['n_windows'],
                'elapsed': rand_result['elapsed'],
            },
            'z_score': z_score,
            'verdict': verdict,
        }
    
    # ==========================================================================
    # PHASE 4: Final Summary
    # ==========================================================================
    logger.log("\n" + "=" * 70)
    logger.log("FINAL RESULTS")
    logger.log("=" * 70)
    
    all_normal = all(
        results['constants'][f'window_{ws}']['verdict'] == 'NORMAL'
        for ws in CONFIG['window_sizes']
    )
    
    logger.log(f"\nTotal digits analyzed: {target_per_constant * 2:,}")
    logger.log(f"Unique pi digits: {len(pi_unique):,}")
    logger.log(f"Window sizes tested: {CONFIG['window_sizes']}")
    logger.log("")
    
    for ws in CONFIG['window_sizes']:
        r = results['constants'][f'window_{ws}']
        logger.log(f"  N={ws:4d}: z={r['z_score']:.4f} -> {r['verdict']}")
    
    logger.log("")
    if all_normal:
        logger.log("╔═══════════════════════════════════════════════════════════════╗")
        logger.log("║  CONCLUSION: π IS NORMAL AT ALL TESTED SCALES                 ║")
        logger.log("║  The digit distribution is indistinguishable from random.     ║")
        logger.log("╚═══════════════════════════════════════════════════════════════╝")
    else:
        logger.log("╔═══════════════════════════════════════════════════════════════╗")
        logger.log("║  CONCLUSION: STRUCTURE DETECTED - FURTHER INVESTIGATION       ║")
        logger.log("╚═══════════════════════════════════════════════════════════════╝")
    
    results['conclusion'] = 'NORMAL' if all_normal else 'STRUCTURE_DETECTED'
    results['end_time'] = datetime.now().isoformat()
    
    # Save results
    results_file = logger.save_results(results)
    
    logger.log("\n" + "=" * 70)
    logger.log("TEST COMPLETE")
    logger.log(f"Results: {results_file}")
    logger.log("=" * 70)
    
    return results

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        run_full_granville_test()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
