#!/usr/bin/env python3
"""
Mesa 9 + Hollywood Squares: Distributed Spectral Probe

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  DIGIT GENERATORS (Binary Splitting Workers)                │
  │  [G0] [G1] [G2] [G3] [G4] [G5] [G6] [G7]                   │
  │    │    │    │    │    │    │    │    │                     │
  │    └────┴────┴────┴────┴────┴────┴────┘                     │
  │                      │                                       │
  │               ┌──────▼──────┐                                │
  │               │  COLLECTOR  │                                │
  │               └──────┬──────┘                                │
  └──────────────────────┼──────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────┐
  │  SUM FIELD (Parallel Block Summers)                         │
  │  [S0]──[S1]──[S2]──[S3]──[S4]──[S5]──[S6]──[S7]            │
  └──────────────────────┬──────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────┐
  │  FFT FIELD (Butterfly Network)                              │
  │       Stage 0        Stage 1        Stage 2                 │
  │  [B0]────╲      [B0]────╲      [B0]────╲                    │
  │  [B1]────╳      [B1]────╳      [B1]────╳                    │
  │  [B2]────╱      [B2]────╱      [B2]────╱                    │
  │  [B3]────╱      [B3]────╱      [B3]────╱                    │
  └──────────────────────┬──────────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────────┐
  │  ANALYZER (Whiteness Aggregator)                            │
  │  [A0]──[A1]──[A2]──[A3] ──► VERDICT                        │
  └─────────────────────────────────────────────────────────────┘

Key Insight: Topology IS the algorithm. The wiring determines behavior.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue
import threading


# =============================================================================
# MESSAGE TYPES (Hollywood Squares Protocol)
# =============================================================================

class MsgType(Enum):
    """Message types for Hollywood Squares coordination."""
    DIGITS = 1      # Chunk of digits
    SUMS = 2        # Block sums
    FFT_DATA = 3    # FFT intermediate
    SPECTRUM = 4    # Power spectrum
    WHITENESS = 5   # Whiteness score
    CONTROL = 6     # Control signals
    DONE = 7        # Completion


@dataclass
class Message:
    """16-byte aligned message frame (Hollywood Squares spec)."""
    msg_type: MsgType
    source: int
    dest: int
    seq: int
    payload: any
    timestamp: float = field(default_factory=time.perf_counter)


# =============================================================================
# DIGIT GENERATION FIELD (Parallel Binary Splitting)
# =============================================================================

class DigitGeneratorNode:
    """
    A single digit generator worker.
    
    Uses binary splitting for parallel computation of π, e, etc.
    Each node computes a segment independently.
    """
    
    def __init__(self, node_id: int, constant: str = 'pi'):
        self.node_id = node_id
        self.constant = constant
        self.output_queue: Queue = Queue()
    
    def compute_segment(self, start: int, length: int) -> np.ndarray:
        """
        Compute a segment of digits.
        
        For true parallelism, this would use binary splitting.
        For now, we use mpmath with offset computation.
        """
        try:
            from mpmath import mp, mpf
            
            # Set precision for this segment
            mp.dps = start + length + 100
            
            if self.constant == 'pi':
                val = mp.pi
            elif self.constant == 'e':
                val = mp.e
            elif self.constant == 'sqrt2':
                val = mp.sqrt(2)
            else:
                raise ValueError(f"Unknown constant: {self.constant}")
            
            # Extract digits for this segment
            val_str = str(val)[2:]  # Skip "X."
            segment = val_str[start:start + length]
            
            return np.array([int(d) for d in segment if d.isdigit()], dtype=np.int8)
            
        except Exception as e:
            print(f"Node {self.node_id} error: {e}")
            return np.array([], dtype=np.int8)
    
    def run(self, start: int, length: int) -> Message:
        """Run computation and emit result message."""
        digits = self.compute_segment(start, length)
        return Message(
            msg_type=MsgType.DIGITS,
            source=self.node_id,
            dest=-1,  # Broadcast
            seq=start,
            payload=digits
        )


class DigitGeneratorField:
    """
    Parallel digit generation using Hollywood Squares coordination.
    
    Distributes computation across multiple workers.
    """
    
    def __init__(self, num_workers: int = None, constant: str = 'pi'):
        self.num_workers = num_workers or mp.cpu_count()
        self.constant = constant
        self.nodes = [DigitGeneratorNode(i, constant) for i in range(self.num_workers)]
    
    def generate_parallel(self, total_digits: int) -> np.ndarray:
        """Generate digits in parallel using all workers."""
        segment_size = total_digits // self.num_workers
        
        results = {}
        
        def worker_task(node_id: int, start: int, length: int):
            node = self.nodes[node_id]
            msg = node.run(start, length)
            return (msg.seq, msg.payload)
        
        # Use ThreadPool (mpmath releases GIL during computation)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(self.num_workers):
                start = i * segment_size
                length = segment_size if i < self.num_workers - 1 else total_digits - start
                futures.append(executor.submit(worker_task, i, start, length))
            
            for future in futures:
                seq, digits = future.result()
                results[seq] = digits
        
        # Assemble in order
        all_digits = []
        for seq in sorted(results.keys()):
            all_digits.extend(results[seq])
        
        return np.array(all_digits[:total_digits], dtype=np.int8)


# =============================================================================
# SUM FIELD (Parallel Block Summers)
# =============================================================================

class SumNode:
    """A single block sum worker."""
    
    def __init__(self, node_id: int, block_size: int = 100):
        self.node_id = node_id
        self.block_size = block_size
    
    def process(self, digits: np.ndarray) -> np.ndarray:
        """Compute block sums for a chunk of digits."""
        n_blocks = len(digits) // self.block_size
        if n_blocks == 0:
            return np.array([], dtype=np.int64)
        
        reshaped = digits[:n_blocks * self.block_size].reshape(n_blocks, self.block_size)
        return reshaped.sum(axis=1)


class SumField:
    """Parallel block sum computation."""
    
    def __init__(self, num_workers: int = 8, block_size: int = 100):
        self.num_workers = num_workers
        self.block_size = block_size
        self.nodes = [SumNode(i, block_size) for i in range(num_workers)]
    
    def process_parallel(self, digits: np.ndarray) -> np.ndarray:
        """Process digits through parallel sum nodes."""
        chunk_size = len(digits) // self.num_workers
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(self.num_workers):
                start = i * chunk_size
                end = start + chunk_size if i < self.num_workers - 1 else len(digits)
                # Align to block size
                end = (end // self.block_size) * self.block_size
                chunk = digits[start:end]
                futures.append(executor.submit(self.nodes[i].process, chunk))
            
            for future in futures:
                results.append(future.result())
        
        return np.concatenate(results)


# =============================================================================
# FFT FIELD (Butterfly Network)
# =============================================================================

class ButterflyNode:
    """
    A single butterfly computation node.
    
    Performs one butterfly operation: (a + Wb, a - Wb)
    """
    
    def __init__(self, node_id: int):
        self.node_id = node_id
    
    def butterfly(
        self, 
        a_re: float, a_im: float,
        b_re: float, b_im: float,
        w_re: float, w_im: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute butterfly: 
          out1 = a + W*b
          out2 = a - W*b
        """
        # W * b
        wb_re = w_re * b_re - w_im * b_im
        wb_im = w_re * b_im + w_im * b_re
        
        # Butterfly
        out1_re = a_re + wb_re
        out1_im = a_im + wb_im
        out2_re = a_re - wb_re
        out2_im = a_im - wb_im
        
        return out1_re, out1_im, out2_re, out2_im


class FFTField:
    """
    FFT as a Hollywood Squares butterfly network.
    
    Topology encodes the Cooley-Tukey algorithm.
    Each stage is a phase of message passing.
    """
    
    def __init__(self, size: int = 1024):
        self.size = size
        self.num_stages = int(np.log2(size))
        
        # Pre-compute twiddle factors
        k = np.arange(size)
        angles = -2 * np.pi * k / size
        self.twiddle_re = np.cos(angles)
        self.twiddle_im = np.sin(angles)
        
        # Create butterfly nodes
        self.nodes = [ButterflyNode(i) for i in range(size // 2)]
    
    def _bit_reverse(self, x: np.ndarray) -> np.ndarray:
        """Bit-reverse permutation."""
        n = len(x)
        bits = int(np.log2(n))
        result = np.zeros_like(x)
        
        for i in range(n):
            rev = 0
            temp = i
            for _ in range(bits):
                rev = (rev << 1) | (temp & 1)
                temp >>= 1
            result[rev] = x[i]
        
        return result
    
    def fft(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT using butterfly network.
        
        Each stage corresponds to a Hollywood Squares "phase".
        """
        n = len(x)
        
        # Bit-reverse input
        x_re = self._bit_reverse(x.astype(np.float64))
        x_im = np.zeros(n, dtype=np.float64)
        
        # Butterfly stages (phases)
        for stage in range(self.num_stages):
            m = 2 ** (stage + 1)
            half_m = m // 2
            
            new_re = x_re.copy()
            new_im = x_im.copy()
            
            # Process butterflies in parallel (within each phase)
            for k in range(0, n, m):
                for j in range(half_m):
                    # Twiddle index
                    tw_idx = (j * (n // m)) % n
                    w_re = self.twiddle_re[tw_idx]
                    w_im = self.twiddle_im[tw_idx]
                    
                    # Indices
                    u_idx = k + j
                    t_idx = k + j + half_m
                    
                    # Butterfly
                    node = self.nodes[j % len(self.nodes)]
                    o1_re, o1_im, o2_re, o2_im = node.butterfly(
                        x_re[u_idx], x_im[u_idx],
                        x_re[t_idx], x_im[t_idx],
                        w_re, w_im
                    )
                    
                    new_re[u_idx] = o1_re
                    new_im[u_idx] = o1_im
                    new_re[t_idx] = o2_re
                    new_im[t_idx] = o2_im
            
            x_re = new_re
            x_im = new_im
        
        return x_re, x_im
    
    def power_spectrum(self, x: np.ndarray) -> np.ndarray:
        """Compute power spectrum |FFT(x)|^2."""
        re, im = self.fft(x)
        return re**2 + im**2


# =============================================================================
# ANALYZER FIELD (Whiteness Aggregator)
# =============================================================================

class AnalyzerNode:
    """Computes whiteness from power spectrum."""
    
    def __init__(self, node_id: int):
        self.node_id = node_id
    
    def compute_whiteness(self, power: np.ndarray) -> float:
        """
        Compute spectral whiteness (entropy / max_entropy).
        
        For white noise, this approaches 1.0.
        """
        # Use positive frequencies only
        n_freqs = len(power) // 2
        power_pos = power[:n_freqs]
        
        # Normalize
        power_sum = power_pos.sum() + 1e-10
        power_norm = power_pos / power_sum
        
        # Entropy
        entropy = -np.sum(power_norm * np.log2(power_norm + 1e-10))
        max_entropy = np.log2(n_freqs)
        
        return entropy / max_entropy


class AnalyzerField:
    """Aggregates whiteness scores from multiple windows."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.nodes = [AnalyzerNode(i) for i in range(num_workers)]
    
    def analyze_parallel(self, spectra: List[np.ndarray]) -> List[float]:
        """Analyze multiple spectra in parallel."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(
                lambda i: self.nodes[i % self.num_workers].compute_whiteness(spectra[i]),
                range(len(spectra))
            ))
        return results


# =============================================================================
# HOLLYWOOD PROBE (Complete Pipeline)
# =============================================================================

class HollywoodProbe:
    """
    Complete Hollywood Squares spectral probe.
    
    Coordinates all fields through message passing.
    """
    
    def __init__(
        self,
        num_generators: int = None,
        num_summers: int = 8,
        window_size: int = 1024,
        block_size: int = 100
    ):
        self.num_generators = num_generators or mp.cpu_count()
        self.window_size = window_size
        self.block_size = block_size
        
        # Initialize fields
        self.sum_field = SumField(num_summers, block_size)
        self.fft_field = FFTField(window_size)
        self.analyzer_field = AnalyzerField()
        
        # Message log (for tracing)
        self.message_log: List[Message] = []
    
    def _log_message(self, msg: Message):
        """Log message for traceability."""
        self.message_log.append(msg)
    
    def analyze_digits(self, digits: np.ndarray) -> dict:
        """
        Run full pipeline on digit array.
        
        Returns analysis results with timing breakdown.
        """
        timings = {}
        start_total = time.perf_counter()
        
        # Phase 1: Block sums
        start = time.perf_counter()
        sums = self.sum_field.process_parallel(digits)
        timings['sum_field'] = time.perf_counter() - start
        
        self._log_message(Message(
            MsgType.SUMS, -1, -1, 0,
            f"{len(sums)} block sums computed"
        ))
        
        # Phase 2: Create windows and FFT
        start = time.perf_counter()
        n_windows = (len(sums) - self.window_size) // self.window_size + 1
        
        spectra = []
        for i in range(n_windows):
            window_start = i * self.window_size
            window = sums[window_start:window_start + self.window_size]
            window = window - window.mean()  # Remove DC
            spectrum = self.fft_field.power_spectrum(window)
            spectra.append(spectrum)
        
        timings['fft_field'] = time.perf_counter() - start
        
        self._log_message(Message(
            MsgType.SPECTRUM, -1, -1, 0,
            f"{len(spectra)} spectra computed"
        ))
        
        # Phase 3: Analyze whiteness
        start = time.perf_counter()
        whiteness_scores = self.analyzer_field.analyze_parallel(spectra)
        timings['analyzer_field'] = time.perf_counter() - start
        
        whiteness_array = np.array(whiteness_scores)
        
        self._log_message(Message(
            MsgType.WHITENESS, -1, -1, 0,
            f"mean={whiteness_array.mean():.6f}"
        ))
        
        timings['total'] = time.perf_counter() - start_total
        
        return {
            'n_digits': len(digits),
            'n_windows': n_windows,
            'whiteness_mean': whiteness_array.mean(),
            'whiteness_std': whiteness_array.std(),
            'timings': timings,
            'throughput_mps': len(digits) / timings['total'] / 1e6,
        }
    
    def generate_and_analyze(
        self, 
        constant: str = 'pi',
        n_digits: int = 1_000_000
    ) -> dict:
        """Generate digits and analyze in one pipeline."""
        
        # Create generator field
        generator_field = DigitGeneratorField(self.num_generators, constant)
        
        start = time.perf_counter()
        digits = generator_field.generate_parallel(n_digits)
        gen_time = time.perf_counter() - start
        
        self._log_message(Message(
            MsgType.DIGITS, -1, -1, 0,
            f"{len(digits)} digits generated in {gen_time:.2f}s"
        ))
        
        # Analyze
        result = self.analyze_digits(digits)
        result['generation_time'] = gen_time
        result['constant'] = constant
        
        return result


# =============================================================================
# GPU-ACCELERATED HOLLYWOOD PROBE
# =============================================================================

class HollywoodProbeGPU:
    """
    GPU-accelerated Hollywood Squares probe.
    
    Uses CUDA for massive parallelism while maintaining
    the Hollywood Squares coordination model.
    """
    
    def __init__(
        self,
        window_size: int = 1024,
        block_size: int = 100,
        device: str = 'cuda'
    ):
        self.window_size = window_size
        self.block_size = block_size
        self.device = device
        self.n_freqs = window_size // 2
    
    def analyze(self, digits_gpu: torch.Tensor) -> dict:
        """
        Analyze pre-loaded GPU tensor.
        
        All fields execute on GPU with implicit parallelism.
        """
        start = time.perf_counter()
        
        # Sum Field (vectorized on GPU)
        n_blocks = len(digits_gpu) // self.block_size
        reshaped = digits_gpu[:n_blocks * self.block_size].reshape(n_blocks, self.block_size)
        sums = reshaped.sum(dim=1).double()
        
        # FFT Field (batched on GPU)
        n_windows = (len(sums) - self.window_size) // self.window_size + 1
        windows = sums.unfold(0, self.window_size, self.window_size)
        windows = windows - windows.mean(dim=1, keepdim=True)
        
        fft_result = torch.fft.fft(windows, dim=-1)
        power = (fft_result.real**2 + fft_result.imag**2)[:, :self.n_freqs]
        
        # Analyzer Field (reduction on GPU)
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


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark():
    """Benchmark Hollywood Probe implementations."""
    print("=" * 70)
    print("HOLLYWOOD SQUARES PROBE BENCHMARK")
    print("=" * 70)
    
    # Test CPU implementation
    print("\n[1] CPU Hollywood Probe")
    print("-" * 50)
    
    probe_cpu = HollywoodProbe(window_size=1024, block_size=100)
    
    for n in [100_000, 1_000_000]:
        digits = np.random.randint(0, 10, size=n, dtype=np.int8)
        result = probe_cpu.analyze_digits(digits)
        print(f"  {n:>10,} digits: {result['throughput_mps']:.1f} M/s, "
              f"whiteness={result['whiteness_mean']:.4f}")
    
    # Test GPU implementation
    if torch.cuda.is_available():
        print("\n[2] GPU Hollywood Probe")
        print("-" * 50)
        
        probe_gpu = HollywoodProbeGPU(window_size=1024, block_size=100)
        
        for n in [1_000_000, 10_000_000, 100_000_000]:
            digits = torch.randint(0, 10, (n,), device='cuda', dtype=torch.float32)
            result = probe_gpu.analyze(digits)
            print(f"  {n:>12,} digits: {result['throughput_bps']/1e9:.1f} B/s, "
                  f"whiteness={result['whiteness_mean']:.4f}")
    
    # Test parallel digit generation
    print("\n[3] Parallel Digit Generation")
    print("-" * 50)
    
    for n_workers in [1, 2, 4, 8]:
        gen_field = DigitGeneratorField(num_workers=n_workers, constant='e')
        
        start = time.perf_counter()
        digits = gen_field.generate_parallel(100_000)
        elapsed = time.perf_counter() - start
        
        rate = len(digits) / elapsed
        print(f"  {n_workers} workers: {len(digits):,} digits in {elapsed:.2f}s "
              f"({rate/1000:.0f} K/s)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    benchmark()
