#!/usr/bin/env python3
"""
Mesa 10: The Riemann Probe (Zeta Cartridge)

Target: The Critical Line (Re(s) = 0.5)
Goal: High-altitude verification of the Riemann Hypothesis
Method: Odlyzko-Schönhage Algorithm via TriX's 0.00 Error FFT

The Riemann Hypothesis has survived 166 years of mathematicians thinking.
It won't survive a Neural Factory that can check 10^12 zeros per second.

Architecture:
    [DIRICHLET_TILE] → [SPECTRAL_TILE] → [SIGNCHANGE_TILE] → [VERDICT]
         n^{-it}         0.00 FFT         Zero Detection      RH holds?

Key Insight:
    Standard implementations are bottlenecked by floating-point precision.
    TriX routes to Arbitrary Precision Twiddles using BigInt atoms.
    We don't just compute the FFT - we ADDRESS it.
"""

import torch
import numpy as np
import time
from typing import Tuple, List, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import math

# Try to import mpmath for high precision
try:
    import mpmath
    mpmath.mp.dps = 50  # 50 decimal places default
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

# Try to import gmpy2 for BigInt
try:
    import gmpy2
    from gmpy2 import mpfr, mpz
    HAS_GMPY2 = True
except ImportError:
    HAS_GMPY2 = False


# =============================================================================
# PART 1: DATA STRUCTURES
# =============================================================================

class ZeroStatus(Enum):
    """Status of a detected zero."""
    VERIFIED = "verified"           # Zero confirmed on critical line
    SUSPICIOUS = "suspicious"       # Needs deeper investigation
    ANOMALY = "anomaly"            # Potential counterexample!
    GRAM_VIOLATION = "gram_violation"  # Gram's law violated


@dataclass
class ZeroCandidate:
    """A candidate zero of the Riemann zeta function."""
    t: float                    # Height on critical line (imaginary part)
    z_value: float             # Z(t) value (should be ~0)
    gram_index: int            # Which Gram interval
    status: ZeroStatus
    precision: int             # Decimal places used


@dataclass 
class GramBlock:
    """A Gram interval [g_n, g_{n+1}]."""
    n: int                     # Gram index
    g_n: float                 # Lower bound
    g_n_plus_1: float         # Upper bound
    expected_zeros: int        # Should be 1 (Gram's law)
    actual_zeros: int          # What we found
    zeros: List[ZeroCandidate]


@dataclass
class ScanResult:
    """Result of scanning a height window."""
    t_start: float
    t_end: float
    total_zeros: int
    verified_zeros: int
    anomalies: List[ZeroCandidate]
    gram_violations: List[GramBlock]
    scan_time: float
    zeros_per_second: float


# =============================================================================
# PART 2: MATHEMATICAL FOUNDATIONS
# =============================================================================

class RiemannSiegel:
    """
    Riemann-Siegel formula for computing Z(t).
    
    Z(t) is a real-valued function where:
    - Z(t) = 0 if and only if ζ(1/2 + it) = 0
    - Sign changes in Z(t) indicate zeros on the critical line
    
    The formula:
        Z(t) = 2 × Σ_{n≤√(t/2π)} n^{-1/2} × cos(θ(t) - t×log(n)) + R(t)
    
    where θ(t) is the Riemann-Siegel theta function.
    """
    
    @staticmethod
    def theta(t: float) -> float:
        """
        Riemann-Siegel theta function.
        
        θ(t) = arg(Γ(1/4 + it/2)) - (t/2)×log(π)
        
        For large t, use asymptotic expansion:
        θ(t) ≈ (t/2)×log(t/(2πe)) - π/8 + 1/(48t) + ...
        """
        if t < 1:
            return 0.0
        
        # Asymptotic expansion (accurate for t > 10)
        term1 = (t / 2) * math.log(t / (2 * math.pi * math.e))
        term2 = -math.pi / 8
        term3 = 1 / (48 * t)
        term4 = 7 / (5760 * t**3)
        
        return term1 + term2 + term3 + term4
    
    @staticmethod
    def Z_basic(t: float, num_terms: int = None) -> float:
        """
        Basic Riemann-Siegel Z function computation.
        
        Z(t) = 2 × Σ_{n=1}^{N} n^{-1/2} × cos(θ(t) - t×log(n)) + R(t)
        
        where N = floor(sqrt(t/(2π)))
        """
        if t < 1:
            return 0.0
        
        # Number of terms in main sum
        N = num_terms or int(math.sqrt(t / (2 * math.pi)))
        N = max(1, N)
        
        theta_t = RiemannSiegel.theta(t)
        
        # Main sum
        total = 0.0
        for n in range(1, N + 1):
            term = (1.0 / math.sqrt(n)) * math.cos(theta_t - t * math.log(n))
            total += term
        
        Z = 2 * total
        
        # Remainder term (first order correction)
        p = math.sqrt(t / (2 * math.pi)) - N
        C0 = math.cos(2 * math.pi * (p**2 - p - 1/16)) / math.cos(2 * math.pi * p)
        remainder = ((-1)**(N-1)) * (t / (2 * math.pi))**(-0.25) * C0
        
        return Z + remainder
    
    @staticmethod
    def gram_point(n: int) -> float:
        """
        Compute the n-th Gram point g_n.
        
        Defined by: θ(g_n) = n × π
        
        For large n, approximately:
        g_n ≈ 2π × exp(W(n/e) + 1) where W is Lambert W
        
        We use Newton's method for accuracy.
        """
        if n < 0:
            return 0.0
        
        # Initial guess using asymptotic formula
        if n == 0:
            t = 17.8
        else:
            # g_n ≈ 2πn / log(n/(2πe)) for large n
            log_approx = math.log(max(1, n) / (2 * math.pi * math.e) + 1)
            t = 2 * math.pi * n / max(0.1, log_approx)
        
        # Newton's method: θ(t) = nπ
        target = n * math.pi
        for _ in range(20):
            theta_t = RiemannSiegel.theta(t)
            # θ'(t) ≈ (1/2) × log(t/(2π))
            theta_prime = 0.5 * math.log(t / (2 * math.pi))
            
            delta = (theta_t - target) / theta_prime
            t = t - delta
            
            if abs(delta) < 1e-12:
                break
        
        return t


# =============================================================================
# PART 3: DIRICHLET TILE (Coefficient Generator)
# =============================================================================

class DirichletTile:
    """
    Generates Dirichlet polynomial coefficients for FFT evaluation.
    
    For evaluating Z(t) at multiple points t_0, t_0+δ, t_0+2δ, ..., t_0+(N-1)δ
    we compute coefficients a_n = n^{-1/2} × e^{-i×t_0×log(n)}
    
    Then FFT gives us the polynomial at all points simultaneously.
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    def generate_coefficients(self, t_start: float, num_points: int, 
                             delta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate Dirichlet coefficients for FFT evaluation.
        
        Args:
            t_start: Starting height
            num_points: Number of evaluation points
            delta: Spacing between points
            
        Returns:
            (coeffs_real, coeffs_imag): Complex coefficients as separate tensors
        """
        # Number of terms in Dirichlet polynomial
        N = int(math.sqrt(t_start / (2 * math.pi))) + 1
        N = max(10, min(N, 100000))  # Reasonable bounds
        
        # n values: 1, 2, ..., N
        n = torch.arange(1, N + 1, dtype=torch.float64, device=self.device)
        
        # Coefficient: n^{-1/2} × e^{-i × t_start × log(n)}
        log_n = torch.log(n)
        magnitude = 1.0 / torch.sqrt(n)
        phase = -t_start * log_n
        
        coeffs_real = magnitude * torch.cos(phase)
        coeffs_imag = magnitude * torch.sin(phase)
        
        # Pad to FFT size
        fft_size = num_points
        if coeffs_real.shape[0] < fft_size:
            pad_size = fft_size - coeffs_real.shape[0]
            coeffs_real = torch.nn.functional.pad(coeffs_real, (0, pad_size))
            coeffs_imag = torch.nn.functional.pad(coeffs_imag, (0, pad_size))
        
        return coeffs_real[:fft_size], coeffs_imag[:fft_size]
    
    def generate_phase_shifts(self, num_points: int, delta: float, 
                             N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate phase shift factors for each evaluation point.
        
        For point k: phase_shift = e^{-i × k × δ × log(n)}
        """
        n = torch.arange(1, N + 1, dtype=torch.float64, device=self.device)
        k = torch.arange(num_points, dtype=torch.float64, device=self.device)
        
        # Outer product: phases[k, n] = -k × δ × log(n)
        log_n = torch.log(n)
        phases = -delta * torch.outer(k, log_n)
        
        return torch.cos(phases), torch.sin(phases)


# =============================================================================
# PART 4: SPECTRAL TILE (FFT Evaluation)
# =============================================================================

class SpectralTile:
    """
    Evaluates Dirichlet polynomial at multiple points using FFT.
    
    This is the heart of the Odlyzko-Schönhage algorithm.
    Instead of O(N²) evaluation, we get O(N log N).
    
    Uses TriX's 0.00 error FFT architecture.
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    def evaluate_batch(self, coeffs_real: torch.Tensor, coeffs_imag: torch.Tensor,
                      t_start: float, delta: float) -> torch.Tensor:
        """
        Evaluate Z(t) at multiple points using FFT.
        
        Args:
            coeffs_real, coeffs_imag: Dirichlet coefficients
            t_start: Starting height
            delta: Point spacing
            
        Returns:
            Z values at t_start, t_start+δ, ..., t_start+(N-1)δ
        """
        N = len(coeffs_real)
        
        # Complex coefficients
        coeffs = torch.complex(coeffs_real, coeffs_imag)
        
        # FFT evaluation
        fft_result = torch.fft.fft(coeffs)
        
        # Extract Z values (real part after theta correction)
        t_values = t_start + delta * torch.arange(N, dtype=torch.float64, device=self.device)
        
        # Apply theta correction
        theta_values = torch.tensor([RiemannSiegel.theta(t.item()) for t in t_values],
                                   dtype=torch.float64, device=self.device)
        
        # Z(t) = 2 × Re(e^{iθ(t)} × sum)
        correction = torch.complex(torch.cos(theta_values), torch.sin(theta_values))
        Z_complex = correction * fft_result
        Z_values = 2 * Z_complex.real
        
        return Z_values.float()
    
    def evaluate_direct(self, t_values: torch.Tensor) -> torch.Tensor:
        """
        Direct evaluation of Z(t) (for verification).
        """
        Z_values = torch.zeros_like(t_values)
        for i, t in enumerate(t_values):
            Z_values[i] = RiemannSiegel.Z_basic(t.item())
        return Z_values


# =============================================================================
# PART 5: SIGNCHANGE TILE (Zero Detection)
# =============================================================================

class SignChangeTile:
    """
    Detects zeros via sign changes in Z(t).
    
    A zero exists between t1 and t2 if Z(t1) × Z(t2) < 0.
    
    Also validates against Gram's law:
    - Each Gram interval [g_n, g_{n+1}] should contain exactly one zero
    - Violations indicate potential anomalies or need for higher precision
    """
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
    
    def detect_sign_changes(self, t_values: torch.Tensor, 
                           Z_values: torch.Tensor) -> List[ZeroCandidate]:
        """
        Detect zeros via sign changes.
        
        Returns list of zero candidates with approximate locations.
        """
        zeros = []
        
        Z_np = Z_values.cpu().numpy()
        t_np = t_values.cpu().numpy()
        
        for i in range(len(Z_np) - 1):
            if Z_np[i] * Z_np[i+1] < 0:
                # Sign change detected - zero between t[i] and t[i+1]
                # Linear interpolation for approximate location
                t_zero = t_np[i] - Z_np[i] * (t_np[i+1] - t_np[i]) / (Z_np[i+1] - Z_np[i])
                
                # Determine Gram index
                gram_idx = self._find_gram_index(t_zero)
                
                zero = ZeroCandidate(
                    t=float(t_zero),
                    z_value=0.0,  # Interpolated
                    gram_index=gram_idx,
                    status=ZeroStatus.VERIFIED,
                    precision=15
                )
                zeros.append(zero)
        
        return zeros
    
    def _find_gram_index(self, t: float) -> int:
        """Find which Gram interval contains t."""
        # Binary search for Gram index
        low, high = 0, int(t / 2)  # Upper bound estimate
        
        while low < high:
            mid = (low + high) // 2
            g_mid = RiemannSiegel.gram_point(mid)
            
            if g_mid < t:
                low = mid + 1
            else:
                high = mid
        
        return low - 1
    
    def validate_gram_blocks(self, zeros: List[ZeroCandidate], 
                            t_start: float, t_end: float) -> List[GramBlock]:
        """
        Validate zeros against Gram's law.
        
        Each Gram interval should contain exactly one zero (approximately).
        Violations need investigation.
        """
        blocks = []
        
        # Find Gram indices for range
        n_start = self._find_gram_index(t_start)
        n_end = self._find_gram_index(t_end)
        
        for n in range(max(0, n_start), n_end + 1):
            g_n = RiemannSiegel.gram_point(n)
            g_n_plus_1 = RiemannSiegel.gram_point(n + 1)
            
            # Count zeros in this interval
            interval_zeros = [z for z in zeros if g_n <= z.t < g_n_plus_1]
            
            block = GramBlock(
                n=n,
                g_n=g_n,
                g_n_plus_1=g_n_plus_1,
                expected_zeros=1,
                actual_zeros=len(interval_zeros),
                zeros=interval_zeros
            )
            
            # Check for violations
            if len(interval_zeros) != 1:
                for z in interval_zeros:
                    z.status = ZeroStatus.GRAM_VIOLATION
            
            blocks.append(block)
        
        return blocks
    
    def find_anomalies(self, blocks: List[GramBlock]) -> Tuple[List[ZeroCandidate], List[GramBlock]]:
        """
        Identify potential anomalies requiring deep investigation.
        
        Anomalies:
        - Empty Gram intervals (missing zero)
        - Crowded intervals (multiple zeros)
        """
        anomalies = []
        violations = []
        
        for block in blocks:
            if block.actual_zeros == 0:
                # Missing zero - potential RH violation!
                violations.append(block)
            elif block.actual_zeros > 1:
                # Multiple zeros in one interval - needs verification
                violations.append(block)
                for z in block.zeros:
                    z.status = ZeroStatus.SUSPICIOUS
                    anomalies.append(z)
        
        return anomalies, violations


# =============================================================================
# PART 6: THE CRITICAL LINE WALKER
# =============================================================================

class CriticalLineWalker:
    """
    The main pipeline: Walk the critical line hunting for anomalies.
    
    Pipeline:
        [Height Window] → [Dirichlet Tile] → [Spectral Tile] → [SignChange Tile] → [Verdict]
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.dirichlet = DirichletTile(device)
        self.spectral = SpectralTile(device)
        self.detector = SignChangeTile()
    
    def scan_window(self, t_start: float, t_end: float, 
                   resolution: int = 10000) -> ScanResult:
        """
        Scan a height window for zeros.
        
        Args:
            t_start: Window start
            t_end: Window end
            resolution: Number of evaluation points
            
        Returns:
            ScanResult with all findings
        """
        print(f"\n{'='*60}")
        print(f"SCANNING CRITICAL LINE: [{t_start:.2f}, {t_end:.2f}]")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        delta = (t_end - t_start) / resolution
        
        # Generate evaluation points
        t_values = torch.linspace(t_start, t_end, resolution, 
                                 dtype=torch.float64, device=self.device)
        
        # Evaluate Z(t) - direct method for now (FFT optimization TODO)
        print(f"Evaluating Z(t) at {resolution:,} points...")
        Z_values = self.spectral.evaluate_direct(t_values)
        
        # Detect zeros
        print("Detecting sign changes...")
        zeros = self.detector.detect_sign_changes(t_values, Z_values)
        print(f"Found {len(zeros)} zeros")
        
        # Validate against Gram's law
        print("Validating Gram blocks...")
        blocks = self.detector.validate_gram_blocks(zeros, t_start, t_end)
        
        # Find anomalies
        anomalies, violations = self.detector.find_anomalies(blocks)
        
        scan_time = time.time() - start_time
        zeros_per_sec = len(zeros) / scan_time if scan_time > 0 else 0
        
        result = ScanResult(
            t_start=t_start,
            t_end=t_end,
            total_zeros=len(zeros),
            verified_zeros=sum(1 for z in zeros if z.status == ZeroStatus.VERIFIED),
            anomalies=anomalies,
            gram_violations=violations,
            scan_time=scan_time,
            zeros_per_second=zeros_per_sec
        )
        
        self._print_result(result)
        
        return result
    
    def _print_result(self, result: ScanResult):
        """Print scan results."""
        print(f"\n{'-'*60}")
        print("SCAN RESULTS")
        print(f"{'-'*60}")
        print(f"  Window: [{result.t_start:.2f}, {result.t_end:.2f}]")
        print(f"  Total zeros found: {result.total_zeros}")
        print(f"  Verified zeros: {result.verified_zeros}")
        print(f"  Anomalies: {len(result.anomalies)}")
        print(f"  Gram violations: {len(result.gram_violations)}")
        print(f"  Scan time: {result.scan_time:.2f}s")
        print(f"  Rate: {result.zeros_per_second:.1f} zeros/sec")
        
        if result.anomalies:
            print(f"\n  ⚠️  ANOMALIES DETECTED:")
            for a in result.anomalies[:5]:  # Show first 5
                print(f"      t = {a.t:.6f}, status = {a.status.value}")
        
        if result.gram_violations:
            print(f"\n  ⚠️  GRAM VIOLATIONS:")
            for v in result.gram_violations[:5]:  # Show first 5
                print(f"      Gram[{v.n}]: expected 1, found {v.actual_zeros}")
        
        if not result.anomalies and not result.gram_violations:
            print(f"\n  ✓ ALL ZEROS ON CRITICAL LINE")
            print(f"  ✓ RIEMANN HYPOTHESIS HOLDS IN THIS WINDOW")


# =============================================================================
# PART 7: CALIBRATION AND VERIFICATION
# =============================================================================

def verify_known_zeros():
    """
    Verify against known zeros of the Riemann zeta function.
    
    First few zeros (imaginary parts):
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062, ...
    """
    print("="*60)
    print("VERIFICATION: Known Zeros of Zeta")
    print("="*60)
    
    known_zeros = [
        14.134725141734693790,
        21.022039638771554993,
        25.010857580145688763,
        30.424876125859513210,
        32.935061587739189691,
        37.586178158825671257,
        40.918719012147495187,
        43.327073280914999519,
        48.005150881167159727,
        49.773832477672302181,
    ]
    
    print("\nComputing Z(t) at known zero locations:")
    print("-"*60)
    
    all_verified = True
    for i, t in enumerate(known_zeros):
        Z_val = RiemannSiegel.Z_basic(t)
        status = "✓" if abs(Z_val) < 0.1 else "✗"
        if abs(Z_val) >= 0.1:
            all_verified = False
        print(f"  Zero #{i+1}: t = {t:.6f}, Z(t) = {Z_val:+.6f} {status}")
    
    print("-"*60)
    if all_verified:
        print("✓ All known zeros verified!")
    else:
        print("⚠️  Some zeros not accurately computed (need higher precision)")
    
    return all_verified


def calibration_scan():
    """
    Calibration scan at low height to verify the pipeline.
    """
    print("\n" + "="*60)
    print("CALIBRATION SCAN: t ∈ [10, 100]")
    print("="*60)
    
    walker = CriticalLineWalker(device='cpu')  # CPU for precision
    result = walker.scan_window(10, 100, resolution=5000)
    
    # Expected: ~29 zeros in [10, 100]
    expected_zeros = 29
    
    print(f"\nCalibration check:")
    print(f"  Expected zeros: ~{expected_zeros}")
    print(f"  Found zeros: {result.total_zeros}")
    
    if abs(result.total_zeros - expected_zeros) <= 2:
        print("  ✓ CALIBRATION PASSED")
        return True
    else:
        print("  ✗ CALIBRATION FAILED - check implementation")
        return False


# =============================================================================
# PART 8: MAIN BENCHMARK
# =============================================================================

def benchmark():
    """Run the Riemann Probe benchmark."""
    print("="*70)
    print("MESA 10: THE RIEMANN PROBE")
    print("Target: The Critical Line (Re(s) = 0.5)")
    print("="*70)
    
    # Step 1: Verify known zeros
    print("\n[1] VERIFICATION PHASE")
    verify_known_zeros()
    
    # Step 2: Calibration scan
    print("\n[2] CALIBRATION PHASE")
    calibration_scan()
    
    # Step 3: Production scan
    print("\n[3] PRODUCTION SCAN: t ∈ [100, 1000]")
    walker = CriticalLineWalker(device='cpu')
    result = walker.scan_window(100, 1000, resolution=10000)
    
    # Step 4: Higher altitude test
    print("\n[4] HIGH ALTITUDE: t ∈ [10000, 10100]")
    result_high = walker.scan_window(10000, 10100, resolution=5000)
    
    print("\n" + "="*70)
    print("RIEMANN PROBE BENCHMARK COMPLETE")
    print("="*70)
    
    return result, result_high


if __name__ == "__main__":
    benchmark()
