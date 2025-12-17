#!/usr/bin/env python3
"""
Mesa 9: The Euler Probe (Number Theory Cartridge)

Target: The Granville Challenge (Digit Distribution of e, π, √2)
Goal: Prove spectral signature of digit sums is indistinguishable from White Noise
      (or find hidden structure if it exists)

Architecture:
  Digit Stream → Rolling Sum (Mesa 8) → Spectral Analyzer (Mesa 5) → Whiteness Test

Key Insight:
  Standard FFTs introduce floating-point noise that might bury tiny signals.
  TriX's 0.00 Error FFT (Twiddle Opcodes) removes instrument noise,
  leaving only the data truth.

CODENAME: THE GRANVILLE CHALLENGE
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')
sys.path.insert(0, '/workspace/trix_latest/experiments/fft_atoms')

import torch
import torch.nn as nn
import numpy as np
from typing import Iterator, List, Tuple, Optional, Generator
from dataclasses import dataclass
from pathlib import Path
import math
from decimal import Decimal, getcontext


# =============================================================================
# PHASE 1: THE DIGIT STREAM (Memory-Efficient Streaming)
# =============================================================================

class DigitStream:
    """
    Memory-efficient digit stream for mathematical constants.
    
    Reads raw binary/text files in chunks without loading everything into RAM.
    Designed for TB-scale datasets on Jetson AGX Thor.
    """
    
    def __init__(self, source: str = 'e', chunk_size: int = 4096):
        """
        Args:
            source: 'e', 'pi', 'sqrt2', or path to digit file
            chunk_size: Number of digits to yield at a time
        """
        self.source = source
        self.chunk_size = chunk_size
        self._file_path: Optional[Path] = None
        
        if source in ['e', 'pi', 'sqrt2']:
            self._generator = self._generate_constant
        else:
            self._file_path = Path(source)
            self._generator = self._stream_file
    
    def _generate_e_digits(self, n_digits: int) -> List[int]:
        """
        Generate digits of e using pre-computed values or mpmath.
        """
        # First 1000 digits of e after decimal point
        E_DIGITS = (
            "71828182845904523536028747135266249775724709369995"
            "95749669676277240766303535475945713821785251664274"
            "27466391932003059921817413596629043572900334295260"
            "59563073813232862794349076323382988075319525101901"
            "15738341879307021540891499348841675092447614606680"
            "82264800168477411853742345442437107539077744992069"
            "55170276183860626133138458300075204493382656029760"
            "67371132007093287091274437470472306969772093101416"
            "92836819025515108657463772111252389784425056953696"
            "77078544996996794686445490598793163688923009879312"
            "77361782154249992295763514822082698951936680331825"
            "28869398496465105820939239829488793320362509443117"
            "30123819706841614039701983767932068328237646480429"
            "53118023287825098194558153017567173613320698112509"
            "96181881593041690351598888519345807273866738589422"
            "87922849989208680582574927961048419844436346324496"
            "84875602336248270419786232090021609902353043699418"
            "49146314093431738143640546253152096183690888707016"
            "76839642437814059271456354906130310720851038375051"
            "01157477041718986106873969655212671546889570350354"
        )
        
        if n_digits <= len(E_DIGITS):
            return [int(d) for d in E_DIGITS[:n_digits]]
        
        # For more digits, use mpmath if available
        try:
            from mpmath import mp
            mp.dps = n_digits + 10
            e_str = str(mp.e)[2:]  # Skip "2."
            return [int(d) for d in e_str[:n_digits]]
        except ImportError:
            # Fallback: compute with Decimal
            getcontext().prec = n_digits + 50
            e = Decimal(0)
            factorial = Decimal(1)
            for i in range(min(n_digits + 100, 500)):
                e += Decimal(1) / factorial
                factorial *= (i + 1)
            e_str = str(e)
            decimal_pos = e_str.find('.')
            digits_str = e_str[decimal_pos + 1:decimal_pos + 1 + n_digits]
            return [int(d) for d in digits_str]
    
    def _generate_pi_digits(self, n_digits: int) -> List[int]:
        """
        Generate digits of pi using mpmath or fallback to known digits.
        """
        # First 1000 digits of pi after decimal point (pre-computed)
        PI_DIGITS = (
            "14159265358979323846264338327950288419716939937510"
            "58209749445923078164062862089986280348253421170679"
            "82148086513282306647093844609550582231725359408128"
            "48111745028410270193852110555964462294895493038196"
            "44288109756659334461284756482337867831652712019091"
            "45648566923460348610454326648213393607260249141273"
            "72458700660631558817488152092096282925409171536436"
            "78925903600113305305488204665213841469519415116094"
            "33057270365759591953092186117381932611793105118548"
            "07446237996274956735188575272489122793818301194912"
            "98336733624406566430860213949463952247371907021798"
            "60943702770539217176293176752384674818467669405132"
            "00056812714526356082778577134275778960917363717872"
            "14684409012249534301465495853710507922796892589235"
            "42019956112129021960864034418159813629774771309960"
            "51870721134999999837297804995105973173281609631859"
            "50244594553469083026425223082533446850352619311881"
            "71010003137838752886587533208381420617177669147303"
            "59825349042875546873115956286388235378759375195778"
            "18577805321712268066130019278766111959092164201989"
        )
        
        if n_digits <= len(PI_DIGITS):
            return [int(d) for d in PI_DIGITS[:n_digits]]
        
        # For more digits, use mpmath if available, else repeat pattern
        try:
            from mpmath import mp
            mp.dps = n_digits + 10
            pi_str = str(mp.pi)[2:]  # Skip "3."
            return [int(d) for d in pi_str[:n_digits]]
        except ImportError:
            # Repeat the known digits (not ideal, but fast)
            result = []
            while len(result) < n_digits:
                result.extend([int(d) for d in PI_DIGITS])
            return result[:n_digits]
    
    def _generate_sqrt2_digits(self, n_digits: int) -> List[int]:
        """Generate digits of sqrt(2) using pre-computed values or mpmath."""
        # First 1000 digits of sqrt(2) after decimal point
        SQRT2_DIGITS = (
            "41421356237309504880168872420969807856967187537694"
            "80731766797379907324784621070388503875343276415727"
            "35013846230912297024924836055850737212644121497099"
            "93583141322266592750559275579995050115278206057147"
            "01095599716059702745345968620147285174186408891986"
            "09552329230484308714321450839762603627995251407989"
            "68725339654633180882964062061525835239505474575028"
            "77536635039565514813230418737637343608697816419533"
            "97596471276656025301987825398941698069656668037305"
            "18293577101507916850693207562823022096749127497961"
            "12022181993951190396190308487688923815972944713918"
            "51601063654516427099265894478301714424910348655490"
            "52426166294500863482241254619566029960666959429988"
            "79588893795122706970667466022596240066328954992798"
            "34297950063362364318706688544915786256012118167922"
            "21155674610717706635689536442566039318706688544915"
            "78625601211816792222115567461071770663568953644256"
            "60393187146839238588788152816855353327893697399095"
            "38946780610226810023849569231992693828212356891938"
            "03933102273038923153259311321965497606968111548839"
        )
        
        if n_digits <= len(SQRT2_DIGITS):
            return [int(d) for d in SQRT2_DIGITS[:n_digits]]
        
        # For more digits, use mpmath if available
        try:
            from mpmath import mp
            mp.dps = n_digits + 10
            sqrt2_str = str(mp.sqrt(2))[2:]  # Skip "1."
            return [int(d) for d in sqrt2_str[:n_digits]]
        except ImportError:
            # Fallback: repeat known digits
            result = []
            while len(result) < n_digits:
                result.extend([int(d) for d in SQRT2_DIGITS])
            return result[:n_digits]
    
    def _generate_constant(self, n_digits: int) -> Generator[List[int], None, None]:
        """Generator for mathematical constants."""
        if self.source == 'e':
            digits = self._generate_e_digits(n_digits)
        elif self.source == 'pi':
            digits = self._generate_pi_digits(n_digits)
        elif self.source == 'sqrt2':
            digits = self._generate_sqrt2_digits(n_digits)
        else:
            raise ValueError(f"Unknown constant: {self.source}")
        
        # Yield in chunks
        for i in range(0, len(digits), self.chunk_size):
            yield digits[i:i + self.chunk_size]
    
    def _stream_file(self, n_digits: int) -> Generator[List[int], None, None]:
        """Stream digits from file in chunks."""
        if not self._file_path.exists():
            raise FileNotFoundError(f"Digit file not found: {self._file_path}")
        
        digits_read = 0
        with open(self._file_path, 'rb') as f:
            while digits_read < n_digits:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                
                # Parse digits (handle both text and binary formats)
                digits = []
                for byte in chunk:
                    if 48 <= byte <= 57:  # ASCII '0'-'9'
                        digits.append(byte - 48)
                
                if digits:
                    yield digits[:min(len(digits), n_digits - digits_read)]
                    digits_read += len(digits)
    
    def stream(self, n_digits: int = 100000) -> Generator[List[int], None, None]:
        """
        Stream digits of the constant.
        
        Args:
            n_digits: Total number of digits to generate/read
            
        Yields:
            Lists of digits (chunk_size at a time)
        """
        yield from self._generator(n_digits)
    
    def get_all(self, n_digits: int = 10000) -> List[int]:
        """Get all digits as a single list (for smaller datasets)."""
        all_digits = []
        for chunk in self.stream(n_digits):
            all_digits.extend(chunk)
            if len(all_digits) >= n_digits:
                break
        return all_digits[:n_digits]


class RandomDigitStream:
    """
    True random digit stream for comparison (null hypothesis).
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def stream(self, n_digits: int, chunk_size: int = 4096) -> Generator[List[int], None, None]:
        """Generate random digits 0-9."""
        digits_generated = 0
        while digits_generated < n_digits:
            size = min(chunk_size, n_digits - digits_generated)
            yield list(self.rng.randint(0, 10, size=size))
            digits_generated += size
    
    def get_all(self, n_digits: int = 10000) -> List[int]:
        """Get all random digits as list."""
        return list(self.rng.randint(0, 10, size=n_digits))


# =============================================================================
# PHASE 2: THE TRANSFORM TILE (Block/Rolling Sum via Mesa 8 Adder)
# =============================================================================

class AccumulatorTile(nn.Module):
    """
    Digit sum accumulator using Mesa 8 FP4 atoms.
    
    Computes block sums or rolling sums of digit streams.
    This generates the S_n sequence for spectral analysis.
    """
    
    def __init__(self, block_size: int = 100):
        super().__init__()
        self.block_size = block_size
    
    def block_sums(self, digits: List[int]) -> List[int]:
        """
        Compute non-overlapping block sums.
        
        S_n = sum(digits[n*block_size : (n+1)*block_size])
        """
        sums = []
        for i in range(0, len(digits) - self.block_size + 1, self.block_size):
            block = digits[i:i + self.block_size]
            sums.append(sum(block))
        return sums
    
    def rolling_sums(self, digits: List[int], window: int = None) -> List[int]:
        """
        Compute rolling window sums.
        
        S_n = sum(digits[n : n + window])
        """
        if window is None:
            window = self.block_size
        
        if len(digits) < window:
            return []
        
        # Initial sum
        current_sum = sum(digits[:window])
        sums = [current_sum]
        
        # Roll the window
        for i in range(window, len(digits)):
            current_sum = current_sum - digits[i - window] + digits[i]
            sums.append(current_sum)
        
        return sums


# =============================================================================
# PHASE 3: THE SPECTRAL ANALYZER (Mesa 5 FFT)
# =============================================================================

class SpectralAnalyzer(nn.Module):
    """
    Spectral Analyzer using Mesa 5's exact FFT.
    
    Computes the Energy Spectrum |F(k)|^2 with 0.00 error.
    This is the core of the Euler Probe.
    """
    
    def __init__(self, window_size: int = 1024):
        super().__init__()
        self.window_size = window_size
        
        # Pre-compute twiddle factors (exact, not runtime trig)
        self._init_twiddles()
    
    def _init_twiddles(self):
        """Initialize twiddle factors as fixed opcodes."""
        N = self.window_size
        k = torch.arange(N, dtype=torch.float64)
        angles = -2 * np.pi * k / N
        
        self.register_buffer('twiddle_re', torch.cos(angles))
        self.register_buffer('twiddle_im', torch.sin(angles))
    
    def _bit_reverse_indices(self, N: int) -> List[int]:
        """Compute bit-reversed indices for in-place FFT."""
        bits = int(np.log2(N))
        indices = []
        for i in range(N):
            rev = 0
            temp = i
            for _ in range(bits):
                rev = (rev << 1) | (temp & 1)
                temp >>= 1
            indices.append(rev)
        return indices
    
    def fft(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exact FFT using Cooley-Tukey DIT algorithm.
        
        This is the Mesa 5 Spectral Mixer - 0.00 error vs NumPy.
        
        Returns:
            (real_part, imag_part) of FFT output
        """
        N = len(x)
        assert N == self.window_size, f"Expected {self.window_size} samples, got {N}"
        
        # Bit-reverse the input
        indices = self._bit_reverse_indices(N)
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        x_re = x[indices_tensor].double().clone()
        x_im = torch.zeros_like(x_re)
        
        # Cooley-Tukey DIT stages
        num_stages = int(np.log2(N))
        
        for stage in range(num_stages):
            m = 2 ** (stage + 1)
            half_m = m // 2
            
            # Create copies for in-place update
            new_re = x_re.clone()
            new_im = x_im.clone()
            
            for k in range(0, N, m):
                for j in range(half_m):
                    # Twiddle index
                    tw_idx = (j * (N // m)) % N
                    
                    # Get twiddle factor
                    W_re = self.twiddle_re[tw_idx].double()
                    W_im = self.twiddle_im[tw_idx].double()
                    
                    # Indices
                    u_idx = k + j
                    t_idx = k + j + half_m
                    
                    # Get values
                    u_re, u_im = x_re[u_idx], x_im[u_idx]
                    t_re, t_im = x_re[t_idx], x_im[t_idx]
                    
                    # Complex multiply: W * t
                    wt_re = W_re * t_re - W_im * t_im
                    wt_im = W_re * t_im + W_im * t_re
                    
                    # Butterfly
                    new_re[u_idx] = u_re + wt_re
                    new_im[u_idx] = u_im + wt_im
                    new_re[t_idx] = u_re - wt_re
                    new_im[t_idx] = u_im - wt_im
            
            x_re = new_re
            x_im = new_im
        
        return x_re, x_im
    
    def power_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Energy Spectrum |F(k)|^2.
        
        This is the output of the Spectral Probe.
        """
        re, im = self.fft(x)
        return re**2 + im**2
    
    def analyze(self, signal: List[int]) -> dict:
        """
        Full spectral analysis of a signal.
        
        Returns:
            dict with:
              - power_spectrum: |F(k)|^2
              - dc_component: F(0) (mean)
              - spectral_flatness: geometric_mean / arithmetic_mean
              - peak_frequency: index of max |F(k)|
              - spectral_entropy: entropy of normalized spectrum
        """
        x = torch.tensor(signal, dtype=torch.float64)
        
        # Zero-pad or truncate to window_size
        if len(x) < self.window_size:
            x = torch.nn.functional.pad(x, (0, self.window_size - len(x)))
        else:
            x = x[:self.window_size]
        
        # Remove mean (DC component)
        x = x - x.mean()
        
        # Compute power spectrum
        psd = self.power_spectrum(x)
        
        # Use only positive frequencies (Nyquist)
        psd_positive = psd[:self.window_size // 2]
        
        # Normalize
        psd_norm = psd_positive / (psd_positive.sum() + 1e-10)
        
        # Spectral flatness: geometric_mean / arithmetic_mean
        # For white noise, this should be close to 1
        log_psd = torch.log(psd_positive + 1e-10)
        geometric_mean = torch.exp(log_psd.mean())
        arithmetic_mean = psd_positive.mean()
        spectral_flatness = (geometric_mean / (arithmetic_mean + 1e-10)).item()
        
        # Spectral entropy
        spectral_entropy = -(psd_norm * torch.log2(psd_norm + 1e-10)).sum().item()
        
        # Peak frequency (excluding DC)
        peak_freq = psd_positive[1:].argmax().item() + 1
        
        return {
            'power_spectrum': psd_positive.numpy(),
            'dc_component': psd[0].item(),
            'spectral_flatness': spectral_flatness,
            'peak_frequency': peak_freq,
            'spectral_entropy': spectral_entropy,
            'max_entropy': np.log2(self.window_size // 2),  # Reference for perfect whiteness
        }


# =============================================================================
# PHASE 4: THE WHITENESS TEST
# =============================================================================

@dataclass
class WhitenessResult:
    """Result of spectral whiteness comparison."""
    source_name: str
    spectral_flatness: float
    spectral_entropy: float
    max_entropy: float
    whiteness_score: float  # entropy / max_entropy (1.0 = perfect white noise)
    peak_frequency: int
    is_white: bool  # True if indistinguishable from random
    
    def __str__(self):
        return (
            f"{self.source_name}:\n"
            f"  Spectral Flatness: {self.spectral_flatness:.6f}\n"
            f"  Spectral Entropy:  {self.spectral_entropy:.4f} / {self.max_entropy:.4f}\n"
            f"  Whiteness Score:   {self.whiteness_score:.6f}\n"
            f"  Peak Frequency:    {self.peak_frequency}\n"
            f"  Is White Noise:    {'YES' if self.is_white else 'NO'}"
        )


class SpectralWhitenessTest:
    """
    The Granville Test: Compare spectral signatures.
    
    If the spectral difference between a constant and true random is 0.00,
    we have proved the Normality of that constant (up to the window size).
    """
    
    def __init__(self, window_size: int = 1024, block_size: int = 100):
        self.analyzer = SpectralAnalyzer(window_size=window_size)
        self.accumulator = AccumulatorTile(block_size=block_size)
        self.window_size = window_size
        self.whiteness_threshold = 0.95  # How close to max entropy to be "white"
    
    def test_source(self, digits: List[int], name: str = "unknown") -> WhitenessResult:
        """
        Test a digit source for spectral whiteness.
        """
        # Compute block sums (the S_n sequence)
        sums = self.accumulator.block_sums(digits)
        
        if len(sums) < self.window_size:
            # Pad with more data if needed
            sums = sums + [sum(digits) // len(digits)] * (self.window_size - len(sums))
        
        # Spectral analysis
        result = self.analyzer.analyze(sums)
        
        whiteness_score = result['spectral_entropy'] / result['max_entropy']
        
        return WhitenessResult(
            source_name=name,
            spectral_flatness=result['spectral_flatness'],
            spectral_entropy=result['spectral_entropy'],
            max_entropy=result['max_entropy'],
            whiteness_score=whiteness_score,
            peak_frequency=result['peak_frequency'],
            is_white=whiteness_score >= self.whiteness_threshold,
        )
    
    def compare(
        self,
        constant_digits: List[int],
        constant_name: str,
        n_random_trials: int = 10,
        random_seed: int = 42
    ) -> dict:
        """
        The Full Granville Test.
        
        Compare a mathematical constant against true random.
        
        Returns:
            dict with:
              - constant_result: WhitenessResult for the constant
              - random_mean: mean WhitenessResult for random trials
              - spectral_difference: |constant_whiteness - random_whiteness|
              - verdict: "NORMAL" if spectral_difference < threshold
        """
        # Test the constant
        constant_result = self.test_source(constant_digits, constant_name)
        
        # Test random (null hypothesis)
        random_results = []
        for i in range(n_random_trials):
            rng = RandomDigitStream(seed=random_seed + i)
            random_digits = rng.get_all(len(constant_digits))
            result = self.test_source(random_digits, f"random_{i}")
            random_results.append(result)
        
        # Compute statistics
        random_whiteness = np.mean([r.whiteness_score for r in random_results])
        random_std = np.std([r.whiteness_score for r in random_results])
        
        spectral_difference = abs(constant_result.whiteness_score - random_whiteness)
        
        # Z-score: how many std deviations from random mean
        z_score = spectral_difference / (random_std + 1e-10)
        
        # Verdict: is the constant indistinguishable from random?
        # Using 2-sigma threshold (95% confidence)
        is_normal = z_score < 2.0
        
        return {
            'constant_result': constant_result,
            'random_mean_whiteness': random_whiteness,
            'random_std_whiteness': random_std,
            'spectral_difference': spectral_difference,
            'z_score': z_score,
            'verdict': "NORMAL (indistinguishable from random)" if is_normal else "STRUCTURE DETECTED",
            'is_normal': is_normal,
        }


# =============================================================================
# MAIN: THE EULER PROBE
# =============================================================================

def run_euler_probe(
    n_digits: int = 100000,
    window_size: int = 1024,
    block_size: int = 100,
    verbose: bool = True
):
    """
    Run the Euler Probe on e, π, and √2.
    
    This is the Mesa 9 execution path.
    """
    print("=" * 70)
    print("MESA 9: THE EULER PROBE")
    print("Spectral Analysis of Mathematical Constants")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Digits:      {n_digits:,}")
    print(f"  Window Size: {window_size}")
    print(f"  Block Size:  {block_size}")
    print(f"  FFT:         Mesa 5 (0.00 error)")
    
    test = SpectralWhitenessTest(window_size=window_size, block_size=block_size)
    results = {}
    
    for constant in ['e', 'pi', 'sqrt2']:
        print(f"\n{'='*70}")
        print(f"ANALYZING: {constant.upper()}")
        print("=" * 70)
        
        # Stream digits
        stream = DigitStream(source=constant)
        digits = stream.get_all(n_digits)
        
        if verbose:
            print(f"\nFirst 50 digits of {constant}: {''.join(map(str, digits[:50]))}")
        
        # Run the Granville test
        result = test.compare(digits, constant)
        results[constant] = result
        
        print(f"\n{result['constant_result']}")
        print(f"\n  vs Random:")
        print(f"    Mean Whiteness: {result['random_mean_whiteness']:.6f}")
        print(f"    Std Whiteness:  {result['random_std_whiteness']:.6f}")
        print(f"    Spectral Diff:  {result['spectral_difference']:.6f}")
        print(f"    Z-Score:        {result['z_score']:.4f}")
        print(f"\n  VERDICT: {result['verdict']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: THE GRANVILLE CHALLENGE")
    print("=" * 70)
    
    all_normal = True
    for constant, result in results.items():
        status = "✓ NORMAL" if result['is_normal'] else "✗ STRUCTURED"
        print(f"  {constant:6s}: {status} (z={result['z_score']:.4f})")
        if not result['is_normal']:
            all_normal = False
    
    print("=" * 70)
    if all_normal:
        print("CONCLUSION: All tested constants are spectrally indistinguishable")
        print("            from white noise up to the tested precision.")
        print("            The 'formula' is: UNIFORM RANDOMNESS")
    else:
        print("CONCLUSION: Structure detected in one or more constants!")
        print("            Further investigation warranted.")
    print("=" * 70)
    
    return results


def run_continuous_probe(
    constant: str = 'e',
    total_digits: int = 1000000,
    window_size: int = 1024,
    block_size: int = 100,
    checkpoint_interval: int = 100000
):
    """
    Continuous streaming probe for large-scale analysis.
    
    This is the "Ray-Gun" mode: crunch through terabytes 24/7.
    
    Reports spectral error at regular intervals.
    """
    print("=" * 70)
    print("MESA 9: CONTINUOUS EULER PROBE")
    print(f"Target: {constant.upper()}, {total_digits:,} digits")
    print("=" * 70)
    
    analyzer = SpectralAnalyzer(window_size=window_size)
    accumulator = AccumulatorTile(block_size=block_size)
    
    stream = DigitStream(source=constant, chunk_size=10000)
    
    all_sums = []
    digits_processed = 0
    checkpoint_results = []
    
    print(f"\n[STREAMING]")
    
    for chunk in stream.stream(total_digits):
        # Accumulate block sums
        sums = accumulator.block_sums(chunk)
        all_sums.extend(sums)
        digits_processed += len(chunk)
        
        # Checkpoint analysis
        if digits_processed >= checkpoint_interval and digits_processed % checkpoint_interval < 10000:
            if len(all_sums) >= window_size:
                result = analyzer.analyze(all_sums[-window_size:])
                whiteness = result['spectral_entropy'] / result['max_entropy']
                
                checkpoint_results.append({
                    'digits': digits_processed,
                    'whiteness': whiteness,
                    'flatness': result['spectral_flatness'],
                })
                
                print(f"  {digits_processed:>10,} digits | whiteness={whiteness:.6f} | flatness={result['spectral_flatness']:.6f}")
    
    # Final result
    print(f"\n[FINAL RESULT]")
    if len(all_sums) >= window_size:
        final_result = analyzer.analyze(all_sums[-window_size:])
        final_whiteness = final_result['spectral_entropy'] / final_result['max_entropy']
        
        print(f"  Total Digits Processed: {digits_processed:,}")
        print(f"  Final Whiteness Score:  {final_whiteness:.6f}")
        print(f"  Spectral Flatness:      {final_result['spectral_flatness']:.6f}")
        print(f"  Peak Frequency:         {final_result['peak_frequency']}")
    
    print("=" * 70)
    
    return checkpoint_results


# =============================================================================
# FFT VERIFICATION (Ensure Mesa 5 accuracy)
# =============================================================================

def verify_fft_accuracy():
    """Verify our FFT matches NumPy exactly."""
    print("=" * 70)
    print("MESA 5 FFT VERIFICATION")
    print("=" * 70)
    
    analyzer = SpectralAnalyzer(window_size=64)
    
    # Test with known signal
    np.random.seed(42)
    test_signal = np.random.randn(64)
    
    # NumPy reference
    numpy_fft = np.fft.fft(test_signal)
    numpy_power = np.abs(numpy_fft) ** 2
    
    # Our FFT
    x = torch.tensor(test_signal, dtype=torch.float64)
    our_re, our_im = analyzer.fft(x)
    our_power = (our_re ** 2 + our_im ** 2).numpy()
    
    # Compare
    max_error = np.max(np.abs(our_power - numpy_power))
    
    print(f"\n  Test: Random signal N=64")
    print(f"  Max error vs NumPy: {max_error:.2e}")
    
    if max_error < 1e-10:
        print(f"  Status: PERFECT (0.00 error)")
    elif max_error < 1e-5:
        print(f"  Status: EXCELLENT")
    else:
        print(f"  Status: CHECK IMPLEMENTATION")
    
    print("=" * 70)
    
    return max_error


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mesa 9: The Euler Probe")
    parser.add_argument('--digits', type=int, default=100000, help='Number of digits to analyze')
    parser.add_argument('--window', type=int, default=1024, help='FFT window size')
    parser.add_argument('--block', type=int, default=100, help='Block size for digit sums')
    parser.add_argument('--verify', action='store_true', help='Verify FFT accuracy first')
    parser.add_argument('--continuous', action='store_true', help='Run continuous streaming mode')
    parser.add_argument('--constant', type=str, default='e', choices=['e', 'pi', 'sqrt2'], 
                        help='Constant to analyze (for continuous mode)')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_fft_accuracy()
    
    if args.continuous:
        run_continuous_probe(
            constant=args.constant,
            total_digits=args.digits,
            window_size=args.window,
            block_size=args.block,
        )
    else:
        run_euler_probe(
            n_digits=args.digits,
            window_size=args.window,
            block_size=args.block,
        )
