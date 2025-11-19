#@title Cell 1: Shared Functions & Core Utilities

"""
================================================================================
CELL 1: SHARED FUNCTIONS & CORE UTILITIES
================================================================================
This cell contains all shared functions used across experiments.
Run this cell FIRST before any experiment cells.

Contents:
- PrimeCache class (efficient prime storage)
- Kahan summation (numerical stability)
- Segmented sieve (prime generation)
- S(T) computation methods (Direct, RS, Euler)
- smooth_RVM (baseline approximation)
- Helper functions
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, chisquare, linregress
from scipy.ndimage import gaussian_filter1d
import time
import pickle
import os
import warnings
from mpmath import mp
warnings.filterwarnings('ignore')

# High precision for reference calculations
mp.dps = 50
mp_pi = mp.pi

# Plotting defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
sns.set_palette("husl")

print("="*80)
print("CELL 1: SHARED FUNCTIONS LOADING...")
print("="*80 + "\n")

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

# Mount Google Drive (if not already mounted)
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
        print("✓ Google Drive mounted")
    else:
        print("✓ Google Drive already mounted")
except:
    print("⚠ Not running in Colab - Drive mount skipped")

# Create directory structure
BASE_DIR = '/content/drive/MyDrive/riemann_experiments'
DIRS = {
    'base': BASE_DIR,
    'cache': f'{BASE_DIR}/cache',
    'results': f'{BASE_DIR}/results',
    'figures': f'{BASE_DIR}/figures',
    'tables': f'{BASE_DIR}/tables'
}

for name, path in DIRS.items():
    os.makedirs(path, exist_ok=True)
    print(f"✓ Directory ready: {name} → {path}")

print()

# ============================================================================
# NUMERICAL UTILITIES
# ============================================================================

def kahan_sum(arr):
    """
    Kahan compensated summation for numerical stability.
    Essential for oscillatory sums with many cancellations.

    Parameters:
    -----------
    arr : array-like
        Array of numbers to sum

    Returns:
    --------
    float : Compensated sum
    """
    total = 0.0
    c = 0.0
    for x in arr:
        y = x - c
        t = total + y
        c = (t - total) - y
        total = t
    return total

def smooth_RVM(T):
    """
    Riemann-von Mangoldt smooth approximation for N(T).

    N_smooth(T) = (T/2π) log(T/2π) - T/2π + 7/8

    This is the asymptotic approximation without oscillatory S(T) term.
    """
    return (T/(2*np.pi)) * np.log(T/(2*np.pi)) - T/(2*np.pi) + 7.0/8.0

print("✓ Numerical utilities loaded (kahan_sum, smooth_RVM)")

# ============================================================================
# PRIME GENERATION
# ============================================================================

def segmented_sieve(limit: int) -> np.ndarray:
    """
    Generate all primes up to limit using segmented sieve.

    Memory-efficient algorithm that processes primes in segments.

    Parameters:
    -----------
    limit : int
        Upper bound (inclusive) for prime generation

    Returns:
    --------
    np.ndarray : Array of all primes p <= limit
    """
    if limit < 2:
        return np.array([], dtype=np.int64)

    # Generate small primes up to sqrt(limit)
    sqrt_limit = int(np.sqrt(limit)) + 1
    is_prime = np.ones(sqrt_limit, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(np.sqrt(sqrt_limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    small_primes = np.where(is_prime)[0]

    if limit <= sqrt_limit:
        return small_primes[small_primes <= limit]

    # Segment size (cache-friendly)
    segment_size = max(sqrt_limit, 32768)
    result = [small_primes]

    # Process segments
    for low in range(sqrt_limit, limit + 1, segment_size):
        high = min(low + segment_size, limit + 1)
        segment = np.ones(high - low, dtype=bool)

        for p in small_primes:
            if p < 2:
                continue
            start = max(p * p, ((low + p - 1) // p) * p)
            if start < high:
                segment[start - low::p] = False

        primes_in_segment = np.where(segment)[0] + low
        if len(primes_in_segment) > 0:
            result.append(primes_in_segment)

    return np.concatenate(result)

print("✓ Prime generation loaded (segmented_sieve)")

# ============================================================================
# PRIME CACHE CLASS
# ============================================================================

class PrimeCache:
    """
    Efficient cache for primes with precomputed logarithms.

    Features:
    - Generates all primes up to P_max once
    - Precomputes log(p) for efficiency
    - Provides subset() method for P_max' < P_max
    - Memory-efficient storage

    Attributes:
    -----------
    P_max : int
        Maximum prime bound
    primes : np.ndarray
        All primes up to P_max
    logp : np.ndarray
        Precomputed log(p) values
    """

    def __init__(self, P_max: int, verbose=True):
        """
        Initialize prime cache.

        Parameters:
        -----------
        P_max : int
            Generate all primes up to this bound
        verbose : bool
            Print progress messages
        """
        if verbose:
            print(f"Generating primes up to {P_max:,}...")

        start = time.time()
        self.P_max = int(P_max)
        self.primes = segmented_sieve(self.P_max)
        self.logp = np.log(self.primes)
        elapsed = time.time() - start

        if verbose:
            memory_mb = (self.primes.nbytes + self.logp.nbytes) / 1e6
            print(f"✓ Generated {len(self.primes):,} primes in {elapsed:.1f}s")
            print(f"  Max prime: {self.primes[-1]:,}")
            print(f"  Memory: {memory_mb:.1f} MB\n")

    def subset(self, P_max_subset: int):
        """
        Return view of primes up to smaller P_max.

        Parameters:
        -----------
        P_max_subset : int
            Smaller bound (must be <= self.P_max)

        Returns:
        --------
        primes, logp : tuple of np.ndarray
            Views into arrays (no copying)
        """
        if P_max_subset > self.P_max:
            raise ValueError(f"Requested P_max={P_max_subset} > cached P_max={self.P_max}")

        idx = np.searchsorted(self.primes, P_max_subset, side='right')
        return self.primes[:idx], self.logp[:idx]

    def __len__(self):
        return len(self.primes)

    def __repr__(self):
        return f"PrimeCache(P_max={self.P_max:,}, n_primes={len(self.primes):,})"

print("✓ PrimeCache class loaded")

# ============================================================================
# S(T) COMPUTATION METHODS
# ============================================================================

def S_direct(T, verbose=False):
    """
    Direct computation: S(T) = (1/π) Im log ζ(1/2+iT)

    Uses high-precision mpmath (50 decimal places).
    This is the REFERENCE method - always correct.

    Parameters:
    -----------
    T : float
        Height on critical line
    verbose : bool
        Print computation time

    Returns:
    --------
    float : S(T) value

    Note:
    -----
    Has discontinuous jumps at zero ordinates.
    Use at T values away from zeros (± 0.5).
    """
    if verbose:
        start = time.time()

    s = mp.mpf('0.5') + 1j * mp.mpf(T)
    z = mp.zeta(s)
    result = float(mp.im(mp.log(z)) / mp_pi)

    if verbose:
        elapsed = time.time() - start
        print(f"  S_direct({T:,}) = {result:+.6f} (computed in {elapsed:.3f}s)")

    return result

def S_riemann_siegel(T, verbose=False):
    """
    Riemann-Siegel formula for S(T).

    Uses direct sine-series formulation:
    S(T) = -(1/π) Σ_{n=1}^N n^{-1/2} sin(T log n)
    where N = floor(sqrt(T/(2π)))

    Complexity: O(sqrt(T))
    Accuracy: ~1-5% typically

    Parameters:
    -----------
    T : float
        Height on critical line
    verbose : bool
        Print details

    Returns:
    --------
    float : S(T) approximation
    """
    N = int(np.sqrt(T / (2 * np.pi)))
    if N < 1:
        N = 1

    n = np.arange(1, N + 1, dtype=np.float64)
    log_n = np.log(n)
    terms = (n ** (-0.5)) * np.sin(T * log_n)

    result = -np.sum(terms) / np.pi

    if verbose:
        print(f"  S_riemann_siegel({T:,}) = {result:+.6f} (N={N} terms)")

    return result

def S_euler_k1(T, P_max, prime_cache, verbose=False):
    """
    Euler product k=1 (primes only):
    S(T) ≈ -(1/π) Σ_p p^{-1/2} sin(T log p)

    This is the MAIN method we're optimizing.

    Complexity: O(1) - independent of T!
    Accuracy: Depends on P_max (our research question!)

    Parameters:
    -----------
    T : float
        Height on critical line
    P_max : float
        Prime truncation bound
    prime_cache : PrimeCache
        Cache with primes >= P_max
    verbose : bool
        Print details

    Returns:
    --------
    float : S(T) approximation
    """
    primes, logp = prime_cache.subset(int(P_max))

    if len(primes) == 0:
        return 0.0

    terms = (primes ** (-0.5)) * np.sin(T * logp)
    result = -kahan_sum(terms) / np.pi

    if verbose:
        print(f"  S_euler_k1({T:,}, P_max={P_max:.2e}) = {result:+.6f} ({len(primes):,} primes)")

    return result

def S_euler_kK(T, P_max, K, prime_cache, verbose=False):
    """
    Euler product k<=K (with prime powers):
    S(T) ≈ -(1/π) Σ_p Σ_{k=1}^K (1/k) p^{-k/2} sin(k T log p)

    Includes prime powers: p^2, p^3, ..., p^K

    Parameters:
    -----------
    T : float
        Height on critical line
    P_max : float
        Prime truncation bound (for k=1)
    K : int
        Maximum power (typically 5)
    prime_cache : PrimeCache
        Cache with primes >= P_max
    verbose : bool
        Print details

    Returns:
    --------
    float : S(T) approximation
    dict : Breakdown by k
    """
    total_sum = 0.0
    breakdown = {}

    for k in range(1, K+1):
        # For each k, use primes p where p^k <= P_max
        p_bound = int(P_max ** (1.0 / k))
        primes, logp = prime_cache.subset(p_bound)

        if len(primes) == 0:
            breakdown[k] = {'sum': 0.0, 'n_primes': 0}
            continue

        # Coefficient: 1/k (from Euler product expansion)
        coeff = (1.0/k) * (primes ** (-0.5*k))
        phases = np.sin(k * T * logp)
        terms = coeff * phases

        k_sum = kahan_sum(terms)
        total_sum += k_sum

        breakdown[k] = {
            'sum': k_sum,
            'n_primes': len(primes),
            'p_max': primes[-1] if len(primes) > 0 else 0
        }

    result = -total_sum / np.pi

    if verbose:
        print(f"  S_euler_kK({T:,}, P_max={P_max:.2e}, K={K}) = {result:+.6f}")
        for k, info in breakdown.items():
            rel = abs(info['sum'] / breakdown[1]['sum'] * 100) if breakdown[1]['sum'] != 0 else 0
            print(f"    k={k}: {info['sum']:+.6e} ({info['n_primes']:,} primes, {rel:.2f}% of k=1)")

    return result, breakdown

print("✓ S(T) methods loaded (S_direct, S_riemann_siegel, S_euler_k1, S_euler_kK)")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_time(seconds):
    """Format seconds as human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def load_cached(filepath, loader_func=None, verbose=True):
    """
    Load cached data if exists, otherwise return None.

    Parameters:
    -----------
    filepath : str
        Path to cached file
    loader_func : callable
        Function to load file (default: pickle.load)
    verbose : bool
        Print messages

    Returns:
    --------
    data or None
    """
    if not os.path.exists(filepath):
        if verbose:
            print(f"⚠ Cache miss: {filepath}")
        return None

    if verbose:
        print(f"✓ Loading cached: {filepath}")

    try:
        if loader_func is None:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            return loader_func(filepath)
    except Exception as e:
        if verbose:
            print(f"✗ Failed to load cache: {e}")
        return None

def save_cached(data, filepath, saver_func=None, verbose=True):
    """
    Save data to cache.

    Parameters:
    -----------
    data : any
        Data to save
    filepath : str
        Path to save to
    saver_func : callable
        Function to save (default: pickle.dump)
    verbose : bool
        Print messages
    """
    try:
        if saver_func is None:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            saver_func(data, filepath)

        if verbose:
            print(f"✓ Saved cache: {filepath}")
    except Exception as e:
        if verbose:
            print(f"✗ Failed to save cache: {e}")

def check_prerequisites(required_files, verbose=True):
    """
    Check if required files exist.

    Parameters:
    -----------
    required_files : list of str
        List of file paths
    verbose : bool
        Print messages

    Returns:
    --------
    bool : True if all exist, False otherwise
    """
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)

    if missing:
        if verbose:
            print("✗ Missing required files:")
            for f in missing:
                print(f"  - {f}")
        return False

    if verbose:
        print("✓ All prerequisites satisfied")
    return True

print("✓ Helper functions loaded")

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "="*80)
print("CELL 1 COMPLETE ✓")
print("="*80)
print("\nShared functions ready:")
print("  • kahan_sum() - numerical stability")
print("  • smooth_RVM() - baseline N(T)")
print("  • segmented_sieve() - prime generation")
print("  • PrimeCache - efficient prime storage")
print("  • S_direct() - reference S(T)")
print("  • S_riemann_siegel() - O(sqrt(T)) approximation")
print("  • S_euler_k1() - primes-only approximation")
print("  • S_euler_kK() - with prime powers")
print("  • Helper functions for caching & I/O")
print("\nDirectory structure:")
for name, path in DIRS.items():
    print(f"  • {name}: {path}")
print("\n✓ Ready to run experiments!")
print("="*80)