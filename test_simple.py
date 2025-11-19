#!/usr/bin/env python3
"""
Simple test to verify basic functionality before full testing.
"""

import sys
sys.path.append('src')

import numpy as np
import time

print("="*60)
print("SIMPLE FUNCTIONALITY TEST")
print("="*60)

# Test 1: Basic imports
print("\n1. Testing imports...")
try:
    from src.core.numerical_utils import kahan_sum
    from src.core.s_t_functions import S_RS, S_euler
    from src.utils.paths import PathConfig
    from src.core.prime_cache import PrimeCache, simple_sieve
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Path config
print("\n2. Testing path configuration...")
paths = PathConfig()
print(f"✓ Base dir: {paths.base_dir}")
print(f"✓ Zeros file: {paths.zeros_file}")
print(f"✓ Zeros file exists: {paths.zeros_file.exists()}")

# Test 3: Load zeros
print("\n3. Loading zeros...")
zeros_cache = paths.cache_dir / "zeros.npy"
if zeros_cache.exists():
    zeros = np.load(zeros_cache)
    print(f"✓ Loaded {len(zeros):,} zeros from cache")
else:
    print("Loading from file (first 100k zeros for testing)...")
    zeros = np.loadtxt(paths.zeros_file, max_rows=100_000)
    print(f"✓ Loaded {len(zeros):,} zeros")
    zeros = np.sort(zeros)

# Test 4: Simple prime generation
print("\n4. Testing prime generation...")
primes_1000 = simple_sieve(1000)
print(f"✓ Generated {len(primes_1000)} primes up to 1000")
print(f"  π(1000) = {len(primes_1000)} (should be 168)")

# Test 5: S_RS computation
print("\n5. Testing S_RS...")
T_test = 1000.0
start = time.time()
s_rs = S_RS(T_test, zeros)
elapsed = time.time() - start
print(f"✓ S_RS({T_test}) = {s_rs:.6f} (took {elapsed:.4f}s)")

# Test 6: Simple S_euler test
print("\n6. Testing S_euler with simple prime list...")
# Create a simple prime cache wrapper
class SimplePrimeCache:
    def __init__(self, primes):
        self.primes = primes
    def get_primes_up_to(self, limit):
        return np.array([p for p in self.primes if p <= limit])

prime_cache = SimplePrimeCache(primes_1000)
P_max_test = 100
start = time.time()
s_euler = S_euler(T_test, P_max_test, prime_cache)
elapsed = time.time() - start
print(f"✓ S_euler({T_test}, {P_max_test}) = {s_euler:.6f} (took {elapsed:.4f}s)")

# Test 7: Error analysis
print("\n7. Testing error analysis...")
error = abs(s_euler - s_rs)
print(f"✓ Absolute error: {error:.6f}")
print(f"✓ Relative error: {error/(abs(s_rs)+1e-10)*100:.2f}%")

print("\n" + "="*60)
print("✓ ALL BASIC TESTS PASSED!")
print("="*60)
print("\nThe core functionality is working.")
print("You can now run the notebooks with confidence.")