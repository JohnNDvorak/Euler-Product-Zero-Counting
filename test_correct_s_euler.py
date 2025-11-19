#!/usr/bin/env python3
"""
Test the corrected S_euler implementation that matches the original notebook.
"""

import sys
sys.path.append('src')

import numpy as np

# Load legacy results
from utils.legacy_results_loader import LegacyResultsLoader

print("Loading legacy results...")
loader = LegacyResultsLoader()
results = loader.load_all_results()

# Generate primes
from core.prime_cache import simple_sieve

print("\nGenerating primes up to 1 million...")
primes = np.array(simple_sieve(1_000_000))
print(f"✓ Generated {len(primes):,} primes")

# Import corrected implementation
from core.s_t_functions_correct import S_euler_correct

print("\n" + "="*80)
print("TESTING CORRECTED S_EULER IMPLEMENTATION")
print("="*80)

# Test against legacy dense sampling data
dense = results['dense']

print("\nComparing with legacy dense sampling data:")
print("-" * 70)

# Test a few specific cases
test_cases = [
    (100.21626824161686, 1000),
    (497.855, 10000),
    (996.032, 100000),
]

for T_test, P_max in test_cases:
    # Find closest match in legacy data
    matches = dense[
        (abs(dense['T'] - T_test) < 5) & (dense['P_max'] == P_max)
    ]

    if len(matches) > 0:
        match = matches.iloc[0]
        T_legacy = match['T']

        print(f"\nT = {T_legacy:.3f}, P_max = {int(P_max):,}")
        print(f"  Legacy S_ref:    {match['S_ref']:.6f}")
        print(f"  Legacy S_approx: {match['S_approx']:.6f}")
        print(f"  Legacy error:    {match['error_S']:.6f}")

        # Calculate with corrected implementation
        s_correct = S_euler_correct(T_legacy, P_max, primes, k_max=1, method='kahan')
        our_error = abs(s_correct - match['S_ref'])

        print(f"  Corrected S_euler: {s_correct:.6f}")
        print(f"  Our error:         {our_error:.6f}")
        print(f"  Error diff:        {abs(our_error - match['error_S']):.6f}")
        print(f"  Matches legacy?    {'✓' if abs(our_error - match['error_S']) < 1e-4 else '✗'}")

# Test prime powers contribution
print("\n\nTesting prime powers (T=1000, P_max=10000):")
print("-" * 50)

for k_max in [1, 2, 3, 4, 5]:
    s = S_euler_correct(1000, 10000, primes, k_max=k_max, method='kahan')
    print(f"  k_max = {k_max}: S = {s:10.6f}")

# Test normalization (values should be reasonable)
print("\n\nTesting range of S values:")
print("-" * 50)

test_values = [
    (100, 1000),
    (1000, 10000),
    (10000, 100000),
    (100000, 1000000),
]

for T, P in test_values:
    s = S_euler_correct(T, P, primes)
    print(f"  T={T:6.0f}, P_max={P:8,.0f}: S = {s:8.4f}")

# Performance test
print("\n\nPerformance test:")
print("-" * 50)

import time
n_tests = 100

start = time.time()
for _ in range(n_tests):
    S_euler_correct(1000, 100000, primes)
elapsed = time.time() - start

print(f"  Average time per computation: {elapsed/n_tests*1000:.2f} ms")

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

print("\n✅ Corrected implementation uses proper formula:")
print("   S(T) ≈ -(1/π) Σ_p p^(-1/2) sin(T log p)")
print("\n✅ NOT the complex Euler product argument computation")
print("\nNext: Verify accuracy matches legacy results precisely")