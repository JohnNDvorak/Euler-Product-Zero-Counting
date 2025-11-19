#!/usr/bin/env python3
"""
Test S_euler accuracy against legacy data using actual T values from the data.
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd

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

# Import stable implementation
from core.s_t_functions_stable import S_euler_stable

print("\n" + "="*80)
print("TESTING S_EULER ACCURACY")
print("="*80)

# Get a sample of dense sampling data
dense = results['dense']

# Test a few specific cases from the legacy data
test_cases = [
    (100.21626824161686, 1000),
    (500.5, 10000),
    (1000.0, 100000),
    (5000.0, 1000000),
    (10000.0, 1000000)
]

print("\nComparing with legacy dense sampling data:")
print("-" * 70)

for T_test, P_max in test_cases:
    # Find closest match in legacy data
    matches = dense[
        (abs(dense['T'] - T_test) < 5) & (dense['P_max'] == P_max)
    ]

    if len(matches) > 0:
        match = matches.iloc[0]
        T_legacy = match['T']

        print(f"\nT = {T_legacy:.3f}, P_max = {int(P_max):,}")
        print(f"  Legacy S_ref:  {match['S_ref']:.6f}")
        print(f"  Legacy S_euler:{match['S_approx']:.6f}")
        print(f"  Legacy error:  {match['error_S']:.6f}")

        # Calculate with our implementation
        s_our = S_euler_stable(T_legacy, P_max, primes, k_max=1, method='logarithmic')
        our_error = abs(s_our - match['S_ref'])

        print(f"  Our S_euler:   {s_our:.6f}")
        print(f"  Our error:     {our_error:.6f}")
        print(f"  Error diff:    {abs(our_error - match['error_S']):.6f}")

        # Test different methods
        print(f"  Methods:")
        for method in ['direct', 'kahan', 'logarithmic']:
            s = S_euler_stable(T_legacy, P_max, primes, k_max=1, method=method)
            err = abs(s - match['S_ref'])
            print(f"    {method:12s}: S={s:.6f}, error={err:.6f}")

# Test with optimal results
print("\n\nComparing with optimal P_max results:")
print("-" * 70)

optimal = results['optimal']

for _, row in optimal.iterrows():
    T = row['T']
    P_opt = row['P_max_optimal']
    expected_error = row['error_minimum']
    S_ref = row['S_ref']

    print(f"\nT = {T:.0f}, Optimal P_max = {int(P_opt):,}")
    print(f"  Expected error: {expected_error:.6f}")

    # Try with limited primes (we only have up to 1M)
    if P_opt <= len(primes):
        s_our = S_euler_stable(T, P_opt, primes, k_max=1, method='logarithmic')
        our_error = abs(s_our - S_ref)

        print(f"  Our S_euler:     {s_our:.6f}")
        print(f"  Our error:       {our_error:.6f}")
        print(f"  Error ratio:     {our_error/expected_error:.2f}x")
    else:
        print(f"  Skipped: P_opt exceeds available primes")

print("\n" + "="*80)
print("TESTING NORMALIZATION")
print("="*80)

# Test phase wrapping normalization
print("\nPhase wrapping test (T=1000, various P_max):")
for P in [1000, 10000, 100000, 1000000]:
    s = S_euler_stable(1000, P, primes, method='logarithmic')
    print(f"  P_max = {P:7,.0f}: S = {s:8.4f} ({'OK' if -0.5 <= s <= 0.5 else 'OUT OF RANGE'})")

# Test prime powers
print("\nPrime powers test (T=1000, P_max=10000):")
for k_max in [1, 2, 3, 4, 5]:
    s = S_euler_stable(1000, 10000, primes, k_max=k_max)
    print(f"  k_max = {k_max}: S = {s:.6f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nKey observations:")
print("1. Check if S values are in [-0.5, 0.5] range")
print("2. Compare our errors with legacy errors")
print("3. Verify different methods give similar results")
print("4. Test prime powers contribution")