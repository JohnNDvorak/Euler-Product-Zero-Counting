#!/usr/bin/env python3
"""
Debug S_euler implementation to understand accuracy issues.
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import time

# Load legacy results
from utils.legacy_results_loader import LegacyResultsLoader

print("Loading legacy results...")
loader = LegacyResultsLoader()
results = loader.load_all_results()

# Generate test data
from core.prime_cache import simple_sieve

print("\nGenerating test data...")
primes = np.array(simple_sieve(10_000_000))  # Need more primes for large P_max
print(f"✓ Generated {len(primes):,} primes")

# Import stable implementation
from core.s_t_functions_stable import S_euler_stable

print("\n" + "="*80)
print("DEBUGGING S_EULER IMPLEMENTATION")
print("="*80)

# Test 1: Simple case with small P_max
print("\n1. Testing simple case (T=1000, P_max=1000):")
T_test = 1000
P_test = 1000

# Try different methods
methods = ['direct', 'kahan', 'logarithmic']
for method in methods:
    s = S_euler_stable(T_test, P_test, primes, k_max=1, method=method)
    print(f"  {method:12s}: S = {s:.6f}")

# Test 2: Check what the reference S should be
print("\n2. Finding reference data:")
dense = results['dense']

# Find closest match
T_closest = dense.iloc[dense['T'].sub(T_test).abs().idxmin()]
print(f"\nClosest T in legacy data: {T_closest['T']:.6f}")
print(f"P_max values available: {sorted(dense['P_max'].unique())}")

if P_test in dense['P_max'].values:
    matches = dense[
        (abs(dense['T'] - T_test) < 1) & (dense['P_max'] == P_test)
    ]
    if len(matches) > 0:
        match = matches.iloc[0]
        print(f"\nLegacy data for T≈{T_test:.0f}, P_max={P_test}:")
        print(f"  S_ref:    {match['S_ref']:.6f}")
        print(f"  S_euler:  {match['S_euler']:.6f}")
        print(f"  Error:    {match['error']:.6f}")

        # Compare with our calculation
        s_our = S_euler_stable(T_test, P_test, primes, k_max=1, method='logarithmic')
        our_error = abs(s_our - match['S_ref'])
        print(f"\nOur calculation:")
        print(f"  S_our:    {s_our:.6f}")
        print(f"  Our error: {our_error:.6f}")
        print(f"  Difference from legacy error: {abs(our_error - match['error']):.6f}")
    else:
        print(f"\nNo legacy data found for T≈{T_test:.0f}, P_max={P_test}")
        # Use exact T from legacy data instead
        T_legacy = dense[dense['P_max'] == P_test]['T'].iloc[0]
        match = dense[dense['T'] == T_legacy].iloc[0]
        print(f"\nUsing legacy T={T_legacy:.0f} instead:")
        print(f"  S_ref:    {match['S_ref']:.6f}")
        print(f"  S_euler:  {match['S_euler']:.6f}")
        print(f"  Error:    {match['error']:.6f}")

# Test 3: Phase wrapping issues
print("\n3. Testing phase wrapping:")
print("S(T) should be in [-0.5, 0.5] range")

for T in [1000, 5000, 10000]:
    for P in [1000, 10000, 100000]:
        s = S_euler_stable(T, P, primes, method='logarithmic')
        print(f"  T={T:5.0f}, P={P:7.0f}: S={s:8.4f} ({'OK' if -0.5 <= s <= 0.5 else 'OUT OF RANGE'})")

# Test 4: Try mpmath for high precision
print("\n4. Testing with mpmath for reference:")
try:
    s_mpmath = S_euler_stable(1000, 1000, primes[:1000], use_mpmath=True)
    s_log = S_euler_stable(1000, 1000, primes[:1000], method='logarithmic')
    print(f"  mpmath (high precision): S = {s_mpmath:.6f}")
    print(f"  logarithmic:           S = {s_log:.6f}")
    print(f"  Difference:             {abs(s_mpmath - s_log):.6f}")
except Exception as e:
    print(f"  mpmath failed: {e}")

# Test 5: Check normalization
print("\n5. Testing normalization (arctan2 vs angle):")
def test_normalization(T, P):
    """Test different ways to compute S from complex zeta."""
    from core.s_t_functions_stable import _S_euler_logarithmic

    # Get primes up to P
    prime_list = primes[primes <= P]

    # Compute using logarithmic method but return raw values
    log_zeta_real = 0.0
    log_zeta_imag = 0.0

    for p in prime_list:
        phase = T * np.log(p)
        phase_mod = phase % (2 * np.pi)

        if abs(phase_mod - np.pi) < 1e-10:
            log_term_real = np.log(2)
            log_term_imag = 0
        else:
            sin_half = np.sin(phase_mod / 2)
            log_term_real = np.log(2 * abs(sin_half))
            log_term_imag = -(np.pi - phase_mod) / 2

        log_zeta_real -= log_term_real
        log_zeta_imag -= log_term_imag

    # Different ways to get S
    s1 = np.arctan2(log_zeta_imag, log_zeta_real) / np.pi
    s2 = np.angle(log_zeta_real + 1j * log_zeta_imag) / np.pi

    # Normalized versions
    s1_norm = ((s1 + 0.5) % 1.0) - 0.5
    s2_norm = ((s2 + 0.5) % 1.0) - 0.5

    print(f"\nT={T}, P_max={P:,}:")
    print(f"  arctan2:      {s1:.6f} (normalized: {s1_norm:.6f})")
    print(f"  angle:        {s2:.6f} (normalized: {s2_norm:.6f})")
    print(f"  Stable impl:  {S_euler_stable(T, P, primes, method='logarithmic'):.6f}")

test_normalization(1000, 10000)
test_normalization(10000, 100000)

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)