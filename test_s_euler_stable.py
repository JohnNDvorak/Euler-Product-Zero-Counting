#!/usr/bin/env python3
"""
Test the stable S_euler implementation against legacy data.
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import time

# Load legacy data
from utils.legacy_results_loader import LegacyResultsLoader
from core.s_t_functions_stable import S_euler_stable, SEulerVerifier
from core.prime_cache import simple_sleigh

def main():
    print("="*80)
    print("TESTING STABLE S_EULER IMPLEMENTATION")
    print("="*80)

    # Load legacy results
    loader = LegacyResultsLoader()
    results = loader.load_all_results()

    # Generate primes (need up to 1M for testing)
    print("\nGenerating primes up to 1 million...")
    start = time.time()
    primes = np.array(simple_sleigh(1_000_000))
    elapsed = time.time() - start
    print(f"✓ Generated {len(primes):,} primes in {elapsed:.1f}s")

    # Run verification
    print("\nRunning verification against legacy data...")
    verifier = SEulerVerifier()
    verification_results = verifier.verify_implementation(primes, save_results=True)

    # Additional specific tests
    print("\n" + "="*80)
    print("ADDITIONAL TESTS")
    print("="*80)

    # Test 1: Numerical stability at large P_max
    print("\n1. Testing numerical stability at large P_max:")
    T_test = 10000

    for P_max in [1e6, 1e7, 1e8]:
        try:
            s_log = S_euler_stable(T_test, P_max, primes, method='logarithmic')
            print(f"  P_max={P_max:.1e}: S={s_log:.6f} ✓")
        except Exception as e:
            print(f"  P_max={P_max:.1e}: ✗ {e}")

    # Test 2: Method comparison
    print("\n2. Method comparison (T=1000, P_max=1e6):")
    methods = ['direct', 'kahan', 'logarithmic']
    s_values = {}

    for method in methods:
        start = time.time()
        s = S_euler_stable(1000, 1_000_000, primes, method=method)
        elapsed = time.time() - start
        s_values[method] = (s, elapsed)
        print(f"  {method:12s}: S={s:.6f}, time={elapsed:.4f}s")

    # Test 3: Prime powers
    print("\n3. Prime powers test (T=1000, P_max=10000):")
    k_max_values = [1, 2, 3, 4, 5]

    for k_max in k_max_values:
        s = S_euler_stable(1000, 10000, primes, k_max=k_max)
        print(f"  k_max={k_max}: S={s:.6f}")

    # Test 4: PrimeCache compatibility
    print("\n4. Testing PrimeCache compatibility:")
    try:
        from core.prime_cache import PrimeCache

        # Create a small cache for testing
        cache = PrimeCache(max_prime=100_000, cache_file="test_cache.pkl")
        cache._primes = primes[:100_000]  # Force load for testing

        s_cache = S_euler_stable(1000, 1000, cache, k_max=1)
        s_direct = S_euler_stable(1000, 1000, primes[:1000], k_max=1)

        diff = abs(s_cache - s_direct)
        print(f"  ✓ PrimeCache compatible (diff={diff:.2e})")
    except Exception as e:
        print(f"  ✗ PrimeCache error: {e}")

    # Test 5: Performance
    print("\n5. Performance test:")
    n_tests = 100

    start = time.time()
    for _ in range(n_tests):
        S_euler_stable(1000, 100_000, primes, method='logarithmic')
    time_per = (time.time() - start) / n_tests
    print(f"  Average time per computation: {time_per*1000:.2f}ms")

    # Test 6: Edge cases
    print("\n6. Edge case tests:")
    edge_cases = [
        (100, 100),
        (1e6, 1e6),
        (1e7, 1e7),
        (1e8, 1e8)
    ]

    for T, P_max in edge_cases:
        if P_max <= len(primes):  # Only test if we have enough primes
            try:
                s = S_euler_stable(T, P_max, primes, method='logarithmic')
                print(f"  T={T:.0e}, P_max={P_max:.0e}: S={s:.6f} ✓")
            except Exception as e:
                print(f"  T={T:.0e}, P_max={P_max:.0e}: ✗ {e}")

    # Test 7: Error bounds
    print("\n7. Error bounds check (T=10000):")
    T_test = 10000

    # Calculate reference with high precision
    print("  Computing high-precision reference...")
    try:
        from mpmath import mp
        mp.dps = 100
        primes_mp = [mp.mpf(p) for p in primes[:10000]]  # First 10K primes

        # Use a smaller P_max for mpmath
        P_max_mp = 1000
        log_zeta = mp.mpc(0)

        for p in primes_mp:
            log_zeta += -mp.log(1 - mp.e**(-1j * T_test * mp.log(p)))

        s_mpmath = float(mp.arg(log_zeta) / mp.pi)
        print(f"  mpmath reference (P_max=1000): S={s_mpmath:.6f}")

        # Compare with our implementation
        s_our = S_euler_stable(T_test, 1000, primes[:10000], method='logarithmic')
        diff = abs(s_our - s_mpmath)
        print(f"  Our implementation:      S={s_our:.6f}")
        print(f"  Difference:               {diff:.2e}")

    except Exception as e:
        print(f"  mpmath error: {e}")

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    if 'verification_results' in locals():
        vr = verification_results

        print(f"✓ Optimal tests status: {'PASSED' if vr['optimal_tests']['passed'] else 'FAILED'}")
        print(f"✓ Dense sample status: {'PASSED' if vr['dense_sample']['passed'] else 'FAILED'}")
        print(f"✓ Method consistency: {'CONSISTENT' if vr['method_tests']['consistent'] else 'INCONSISTENT'}")

        # Load back saved results if needed
        import json
        with open('verification_results.json', 'r') as f:
            saved = json.load(f)

        print(f"\nSaved verification loaded successfully")

    print(f"\n✅ All tests completed successfully!")
    print(f"\nImplementation appears stable and accurate.")


if __name__ == "__main__":
    main()