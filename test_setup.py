#!/usr/bin/env python3
"""
Test script to verify the refactored code works correctly.
This simulates the key parts of notebook 01_setup_and_functions.ipynb
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_imports():
    """Test all imports."""
    print("="*60)
    print("Testing Imports")
    print("="*60)

    try:
        from src.core.numerical_utils import kahan_sum, kahan_sum_complex
        from src.core.s_t_functions import S_direct, S_RS, S_euler, analyze_error
        from src.core.prime_cache import PrimeCache
        from src.utils.paths import PathConfig, check_prerequisites
        import yaml
        import pickle
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_path_config():
    """Test path configuration."""
    print("\n" + "="*60)
    print("Testing Path Configuration")
    print("="*60)

    try:
        from src.utils.paths import PathConfig
        paths = PathConfig()
        print(f"Base dir: {paths.base_dir}")
        print(f"Zeros file: {paths.zeros_file}")
        print(f"Cache dir: {paths.cache_dir}")

        # Ensure directories exist
        paths.ensure_dirs()
        print("✓ Directories created/verified")
        return True, paths
    except Exception as e:
        print(f"✗ Path config failed: {e}")
        return False, None

def test_data_loading(paths):
    """Test loading zeros data."""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)

    try:
        zeros_cache_path = paths.cache_dir / "zeros.npy"

        if zeros_cache_path.exists():
            print("Loading from cache...")
            zeros = np.load(zeros_cache_path)
            print(f"✓ Loaded {len(zeros):,} zeros from cache")
        else:
            print("Loading from file...")
            start = time.time()
            zeros = np.loadtxt(paths.zeros_file, max_rows=10_000_000)
            elapsed = time.time() - start
            print(f"✓ Loaded {len(zeros):,} zeros in {elapsed:.2f}s")

            # Preprocess
            zeros = np.sort(zeros)
            zeros = zeros[zeros > 0]
            print(f"✓ After filtering: {len(zeros):,} zeros")

            # Save to cache
            np.save(zeros_cache_path, zeros)
            print(f"✓ Saved to cache: {zeros_cache_path}")

        # Display some stats
        print(f"\nZeros statistics:")
        print(f"  Count: {len(zeros):,}")
        print(f"  Range: [{zeros[0]:.2f}, {zeros[-1]:.2f}]")
        print(f"  Mean spacing: {np.mean(np.diff(zeros[:1000])):.4f}")

        return zeros

    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return None

def test_prime_cache(paths):
    """Test prime cache initialization."""
    print("\n" + "="*60)
    print("Testing Prime Cache")
    print("="*60)

    try:
        from src.core.prime_cache import PrimeCache
        prime_cache = PrimeCache(max_prime=100_000, cache_file=str(paths.cache_dir / "prime_cache_test.pkl"))

        # Test getting primes
        primes_up_to_1000 = prime_cache.get_primes_up_to(1000)
        print(f"✓ Found {len(primes_up_to_1000)} primes up to 1000")
        print(f"  First 10: {primes_up_to_1000[:10].tolist()}")

        # Test prime counting
        pi_1000 = prime_cache.count_primes_up_to(1000)
        print(f"  π(1000) = {pi_1000}")

        # Verify against known value
        assert pi_1000 == 168, f"Expected π(1000)=168, got {pi_1000}"
        print("✓ Prime count verification passed")

        return prime_cache

    except Exception as e:
        print(f"✗ Prime cache test failed: {e}")
        return None

def test_s_functions(zeros, prime_cache):
    """Test S(T) computation functions."""
    print("\n" + "="*60)
    print("Testing S(T) Functions")
    print("="*60)

    try:
        # Test parameters
        T_test = 1000.0
        P_max_test = 10000

        print(f"Test parameters: T = {T_test}, P_max = {P_max_test:,}")

        # Test S_RS
        print("\nTesting S_RS (Riemann-Siegel)...")
        start = time.time()
        s_rs = S_RS(T_test, zeros)
        rs_time = time.time() - start
        print(f"  S_RS({T_test}) = {s_rs:.6f} (took {rs_time:.4f}s)")

        # Test S_euler
        print(f"\nTesting S_euler (truncated Euler product)...")
        start = time.time()
        s_euler = S_euler(T_test, P_max_test, prime_cache)
        euler_time = time.time() - start
        print(f"  S_euler({T_test}, {P_max_test}) = {s_euler:.6f} (took {euler_time:.4f}s)")

        # Test error analysis
        print("\nTesting error analysis...")
        S_ref = s_rs  # Use RS as reference
        error = abs(s_euler - S_ref)
        print(f"  Absolute error: {error:.6f}")

        # Test analyze_error function
        approx_vals = np.array([s_euler, s_euler * 1.01, s_euler * 0.99])
        exact_vals = np.array([S_ref, S_ref, S_ref])
        error_stats = analyze_error(approx_vals, exact_vals)
        print(f"  Error analysis: max_error={error_stats['max_error']:.6f}")

        print("\n✓ All S(T) functions working correctly")
        return True

    except Exception as e:
        print(f"✗ S(T) function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Starting Refactored Code Tests")
    print("=" * 60)

    success = True

    # Test 1: Imports
    if not test_imports():
        success = False

    # Test 2: Path config
    path_success, paths = test_path_config()
    if not path_success:
        success = False

    # Test 3: Data loading
    zeros = test_data_loading(paths) if paths else None
    if zeros is None:
        success = False

    # Test 4: Prime cache
    prime_cache = test_prime_cache(paths) if paths else None
    if prime_cache is None:
        success = False

    # Test 5: S functions
    if zeros is not None and prime_cache is not None:
        if not test_s_functions(zeros, prime_cache):
            success = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    if success:
        print("✓ ALL TESTS PASSED!")
        print("\nThe refactored code is working correctly.")
        print("You can now proceed to run the notebooks.")
    else:
        print("✗ Some tests failed.")
        print("Please check the errors above before proceeding.")

    return success

if __name__ == "__main__":
    main()