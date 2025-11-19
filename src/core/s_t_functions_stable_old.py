"""
Stable implementations of S(T) functions.

These implementations address numerical stability issues found in the original
code, particularly for large P_max values.
"""

import numpy as np
import pandas as pd
import cmath
import mpmath as mp
from typing import Union, Optional
from functools import lru_cache

# High precision for reference calculations
mp.dps = 50

# Set up mpmath constants
mp_pi = mp.pi


def S_euler_stable(T: float, P_max: float, primes: Union[np.ndarray, object],
                  k_max: int = 1, method: str = 'adaptive',
                  use_mpmath: bool = False) -> float:
    """
    Compute S(T) using truncated Euler product with numerical stability.

    Parameters:
    -----------
    T : float
        Height on critical line
    P_max : float
        Maximum prime/power to include in product
    primes : np.ndarray or PrimeCache object
        Prime numbers or cache object
    k_max : int
        Maximum exponent for prime powers (1 for primes only)
    method : str
        Method to use:
        - 'adaptive': Choose best method based on P_max
        - 'logarithmic': Use logarithmic computation (stable for large P_max)
        - 'kahan': Use Kahan summation (good for small P_max)
        - 'mpmath': Use high-precision arithmetic (slow but accurate)
    use_mpmath : bool
        Whether to use mpmath for critical computations

    Returns:
    --------
    float : S(T) approximation = (1/π) * arg ζ(1/2 + iT)
    """
    # Choose method automatically if requested
    if method == 'adaptive':
        if P_max > 1_000_000 or use_mpmath:
            method = 'logarithmic'
        elif P_max > 100_000:
            method = 'kahan'
        else:
            method = 'direct'

    # Extract primes
    if hasattr(primes, 'get_primes_up_to'):
        # PrimeCache object
        prime_list = primes.get_primes_up_to(P_max)
    else:
        # Assume numpy array
        prime_list = primes[primes <= P_max]

    if use_mpmath:
        return _S_euler_mpmath(T, P_max, prime_list, k_max)
    elif method == 'logarithmic':
        return _S_euler_logarithmic(T, P_max, prime_list, k_max)
    elif method == 'kahan':
        return _S_euler_kahan(T, P_max, prime_list, k_max)
    else:
        return _S_euler_direct(T, P_max, prime_list, k_max)


def _S_euler_mpmath(T: float, P_max: float, prime_list: list, k_max: int) -> float:
    """
    High-precision Euler product using mpmath.

    This is slow but provides a reference for verification.
    """
    # Set precision based on computation size
    mp.dps = 80 if P_max > 1_000_000 else 50

    log_zeta = mp.mpc(0, 0)

    for p in prime_list:
        p_k = p
        for k in range(1, k_max + 1):
            if p_k > P_max:
                break

            # Compute -log(1 - p^(-iT))
            # p^(-iT) = e^(-iT*log(p)) = cos(T*log(p)) - i*sin(T*log(p))
            angle = k * T * mp.log(p)
            complex_p = mp.e**(-1j * angle)

            # Numerically stable log(1 - z)
            if abs(complex_p - 1) < 1e-30:
                # Use series expansion when z is close to 1
                # log(1 - z) ≈ -z - z^2/2 - z^3/3 - ...
                z = complex_p - 1
                log_term = -z - z**2/2 - z**3/3
            else:
                log_term = mp.log(1 - complex_p)

            log_zeta -= log_term / k

            p_k *= p

    # Compute argument
    arg = mp.arg(log_zeta)
    return float(arg / mp.pi)


def _S_euler_logarithmic(T: float, P_max: float, prime_list: list, k_max: int) -> float:
    """
    Compute S(T) using logarithmic method to avoid overflow.

    This is the most stable method for large P_max.
    """
    log_zeta_real = 0.0
    log_zeta_imag = 0.0

    for p in prime_list:
        p_k = p
        for k in range(1, k_max + 1):
            if p_k > P_max:
                break

            # Compute phase: k * T * log(p)
            phase = k * T * np.log(p)

            # Handle numerical issues for large phases
            phase_mod = phase % (2 * np.pi)

            if phase_mod < 1e-10 or phase_mod > 2*np.pi - 1e-10:
                # Near 0 or 2π, use series expansion
                # log(1 - e^(-iθ)) = -iθ/2 - θ^2/8 + O(θ^4)
                theta = phase_mod
                if theta < np.pi:
                    log_term_real = 0
                    log_term_imag = -theta / 2
                else:
                    log_term_real = 0
                    log_term_imag = -(theta - 2*np.pi) / 2
            elif abs(phase_mod - np.pi) < 1e-10:
                # Near π, use series expansion
                # log(1 - e^(-iπ)) = log(2)
                log_term_real = np.log(2)
                log_term_imag = 0
            else:
                # General case - use arctangent formulation
                # 1 - e^(-iθ) = 2|sin(θ/2)| * e^{-i(π-θ)/2}
                sin_half = np.sin(phase_mod / 2)
                log_term_real = np.log(2 * abs(sin_half))
                log_term_imag = -(np.pi - phase_mod) / 2

            # Add contribution scaled by 1/k
            log_zeta_real -= log_term_real / k
            log_zeta_imag -= log_term_imag / k

            p_k *= p

    # Convert from log form to argument
    s_value = np.arctan2(log_zeta_imag, log_zeta_real) / np.pi

    # Normalize to [-0.5, 0.5] range if needed
    # This handles branch cuts in arctan2
    s_value = ((s_value + 0.5) % 1.0) - 0.5

    return s_value


def _S_euler_kahan(T: float, P_max: float, prime_list: list, k_max: int) -> float:
    """
    Compute S(T) using Kahan summation.

    Good for medium-sized P_max (up to ~1M).
    """
    zeta_prod = 1.0 + 0.0j
    compensation = 0.0 + 0.0j

    for p in prime_list:
        p_k = p
        for k in range(1, k_max + 1):
            if p_k > P_max:
                break

            # Euler factor: (1 - p^(-iT))^{-1}
            phase = k * T * np.log(p)
            factor = 1.0 / (1.0 - np.exp(-1j * phase))

            # Kahan summation for complex numbers
            y = factor - 1.0
            temp = zeta_prod + y
            zeta_prod = temp
            compensation = (temp - zeta_prod) - compensation

            p_k *= p

    # Add back the compensation
    zeta_prod += compensation

    return np.angle(zeta_prod) / np.pi


def _S_euler_direct(T: float, P_max: float, prime_list: list, k_max: int) -> float:
    """
    Direct computation of Euler product.

    Only for very small P_max where numerical issues are minimal.
    """
    zeta_prod = 1.0 + 0.0j

    for p in prime_list:
        p_k = p
        for k in range(1, k_max + 1):
            if p_k > P_max:
                break

            phase = k * T * np.log(p)
            zeta_prod *= 1.0 / (1.0 - np.exp(-1j * phase))

            p_k *= p

    return np.angle(zeta_prod) / np.pi


def S_euler_prime_powers(T: float, P_max: float, primes: Union[np.ndarray, object],
                         k_max: int = 5, include_k1: bool = True) -> float:
    """
    Compute S(T) with prime powers, separating k=1 and k>1 contributions.

    This matches the implementation in the original notebook.
    """
    # Get primes up to P_max
    if hasattr(primes, 'get_primes_up_to'):
        prime_list = primes.get_primes_up_to(P_max)
    else:
        prime_list = primes[primes <= P_max]

    # k=1 contribution (primes only)
    if include_k1:
        s_k1 = S_euler_stable(T, P_max, prime_list, k_max=1, method='adaptive')
    else:
        s_k1 = 0.0

    # k>1 contributions (prime powers)
    if k_max > 1:
        s_k_gt1 = _S_euler_prime_powers_only(T, P_max, prime_list, k_max)
    else:
        s_k_gt1 = 0.0

    return s_k1 + s_k_gt1


def _S_euler_prime_powers_only(T: float, P_max: float, prime_list: list,
                             k_max: int) -> float:
    """
    Compute only the k>1 (prime power) contributions to S(T).
    """
    contribution = 0.0

    for p in prime_list:
        p_k = p * p  # Start with p^2
        for k in range(2, k_max + 1):
            if p_k > P_max:
                break

            # Contribution from p^k
            phase = k * T * np.log(p)
            term = -np.exp(1j * 2 * np.pi * phase) / k

            contribution += np.imag(term) / (2 * np.pi)

            p_k *= p

    return contribution


class SEulerVerifier:
    """
    Verify S_euler implementation against legacy data.
    """

    def __init__(self):
        from ..utils.legacy_results_loader import LegacyResultsLoader
        self.loader = LegacyResultsLoader()
        self.legacy_data = self.loader.load_all_results()

    def verify_implementation(self, primes: np.ndarray,
                            save_results: bool = True) -> dict:
        """
        Verify S_euler implementation against legacy results.

        Parameters:
        -----------
        primes : np.ndarray
            Prime numbers to use
        save_results : bool
            Whether to save comparison results

        Returns:
        --------
        dict : Verification results
        """
        print("="*80)
        print("VERIFYING S_EULER IMPLEMENTATION")
        print("="*80)

        # Load reference data
        dense = self.legacy_data['dense']
        optimal = self.legacy_data['optimal']

        verification_results = {
            'method_tests': {},
            'optimal_tests': {},
            'dense_sample': {},
            'errors': []
        }

        # Test 1: Compare with optimal P_max results
        print("\n1. Testing against optimal P_max results...")
        self._test_optimal_results(optimal, primes, verification_results)

        # Test 2: Sample dense sampling comparisons
        print("\n2. Testing dense sampling samples...")
        self._test_dense_sampling(dense, primes, verification_results)

        # Test 3: Method comparison
        print("\n3. Testing different methods...")
        self._test_methods(primes, verification_results)

        # Test 4: Prime powers
        print("\n4. Testing prime powers...")
        self._test_prime_powers(primes, verification_results)

        # Save results
        if save_results:
            self._save_verification_results(verification_results)

        # Summary
        self._print_summary(verification_results)

        return verification_results

    def _test_optimal_results(self, optimal_df: pd.DataFrame,
                              primes: np.ndarray,
                              results: dict):
        """Test against optimal P_max results."""
        tests = []

        for _, row in optimal_df.iterrows():
            T = row['T']
            P_opt = row['P_max_optimal']
            expected_error = row['error_minimum']
            S_ref = row['S_ref']

            # Test with our implementation
            S_calc = S_euler_stable(T, P_opt, primes, k_max=1, method='logarithmic')
            calc_error = abs(S_calc - S_ref)

            tests.append({
                'T': T,
                'P_max': P_opt,
                'expected_error': expected_error,
                'calculated_error': calc_error,
                'error_diff': abs(calc_error - expected_error),
                'S_ref': S_ref,
                'S_optimal': row['S_optimal'],
                'S_calculated': S_calc
            })

            # Check if close
            if calc_error < 10 * expected_error:  # Allow some tolerance
                print(f"  ✓ T={T:.0f}, P={P_opt:.1e}: expected={expected_error:.6f}, "
                      f"calc={calc_error:.6f}")
            else:
                print(f"  ✗ T={T:.0f}, P={P_opt:.1e}: expected={expected_error:.6f}, "
                      f"calc={calc_error:.6f} (MISS)")

        results['optimal_tests'] = {
            'tests': tests,
            'mean_error_diff': np.mean([t['error_diff'] for t in tests]),
            'max_error_diff': np.max([t['error_diff'] for t in tests]),
            'passed': np.mean([t['error_diff'] for t in tests]) < 0.01
        }

    def _test_dense_sampling(self, dense_df: pd.DataFrame,
                             primes: np.ndarray,
                             results: dict):
        """Test sample of dense sampling results."""
        # Sample 10 random T values
        T_sample = dense_df['T'].unique()
        if len(T_sample) > 10:
            T_sample = np.random.choice(T_sample, 10, replace=False)

        tests = []

        for T in T_sample:
            for P_max in [100, 10000, 1000000]:
                # Get expected value
                expected_row = dense_df[
                    (dense_df['T'] == T) & (dense_df['P_max'] == P_max)
                ].iloc[0]

                # Calculate with our implementation
                S_calc = S_euler_stable(T, P_max, primes, k_max=1)
                calc_error = abs(S_calc - expected_row['S_ref'])

                # Compare errors
                error_diff = abs(calc_error - expected_row['error_S'])

                tests.append({
                    'T': T,
                    'P_max': P_max,
                    'expected_error': expected_row['error_S'],
                    'calculated_error': calc_error,
                    'error_diff': error_diff
                })

        results['dense_sample'] = {
            'tests': tests,
            'mean_error_diff': np.mean([t['error_diff'] for t in tests]),
            'passed': np.mean([t['error_diff'] for t in tests]) < 0.01
        }

    def _test_methods(self, primes: np.ndarray, results: dict):
        """Test different computational methods."""
        T_test = 1000
        P_max = 1_000_000

        methods = ['direct', 'kahan', 'logarithmic', 'mpmath']
        s_values = {}

        print(f"  Testing T={T_test}, P_max={P_max:,}")

        for method in methods:
            if method == 'mpmath':
                s_values[method] = S_euler_stable(
                    T_test, P_max, primes, k_max=1, method='adaptive',
                    use_mpmath=True
                )
            else:
                s_values[method] = S_euler_stable(
                    T_test, P_max, primes, k_max=1, method=method
                )

            print(f"    {method:12s}: S = {s_values[method]:.6f}")

        # Check consistency
        ref_method = 'logarithmic'
        consistent = all(
            abs(s_values[m] - s_values[ref_method]) < 0.01
            for m in methods if m != 'mpmath'
        )

        results['method_tests'] = {
            'values': s_values,
            'consistent': consistent
        }

    def _test_prime_powers(self, primes: np.ndarray, results: dict):
        """Test inclusion of prime powers (k > 1)."""
        T_test = 1000
        P_max = 10_000

        print(f"  Testing T={T_test}, P_max={P_max:,}")

        # Test different k_max values
        s_values = {}
        for k in [1, 2, 3, 4, 5]:
            s_values[k] = S_euler_stable(T_test, P_max, primes, k_max=k)
            print(f"    k={k}: S = {s_values[k]:.6f}")

        # Check monotonicity (more terms should change value)
        monotonic = True
        for k in range(1, 5):
            if abs(s_values[k] - s_values[k+1]) < 1e-10:
                monotonic = False
                break

        results['prime_powers'] = {
            'values': s_values,
            'monotonic': monotonic,
            'change_k2_to_k1': s_values[2] - s_values[1]
        }

    def _save_verification_results(self, results: dict):
        """Save verification results."""
        import json
        from pathlib import Path

        # Convert numpy arrays to lists for JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        json_results = convert_numpy(results)

        # Save
        file_path = Path('verification_results.json')
        with open(file_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\n✓ Verification results saved to: {file_path}")

    def _print_summary(self, results: dict):
        """Print verification summary."""
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)

        # Optimal tests
        if 'optimal_tests' in results:
            opt = results['optimal_tests']
            print(f"\nOptimal P_max Tests:")
            print(f"  Tests: {len(opt['tests'])}")
            print(f"  Mean error difference: {opt['mean_error_diff']:.2e}")
            print(f"  Status: {'✓ PASSED' if opt['passed'] else '✗ FAILED'}")

        # Dense sampling tests
        if 'dense_sample' in results:
            dense = results['dense_sample']
            print(f"\nDense Sampling Tests:")
            print(f"  Tests: {len(dense['tests'])}")
            print(f"  Mean error difference: {dense['mean_error_diff']:.2e}")
            print(f"  Status: {'✓ PASSED' if dense['passed'] else '✗ FAILED'}")

        # Method consistency
        if 'method_tests' in results:
            meth = results['method_tests']
            print(f"\nMethod Consistency:")
            print(f"  Status: {'✓ CONSISTENT' if meth['consistent'] else '✗ INCONSISTENT'}")

        # Prime powers
        if 'prime_powers' in results:
            pp = results['prime_powers']
            print(f"\nPrime Powers Test:")
            print(f"  k=1: S = {pp['values'][1]:.6f}")
            print(f"  k=2: S = {pp['values'][2]:.6f} (change = {pp['change_k2_to_k1']:+.6f})")
            print(f"  Monotonic: {'✓' if pp['monotonic'] else '✗'}")

        print("\n" + "="*80)


# Convenience function for quick verification
def quick_verification():
    """Quick verification of S_euler implementation."""
    print("Loading legacy results for verification...")
    verifier = SEulerVerifier()

    # Need primes - generate a small set for testing
    from ..core.prime_cache import simple_sieve
    primes = np.array(simple_sieve(1_000_000))

    print("Running verification...")
    results = verifier.verify_implementation(primes)

    return results