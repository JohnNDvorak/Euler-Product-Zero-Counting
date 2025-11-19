#!/usr/bin/env python3
"""
Deep investigation of S_euler implementation to find the source of errors.
"""

import sys
sys.path.append('src')

import numpy as np
import cmath

# Generate primes
from core.prime_cache import simple_sieve

print("Generating primes...")
primes = np.array(simple_sieve(100_000))
print(f"✓ Generated {len(primes):,} primes")

# Import our implementation
from core.s_t_functions_stable import S_euler_stable

print("\n" + "="*80)
print("INVESTIGATING S_EULER IMPLEMENTATION")
print("="*80)

# Test simple case where we can compute manually
print("\n1. Simple case: T=100, P_max=1000 (primes only)")
T_test = 100.0
P_test = 1000

# Get primes up to P_max
prime_subset = primes[primes <= P_test]
print(f"Using {len(prime_subset)} primes up to {P_test}")

# Manual computation using direct formula
zeta_prod = 1.0 + 0.0j
for p in prime_subset:
    phase = T_test * np.log(p)
    factor = 1.0 / (1.0 - np.exp(-1j * phase))
    zeta_prod *= factor

S_manual = np.angle(zeta_prod) / np.pi
print(f"Manual direct product: S = {S_manual:.6f}")

# Test our methods
for method in ['direct', 'kahan', 'logarithmic']:
    S = S_euler_stable(T_test, P_test, primes, k_max=1, method=method)
    print(f"  {method:12s}: S = {S:.6f}")

# Check contributions
print("\n2. Checking individual prime contributions:")
print("First 5 contributions:")
contributions = []
for i, p in enumerate(prime_subset[:5]):
    phase = T_test * np.log(p)
    factor = 1.0 / (1.0 - np.exp(-1j * phase))
    z_temp = zeta_prod * (factor / zeta_prod)  # This is just factor

    # Log contribution
    log_factor = np.log(1.0 / (1.0 - np.exp(-1j * phase)))

    contributions.append({
        'prime': p,
        'phase': phase,
        'factor': factor,
        'log_factor': log_factor
    })

    print(f"  p={p:3d}: phase={phase:7.3f}, |factor|={abs(factor):.3f}, arg(factor)/π={np.angle(factor)/np.pi:7.3f}")

# Analyze phase distribution
print("\n3. Phase analysis:")
phases = [T_test * np.log(p) for p in prime_subset]
phases_mod = [p % (2*np.pi) for p in phases]
phases_wrapped = [(p if p <= np.pi else p - 2*np.pi) for p in phases_mod]

print(f"  Phase range: [{min(phases_wrapped):.3f}, {max(phases_wrapped):.3f}]")
print(f"  Mean phase: {np.mean(phases_wrapped):.3f}")
print(f"  Std phase: {np.std(phases_wrapped):.3f}")

# Check where phases are near critical values
near_0 = sum(1 for p in phases_mod if p < 0.1 or p > 2*np.pi - 0.1)
near_pi = sum(1 for p in phases_mod if abs(p - np.pi) < 0.1)
print(f"  Phases near 0 or 2π: {near_0}/{len(phases_mod)}")
print(f"  Phases near π: {near_pi}/{len(phases_mod)}")

# Test logarithmic method manually
print("\n4. Testing logarithmic method manually:")
log_zeta_real = 0.0
log_zeta_imag = 0.0

for p in prime_subset:
    phase = T_test * np.log(p)
    phase_mod = phase % (2 * np.pi)

    # Use arctangent formulation
    sin_half = np.sin(phase_mod / 2)
    log_term_real = np.log(2 * abs(sin_half))
    log_term_imag = -(np.pi - phase_mod) / 2

    log_zeta_real -= log_term_real
    log_zeta_imag -= log_term_imag

S_log_manual = np.arctan2(log_zeta_imag, log_zeta_real) / np.pi
S_log_norm = ((S_log_manual + 0.5) % 1.0) - 0.5

print(f"Manual logarithmic: S = {S_log_manual:.6f}")
print(f"After normalization: S = {S_log_norm:.6f}")
print(f"Our implementation:  S = {S_euler_stable(T_test, P_test, primes, method='logarithmic'):.6f}")

# Test with mpmath for reference
print("\n5. mpmath reference calculation:")
try:
    import mpmath as mp
    mp.dps = 50

    log_zeta_mp = mp.mpc(0)
    for p in prime_subset[:10]:  # Just first 10 for speed
        log_zeta_mp += -mp.log(1 - mp.e**(-1j * T_test * mp.log(p)))

    S_mp = float(mp.arg(log_zeta_mp) / mp.pi)
    print(f"mpmath (first 10 primes): S = {S_mp:.6f}")
except Exception as e:
    print(f"  mpmath failed: {e}")

# Check the formula interpretation
print("\n6. Formula interpretation check:")
print("S(T) = (1/π) * arg ζ(1/2 + iT)")
print("With Euler product: ζ(s) = ∏(1 - p^(-s))^(-1)")
print("So log ζ(s) = -∑ log(1 - p^(-s))")
print("\nFor s = 1/2 + iT:")
print("p^(-s) = p^(-1/2) * p^(-iT)")
print("But we're computing p^(-iT) only - this might be the issue!")

# Test including p^(-1/2) factor
print("\n7. Testing with p^(-1/2) factor:")
zeta_prod_full = 1.0 + 0.0j
for p in prime_subset[:10]:  # Just first 10
    # Full factor: p^(-s) = p^(-1/2) * p^(-iT)
    p_power = p**(-0.5 - 1j * T_test)
    factor = 1.0 / (1.0 - p_power)
    zeta_prod_full *= factor

S_full = np.angle(zeta_prod_full) / np.pi
print(f"With p^(-1/2): S = {S_full:.6f} (first 10 primes)")

# Compare with our current method (no p^(-1/2))
zeta_prod_no_sqrt = 1.0 + 0.0j
for p in prime_subset[:10]:
    phase = T_test * np.log(p)
    factor = 1.0 / (1.0 - np.exp(-1j * phase))
    zeta_prod_no_sqrt *= factor

S_no_sqrt = np.angle(zeta_prod_no_sqrt) / np.pi
print(f"Without p^(-1/2): S = {S_no_sqrt:.6f} (first 10 primes)")
print(f"Difference: {abs(S_full - S_no_sqrt):.6f}")

print("\n" + "="*80)
print("FINDINGS")
print("="*80)
print("\n1. Check if we're missing the p^(-1/2) factor")
print("2. Verify normalization of complex arguments")
print("3. Consider branch cut issues in arctan2/angle")
print("4. The large errors suggest a fundamental formula difference")