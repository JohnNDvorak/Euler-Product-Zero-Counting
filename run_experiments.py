#!/usr/bin/env python3
"""
Run the main experiments for Prime Reduction Estimates for S(T).

This script runs all three main experiments:
1. Optimal Truncation Search
2. Phase Cancellation Validation
3. Method Comparison
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress, kstest

# Import modules
from src.utils.paths import PathConfig
from src.core.s_t_functions import S_RS, S_euler, analyze_error
from src.core.prime_cache import simple_sieve
from src.core.numerical_utils import kahan_sum_complex

print("="*80)
print("PRIME REDUCTION ESTIMATES FOR S(T)")
print("Running Main Experiments")
print("="*80)

# Initialize paths
paths = PathConfig()
paths.ensure_dirs()

# Load cached zeros
print("\nLoading zeros from cache...")
zeros = np.load(paths.cache_dir / "zeros.npy")
print(f"✓ Loaded {len(zeros):,} zeros")

# Generate primes (up to 200M for main experiments)
print("\nGenerating primes up to 200 million...")
MAX_PRIME = 200_000_000
primes_list = simple_sieve(MAX_PRIME)
primes = np.array(primes_list)
print(f"✓ Generated {len(primes):,} primes")

# Save prime cache
prime_cache_path = paths.cache_dir / "prime_cache_200M.pkl"
with open(prime_cache_path, 'wb') as f:
    pickle.dump({'primes': primes}, f)
print(f"✓ Saved to {prime_cache_path}")

# ==============================
# EXPERIMENT 1: OPTIMAL TRUNCATION
# ==============================

print("\n" + "="*80)
print("EXPERIMENT 1: OPTIMAL TRUNCATION SEARCH")
print("="*80)

# Test parameters
T_TEST = [1_000, 10_000, 100_000, 1_000_000]
P_MAX_MIN = 1e6
P_MAX_MAX = 2e8
N_POINTS = 30  # Reduced for faster execution
P_MAX_RANGE = np.logspace(np.log10(P_MAX_MIN), np.log10(P_MAX_MAX), N_POINTS)

print(f"\nT values: {T_TEST}")
print(f"P_max range: {P_MAX_MIN:.0e} to {P_MAX_MAX:.0e} ({N_POINTS} points)")

def run_optimal_truncation(T, P_max_range, zeros, primes):
    """Find optimal P_max for a given T."""
    print(f"\nFinding optimal P_max for T = {T:,}")
    print("-" * 50)

    # Reference value (using Riemann-Siegel)
    S_ref = S_RS(T, zeros)
    print(f"Reference S_RS({T}) = {S_ref:.6f}")

    # Test different P_max values
    errors = []
    S_values = []

    for i, P_max in enumerate(P_max_range):
        if i % 5 == 0:
            print(f"  Progress: {i}/{len(P_max_range)} - P_max = {P_max:.2e}")

        # Compute S_euler at this P_max
        S_e = S_euler_fast(T, P_max, primes)
        S_values.append(S_e)

        # Compute error
        error = abs(S_e - S_ref)
        errors.append(error)

    # Convert to arrays
    errors = np.array(errors)
    S_values = np.array(S_values)

    # Find optimal P_max (minimum error)
    optimal_idx = np.argmin(errors)
    optimal_P_max = P_max_range[optimal_idx]
    min_error = errors[optimal_idx]

    print(f"  Optimal P_max: {optimal_P_max:.2e} (error = {min_error:.6f})")

    return {
        'T': T,
        'P_max_range': P_max_range,
        'errors': errors,
        'S_values': S_values,
        'S_ref': S_ref,
        'optimal_P_max': optimal_P_max,
        'optimal_error': min_error,
        'optimal_idx': optimal_idx
    }

# Run Experiment 1
exp1_results = []
start_time = time.time()

for T in T_TEST:
    result = run_optimal_truncation(T, P_MAX_RANGE, zeros, primes)
    exp1_results.append(result)

elapsed = time.time() - start_time
print(f"\nExperiment 1 completed in {elapsed:.1f} seconds")

# Save Experiment 1 results
with open(paths.results_dir / 'exp1_optimal_results.pkl', 'wb') as f:
    pickle.dump(exp1_results, f)

# Create summary
summary_data = []
for result in exp1_results:
    summary_data.append({
        'T': result['T'],
        'optimal_P_max': result['optimal_P_max'],
        'optimal_error': result['optimal_error'],
        'S_ref': result['S_ref'],
        'log10_T': np.log10(result['T']),
        'log10_P_opt': np.log10(result['optimal_P_max'])
    })

df_exp1 = pd.DataFrame(summary_data)
df_exp1.to_csv(paths.results_dir / 'exp1_summary.csv', index=False)

# Analyze scaling
log_T = np.log10(df_exp1['T'])
log_P_opt = np.log10(df_exp1['optimal_P_max'])
slope, intercept, r_value, _, _ = linregress(log_T, log_P_opt)

print(f"\nExperiment 1 Results Summary:")
print("="*50)
print(df_exp1[['T', 'optimal_P_max', 'optimal_error']].to_string(index=False, float_format='{:,.6e}'.format))
print(f"\nScaling Analysis:")
print(f"  log10(P_opt) = {slope:.3f} * log10(T) + {intercept:.3f}")
print(f"  R² = {r_value**2:.4f}")
print(f"  P_opt ≈ T^{slope:.3f}")
print(f"  Expected exponent: 0.25 (T^1/4)")

# ==============================
# EXPERIMENT 2: PHASE VALIDATION
# ==============================

print("\n" + "="*80)
print("EXPERIMENT 2: PHASE CANCELLATION VALIDATION")
print("="*80)

def analyze_phases(T, primes, max_primes=50_000):
    """Analyze phase distribution for given T."""
    print(f"\nAnalyzing phases for T = {T:,}")

    # Take subset of primes
    primes_subset = primes[:max_primes]

    # Compute phases
    phases = (T * np.log(primes_subset) / (2 * np.pi)) % 1

    # KS test for uniformity
    ks_stat, ks_pvalue = kstest(phases, 'uniform')

    # Partial sums
    phases_complex = np.exp(2j * np.pi * phases)
    partial_sums = np.cumsum(phases_complex)
    partial_magnitudes = np.abs(partial_sums)

    print(f"  KS statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")
    print(f"  Final magnitude: {partial_magnitudes[-1]:.2f}")

    return {
        'T': T,
        'phases': phases,
        'partial_magnitudes': partial_magnitudes,
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue
    }

# Test phase uniformity
exp2_results = []
for T in T_TEST[:3]:  # Test first 3 values
    result = analyze_phases(T, primes)
    exp2_results.append(result)

# ==============================
# EXPERIMENT 3: METHOD COMPARISON
# ==============================

print("\n" + "="*80)
print("EXPERIMENT 3: METHOD COMPARISON")
print("="*80)

def compare_methods(T_list, optimal_results, zeros, primes):
    """Compare different S(T) computation methods."""
    comparison = []

    for i, T in enumerate(T_list):
        print(f"\nComparing methods for T = {T:,}")

        # Get optimal P_max from Experiment 1
        optimal_P_max = optimal_results[i]['optimal_P_max']
        S_ref = optimal_results[i]['S_ref']

        # Method 1: Riemann-Siegel
        start = time.time()
        S_rs = S_RS(T, zeros)
        time_rs = time.time() - start
        error_rs = abs(S_rs - S_ref)

        # Method 2: Euler at optimal P_max
        start = time.time()
        S_euler_opt = S_euler_fast(T, optimal_P_max, primes)
        time_euler_opt = time.time() - start
        error_euler_opt = abs(S_euler_opt - S_ref)

        # Method 3: Euler at fixed P_max
        P_fixed = 50_000_000
        start = time.time()
        S_euler_fixed = S_euler_fast(T, P_fixed, primes)
        time_euler_fixed = time.time() - start
        error_euler_fixed = abs(S_euler_fixed - S_ref)

        print(f"  RS:       {S_rs:.6f}, error={error_rs:.6f}, time={time_rs:.4f}s")
        print(f"  Euler opt: {S_euler_opt:.6f}, error={error_euler_opt:.6f}, time={time_euler_opt:.4f}s")
        print(f"  Euler fix: {S_euler_fixed:.6f}, error={error_euler_fixed:.6f}, time={time_euler_fixed:.4f}s")

        # Calculate improvement
        if error_euler_fixed > 0:
            improvement = (error_euler_fixed - error_euler_opt) / error_euler_fixed * 100
        else:
            improvement = 0

        print(f"  Improvement (opt vs fixed): {improvement:.1f}%")

        comparison.append({
            'T': T,
            'S_RS': S_rs,
            'S_euler_opt': S_euler_opt,
            'S_euler_fixed': S_euler_fixed,
            'error_RS': error_rs,
            'error_euler_opt': error_euler_opt,
            'error_euler_fixed': error_euler_fixed,
            'improvement': improvement,
            'time_RS': time_rs,
            'time_euler_opt': time_euler_opt,
            'time_euler_fixed': time_euler_fixed,
            'optimal_P_max': optimal_P_max
        })

    return comparison

# Run comparison
exp3_results = compare_methods(T_TEST[:3], exp1_results[:3], zeros, primes)
df_exp3 = pd.DataFrame(exp3_results)
df_exp3.to_csv(paths.results_dir / 'exp3_comparison.csv', index=False)

# ==============================
# GENERATE FIGURES
# ==============================

print("\n" + "="*80)
print("GENERATING FIGURES")
print("="*80)

# Figure 1: Optimal truncation results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Error curves for Experiment 1
colors = plt.cm.viridis(np.linspace(0, 1, len(exp1_results)))
for i, result in enumerate(exp1_results):
    ax1.loglog(result['P_max_range'], result['errors'],
               color=colors[i], label=f"T={result['T']:.0e}")
    ax1.axvline(result['optimal_P_max'], color=colors[i],
                linestyle='--', alpha=0.5)

ax1.set_xlabel('P_max')
ax1.set_ylabel('Error |S_euler - S_ref|')
ax1.set_title('Experiment 1: Error vs P_max')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Scaling relationship
ax2.loglog(df_exp1['T'], df_exp1['optimal_P_max'], 'bo-', markersize=8)
T_fit = np.logspace(np.log10(df_exp1['T'].min()), np.log10(df_exp1['T'].max()), 100)
P_fit = 10**intercept * T_fit**slope
ax2.loglog(T_fit, P_fit, 'r--', label=f'Fit: T^{slope:.3f}')
ax2.set_xlabel('T')
ax2.set_ylabel('Optimal P_max')
ax2.set_title(f'Optimal P_max Scaling (Exponent = {slope:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Phase distributions
for i, result in enumerate(exp2_results):
    ax3.hist(result['phases'], bins=30, density=True, alpha=0.5,
             label=f"T={result['T']:.0e}", color=colors[i])
ax3.axhline(y=1, color='red', linestyle='--', label='Uniform')
ax3.set_xlabel('Phase mod 1')
ax3.set_ylabel('Density')
ax3.set_title('Phase Distribution Uniformity')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Method comparison
x = np.arange(len(df_exp3))
width = 0.25
ax4.bar(x - width, df_exp3['error_RS'], width, label='Riemann-Siegel')
ax4.bar(x, df_exp3['error_euler_opt'], width, label='Euler (Optimal)')
ax4.bar(x + width, df_exp3['error_euler_fixed'], width, label='Euler (Fixed)')
ax4.set_xlabel('T')
ax4.set_ylabel('Error')
ax4.set_title('Method Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels([f'{T:.0e}' for T in df_exp3['T']])
ax4.legend()
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(paths.figures_dir / 'main_experiments.png', dpi=300, bbox_inches='tight')
print("✓ Figure saved to main_experiments.png")

# ==============================
# FINAL SUMMARY
# ==============================

print("\n" + "="*80)
print("EXPERIMENTS COMPLETED")
print("="*80)

print("\nKey Results:")
print(f"\n1. Optimal P_max Scaling:")
print(f"   Exponent: {slope:.3f} (theoretical: 0.25)")
print(f"   R²: {r_value**2:.4f}")

print(f"\n2. Method Performance (T={df_exp3['T'].iloc[1]:,}):")
print(f"   RS error: {df_exp3['error_RS'].iloc[1]:.6f}")
print(f"   Euler optimal error: {df_exp3['error_euler_opt'].iloc[1]:.6f}")
print(f"   Euler fixed error: {df_exp3['error_euler_fixed'].iloc[1]:.6f}")
print(f"   Improvement: {df_exp3['improvement'].iloc[1]:.1f}%")

print(f"\n3. Computational Cost:")
print(f"   RS time: {df_exp3['time_RS'].iloc[1]:.4f}s (scales as √T)")
print(f"   Euler time: {df_exp3['time_euler_opt'].iloc[1]:.4f}s (fixed)")

print("\n" + "="*80)
print("✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
print("="*80)