#!/usr/bin/env python3
"""
Phase 2: Dense Sampling Experiment (Cell 16)

Recreates the complete dense sampling experiment with:
- 599 T values (based on zeta zeros)
- 6 P_max values (100 to 10,000,000)
- Total: 3,594 measurements
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import time
from pathlib import Path

# Load required modules
from utils.legacy_results_loader import LegacyResultsLoader
from core.s_t_functions import S_euler, S_euler_with_breakdown
from core.prime_cache import simple_sieve

print("="*80)
print("PHASE 2: DENSE SAMPLING EXPERIMENT (Cell 16)")
print("="*80)

# 1. Generate T values based on zeta zeros
print("\n1. Generating T values from zeta zeros...")

# Load zeros
try:
    zeros = np.loadtxt("data/10M_zeta_zeros.txt")
    print(f"✓ Loaded {len(zeros):,} zeta zeros")
except FileNotFoundError:
    print("Warning: No zeros file found, using synthetic T values")
    # Generate synthetic zeros using approximation
    zeros = (np.arange(1, 1000) + 0.75) * np.pi

# Select 599 T values from zeros (similar to original notebook)
T_min = 100
T_max = 10000
zeros_in_range = zeros[(zeros >= T_min) & (zeros <= T_max)]

if len(zeros_in_range) >= 599:
    # Take evenly spaced zeros
    indices = np.linspace(0, len(zeros_in_range) - 1, 599, dtype=int)
    T_values = zeros_in_range[indices]
else:
    # Generate evenly spaced T values if not enough zeros
    T_values = np.linspace(T_min, T_max, 599)

print(f"✓ Selected {len(T_values)} T values from {T_values[0]:.1f} to {T_values[-1]:.1f}")

# 2. Prepare primes
print("\n2. Preparing primes...")

# Need primes up to max P_max = 10M
max_prime_needed = 10_000_000

# Check if we already have enough primes
prime_cache_file = "data/primes_up_to_10M.npy"
if Path(prime_cache_file).exists():
    primes = np.load(prime_cache_file)
    print(f"✓ Loaded {len(primes):,} primes from cache")
else:
    print(f"Generating primes up to {max_prime_needed:,}...")
    start_time = time.time()
    primes = np.array(simple_sieve(max_prime_needed))
    elapsed = time.time() - start_time
    print(f"✓ Generated {len(primes):,} primes in {elapsed:.1f}s")

    # Save to cache
    np.save(prime_cache_file, primes)
    print(f"✓ Saved primes to {prime_cache_file}")

# 3. Define P_max values (matching original notebook)
P_max_values = [100, 1000, 10000, 100000, 1000000, 10000000]
print(f"\n3. Testing P_max values: {P_max_values}")

# 4. Run dense sampling experiment
print("\n4. Running dense sampling experiment...")
print(f"Total computations: {len(T_values)} × {len(P_max_values)} = {len(T_values) * len(P_max_values):,}")
print("-" * 60)

# Initialize results storage
results = []
total_computations = 0
start_time = time.time()

# For each T value
for i, T in enumerate(T_values):
    # Compute reference S using largest P_max
    S_ref = S_euler(T, P_max_values[-1], primes, k_max=1)

    # Test each P_max value
    for P_max in P_max_values:
        # Compute S_euler
        if P_max == P_max_values[-1]:
            S_approx = S_ref  # Already computed
        else:
            S_approx = S_euler(T, P_max, primes, k_max=1)

        # Calculate error
        error = abs(S_approx - S_ref)

        # Count primes up to P_max
        n_primes = len(primes[primes <= P_max])

        # Calculate improvement vs P_max=100
        if P_max == 100:
            baseline_error = error
            improvement = 0.0
        else:
            improvement = (1 - error / baseline_error) * 100 if baseline_error > 0 else 0

        # Store result
        results.append({
            'T_idx': i,
            'T': T,
            'P_max': P_max,
            'n_primes': n_primes,
            'S_ref': S_ref,
            'S_approx': S_approx,
            'error_S': error,
            'improvement': improvement,
            'computation_time': 0.0  # Not tracking individual times for speed
        })

        total_computations += 1

    # Progress update
    if (i + 1) % 50 == 0 or i == len(T_values) - 1:
        elapsed = time.time() - start_time
        rate = total_computations / elapsed
        eta = (len(T_values) * len(P_max_values) - total_computations) / rate
        print(f"  Progress: {i+1}/{len(T_values)} ({100*(i+1)/len(T_values):.0f}%) - "
              f"{rate:.1f} comps/sec - ETA: {eta:.0f}s")

# Convert to DataFrame
results_df = pd.DataFrame(results)
total_time = time.time() - start_time

print(f"\n✓ Completed in {total_time:.1f}s")
print(f"  Total computations: {total_computations:,}")
print(f"  Average time: {total_time/total_computations*1000:.2f}ms per computation")

# 5. Save results
print("\n5. Saving results...")
output_file = "data/dense_sampling_results.csv"
results_df.to_csv(output_file, index=False)
print(f"✓ Saved to {output_file}")

# 6. Analysis
print("\n6. Analyzing results...")
print("-" * 50)

# Summary by P_max
summary_by_P = results_df.groupby('P_max').agg({
    'error_S': ['mean', 'std', 'min', 'max'],
    'improvement': 'mean'
}).round(6)

print("\nMean error by P_max:")
for P_max in P_max_values:
    subset = results_df[results_df['P_max'] == P_max]
    mean_error = subset['error_S'].mean()
    print(f"  P_max = {P_max:>10,}: {mean_error:.6f}")

# Best P_max distribution
best_per_T = results_df.loc[results_df.groupby('T')['error_S'].idxmin()]
P_counts = best_per_T['P_max'].value_counts().sort_index()

print(f"\nBest P_max distribution (for {len(best_per_T)} T values):")
for P_max, count in P_counts.items():
    pct = count / len(best_per_T) * 100
    print(f"  P_max = {P_max:>10,}: {count:3d} T values ({pct:.1f}%)")

# Improvement statistics
improvement_stats = results_df[results_df['P_max'] > 100]['improvement'].describe()
print(f"\nImprovement over P_max=100:")
print(f"  Mean: {improvement_stats['mean']:.1f}%")
print(f"  Max: {improvement_stats['max']:.1f}%")
print(f"  Std: {improvement_stats['std']:.1f}%")

# 7. Verification against legacy data
print("\n7. Verifying against legacy data...")
print("-" * 50)

# Load legacy data
loader = LegacyResultsLoader()
legacy_results = loader.load_all_results()
legacy_dense = legacy_results['dense']

# Compare a sample of results
n_test = min(100, len(legacy_dense))
max_diff = 0
matches = 0

for i in range(n_test):
    legacy_row = legacy_dense.iloc[i]
    T = legacy_row['T']
    P_max = legacy_row['P_max']

    # Find our result
    our_row = results_df[(results_df['T'] == T) & (results_df['P_max'] == P_max)]

    if len(our_row) > 0:
        diff = abs(our_row['S_approx'].iloc[0] - legacy_row['S_approx'])
        max_diff = max(max_diff, diff)
        if diff < 1e-10:
            matches += 1

print(f"Compared {n_test} measurements:")
print(f"  Perfect matches: {matches}/{n_test}")
print(f"  Maximum difference: {max_diff:.2e}")

# 8. Create visualization
print("\n8. Creating visualization...")

import matplotlib.pyplot as plt

# Create figures directory
Path("figures").mkdir(exist_ok=True)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dense Sampling Analysis (599 T × 6 P_max)', fontsize=16)

# Plot 1: Error heatmap
pivot_error = results_df.pivot(index='T_idx', columns='P_max', values='error_S')
im1 = axes[0, 0].imshow(np.log10(pivot_error.T), aspect='auto', cmap='viridis_r')
axes[0, 0].set_title('Log10(Error) Heatmap')
axes[0, 0].set_xlabel('T index')
axes[0, 0].set_ylabel('P_max')
axes[0, 0].set_yticks(range(len(P_max_values)))
axes[0, 0].set_yticklabels([f'{p:,}' for p in P_max_values])
plt.colorbar(im1, ax=axes[0, 0])

# Plot 2: Mean error vs P_max
mean_error_by_P = results_df.groupby('P_max')['error_S'].mean()
axes[0, 1].loglog(P_max_values, mean_error_by_P.values, 'o-', linewidth=2)
axes[0, 1].set_title('Mean Error vs P_max')
axes[0, 1].set_xlabel('P_max')
axes[0, 1].set_ylabel('Mean Error')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Improvement distribution
improvement_data = []
labels = []
for P_max in P_max_values[1:]:  # Skip P_max=100 (baseline)
    imp = results_df[results_df['P_max'] == P_max]['improvement']
    improvement_data.append(imp.values)
    labels.append(f'{P_max:,}')

axes[0, 2].boxplot(improvement_data, labels=labels)
axes[0, 2].set_title('Improvement over Baseline')
axes[0, 2].set_xlabel('P_max')
axes[0, 2].set_ylabel('Improvement (%)')
axes[0, 2].tick_params(axis='x', rotation=45)
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Sample S(T) curves
for P_max in [100, 10000, 1000000]:
    subset = results_df[results_df['P_max'] == P_max]
    axes[1, 0].plot(subset['T'], subset['S_approx'],
                    label=f'P_max={P_max:,}', alpha=0.7, linewidth=0.5)
axes[1, 0].set_title('Sample S(T) Computations')
axes[1, 0].set_xlabel('T')
axes[1, 0].set_ylabel('S(T)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Best P_max vs T
axes[1, 1].scatter(np.log10(best_per_T['T']), best_per_T['P_max'],
                   alpha=0.5, s=5)
axes[1, 1].set_title('Optimal P_max vs T')
axes[1, 1].set_xlabel('log10(T)')
axes[1, 1].set_ylabel('Optimal P_max')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/dense_sampling_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to figures/dense_sampling_analysis.png")

print("\n" + "="*80)
print("PHASE 2 COMPLETE")
print("="*80)

print(f"\n📊 SUMMARY:")
print(f"   • Computed {len(results_df):,} S(T) values")
print(f"   • Range: T∈[{T_values[0]:.0f}, {T_values[-1]:.0f}], P_max∈[{P_max_values[0]:,}, {P_max_values[-1]:,}]")
print(f"   • Best P_max: {P_counts.index[0]:,} (used for {P_counts.iloc[0]:.1f}% of T values)")
print(f"   • Mean improvement: {improvement_stats['mean']:.1f}% over baseline")
print(f"   • Verification: {matches}/{n_test} matches with legacy data")

print(f"\n📁 Files created:")
print(f"   • data/dense_sampling_results.csv")
print(f"   • figures/dense_sampling_analysis.png")
print(f"   • data/primes_up_to_10M.npy (prime cache)")

print(f"\n🚀 Ready for Phase 3: Phase Validation")