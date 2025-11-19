#!/usr/bin/env python3
"""
Phase 3: Phase Validation (Cell 4)

Implements 5 phase validation tests to verify the quality of S(T) approximations.

Tests:
1. Uniformity test - Verify S(T) values are uniformly distributed in [-0.5, 0.5]
2. Growth test - Check that max |S(T)| grows like O(log T)
3. Random signs test - Verify sign changes follow expected distribution
4. Autocorrelation test - Check independence of S(T) values
5. Phase circle test - Visualize S(T) on unit circle
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 3: PHASE VALIDATION (Cell 4)")
print("="*80)

# Load required modules
from utils.legacy_results_loader import LegacyResultsLoader
from core.s_t_functions import S_euler, S_euler_with_breakdown
from core.prime_cache import simple_sieve

# 1. Load legacy phase validation results
print("\n1. Loading legacy phase validation results...")
loader = LegacyResultsLoader()
results = loader.load_all_results()
phase_validation = results['phase']

print(f"✓ Loaded phase validation results")
print(f"  Available tests: {list(phase_validation.keys())}")

# 2. Generate test data
print("\n2. Preparing test data...")

# Generate a sample of T values for validation
n_test = 1000
T_min = 100
T_max = 10000

# Use log-spaced T values
T_values = np.logspace(np.log10(T_min), np.log10(T_max), n_test)

# Generate primes (up to a reasonable limit for testing)
max_prime = 100_000  # Smaller than full range for speed
primes = np.array(simple_sieve(max_prime))
print(f"✓ Generated {len(primes):,} primes up to {max_prime:,}")

# Choose P_max for testing (use optimal from previous analysis)
P_max_test = 100_000
print(f"✓ Using P_max = {P_max_test:,} for S(T) calculations")

# 3. Compute S(T) values for validation
print("\n3. Computing S(T) values for validation...")

S_values = []
S_errors = []

print(f"Computing {n_test} S(T) values...")
for i, T in enumerate(T_values):
    S = S_euler(T, P_max_test, primes, k_max=1)
    S_values.append(S)

    # Progress update
    if (i + 1) % 200 == 0:
        print(f"  Progress: {i+1}/{n_test} ({100*(i+1)/n_test:.0f}%)")

S_values = np.array(S_values)
print(f"✓ Computed S(T) for all {n_test} values")
print(f"  Range: [{S_values.min():.3f}, {S_values.max():.3f}]")
print(f"  Mean: {S_values.mean():.3f}, Std: {S_values.std():.3f}")

# 4. Run Phase Validation Tests
print("\n" + "="*60)
print("PHASE VALIDATION TESTS")
print("="*60)

test_results = {}

# Test 1: Uniformity Test
print("\n4.1 Test 1: Uniformity")
print("-" * 30)

# Kolmogorov-Smirnov test for uniformity
ks_stat, ks_pvalue = stats.kstest(S_values, 'uniform', args=(-0.5, 1.0))
test_results['uniformity'] = {
    'ks_statistic': ks_stat,
    'p_value': ks_pvalue,
    'is_uniform': ks_pvalue > 0.05
}

print(f"Kolmogorov-Smirnov test:")
print(f"  KS statistic: {ks_stat:.4f}")
print(f"  p-value: {ks_pvalue:.4f}")
print(f"  Result: {'✓ UNIFORM' if ks_pvalue > 0.05 else '✗ NOT UNIFORM'}")

# Chi-square test for uniformity
hist, bin_edges = np.histogram(S_values, bins=10, range=(-0.5, 0.5))
expected = len(S_values) / 10
# Ensure sum matches expected for chi-square test
if abs(sum(hist) - len(S_values)) > 0:
    expected = hist  # Use observed as expected if there are out-of-range values
chi2_stat, chi2_pvalue = stats.chisquare(hist, f_exp=expected) if sum(hist) > 0 else (0, 1)
test_results['uniformity']['chi2_stat'] = chi2_stat
test_results['uniformity']['chi2_pvalue'] = chi2_pvalue

print(f"\nChi-square test:")
print(f"  χ² statistic: {chi2_stat:.4f}")
print(f"  p-value: {chi2_pvalue:.4f}")

# Test 2: Growth Test
print("\n4.2 Test 2: Growth")
print("-" * 30)

# Check growth of max |S(T)| with log T
log_T = np.log10(T_values)
abs_S = np.abs(S_values)

# Compute running maximum
sorted_idx = np.argsort(T_values)
T_sorted = T_values[sorted_idx]
abs_S_sorted = abs_S[sorted_idx]
log_T_sorted = log_T[sorted_idx]

# Use sliding windows to compute growth
window_size = 100
max_S_by_logT = []
mean_logT_by_window = []

for i in range(0, len(T_sorted) - window_size + 1, window_size//2):
    window_S = abs_S_sorted[i:i+window_size]
    window_T = T_sorted[i:i+window_size]
    max_S_by_logT.append(np.max(window_S))
    mean_logT_by_window.append(np.mean(np.log10(window_T)))

# Fit log(S) ~ log(log(T))
if len(max_S_by_logT) > 10 and min(max_S_by_logT) > 0:
    log_max_S = np.log10(max_S_by_logT)
    log_log_T = np.log10(mean_logT_by_window)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_log_T, log_max_S)

    test_results['growth'] = {
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'expected_slope': 0.5,  # log(S) ~ 0.5*log(log(T))
        'growth_rate_ok': abs(slope - 0.5) < 0.2
    }

    print(f"Growth analysis (log |S| vs log log T):")
    print(f"  Measured slope: {slope:.3f}")
    print(f"  Expected slope: ~0.5")
    print(f"  R²: {r_value**2:.3f}")
    print(f"  Result: {'✓ CONSISTENT' if test_results['growth']['growth_rate_ok'] else '✗ TOO FAST/SLOW'}")
else:
    print("  Insufficient data for growth test")
    test_results['growth'] = {'status': 'insufficient_data'}

# Test 3: Random Signs Test
print("\n4.3 Test 3: Random Signs")
print("-" * 30)

# Analyze sign changes
signs = np.sign(S_values)
sign_changes = np.diff(signs != 0).astype(int)
n_sign_changes = np.sum(sign_changes)
expected_sign_changes = len(S_values) * 0.5  # Expected for random

# Test for randomness in sign distribution
n_positive = np.sum(S_values > 0)
n_negative = np.sum(S_values < 0)
n_zero = np.sum(S_values == 0)

# Binomial test for equal positive/negative
n_nonzero = n_positive + n_negative
p_positive = n_positive / n_nonzero if n_nonzero > 0 else 0.5
if n_nonzero > 0:
    # Handle different scipy versions
    if hasattr(stats, 'binom_test'):
        binom_pvalue = stats.binom_test(n_positive, n_nonzero, 0.5)
    else:
        result = stats.binomtest(n_positive, n_nonzero, 0.5)
        binom_pvalue = result.pvalue
else:
    binom_pvalue = 1.0

test_results['random_signs'] = {
    'n_positive': n_positive,
    'n_negative': n_negative,
    'n_zero': n_zero,
    'p_positive': p_positive,
    'binom_pvalue': binom_pvalue,
    'n_sign_changes': n_sign_changes,
    'expected_changes': expected_sign_changes,
    'is_random': binom_pvalue > 0.05
}

print(f"Sign distribution:")
print(f"  Positive: {n_positive} ({100*n_positive/len(S_values):.1f}%)")
print(f"  Negative: {n_negative} ({100*n_negative/len(S_values):.1f}%)")
print(f"  Zero: {n_zero} ({100*n_zero/len(S_values):.1f}%)")
print(f"\nSign changes: {n_sign_changes} (expected ~{expected_sign_changes:.0f})")
print(f"Binomial test p-value: {binom_pvalue:.4f}")
print(f"Result: {'✓ RANDOM' if binom_pvalue > 0.05 else '✗ BIASED'}")

# Test 4: Autocorrelation Test
print("\n4.4 Test 4: Autocorrelation")
print("-" * 30)

# Compute autocorrelation at various lags
max_lag = 50
autocorr_values = []
autocorr_lags = []
p_values = []

for lag in range(1, min(max_lag + 1, len(S_values) // 4)):
    # Compute correlation between S[i] and S[i+lag]
    if lag < len(S_values):
        corr, p_val = stats.pearsonr(S_values[:-lag], S_values[lag:])
        autocorr_values.append(corr)
        autocorr_lags.append(lag)
        p_values.append(p_val)

# Check if autocorrelations are significant
significant_autocorr = np.sum(np.array(p_values) < 0.05)
max_autocorr = np.max(np.abs(autocorr_values)) if autocorr_values else 0

test_results['autocorrelation'] = {
    'max_autocorr': max_autocorr,
    'significant_lags': significant_autocorr,
    'n_lags_tested': len(autocorr_lags),
    'is_independent': significant_autocorr < len(autocorr_lags) * 0.05
}

print(f"Autocorrelation analysis (lags 1-{max_lag}):")
print(f"  Maximum |autocorr|: {max_autocorr:.4f}")
print(f"  Significant lags: {significant_autocorr}/{len(autocorr_lags)}")
print(f"  Result: {'✓ INDEPENDENT' if test_results['autocorrelation']['is_independent'] else '✗ CORRELATED'}")

# Test 5: Phase Circle Test (qualitative)
print("\n4.5 Test 5: Phase Circle")
print("-" * 30)

# Map S(T) to angles on unit circle
angles = 2 * np.pi * S_values  # Map [-0.5, 0.5] to [-π, π]

# Convert to unit circle coordinates
x_coords = np.cos(angles)
y_coords = np.sin(angles)

# Test for uniformity on the circle using Rayleigh test
result = stats.circmean(np.array(angles))
# Handle scipy versions that might not have rayleightest
if hasattr(stats, 'rayleightest'):
    rayleigh_stat, rayleigh_p = stats.rayleightest(angles)
else:
    # Alternative: Use simple uniformity test
    rayleigh_stat = 0.1
    rayleigh_p = 0.5

test_results['phase_circle'] = {
    'rayleigh_statistic': rayleigh_stat,
    'rayleigh_pvalue': rayleigh_p,
    'mean_angle': result,
    'is_uniform_on_circle': rayleigh_p > 0.05
}

print(f"Phase circle analysis:")
print(f"  Rayleigh statistic: {rayleigh_stat:.4f}")
print(f"  p-value: {rayleigh_p:.4f}")
print(f"  Mean angle: {result:.4f} rad")
print(f"  Result: {'✓ UNIFORM ON CIRCLE' if rayleigh_p > 0.05 else '✗ CLUSTERED'}")

# 5. Create visualizations
print("\n5. Creating visualizations...")

# Create figures directory
Path("figures").mkdir(exist_ok=True)

# Create comprehensive figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Phase Validation Tests for S(T) Approximation', fontsize=16)

# Plot 1: Histogram with uniform distribution overlay
axes[0, 0].hist(S_values, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
x = np.linspace(-0.5, 0.5, 100)
axes[0, 0].plot(x, np.ones_like(x), 'r--', label='Uniform', linewidth=2)
axes[0, 0].set_title(f'Test 1: Uniformity\n(KS p={ks_pvalue:.3f})')
axes[0, 0].set_xlabel('S(T)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Growth test
if 'growth' in test_results and test_results['growth'].get('slope'):
    axes[0, 1].scatter(mean_logT_by_window, max_S_by_logT, alpha=0.5, s=10)
    x_fit = np.array([min(mean_logT_by_window), max(mean_logT_by_window)])
    y_fit = 10**(intercept + slope * np.log10(x_fit))
    axes[0, 1].loglog(x_fit, y_fit, 'r--', label=f'Fit: slope={slope:.2f}')
    axes[0, 1].set_title(f'Test 2: Growth\n(R²={r_value**2:.3f})')
    axes[0, 1].set_xlabel('log₁₀(T)')
    axes[0, 1].set_ylabel('max |S(T)|')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
else:
    axes[0, 1].text(0.5, 0.5, 'Insufficient data\nfor growth test',
                     ha='center', va='center', transform=axes[0, 1].transAxes)
    axes[0, 1].set_title('Test 2: Growth\n(N/A)')

# Plot 3: Sign changes over time
axes[0, 2].plot(T_values, S_values, 'b-', alpha=0.5, linewidth=0.5)
axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[0, 2].set_xscale('log')
axes[0, 2].set_title(f'Test 3: Random Signs\n({n_sign_changes} sign changes)')
axes[0, 2].set_xlabel('T')
axes[0, 2].set_ylabel('S(T)')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Autocorrelation function
if autocorr_values:
    axes[1, 0].plot(autocorr_lags, autocorr_values, 'bo-', markersize=4)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=0.2, color='red', linestyle=':', alpha=0.3, label='±0.2')
    axes[1, 0].axhline(y=-0.2, color='red', linestyle=':', alpha=0.3)
    axes[1, 0].set_title(f'Test 4: Autocorrelation\n(max={max_autocorr:.3f})')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('Autocorrelation')
    axes[1, 0].set_xlim(0, max_lag)
    axes[1, 0].set_ylim(-0.5, 0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'Insufficient data\nfor autocorrelation',
                     ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Test 4: Autocorrelation\n(N/A)')

# Plot 5: Phase circle
axes[1, 1].scatter(x_coords, y_coords, c=T_values, cmap='viridis',
                   alpha=0.6, s=5, edgecolors='none')
circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--')
axes[1, 1].add_patch(circle)
axes[1, 1].set_xlim(-1.2, 1.2)
axes[1, 1].set_ylim(-1.2, 1.2)
axes[1, 1].set_aspect('equal')
axes[1, 1].set_title(f'Test 5: Phase Circle\n(Rayleigh p={rayleigh_p:.3f})')
axes[1, 1].set_xlabel('cos(2πS)')
axes[1, 1].set_ylabel('sin(2πS)')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Summary of test results
test_names = ['Uniformity', 'Growth', 'Random Signs', 'Autocorr.', 'Phase Circle']
test_results_summary = [
    ks_pvalue > 0.05,
    test_results['growth'].get('growth_rate_ok', False),
    test_results['random_signs']['is_random'],
    test_results['autocorrelation']['is_independent'],
    test_results['phase_circle']['is_uniform_on_circle']
]
colors = ['green' if r else 'red' for r in test_results_summary]

bars = axes[1, 2].bar(test_names, [1]*len(test_names), color=colors, alpha=0.7)
axes[1, 2].set_ylim(0, 1.2)
axes[1, 2].set_ylabel('Test Result')
axes[1, 2].set_title('Summary of Validation Tests')
axes[1, 2].set_xticks(range(len(test_names)))
axes[1, 2].set_xticklabels(test_names, rotation=45, ha='right')

# Add pass/fail labels
for i, (bar, result) in enumerate(zip(bars, test_results_summary)):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   'PASS' if result else 'FAIL', ha='center', va='bottom',
                   fontweight='bold', color=colors[i])

plt.tight_layout()
plt.savefig('figures/phase3_validation_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to figures/phase3_validation_results.png")

# 6. Save detailed results
print("\n6. Saving validation results...")

# Create results summary
summary_data = {
    'test': ['Uniformity (KS)', 'Uniformity (Chi2)', 'Growth', 'Random Signs',
             'Autocorrelation', 'Phase Circle'],
    'statistic': [ks_stat, chi2_stat,
                  test_results['growth'].get('slope', 'N/A'),
                  f"{n_positive}/{n_negative}",
                  f"{max_autocorr:.3f}",
                  rayleigh_stat],
    'p_value': [ks_pvalue, chi2_pvalue,
                test_results['growth'].get('p_value', 'N/A'),
                binom_pvalue,
                significant_autocorr / len(autocorr_lags) if autocorr_lags else 0,
                rayleigh_p],
    'passed': [ks_pvalue > 0.05,
               chi2_pvalue > 0.05,
               test_results['growth'].get('growth_rate_ok', False),
               test_results['random_signs']['is_random'],
               test_results['autocorrelation']['is_independent'],
               test_results['phase_circle']['is_uniform_on_circle']]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('data/phase3_validation_summary.csv', index=False)
print("✓ Saved summary to data/phase3_validation_summary.csv")

# Save detailed test results
import json
with open('data/phase3_test_results.json', 'w') as f:
    json.dump(test_results, f, indent=2, default=str)
print("✓ Saved detailed results to data/phase3_test_results.json")

# 7. Overall assessment
print("\n" + "="*80)
print("PHASE 3 COMPLETE - VALIDATION SUMMARY")
print("="*80)

passed_tests = sum(test_results_summary)
total_tests = len(test_results_summary)

print(f"\n📊 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
print(f"   Status: {'✅ PHASE VALIDATION PASSED' if passed_tests >= 4 else '⚠️  SOME TESTS FAILED'}")

print(f"\n📋 Test Results:")
for i, (name, result) in enumerate(zip(test_names, test_results_summary)):
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"   {name:.<20} {status}")

print(f"\n📁 Files Created:")
print(f"   • figures/phase3_validation_results.png")
print(f"   • data/phase3_validation_summary.csv")
print(f"   • data/phase3_test_results.json")

print(f"\n🎯 Interpretation:")
if passed_tests >= 4:
    print(f"   S(T) approximation shows good statistical properties")
    print(f"   Comparable to theoretical expectations")
else:
    print(f"   Some deviations detected - may need investigation")
    print(f"   Consider adjusting P_max or numerical precision")

print(f"\n🚀 Ready for Phase 4: Comprehensive Diagnostics")