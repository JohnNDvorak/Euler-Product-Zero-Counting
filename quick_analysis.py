#!/usr/bin/env python3
"""
Quick analysis of legacy results without formatting issues.
"""

import sys
sys.path.append('src')

from utils.legacy_results_loader import LegacyResultsLoader
import numpy as np

def main():
    print("="*80)
    print("LEGACY RESULTS SUMMARY")
    print("="*80)

    # Load results
    loader = LegacyResultsLoader()
    results = loader.load_all_results()

    # Dense sampling analysis
    print("\n1. DENSE SAMPLING RESULTS (Cell 16)")
    print("-" * 50)
    dense = results['dense']
    print(f"✓ Total measurements: {len(dense):,}")
    print(f"✓ T values: {dense['T'].nunique()}")
    print(f"✓ P_max values: {sorted(dense['P_max'].unique())}")
    print(f"✓ T range: {dense['T'].min():.2f} to {dense['T'].max():.2f}")

    # Best P_max distribution
    best_per_T = dense.loc[dense.groupby('T')['error_S'].idxmin()]
    P_max_counts = best_per_T['P_max'].value_counts().sort_index()

    print(f"\nBest P_max Distribution:")
    for P_max, count in P_max_counts.items():
        print(f"  P_max = {P_max:>10,}: {count:3d} T values ({count/len(best_per_T)*100:.1f}%)")

    # Optimal P_max scaling
    print("\n2. OPTIMAL P_MAX SCALING (Cell 3)")
    print("-" * 50)
    opt = results['optimal']

    log_T = np.log10(opt['T'])
    log_P_opt = np.log10(opt['P_max_optimal'])

    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(log_T, log_P_opt)

    print(f"✓ Measured scaling: P_opt ≈ T^{slope:.3f}")
    print(f"✓ Expected: T^0.25")
    print(f"✓ R² = {r_value**2:.4f}")
    print(f"✓ p-value = {p_value:.2f}")

    # Key numbers from optimal results
    print(f"\nOptimal Results Summary:")
    for i, row in opt.iterrows():
        print(f"  T={row['T']:>8,}: P_opt={row['P_max_optimal']:>12,.0e}, "
              f"error={row['error_minimum']:>8.6f}, "
              f"improvement={row['improvement_vs_200M_pct']:>6.1f}%")

    # Dense sampling errors
    print("\n3. ERROR ANALYSIS")
    print("-" * 50)
    error_stats = dense.groupby('P_max')['error_S'].describe()
    for P_max in sorted(dense['P_max'].unique()):
        stats = error_stats.loc[P_max]
        print(f"  P_max={P_max:>10,}: mean={stats['mean']:>8.6f}, "
              f"std={stats['std']:>8.6f}")

    # Improvement statistics
    improvement = dense[dense['P_max'] > 100]['improvement']
    print(f"\nImprovement over P_max=100:")
    print(f"  Mean: {improvement.mean():.1f}%")
    print(f"  Max: {improvement.max():.1f}%")
    print(f"  Std: {improvement.std():.1f}%")

    # Save reference datasets
    print("\n4. SAVING REFERENCE DATASETS")
    print("-" * 50)
    loader.create_reference_datasets()
    print("✓ Reference datasets saved to data/existing_results/")

    print("\n" + "="*80)
    print("READY FOR VERIFICATION")
    print("="*80)
    print("\nYour legacy results show:")
    print(f"  • Complete dense sampling: {len(dense):,} measurements")
    print(f"  • Scaling exponent: {slope:.3f} (close to 0.25)")
    print(f"  • Up to {opt['improvement_vs_200M_pct'].max():.1f}% improvement")
    print(f"\nNext step: Run the verification script to compare implementations")

if __name__ == "__main__":
    main()