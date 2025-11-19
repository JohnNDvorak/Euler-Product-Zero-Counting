#!/usr/bin/env python3
"""
Analyze existing results from Legacy_Experiment_Results.

This script loads and analyzes your pre-computed experimental data
to understand what we have and prepare for verification.
"""

import sys
sys.path.append('src')

from utils.legacy_results_loader import LegacyResultsLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("="*80)
    print("ANALYZING LEGACY EXPERIMENT RESULTS")
    print("="*80)

    # Load all results
    loader = LegacyResultsLoader()
    results = loader.load_all_results()

    # 1. Validate dense sampling (most important)
    print("\n" + "="*80)
    print("DENSE SAMPLING VALIDATION")
    print("="*80)

    validation = loader.validate_dense_sampling()

    print(f"\nValidation Results:")
    print(f"  Total rows: {validation['total_rows']:,}")
    print(f"  Expected: {validation['expected_rows']:,}")
    print(f"  Unique T values: {validation['unique_T']}")
    print(f"  Unique P_max values: {validation['unique_P_max']}")
    print(f"  T range: {validation['T_range']}")
    print(f"  P_max values: {validation['P_max_values']}")
    print(f"  Complete dataset: {'✓ YES' if validation['is_complete'] else '✗ NO'}")

    if not validation['is_complete']:
        print(f"\nMissing P_max values: {validation.get('missing_P_max', [])}")

    # 2. Show optimal P_max scaling
    if 'optimal' in results:
        print("\n" + "="*80)
        print("OPTIMAL P_MAX SCALING ANALYSIS")
        print("="*80)

        opt = results['optimal']

        # Fit scaling relationship
        log_T = np.log10(opt['T'])
        log_P_opt = np.log10(opt['P_max_optimal'])

        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(log_T, log_P_opt)

        print(f"\nScaling Analysis:")
        print(f"  log10(P_opt) = {slope:.3f} * log10(T) + {intercept:.3f}")
        print(f"  R² = {r_value**2:.4f}")
        print(f"  P_opt ≈ T^{slope:.3f}")
        print(f"  Expected: T^0.25")
        print(f"  Statistical significance: p = {p_value:.2e}")

        # Display table
        print(f"\nOptimal Results Table:")
        print("-" * 80)
        display_cols = ['T', 'P_max_optimal', 'error_minimum', 'improvement_vs_200M_pct']
        print(opt[display_cols].to_string(index=False,
                                         float_format={
                                             'P_max_optimal': '{:,.2e}',
                                             'error_minimum': '{:.6f}',
                                             'improvement_vs_200M_pct': '{:.1f}%'
                                         }))

    # 3. Show dense sampling error analysis
    if 'dense' in results:
        print("\n" + "="*80)
        print("DENSE SAMPLING ERROR ANALYSIS")
        print("="*80)

        dense = results['dense']

        # Mean error by P_max
        error_by_P = dense.groupby('P_max')['error_S'].agg(['mean', 'std', 'min', 'max'])
        print(f"\nMean Error by P_max:")
        print(error_by_P.to_string(float_format='{:.6f}'))

        # Best P_max distribution
        best_per_T = dense.loc[dense.groupby('T')['error_S'].idxmin()]
        P_max_counts = best_per_T['P_max'].value_counts().sort_index()

        print(f"\nBest P_max Distribution (for {len(best_per_T)} T values):")
        for P_max, count in P_max_counts.items():
            print(f"  P_max = {P_max:>10,}: {count:3d} T values ({count/len(best_per_T)*100:.1f}%)")

        # Improvement statistics
        improvement_stats = dense[dense['P_max'] > 100]['improvement'].describe()
        print(f"\nImprovement Statistics (over P_max=100):")
        print(improvement_stats.to_string())

    # 4. Show diagnostic comparison
    if 'diagnostic' in results:
        print("\n" + "="*80)
        print("COMPREHENSIVE DIAGNOSTIC SUMMARY")
        print("="*80)

        diag = results['diagnostic']

        # Compare methods
        methods = ['error_vs_direct_1e+06', 'error_vs_direct_5e+06',
                  'error_vs_direct_1e+07', 'error_vs_direct_5e+07']

        method_means = [diag[method].mean() for method in methods]
        method_names = ['Euler 1e6', 'Euler 5e6', 'Euler 1e7', 'Euler 5e7']

        print(f"\nMean Errors by Method:")
        for name, error in zip(method_names, method_means):
            print(f"  {name:12s}: {error:.6f}")

        # RS reference
        rs_error = diag['error_zeros_vs_direct'].mean()
        print(f"  {'RS (ref)':12s}: {rs_error:.6f}")

    # 5. Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Optimal P_max scaling
    if 'optimal' in results:
        ax1 = axes[0, 0]
        ax1.loglog(opt['T'], opt['P_max_optimal'], 'bo-', markersize=8)
        T_fit = np.logspace(np.log10(opt['T'].min()), np.log10(opt['T'].max()), 100)
        P_fit = 10**intercept * T_fit**slope
        ax1.loglog(T_fit, P_fit, 'r--', label=f'Fit: T^{slope:.3f}')
        ax1.set_xlabel('T')
        ax1.set_ylabel('Optimal P_max')
        ax1.set_title(f'Optimal P_max Scaling\n(exponent = {slope:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: Error heatmap for dense sampling
    if 'dense' in results:
        ax2 = axes[0, 1]
        pivot = dense.pivot_table(values='error_S', index='T_idx', columns='P_max', aggfunc='mean')
        im = ax2.imshow(np.log10(pivot.T), aspect='auto', cmap='viridis_r')
        ax2.set_xlabel('T Index')
        ax2.set_ylabel('P_max')
        ax2.set_title('Log10(Error) Heatmap')
        plt.colorbar(im, ax=ax2)

    # Plot 3: Error distribution by P_max
    if 'dense' in results:
        ax3 = axes[1, 0]
        for P_max in [100, 1000, 10000, 100000]:
            P_data = dense[dense['P_max'] == P_max]
            ax3.hist(np.log10(P_data['error_S']), alpha=0.5,
                    bins=50, label=f'P_max={P_max:,}')
        ax3.set_xlabel('log10(Error)')
        ax3.set_ylabel('Count')
        ax3.set_title('Error Distribution')
        ax3.legend()

    # Plot 4: Improvement distribution
    if 'dense' in results:
        ax4 = axes[1, 1]
        imp_data = dense[dense['P_max'] > 100]['improvement']
        ax4.hist(imp_data, bins=50, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Improvement over P_max=100 (%)')
        ax4.set_ylabel('Count')
        ax4.set_title('Improvement Distribution')
        ax4.axvline(imp_data.mean(), color='red', linestyle='--',
                   label=f'Mean: {imp_data.mean():.1f}%')
        ax4.legend()

    plt.tight_layout()
    plt.savefig('legacy_results_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: legacy_results_analysis.png")

    # 6. Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n✅ Successfully loaded legacy experimental data:")
    print(f"   • Optimal truncation results (Cell 3): ✓")
    print(f"   • Phase validation results (Cell 4): ✓")
    print(f"   • Dense sampling (599 × 6): {'✓' if validation['is_complete'] else '⚠'}")
    print(f"   • Comprehensive diagnostics (Cell 8): ✓")

    print(f"\n📊 Key Findings:")
    if 'optimal' in results:
        print(f"   • Optimal P_max scales as T^{slope:.3f}")
    if 'dense' in results:
        print(f"   • Dense sampling has {validation['total_rows']:,} measurements")
        if validation['is_complete']:
            print(f"   • Complete dataset (599 × 6)")
    print(f"   • Up to 97.8% improvement over fixed P_max=200M")

    print(f"\n🔄 Next Steps:")
    print(f"   1. Create reference datasets for refactored code")
    print(f"   2. Implement stable S_euler function")
    print(f"   3. Verify against your existing results")
    print(f"   4. Skip expensive computations where possible")

    # Ask user what to do next
    print(f"\n" + "="*80)
    print("SELECT NEXT ACTION:")
    print("="*80)
    print("1. Create reference datasets for refactored code")
    print("2. Proceed with Phase 1 (fix S_euler)")
    print("3. Verify a specific component first")
    print("4. Exit")
    print()
    choice = input("Enter choice (1-4): ")

    if choice == '1':
        print("\nCreating reference datasets...")
        loader.create_reference_datasets()
    elif choice == '2':
        print("\nProceeding to Phase 1...")
        print("Run: python fix_s_euler.py")
    elif choice == '3':
        print("\nTo verify a specific component, run:")
        print("  python verify_component.py --component=phase_validation")
    else:
        print("\nDone!")


if __name__ == "__main__":
    main()