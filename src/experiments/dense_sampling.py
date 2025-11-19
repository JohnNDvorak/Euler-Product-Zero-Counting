"""
Dense T Sampling Experiment - Cell 16 Equivalent

This implements the comprehensive experiment with 599 T values
and 6 P_max values as in the original notebook.
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from ..core.s_t_functions import S_RS, S_euler
from ..utils.paths import PathConfig


class DenseSamplingExperiment:
    """
    Implements dense T sampling with 6 P_max values.
    This matches Cell 16 from the original notebook.
    """

    def __init__(self):
        self.paths = PathConfig()
        self.P_MIN_TEST = [100, 1000, 10000, 100000, 1000000, 10000000]
        self.N_T_SAMPLES = 599
        self.results = None

    def generate_T_values(self) -> np.ndarray:
        """
        Generate or load T values for dense sampling.
        Tries to load from existing results, otherwise generates log-spaced values.
        """
        # Try to load T values from existing optimal results
        optimal_file = self.paths.results_dir / 'exp1_optimal_summary_k1.csv'

        if optimal_file.exists():
            print("Loading T values from existing results...")
            df = pd.read_csv(optimal_file)

            if len(df) >= self.N_T_SAMPLES:
                # Sample evenly across the range (log-spaced indices)
                indices = np.round(np.logspace(0, np.log10(len(df)-1), self.N_T_SAMPLES)).astype(int)
                T_values = df['T'].iloc[indices].values
            else:
                # Not enough values, generate new ones
                T_values = np.logspace(3, 6, self.N_T_SAMPLES)
        else:
            # Generate T values from 1K to 1M
            T_values = np.logspace(3, 6, self.N_T_SAMPLES)

        print(f"Selected {len(T_values)} T values")
        print(f"Range: [{T_values.min():.1f}, {T_values.max():.2e}]")

        return T_values

    def run_experiment(self, zeros: np.ndarray, primes: np.ndarray,
                       load_existing: bool = True) -> pd.DataFrame:
        """
        Run the dense sampling experiment.

        Parameters:
        -----------
        zeros : np.ndarray
            Riemann zeros
        primes : np.ndarray
            Prime numbers up to required maximum
        load_existing : bool
            Whether to load existing cached results
        """
        # Generate T values
        T_values = self.generate_T_values()

        # Check for existing results
        cache_file = self.paths.results_dir / 'dense_sampling_results.pkl'

        if load_existing and cache_file.exists():
            print(f"Loading existing results from {cache_file}")
            self.results = pd.read_pickle(cache_file)
            return self.results

        print("\nStarting dense sampling experiment...")
        print(f"Total computations: {len(T_values)} T × {len(self.P_MIN_TEST)} P_max = {len(T_values) * len(self.P_MIN_TEST):,}")
        print(f"6 P_max values: {[f'{p:,}' for p in self.P_MIN_TEST]}")
        print()

        # Initialize results storage
        results_data = []
        total_time = 0
        rate = 0

        # Run experiment
        for t_idx, T in enumerate(T_values):
            start_t = time.time()

            # Print progress
            if t_idx > 0:
                rate = 1 / (time.time() - start_t + 1e-6)
                remaining = (len(T_values) - t_idx) / rate if rate > 0 else 0
                print(f"Progress: {t_idx}/{len(T_values)} ({rate:.0f} T/s, {remaining:.0f}s remaining)")

            # Compute reference value using Riemann-Siegel
            S_ref = S_RS(T, zeros)

            # Test each P_max value
            for P_max in self.P_MIN_TEST:
                p_start = time.time()

                # Compute S_euler
                S_e = S_euler(T, P_max, primes, k_max=1)  # k=1 for now

                # Compute error and improvement
                error = abs(S_e - S_ref)

                # For improvement calculation, compare to P_max=100
                if P_max == 100:
                    baseline_error = error
                    improvement = 0.0
                else:
                    improvement = (baseline_error - error) / baseline_error * 100 if baseline_error > 0 else 0

                p_time = time.time() - p_start

                # Store results
                results_data.append({
                    'T_idx': t_idx,
                    'T': T,
                    'P_max': P_max,
                    'S_ref': S_ref,
                    'S_euler': S_e,
                    'error': error,
                    'improvement': improvement,
                    'computation_time': p_time,
                    'log10_T': np.log10(T),
                    'log10_P': np.log10(P_max)
                })

            total_time += time.time() - start_t

        # Convert to DataFrame
        self.results = pd.DataFrame(results_data)

        # Save results
        self.results.to_pickle(cache_file)
        self.results.to_csv(self.paths.results_dir / 'dense_sampling_results.csv', index=False)

        print(f"\n✓ Experiment completed in {total_time:.1f}s")
        print(f"Average rate: {len(T_values)/total_time:.1f} T/s")
        print(f"Results saved to: {cache_file}")

        return self.results

    def analyze_results(self) -> Dict:
        """
        Analyze the dense sampling results.
        Returns summary statistics.
        """
        if self.results is None:
            raise ValueError("Run experiment first")

        print("\n" + "="*80)
        print("DENSE SAMPLING ANALYSIS")
        print("="*80)

        # Summary by P_max
        summary_by_pmax = self.results.groupby('P_max').agg({
            'error': ['mean', 'std', 'min', 'max'],
            'improvement': 'mean',
            'computation_time': 'mean'
        }).round(6)

        print("\nSummary by P_max:")
        print(summary_by_pmax)

        # Best P_max for each T
        best_per_T = self.results.loc[self.results.groupby('T_idx')['error'].idxmin()]

        print(f"\nOptimal P_max distribution:")
        pmax_counts = best_per_T['P_max'].value_counts().sort_index()
        for p_max, count in pmax_counts.items():
            print(f"  P_max = {p_max:>10,}: {count:3d} T values ({count/len(best_per_T)*100:.1f}%)")

        # Performance metrics
        print(f"\nPerformance Metrics:")
        print(f"  Total computations: {len(self.results):,}")
        print(f"  Mean error: {self.results['error'].mean():.6f}")
        print(f"  Best mean error (P_max={self.results.groupby('P_max')['error'].mean().idxmin():}): {self.results.groupby('P_max')['error'].mean().min():.6f}")
        print(f"  Mean improvement over P_max=100: {self.results[self.results['P_max'] > 100]['improvement'].mean():.2f}%")

        return {
            'summary_by_pmax': summary_by_pmax,
            'best_per_T': best_per_T,
            'pmax_distribution': pmax_counts
        }

    def create_visualization(self, save_path: Path = None):
        """
        Create comprehensive visualization of results.
        """
        if self.results is None:
            raise ValueError("Run experiment first")

        import matplotlib.pyplot as plt

        if save_path is None:
            save_path = self.paths.figures_dir / 'dense_sampling_analysis.png'

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dense Sampling Analysis (599 T × 6 P_max)', fontsize=16)

        # Plot 1: Error heatmap
        pivot_error = self.results.pivot(index='T_idx', columns='P_max', values='error')
        im1 = axes[0, 0].imshow(np.log10(pivot_error.T), aspect='auto', cmap='viridis_r')
        axes[0, 0].set_title('Log10(Error) Heatmap')
        axes[0, 0].set_xlabel('T index')
        axes[0, 0].set_ylabel('P_max')
        axes[0, 0].set_yticks(range(len(self.P_MIN_TEST)))
        axes[0, 0].set_yticklabels([f'{p:,}' for p in self.P_MIN_TEST])
        plt.colorbar(im1, ax=axes[0, 0])

        # Plot 2: Mean error by P_max
        mean_error = self.results.groupby('P_max')['error'].mean()
        axes[0, 1].loglog(self.P_MIN_TEST, mean_error.values, 'o-', linewidth=2)
        axes[0, 1].set_xlabel('P_max')
        axes[0, 1].set_ylabel('Mean Error')
        axes[0, 1].set_title('Mean Error vs P_max')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Improvement distribution
        improvement_data = []
        for p_max in self.P_MIN_TEST[1:]:  # Skip P_max=100 (baseline)
            imp = self.results[self.results['P_max'] == p_max]['improvement']
            improvement_data.append(imp.values)

        axes[0, 2].boxplot(improvement_data, labels=[f'{p:,}' for p in self.P_MIN_TEST[1:]])
        axes[0, 2].set_xlabel('P_max')
        axes[0, 2].set_ylabel('Improvement (%)')
        axes[0, 2].set_title('Improvement over Baseline')
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Optimal P_max vs T
        best_per_T = self.results.loc[self.results.groupby('T_idx')['error'].idxmin()]
        axes[1, 0].scatter(np.log10(best_per_T['T']), best_per_T['P_max'], alpha=0.5, s=5)
        axes[1, 0].set_xlabel('log10(T)')
        axes[1, 0].set_ylabel('Optimal P_max')
        axes[1, 0].set_title('Optimal P_max vs T')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Error distribution
        for p_max in self.P_MIN_TEST[::2]:  # Show every other P_max
            errors = self.results[self.results['P_max'] == p_max]['error']
            axes[1, 1].hist(np.log10(errors), alpha=0.5, bins=50,
                           label=f'P_max={p_max:,}')
        axes[1, 1].set_xlabel('log10(Error)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Computation time
        mean_time = self.results.groupby('P_max')['computation_time'].mean()
        axes[1, 2].loglog(self.P_MIN_TEST, mean_time.values, 's-', linewidth=2)
        axes[1, 2].set_xlabel('P_max')
        axes[1, 2].set_ylabel('Mean Time (s)')
        axes[1, 2].set_title('Computation Time vs P_max')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

        return fig


def main():
    """Run the dense sampling experiment."""
    import sys
    sys.path.append('src')

    from src.core.prime_cache import simple_sieve

    print("Initializing Dense Sampling Experiment...")

    # Load data
    zeros = np.load(PathConfig().cache_dir / "zeros.npy")
    primes = np.array(simple_sieve(10_000_000))  # Need up to 10M for P_max=10M

    # Run experiment
    exp = DenseSamplingExperiment()
    results = exp.run_experiment(zeros, primes)

    # Analyze
    exp.analyze_results()

    # Visualize
    exp.create_visualization()

    return results


if __name__ == "__main__":
    main()