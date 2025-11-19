"""
Load and utilize existing results from Legacy_Experiment_Results folder.

This module provides access to your pre-computed experimental data,
accelerating development and verification.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class LegacyResultsLoader:
    """
    Load existing experimental results from original notebook.
    """

    def __init__(self, legacy_dir: Optional[Path] = None):
        self.legacy_dir = legacy_dir or Path("Legacy_Experiment_Results")
        self.results = {}

    def load_all_results(self) -> Dict:
        """Load all available experimental results."""
        print("Loading legacy experimental results...")
        print("="*60)

        # 1. Load optimal truncation results (Cell 3)
        print("\n1. Loading optimal truncation results (Cell 3)...")
        try:
            opt_file = self.legacy_dir / "exp1_optimal_summary_k1.csv"
            if opt_file.exists():
                self.results['optimal'] = pd.read_csv(opt_file)
                print(f"   ✓ Loaded {len(self.results['optimal'])} optimal P_max results")
                print(f"     T values: {self.results['optimal']['T'].tolist()}")
                print(f"     Columns: {list(self.results['optimal'].columns)}")
            else:
                print("   ✗ Not found")
        except Exception as e:
            print(f"   ✗ Error: {e}")

        # 2. Load phase validation results (Cell 4)
        print("\n2. Loading phase validation results (Cell 4)...")
        try:
            phase_file = self.legacy_dir / "exp2_phase_results_enhanced.pkl"
            if phase_file.exists():
                with open(phase_file, 'rb') as f:
                    self.results['phase'] = pickle.load(f)

                # Also load summary
                summary_file = self.legacy_dir / "exp2_phase_summary_enhanced.csv"
                if summary_file.exists():
                    self.results['phase_summary'] = pd.read_csv(summary_file)

                print(f"   ✓ Loaded phase validation results")
                print(f"     Tests: {list(self.results['phase'].keys())}")
            else:
                print("   ✗ Not found")
        except Exception as e:
            print(f"   ✗ Error: {e}")

        # 3. Load dense sampling results (Cell 16) - MOST IMPORTANT!
        print("\n3. Loading dense sampling results (Cell 16)...")
        try:
            dense_file = self.legacy_dir / "pmin_dense_search.csv"
            if dense_file.exists():
                self.results['dense'] = pd.read_csv(dense_file)
                print(f"   ✓ Loaded {len(self.results['dense'])} dense sampling results")

                # Analyze the data
                n_T = self.results['dense']['T'].nunique()
                n_P = self.results['dense']['P_max'].nunique()
                print(f"     T values: {n_T}")
                print(f"     P_max values: {sorted(self.results['dense']['P_max'].unique())}")
                print(f"     Total combinations: {n_T} × {n_P} = {n_T * n_P}")

                # Check if we have all expected data
                if n_T == 599 and n_P == 6:
                    print(f"     ✓ Complete dataset (599 × 6 = {n_T * n_P})")
                else:
                    print(f"     ⚠ Expected 599 × 6, got {n_T} × {n_P}")
            else:
                print("   ✗ Not found")
        except Exception as e:
            print(f"   ✗ Error: {e}")

        # 4. Load comprehensive diagnostic results (Cell 8)
        print("\n4. Loading comprehensive diagnostic results (Cell 8)...")
        try:
            diag_file = self.legacy_dir / "comprehensive_diagnostic_4pmax.csv"
            if diag_file.exists():
                self.results['diagnostic'] = pd.read_csv(diag_file)
                print(f"   ✓ Loaded {len(self.results['diagnostic'])} diagnostic results")
                print(f"     Methods compared: RS, Euler at 4 P_max values")
            else:
                print("   ✗ Not found")
        except Exception as e:
            print(f"   ✗ Error: {e}")

        # 5. Load publication data
        print("\n5. Loading publication data...")
        try:
            pub_file = self.legacy_dir / "publication_data.json"
            if pub_file.exists():
                import json
                with open(pub_file, 'r') as f:
                    self.results['publication'] = json.load(f)
                print(f"   ✓ Loaded publication data")
            else:
                print("   ✗ Not found")
        except Exception as e:
            print(f"   ✗ Error: {e}")

        print("\n" + "="*60)
        print("Legacy results loading complete!")
        return self.results

    def get_optimal_results(self) -> Optional[pd.DataFrame]:
        """Get optimal P_max results."""
        return self.results.get('optimal')

    def get_dense_sampling_results(self) -> Optional[pd.DataFrame]:
        """Get dense sampling results (599 × 6)."""
        return self.results.get('dense')

    def get_diagnostic_results(self) -> Optional[pd.DataFrame]:
        """Get comprehensive diagnostic results."""
        return self.results.get('diagnostic')

    def get_phase_validation_results(self) -> Optional[Dict]:
        """Get phase validation results."""
        return self.results.get('phase')

    def validate_dense_sampling(self) -> Dict:
        """Validate dense sampling data completeness."""
        if 'dense' not in self.results:
            return {'error': 'No dense sampling data loaded'}

        df = self.results['dense']

        validation = {
            'total_rows': len(df),
            'unique_T': df['T'].nunique(),
            'unique_P_max': df['P_max'].nunique(),
            'expected_rows': 599 * 6,
            'P_max_values': sorted(df['P_max'].unique()),
            'T_range': (df['T'].min(), df['T'].max()),
            'missing_T_values': [],
            'all_P_max_present': set([100, 1000, 10000, 100000, 1000000, 10000000]).issubset(set(df['P_max'].unique()))
        }

        # Check for expected P_max values
        expected_P = [100, 1000, 10000, 100000, 1000000, 10000000]
        validation['missing_P_max'] = [p for p in expected_P if p not in validation['P_max_values']]

        # Check completeness
        validation['is_complete'] = (
            validation['total_rows'] == validation['expected_rows'] and
            validation['unique_T'] == 599 and
            validation['unique_P_max'] == 6 and
            len(validation['missing_P_max']) == 0
        )

        return validation

    def extract_T_values(self) -> np.ndarray:
        """Extract the 599 T values from dense sampling."""
        if 'dense' not in self.results:
            raise ValueError("No dense sampling data loaded")

        return np.sort(self.results['dense']['T'].unique())

    def create_reference_datasets(self, output_dir: Optional[Path] = None):
        """Create standardized reference datasets for the refactored code."""
        if output_dir is None:
            output_dir = Path("data/existing_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Creating reference datasets...")

        # 1. Optimal results
        if 'optimal' in self.results:
            # Reformat to match expected structure
            optimal_df = self.results['optimal'][['T', 'P_max_optimal', 'error_minimum', 'S_ref']]
            optimal_df.columns = ['T', 'optimal_P_max', 'min_error', 'S_ref']
            optimal_df.to_csv(output_dir / 'optimal_results.csv', index=False)
            print(f"   ✓ Created optimal_results.csv")

        # 2. Dense sampling
        if 'dense' in self.results:
            # Reformat to match expected structure
            dense_df = self.results['dense'].copy()

            # Compute T_idx for consistency
            dense_df['T_idx'] = dense_df.groupby('T').cumcount()
            T_unique = dense_df['T'].unique()
            T_to_idx = {T: i for i, T in enumerate(T_unique)}
            dense_df['T_idx'] = dense_df['T'].map(T_to_idx)

            # Select and reorder columns
            columns_needed = ['T_idx', 'T', 'P_max', 'S_ref', 'S_approx', 'error_S',
                           'improvement']
            dense_formatted = dense_df[columns_needed].copy()
            dense_formatted = dense_formatted.rename(columns={
                'S_approx': 'S_euler',
                'error_S': 'error',
                'improvement': 'improvement'
            })

            # Add computation_time (not in legacy data)
            dense_formatted['computation_time'] = 0.0

            dense_formatted.to_csv(output_dir / 'dense_sampling_results.csv', index=False)
            print(f"   ✓ Created dense_sampling_results.csv ({len(dense_formatted)} rows)")

        # 3. Diagnostic results
        if 'diagnostic' in self.results:
            diag_df = self.results['diagnostic'].copy()
            diag_df.to_csv(output_dir / 'diagnostic_results.csv', index=False)
            print(f"   ✓ Created diagnostic_results.csv")

        print(f"\nAll reference datasets saved to: {output_dir}")

    def summary_statistics(self) -> Dict:
        """Provide summary statistics of loaded results."""
        summary = {}

        if 'optimal' in self.results:
            opt = self.results['optimal']
            summary['optimal'] = {
                'T_values': opt['T'].tolist(),
                'mean_optimal_P_max': opt['P_max_optimal'].mean(),
                'P_max_range': (opt['P_max_optimal'].min(), opt['P_max_optimal'].max()),
                'mean_error': opt['error_minimum'].mean(),
                'improvement_stats': opt['improvement_vs_200M_pct'].describe()
            }

        if 'dense' in self.results:
            dense = self.results['dense']
            summary['dense'] = {
                'n_T': dense['T'].nunique(),
                'n_P_max': dense['P_max'].nunique(),
                'T_range': (dense['T'].min(), dense['T'].max()),
                'mean_error_by_P_max': dense.groupby('P_max')['error_S'].mean().to_dict(),
                'best_P_max_frequency': dense.loc[dense.groupby('T')['error_S'].idxmin(), 'P_max'].value_counts().to_dict()
            }

        if 'diagnostic' in self.results:
            diag = self.results['diagnostic']
            summary['diagnostic'] = {
                'n_T': len(diag),
                'methods_compared': ['RS', 'Euler_1e6', 'Euler_5e6', 'Euler_1e7', 'Euler_5e7'],
                'mean_errors': {
                    'RS': diag['error_vs_direct_1e+06'].mean(),
                    'Euler_1e6': diag['error_vs_direct_5e+07'].mean(),
                    'Euler_5e6': diag['error_vs_direct_1e+07'].mean(),
                    'Euler_1e7': diag['error_vs_direct_5e+07'].mean(),
                    'Euler_5e7': diag['error_vs_direct_5e+07'].mean()
                }
            }

        return summary


# Convenience function for quick loading
def load_legacy_results():
    """Quickly load all legacy results."""
    loader = LegacyResultsLoader()
    return loader.load_all_results()