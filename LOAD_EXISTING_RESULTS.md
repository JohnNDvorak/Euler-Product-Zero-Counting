# Loading and Utilizing Existing Results

This guide explains how to load and use your pre-computed results to accelerate development and verification.

## Types of Existing Results

### 1. Results in Notebook Outputs
From your analysis, these include:
- **Cell 2**: Prime cache (50M primes loaded)
- **Cell 3**: Optimal truncation results for T = [1K, 10K, 100K, 1M, 10M]
- **Cell 4**: Phase validation statistics
- **Cell 5**: Method comparison data
- **Cell 8**: Diagnostic results at 4 P_max values
- **Cell 16**: Dense sampling results (some subset)

### 2. Pre-computed Dense Sampling Data
You mentioned you have the intensive calculations for each T at each P level. This is crucial - it's the **3,594 measurements** (599 T × 6 P_max).

## How to Provide Your Results

### Option 1: Copy from Notebook Outputs
```python
# Example for optimal truncation results
optimal_results = {
    1000: {'optimal_P_max': 2.3e6, 'min_error': 0.0412, 'S_ref': -0.401194},
    10000: {'optimal_P_max': 8.7e6, 'min_error': 0.0321, 'S_ref': 0.491911},
    # ... etc
}
```

### Option 2: Export from Notebook
```python
# In your original notebook, run:
import json

# Save optimal results
optimal_df.to_json('optimal_results.json')

# Save dense sampling results
dense_df.to_pickle('dense_sampling_results.pkl')
```

### Option 3: Provide Raw Numbers
If easier, just provide the key numbers in a structured format:
```
Dense Sampling Results (599 T × 6 P_max):
T=1000: P_max=100: error=0.452, P_max=1000: error=0.231, ...
T=1001: P_max=100: error=0.448, P_max=1000: error=0.229, ...
...
```

## Implementation: Results Loader

Let me create a results loader module:

### File: `src/utils/results_loader.py`

```python
"""
Utility to load and validate existing results from the original notebook.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional

class ResultsLoader:
    """Load and manage pre-computed results from original experiments."""

    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or Path("data/existing_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_optimal_results(self) -> pd.DataFrame:
        """
        Load optimal P_max results from Cell 3.

        Expected format:
        T, optimal_P_max, min_error, S_ref
        """
        file_path = self.results_dir / "optimal_results.csv"

        if file_path.exists():
            return pd.read_csv(file_path)

        # Create template with your actual values
        template = {
            'T': [1000, 10000, 100000, 1000000, 10000000],
            'optimal_P_max': [2.3e6, 8.7e6, 3.2e7, 1.1e8, 3.4e8],
            'min_error': [0.0412, 0.0321, 0.0287, 0.0254, 0.0231],
            'S_ref': [-0.401194, 0.491911, -0.487058, 0.492184, -0.498723]
        }

        df = pd.DataFrame(template)
        print(f"Created template for optimal results. Please update with actual values.")
        return df

    def load_dense_sampling_results(self) -> pd.DataFrame:
        """
        Load the complete dense sampling results (599 × 6).

        Expected columns:
        T_idx, T, P_max, S_ref, S_euler, error, improvement, computation_time
        """
        file_path = self.results_dir / "dense_sampling_results.csv"

        if file_path.exists():
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} dense sampling results")

            # Verify we have complete data
            expected = 599 * 6  # 3,594
            if len(df) == expected:
                print("✓ Complete dense sampling results loaded")
            else:
                print(f"⚠ Incomplete: {len(df)}/{expected} results")

            return df

        # Check for pickle version
        pkl_path = self.results_dir / "dense_sampling_results.pkl"
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            # Convert to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data

            print(f"Loaded dense sampling from pickle: {len(df)} rows")
            return df

        print("❌ No dense sampling results found")
        return None

    def load_phase_validation_results(self) -> Dict:
        """Load phase validation results from Cell 4."""
        file_path = self.results_dir / "phase_validation.pkl"

        if file_path.exists():
            with open(file_path, 'rb') as f:
                return pickle.load(f)

        # Create template structure
        template = {
            'test1_uniformity': {
                1000: {'ks_stat': 0.0019, 'ks_pvalue': 0.8816},
                10000: {'ks_stat': 0.0026, 'ks_pvalue': 0.5061},
                # Add more as needed
            },
            'test2_growth': {
                'slope': 0.487,
                'r2': 0.923,
                # Add more as needed
            }
        }

        print(f"Created template for phase results. Please update with actual values.")
        return template

    def save_template_files(self):
        """Create template files with your expected structure."""

        # 1. Optimal results template
        optimal_file = self.results_dir / "optimal_results.csv"
        if not optimal_file.exists():
            optimal_df = self.load_optimal_results()
            optimal_df.to_csv(optimal_file, index=False)
            print(f"Created: {optimal_file}")

        # 2. Dense sampling template
        dense_file = self.results_dir / "dense_sampling_template.csv"
        if not dense_file.exists():
            # Create first few rows as example
            template_data = []
            T_idx = 0

            for T in [1000, 1001, 1002]:  # Example T values
                for P_max in [100, 1000, 10000, 100000, 1000000, 10000000]:
                    template_data.append({
                        'T_idx': T_idx,
                        'T': T,
                        'P_max': P_max,
                        'S_ref': 0.0,  # To be filled
                        'S_euler': 0.0,  # To be filled
                        'error': 0.0,  # To be filled
                        'improvement': 0.0,
                        'computation_time': 0.0
                    })
                T_idx += 1

            df = pd.DataFrame(template_data)
            df.to_csv(dense_file, index=False)
            print(f"Created: {dense_file}")

        print("\nTemplate files created. Please fill with your actual results.")

    def validate_results(self, dense_df: pd.DataFrame) -> Dict:
        """Validate that results match expected format."""
        validation = {
            'total_rows': len(dense_df),
            'unique_T': dense_df['T'].nunique(),
            'unique_P_max': dense_df['P_max'].nunique(),
            'expected_rows': 599 * 6,
            'has_all_columns': all(col in dense_df.columns for col in
                               ['T_idx', 'T', 'P_max', 'S_ref', 'S_euler', 'error', 'improvement']),
            'valid_error': (dense_df['error'] >= 0).all(),
            'valid_improvement': ((dense_df['P_max'] == 100) |
                                  (dense_df['improvement'] >= 0)).all()
        }

        validation['is_complete'] = (
            validation['total_rows'] == validation['expected_rows'] and
            validation['unique_T'] == 599 and
            validation['unique_P_max'] == 6 and
            validation['has_all_columns'] and
            validation['valid_error'] and
            validation['valid_improvement']
        )

        return validation
```

### File: `scripts/load_existing_results.py`

```python
#!/usr/bin/env python3
"""
Script to load and validate existing results from original notebook.
"""

import sys
sys.path.append('src')

from utils.results_loader import ResultsLoader
import pandas as pd

def main():
    loader = ResultsLoader()

    print("="*60)
    print("LOADING EXISTING RESULTS")
    print("="*60)

    # 1. Load optimal results
    print("\n1. Optimal Truncation Results (Cell 3):")
    optimal_df = loader.load_optimal_results()
    print(optimal_df.to_string())

    # 2. Load dense sampling results
    print("\n2. Dense Sampling Results (599 × 6):")
    dense_df = loader.load_dense_sampling_results()

    if dense_df is not None:
        # Validate
        validation = loader.validate_results(dense_df)
        print("\nValidation Results:")
        for key, value in validation.items():
            print(f"  {key}: {value}")

        # Show sample
        print("\nSample Results:")
        print(dense_df.head(10))

    # 3. Load phase validation
    print("\n3. Phase Validation Results:")
    phase_results = loader.load_phase_validation_results()
    if phase_results:
        print("Available phase tests:", list(phase_results.keys()))

    # 4. Create templates if needed
    print("\n4. Creating template files...")
    loader.save_template_files()

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Fill template files with your actual results")
    print("2. Or export results from your original notebook")
    print("3. Run the verification script")

if __name__ == "__main__":
    main()
```

## How to Proceed

### Option A: Export from Your Original Notebook

Run these cells in your original notebook to export data:

```python
# 1. Export optimal results
optimal_df = pd.DataFrame({
    'T': T_TEST,
    'optimal_P_max': [r['optimal_P_max'] for r in optimal_results_k1],
    'min_error': [r['optimal_error'] for r in optimal_results_k1],
    'S_ref': [r['S_ref'] for r in optimal_results_k1]
})
optimal_df.to_csv('/path/to/refactored/data/existing_results/optimal_results.csv', index=False)

# 2. Export dense sampling if available
if 'dense_results_df' in locals():
    dense_results_df.to_csv('/path/to/refactored/data/existing_results/dense_sampling_results.csv', index=False)
    dense_results_df.to_pickle('/path/to/refactored/data/existing_results/dense_sampling_results.pkl')

# 3. Export phase validation
with open('/path/to/refactored/data/existing_results/phase_validation.pkl', 'wb') as f:
    pickle.dump(all_phase_results, f)
```

### Option B: Provide Key Numbers Directly

Just provide me the important results in a structured format, and I'll create the loader files.

### What Results Would Help Most?

1. **Dense Sampling Results** - The full 599 × 6 matrix
2. **Optimal P_max Values** - For each T value tested
3. **Phase Validation Statistics** - KS test p-values, growth rate slope
4. **Method Comparison Data** - Errors and timings

## Benefits of Using Existing Results

1. **Skip 10+ hours of computation** for dense sampling
2. **Verify our implementations** against known good values
3. **Accelerate development** by focusing on code quality
4. **Ensure 100% accuracy** by direct comparison

## Verification Script

Once you provide results, I'll create a verification script that:
```python
# Loads your results
# Runs our implementation on the same parameters
# Compares outputs
# Reports any discrepancies
```

Would you like to:
1. **Export results** from your notebook using the code above?
2. **Provide specific values** for the most important experiments?
3. **Start with Phase 1** and add verification tests later?