# Usage Guide for Prime Reduction Estimates

This guide provides step-by-step instructions for using the refactored codebase.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download required data**:
   - Riemann zeros file: Download from Andrew Odlyzko's website
   - Place in `data/raw/zeros.txt`

3. **Run the notebooks** in order:
   - `01_setup_and_functions.ipynb`
   - `02_main_experiments.ipynb`
   - `03_visualization.ipynb` (optional)

## Detailed Instructions

### 1. Data Setup

The repository already includes the required Riemann zeros file:
- File: `10M_zeta_zeros.txt` (first 10 million zeros)
- Location: Repository root directory
- No download needed - file is already provided

### 2. Running Experiments

#### Notebook 1: Setup and Functions
- Loads all dependencies
- Initializes the prime cache (takes 15-20 minutes first time)
- Tests all core functions
- **Must be run first**

#### Notebook 2: Main Experiments
- Experiment 1: Finds optimal P_max for each T
- Experiment 2: Validates phase cancellation
- Experiment 3: Compares computation methods
- Takes ~30-40 minutes on first run

#### Notebook 3: Visualization
- Generates publication-quality figures
- Creates additional diagnostic plots
- Exports results in various formats

### 3. Using the Python API

You can also use the functions directly in Python:

```python
import sys
sys.path.append('src')

from src.core.s_t_functions import S_euler, S_RS
from src.core.prime_cache import PrimeCache
from src.utils.paths import PathConfig

# Initialize
paths = PathConfig()
prime_cache = PrimeCache(max_prime=1_000_000_000)

# Load zeros
import numpy as np
zeros = np.load(paths.cache_dir / 'zeros.npy')

# Compute S(T)
T = 10000
P_max = 1e8

s_euler = S_euler(T, P_max, prime_cache)
s_rs = S_RS(T, zeros)

print(f"S_euler({T}) = {s_euler:.6f}")
print(f"S_RS({T}) = {s_rs:.6f}")
```

### 4. Configuration

Edit `config.yaml` to modify:
- File paths
- Experiment parameters
- Visualization settings
- Computation options

### 5. Understanding the Output

#### Results Location
- Cache files: `cache/`
- Experiment results: `results/`
- Figures: `results/figures/`

#### Key Files
- `exp1_optimal_summary_k1.csv`: Optimal P_max for each T
- `exp3_method_comparison.csv`: Method comparison table
- `exp3_timing_analysis.csv`: Computational cost analysis

#### Interpreting Results
1. **Optimal P_max scaling**: Should follow T^0.25
2. **Error reduction**: Optimal P_max gives up to 95% improvement
3. **Complexity**: O(1) for Euler vs O(√T) for Riemann-Siegel

## Troubleshooting

### Common Issues

1. **"Zeros file not found"**
   - Download zeros file from Odlyzko's website
   - Place in `data/raw/zeros.txt`

2. **"Prime cache generation takes too long"**
   - First-time generation is normal (~15-20 minutes)
   - Subsequent runs use cached data

3. **Memory errors**
   - Ensure at least 8GB RAM available
   - Reduce `max_prime` in config if needed

4. **Import errors**
   - Run `pip install -r requirements.txt`
   - Check Python version (>=3.8)

### Performance Tips

1. **First run is slow** - caching speeds up subsequent runs
2. **Use GPU acceleration** - Not currently implemented
3. **Parallel processing** - Can be added for independent T values
4. **Memory mapping** - For very large datasets

## Advanced Usage

### Custom Experiments

Create a new notebook:

```python
# Custom experiment template
import sys
sys.path.append('../src')

from src.core.s_t_functions import *
from src.utils.paths import PathConfig

paths = PathConfig()
# Your experiment code here
```

### Extending the Methods

Add new S(T) computation methods in `src/core/s_t_functions.py`:

```python
def S_my_method(T, params):
    """Custom S(T) implementation."""
    # Your implementation
    return s_value
```

### Batch Processing

For processing multiple T values:

```python
T_values = np.logspace(3, 7, 50)
results = []

for T in T_values:
    s_val = S_euler(T, optimal_P_max, prime_cache)
    results.append({'T': T, 'S': s_val})
```

## Citation

If you use this code, cite:

```bibtex
@software{prime_reduction_2024,
  title={Prime Reduction Estimates for S(T) - Computational Code},
  author={Dvorak, John},
  year={2024},
  url={https://github.com/JohnNDvorak/Euler-Product-Zero-Counting}
}
```

## Support

For issues or questions:
1. Check this guide
2. Review docstrings in source code
3. Open an issue on GitHub