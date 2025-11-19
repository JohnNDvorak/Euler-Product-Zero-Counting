# Repository Refactoring Guide

This document outlines recommended improvements for better repository organization and maintainability.

## Current Issues

1. **Monolithic notebook**: 30 cells containing ~6MB of code
2. **Mixed concerns**: Infrastructure, experiments, and figure generation all together
3. **Redundancy**: Multiple experimental approaches not all used in the paper
4. **Hardcoded paths**: Google Drive paths limit reusability

## Recommended Refactoring

### 1. Split the Notebook

```
notebooks/
├── 01_setup_and_functions.ipynb      # Cells 1-2 (prerequisites)
├── 02_main_experiments.ipynb         # Cells 3-5 (core results)
├── 03_visualization.ipynb            # Cell 6 (figures)
├── 04_diagnostics.ipynb              # Cell 8 (comprehensive analysis)
└── 05_publication_figures.ipynb      # Cells 23-30 (final figures)
```

### 2. Extract Python Modules

```
src/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── functions.py          # From Cell 1
│   ├── prime_cache.py        # PrimeCache class
│   └── numerical_utils.py    # Kahan summation, etc.
├── experiments/
│   ├── __init__.py
│   ├── optimal_truncation.py # Experiment 1
│   ├── phase_validation.py   # Experiment 2
│   └── method_comparison.py  # Experiment 3
├── visualization/
│   ├── __init__.py
│   └── plots.py             # Plotting utilities
└── utils/
    ├── __init__.py
    ├── data_loader.py       # Data loading utilities
    └── paths.py            # Path configuration
```

### 3. Configuration Management

Create `config.yaml`:
```yaml
data:
  zeros_file: "data/zeros/combined_zeros_1.txt"
  cache_dir: "cache"
  results_dir: "results"

experiments:
  t_values: [1000, 10000, 100000, 1000000]
  p_max_range: [1e6, 1e9]
  n_points: 50

paths:
  # Remove Google Colab dependencies
  base_dir: "."
```

### 4. Essential Code Mapping

Based on the paper content, here's what to keep:

| Section | Notebook Cells | Priority | Action |
|---------|----------------|----------|--------|
| **Core Infrastructure** | 1-2 | Essential | Keep in `01_setup_and_functions.ipynb` |
| **Main Results** | 3-5 | Essential | Keep in `02_main_experiments.ipynb` |
| **Figures** | 6 | Essential | Keep in `03_visualization.ipynb` |
| **Comprehensive Diagnostics** | 8 | High | Keep in `04_diagnostics.ipynb` |
| **Publication Plots** | 23-30 | Essential | Keep in `05_publication_figures.ipynb` |
| **ML Exploration** | 10-14 | Low | Archive in `exploratory/` |
| **Statistical Tests** | 15-22 | Medium | Consider for supplementary material |
| **Redundant Cells** | 7, 9 | Low | Remove or archive |

### 5. Directory Structure After Refactoring

```
Euler-Product-Zero-Counting/
├── README.md
├── LICENSE
├── requirements.txt
├── config.yaml
├── .gitignore
├── notebooks/                    # Main analysis notebooks
├── src/                         # Python source code
├── data/                        # Raw and processed data
│   ├── raw/
│   ├── processed/
│   └── external/
├── results/                     # Experiment results
│   ├── main/
│   ├── supplementary/
│   └── figures/
├── cache/                       # Computed caches
├── docs/                        # Additional documentation
│   ├── data_source.md
│   └── experimental_notes.md
└── exploratory/                 # Archived experiments
    ├── ml_analysis.ipynb
    └── extended_tests.ipynb
```

### 6. Function Extraction Priorities

From Cell 1, extract to `src/core/functions.py`:

```python
# Essential functions (keep)
- PrimeCache class
- kahan_sum()
- segmented_sieve()
- S_direct()
- S_RS()
- S_euler()
- smooth_RVM()

# Optional but useful
- check_prerequisites()
- compute_N_T()
- analyze_error()
```

### 7. Path Configuration

Create `src/utils/paths.py`:
```python
import os
from pathlib import Path
import yaml

class Config:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    @property
    def base_dir(self):
        return Path(self.config['paths']['base_dir'])

    @property
    def data_dir(self):
        return self.base_dir / "data"

    # ... other path properties
```

### 8. Migration Steps

1. **Phase 1: Extraction**
   - Create new directory structure
   - Extract Python modules from Cell 1
   - Create configuration files

2. **Phase 2: Notebook Splitting**
   - Split notebooks by functionality
   - Update import statements
   - Test each notebook independently

3. **Phase 3: Path Updates**
   - Replace Google Drive paths
   - Implement path configuration
   - Update data loading

4. **Phase 4: Documentation**
   - Update README with new structure
   - Add module documentation
   - Create data source documentation

5. **Phase 5: Cleanup**
   - Archive exploratory code
   - Update .gitignore
   - Tag release version

### 9. .gitignore Recommendations

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Data and cache
cache/
*.pkl
*.npy
data/raw/
!data/raw/README.md

# Jupyter
.ipynb_checkpoints/
*.ipynb_bak

# OS
.DS_Store
Thumbs.db

# Results
results/temp/
```

### 10. Benefits of Refactoring

1. **Maintainability**: Easier to understand and modify individual components
2. **Reusability**: Functions can be imported in other projects
3. **Collaboration**: Multiple researchers can work on different components
4. **Version Control**: Better change tracking with smaller files
5. **Documentation**: Each module can have focused documentation
6. **Testing**: Easier to unit test individual functions

### 11. Quick Refactoring Script

Create `scripts/refactor.py` to automate:
```python
import os
import shutil

def create_structure():
    dirs = [
        'notebooks', 'src', 'data', 'results',
        'cache', 'docs', 'exploratory'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, '.gitkeep'), exist_ok=True)

# Add functions to extract cells, copy files, etc.
```

This refactoring will make the repository more professional, maintainable, and accessible to other researchers while preserving all important results.