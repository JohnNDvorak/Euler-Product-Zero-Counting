# Prime Reduction Estimates for S(T)

This repository contains the computational experiments and supplementary material for the paper:
**"Prime Reduction Estimates for S(T)"** (November 14, 2024).

## Overview

This work explores optimal truncation strategies for Euler product representations in the computation of S(T), the argument of the Riemann zeta function on the critical line. We demonstrate that by carefully selecting an optimal prime bound P_max(T), we can achieve:
- Fixed computational cost O(1) versus O(√T) for Riemann-Siegel
- Improved accuracy compared to arbitrary truncation points
- A principled approach based on quasi-random phase cancellation theory

## Key Results

1. **Optimal P_max Scaling**: The optimal prime truncation bound scales as P_max ≈ T^(1/4)
2. **Error Reduction**: Achieves up to 95% error reduction compared to direct computation at large T
3. **Fixed Cost**: O(1) computational complexity versus O(√T) for traditional methods
4. **Theoretical Foundation**: Based on Weyl equidistribution and random phase heuristics

## 🚨 IMPORTANT: REFACTORING IN PROGRESS

This repository is currently being refactored from a monolithic Jupyter notebook to a professional, modular codebase.

**Current Status**: 33% Complete (2 of 6 phases)
- ✅ Phase 1: Core S_euler implementation (FIXED and verified)
- ✅ Phase 2: Dense sampling analysis (3,594 measurements)
- ⏳ Phase 3: Phase validation tests (pending)
- ⏳ Phase 4: Comprehensive diagnostics (pending)
- ⏳ Phase 5: Statistical analysis (pending)
- ⏳ Phase 6: Publication materials (pending)

See [PROGRESS.md](PROGRESS.md) for detailed status.

## Repository Structure

### Refactored Code (New)
```
├── src/
│   ├── core/
│   │   ├── s_t_functions.py          # ✅ Corrected S_euler implementation
│   │   └── prime_cache.py            # ✅ Prime generation utilities
│   └── utils/
│       └── legacy_results_loader.py  # ✅ Loads original experimental data
├── data/
│   ├── 10M_zeta_zeros.txt            # Riemann zeros (10M values)
│   └── existing_results/             # ✅ Legacy data from original notebook
├── figures/                          # Generated analysis plots
├── phase2_dense_sampling_fast.py     # ✅ Phase 2 analysis script
├── test_phase1_complete.py          # ✅ Phase 1 verification
└── PROGRESS.md                       # Detailed progress report
```

### Original Notebook
```
├── 26OCT_S(T)_Prime_Sums.ipynb    # ⚠️ Original notebook (30MB, being refactored)
└── Prime_Reduction_S_T__Estimates_14NOV.pdf  # Published paper
```

## Notebook Contents

The main notebook is organized into experimental modules:

### Core Infrastructure (Required)
- **Cell 1**: Shared functions & core utilities
- **Cell 2**: Experiment 0 - Setup & data loading (generates caches)

### Main Experiments (Core Results)
- **Cell 3**: Experiment 1 - Optimal truncation search
- **Cell 4**: Experiment 2 - Phase cancellation validation
- **Cell 5**: Experiment 3 - Method comparison at optimal P_max
- **Cell 6**: Experiment 4 - Visualization (all figures)

### Extended Analyses (Supplementary)
- **Cells 7-19**: Additional diagnostic tests, statistical analyses
- **Cells 20-30**: Publication-quality figure generation and export

## Requirements

```python
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
mpmath>=1.2.0

# For reproducibility
Python >= 3.8
```

## Quick Start

### Using Refactored Code (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JohnNDvorak/Euler-Product-Zero-Counting.git
   cd Euler-Product-Zero-Counting
   ```

2. **Set up environment**:
   ```bash
   pip install numpy pandas matplotlib seaborn scipy mpmath
   ```

3. **Run refactored experiments**:
   ```bash
   # Phase 1 verification (already complete)
   python test_phase1_complete.py

   # Phase 2 analysis (already complete)
   python phase2_dense_sampling_fast.py
   ```

### Using Original Notebook

1. **Open Jupyter notebook**:
   ```bash
   jupyter notebook 26OCT_S(T)_Prime_Sums.ipynb
   ```

2. **Execute cells sequentially**:
   - Cell 1: Load functions
   - Cell 2: Setup and data loading (~15-20 minutes first time)
   - Cells 3-6: Main experimental results

## Computational Resources

- **Memory**: ~8GB RAM recommended (for prime cache)
- **Storage**: ~2GB for cached data
- **Time**:
  - Refactored code: <5 minutes (uses cached results)
  - Original notebook: 30-40 minutes (first run)

## Recent Updates

### November 17, 2024
- ✅ **CRITICAL FIX**: Corrected S_euler implementation (was using wrong formula)
- ✅ Completed Phase 1: Core infrastructure with 100% verification
- ✅ Completed Phase 2: Dense sampling analysis (3,594 measurements)
- 📁 Added comprehensive documentation and progress tracking

### Key Technical Discovery
The original notebook uses a simplified approximation, not the full complex Euler product:
```
S(T) ≈ -(1/π) Σ_p p^(-1/2) sin(T log p)
```

This insight was crucial for correct implementation.

## Repository Cleanup Recommendations

### Essential Code to Keep
1. **Cells 1-6**: Core infrastructure and main experiments
2. **Cell 8**: Comprehensive diagnostic with three methods
3. **Cells 23-30**: Publication figure generation

### Experimental Code to Consider Removing
1. **Cells 10-19**: Exploratory ML analyses and extended statistical tests
2. **Cells 7, 9-10**: Redundant visualization attempts
3. **Cells 11-12**: Fourier analysis and adaptive strategies (not in paper)

### Suggested Refactoring
```python
# Split into multiple notebooks:
├── notebooks/
│   ├── 01_core_experiments.ipynb     # Cells 1-6
│   ├── 02_supplementary_analysis.ipynb  # Cells 7-22
│   └── 03_publication_figures.ipynb   # Cells 23-30
├── src/
│   ├── s_t_functions.py              # Core functions from Cell 1
│   ├── experiments.py                # Experiment classes
│   └── utils.py                      # Helper utilities
└── data/                             # Data and cache directory
```

## Key Functions

- `S_direct(T, zeros)`: Direct computation via Im log ζ
- `S_RS(T, zeros)`: Riemann-Siegel formula
- `S_euler(T, P_max, prime_cache, k=1)`: Euler product with truncation
- `find_optimal_P_max(T, P_max_range)`: Optimizes truncation bound
- `kahan_sum(arr)`: Numerically stable summation

## Citation

If you use this code or results, please cite:

```bibtex
@article{prime_reduction_2024,
  title={Prime Reduction Estimates for S(T)},
  author={Dvorak, John},
  journal={arXiv preprint},
  year={2024},
  month={November}
}
```

## License

This repository is licensed under the MIT License - see LICENSE file for details.

## Contact

John Dvorak - [your-email@example.com]

## Acknowledgments

- Data source: Andrew Odlyzko's tables of zeros of the Riemann zeta function
- Numerical algorithms: Based on implementations in mpmath and SciPy