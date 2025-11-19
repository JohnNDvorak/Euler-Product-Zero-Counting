# Migration Plan: From Original to Refactored Code

## Current Status

### ✅ Completed
1. **Basic Infrastructure**: Core functions, path configuration, data loading
2. **Directory Structure**: Professional Python package layout
3. **Simple Tests**: Verified basic functionality works
4. **Dense Sampling Framework**: Created structure for 599 T × 6 P_max experiment

### ❌ Major Gaps Remain

The original notebook contains **significantly more** than what's currently implemented. Here's what's missing:

## 1. Experimental Scope Gap

| Metric | Original | Refactored | Gap |
|--------|----------|------------|-----|
| **T values tested** | 599 | 3-5 | 99% missing |
| **P_max values** | 6 fixed: [100, 1K, 10K, 100K, 1M, 10M] | Continuous search | Different methodology |
| **Prime powers** | k ≤ 5 | k = 1 only | Missing k=2,3,4,5 |
| **Total experiments** | 3,594 measurements | ~15 | 99.6% missing |

## 2. Missing Notebooks (30 cells → need 6 notebooks)

### ✅ Already Created:
- `01_setup_and_functions.ipynb` - Cell 1-2 (partial)
- `02_main_experiments.ipynb` - Cell 3-5 (simplified)
- `04_dense_sampling.ipynb` - Cell 16 framework

### ❌ Still Need:

#### `03_phase_validation.ipynb` (Cell 4)
```python
# 5 Tests needed:
1. Uniformity test (KS statistic)
2. Growth rate (10 points from 10^7 to 10^9)
3. Random signs (500K primes, 100 Monte Carlo trials)
4. Autocorrelation (max lag = 100)
5. Phase circle visualization
```

#### `05_comprehensive_diagnostics.ipynb` (Cell 8)
```python
# 4 P_max values: [1e6, 5e6, 1e7, 5e7]
# 3 methods:
- Riemann-Siegel
- Euler k=1 at optimal P_max
- Euler k=1 at fixed P_max
```

#### `06_statistical_analysis.ipynb` (Cells 17-18)
```python
# Height-level correlation (n=599)
# Cluster-robust logistic regression (n=3,594, 599 clusters)
# Statistical significance testing
# Outlier-resistant methods
```

#### `07_publication_figures.ipynb` (Cells 23-30)
```python
# All publication-quality figures
# Export to JSON for reproducibility
# Three-factor interaction plots
```

## 3. Code Implementation Gaps

### Missing Functions in `src/core/`:

#### `s_t_functions.py` improvements needed:
```python
# Current S_euler has issues with:
1. No logarithmic computation (causes overflow)
2. Doesn't handle PrimeCache correctly
3. k_max parameter not fully implemented

# Needed improvements:
def S_euler_stable(T, P_max, primes, k_max=5):
    """Numerically stable Euler product computation."""
    # Use logarithmic approach to avoid overflow
    # Support both PrimeCache and raw arrays
    # Include prime powers up to k=5
```

#### Missing functions:
```python
def smooth_RVM(T, L=10.0):
    """Smooth Riemann-von Mangoldt estimate."""

def compute_N_T(T, zeros):
    """Compute N(T) from zeros."""

def adaptive_P_max(T, base_P_max):
    """Adaptive P_max strategy from Cell 12."""
```

### Missing `PrimeCache` methods:
```python
class PrimeCache:
    def __init__(self, max_prime=1_000_000_000):
        # Need full implementation with segmented sieve
        # Should handle up to 1 billion primes efficiently
        pass
```

## 4. Data Processing Gap

### Original Data Pipeline:
1. **Cell 2**: Load 10M zeros → preprocess → cache
2. **Cell 2**: Generate 1B prime cache → cache
3. **Cell 3**: Find optimal P_max for 5 T values
4. **Cell 4**: Phase validation tests
5. **Cell 5**: Method comparison
6. **Cell 16**: **DENSE SAMPLING** (599 T × 6 P_max)
7. **Cells 17-18**: Statistical analysis

### Current Pipeline:
1. Load 10M zeros ✅
2. Generate 5.7M primes (not 1B) ❌
3. Simple experiments only ❌

## 5. Migration Action Items

### Phase 1: Fix Core Issues (Priority: HIGH)
1. **Fix S_euler numerical stability**:
   - Implement logarithmic computation
   - Handle large P_max without overflow
   - Support k_max up to 5

2. **Complete PrimeCache**:
   - Implement segmented sieve
   - Support up to 1 billion primes
   - Efficient memory management

3. **Run dense sampling experiment**:
   - Execute `04_dense_sampling.ipynb`
   - Generate 3,594 measurements
   - Verify paper results

### Phase 2: Implement Missing Notebooks (Priority: MEDIUM)
1. Create `03_phase_validation.ipynb`
2. Create `05_comprehensive_diagnostics.ipynb`
3. Create `06_statistical_analysis.ipynb`
4. Create `07_publication_figures.ipynb`

### Phase 3: Full Reproduction (Priority: MEDIUM)
1. Run complete experimental suite
2. Verify all paper figures match
3. Export results in original format
4. Create reproducibility package

## 6. Critical Path

To achieve full paper reproduction:

```
Day 1: Fix S_euler numerical stability
Day 2: Complete PrimeCache implementation
Day 3: Run dense sampling (4-6 hours compute time)
Day 4: Create missing notebooks
Day 5: Statistical analysis
Day 6: Publication figures
Day 7: Full verification
```

## 7. Resource Requirements

### Computational:
- **Memory**: 16GB+ for 1B prime cache
- **CPU**: Dense sampling requires 10-30 hours
- **Storage**: 10GB+ for all caches

### Development:
- Implement segmented sieve
- Add statistical analysis functions
- Create all missing visualizations

## Conclusion

The refactoring created a good **foundation** but only implemented ~1% of the experimental scope. The original notebook contains **extensive statistical analysis** and **comprehensive experiments** that need to be fully ported to match the paper.

**Immediate next steps**:
1. Fix numerical stability in S_euler
2. Run the dense sampling experiment
3. Verify we can reproduce the 599 × 6 measurement matrix

This will ensure the refactored code truly matches the paper's experimental scope.