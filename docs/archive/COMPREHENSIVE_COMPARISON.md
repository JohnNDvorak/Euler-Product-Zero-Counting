# Comprehensive Comparison: Original vs Refactored Code

## Executive Summary

The refactored code currently implements only a **simplified version** of the experiments. The original notebook contains **extensive experiments** with:
- **599 T values** (height levels)
- **6 P_max values**: [100, 1K, 10K, 100K, 1M, 10M]
- **Multiple additional analyses** not yet refactored

## Key Differences

### 1. Experimental Scale

| Aspect | Original Notebook | Refactored Code | Gap |
|--------|------------------|----------------|-----|
| **T values tested** | 599 (dense sampling) | 3-5 (sparse) | ❌ Missing 594+ T values |
| **P_max values** | 6 fixed values [100, 1K, 10K, 100K, 1M, 10M] | Continuous search up to 2×10⁸ | ❌ Different methodology |
| **Prime powers** | k ≤ 5 (includes prime squares, cubes, etc.) | k = 1 (primes only) | ❌ Missing prime powers |
| **Total computations** | 599 × 6 = 3,594 measurements | ~15 computations | ❌ 99.6% missing |

### 2. Missing Experiments

The original notebook contains these experiments **not yet implemented**:

#### Cell 1: Core Functions ✅ (Partially Implemented)
- ❌ PrimeCache with full 1B prime support
- ❌ S_euler with k_max parameter
- ❌ smooth_RVM baseline
- ❌ Complete check_prerequisites

#### Cell 2: Experiment 0 - Setup ❌ (Not Implemented)
- Load 10M zeros with preprocessing
- Generate 1B prime cache
- Multiple S_euler_k1 tests at different P_max

#### Cell 3: Experiment 1 - Optimal Truncation ❌ (Simplified)
- **Original**: 50 P_max points from 10⁶ to 10⁹
- **Original**: Tests both k=1 and k≤5
- **Refactored**: Only 5 points, k=1 only
- **Missing**: Comprehensive error curves

#### Cell 4: Experiment 2 - Phase Validation ❌ (Not Implemented)
- Test 1: Uniformity test
- Test 2: Growth rate (enhanced with 10 points)
- Test 3: Random signs (500K primes, 100 trials)
- Test 4: Autocorrelation
- Test 5: Phase circle visualization

#### Cell 5: Experiment 3 - Method Comparison ❌ (Not Implemented)
- Compares 5 methods:
  1. Direct (Im log ζ)
  2. Riemann-Siegel
  3. Euler k=1 at optimal
  4. Euler k=1 at 200M
  5. Euler k≤5 at optimal

#### Cell 6: Experiment 4 - Visualization ❌ (Not Implemented)
- All publication figures

#### Cell 8: Comprehensive Diagnostic ❌ (Not Implemented)
- 4 P_max values: [1e6, 5e6, 1e7, 5e7]
- Three methods comparison
- Error analysis

#### Cell 16: Dense T Sampling ❌ (Not Implemented)
- **599 T values** from existing results
- **6 P_max values**: [100, 1K, 10K, 100K, 1M, 10M]
- **3,594 total computations**

#### Cells 17-30: Extended Analyses ❌ (Not Implemented)
- Statistical significance testing
- Logistic regression with interaction effects
- Publication-quality figures
- JSON export for reproducibility

### 3. Implementation Differences

#### Prime Powers
```python
# Original (k ≤ 5)
for k in range(1, K+1):
    p_k = p**k
    if p_k > P_max:
        break
    contribution = -np.exp(2j * np.pi * k * phase) / k

# Refactored (k = 1 only)
for p in primes_filtered:
    phase = T * np.log(p) / (2 * np.pi)
    zeta_factor = 1.0 / (1.0 - np.exp(-2j * np.pi * phase))
```

#### Numerical Stability
- **Original**: Uses complex Euler factors directly
- **Refactored**: Same approach but encounters overflow

#### Data Structure
- **Original**: Clustered by T (599 clusters × 6 P_max)
- **Refactored**: No clustering, sparse sampling

## Required Implementation

To fully match the paper's experiments, the refactored code needs:

### 1. Complete S_euler Implementation
```python
def S_euler(T, P_max, prime_cache, k_max=5):
    """
    Compute S(T) using truncated Euler product with prime powers.

    Parameters:
    - k_max: Maximum exponent for prime powers (default: 5)
    """
    primes = prime_cache.get_primes_up_to(P_max)

    zeta_prod = 1.0 + 0.0j

    for p in primes:
        p_k = p
        for k in range(1, k_max + 1):
            if p_k > P_max:
                break

            phase = T * np.log(p_k) / (2 * np.pi)
            zeta_prod *= 1.0 / (1.0 - np.exp(-2j * np.pi * phase))
            p_k *= p

    return np.angle(zeta_prod) / np.pi
```

### 2. Dense T Sampling Experiment
```python
# Cell 16 equivalent
P_MIN_TEST = [100, 1000, 10000, 100000, 1000000, 10000000]
N_T_SAMPLES = 599  # Or load from existing results

# Load existing results to get T values
results_df = pd.read_csv('existing_results.csv')
T_sample = results_df['T'].unique()[:599]

# Run full experiment
for T in T_sample:
    for P_max in P_MIN_TEST:
        # Compute S_euler for all methods
        # Store in structured DataFrame
```

### 3. Missing Notebook Implementations

We need to create these notebooks:
1. **`notebooks/03_diagnostics.ipynb`** - Cell 8 diagnostics
2. **`notebooks/04_dense_sampling.ipynb`** - Cell 16 dense T sampling
3. **`notebooks/05_statistical_analysis.ipynb`** - Cells 17-18
4. **`notebooks/06_publication_figures.ipynb`** - Cells 23-30

### 4. Data Processing Pipeline

The original processes data in this order:
1. Load 10M zeros (Cell 2)
2. Generate 1B prime cache (Cell 2)
3. Find optimal P_max (Cell 3)
4. Validate phases (Cell 4)
5. Compare methods (Cell 5)
6. Generate figures (Cell 6)
7. **Dense sampling on 599 T values** (Cell 16)
8. Statistical analysis (Cells 17-18)
9. Publication figures (Cells 23-30)

## Recommendations

### Immediate Actions

1. **Extract Cell 16 Functionality**:
   - Implement dense T sampling with 599 values
   - Use 6 fixed P_max values [100, 1K, 10K, 100K, 1M, 10M]
   - Create structured output with T_idx, P_max, error, improvement

2. **Add Prime Powers Support**:
   - Update S_euler to accept k_max parameter
   - Test convergence with k=1 vs k=5

3. **Implement Cell 8 Diagnostics**:
   - Three method comparison at 4 P_max values
   - Error analysis and timing

4. **Create Missing Notebooks**:
   - Each major experiment gets its own notebook
   - Preserve original methodology

### Long-term Actions

1. **Numerical Stability**:
   - Implement logarithmic Euler product
   - Use higher precision arithmetic

2. **Performance Optimization**:
   - Vectorized prime operations
   - Parallel processing for independent T values

3. **Reproducibility**:
   - Export all results in JSON format
   - Include all metadata and random seeds

## Conclusion

The current refactored code implements **approximately 1-2%** of the original experimental scope. To fully reproduce the paper results, we need to:

1. ✅ Complete the basic infrastructure (done)
2. 🔄 Implement dense T sampling (599 values)
3. 🔄 Add 6 P_max value testing
4. 🔄 Include prime powers (k ≤ 5)
5. 🔄 Add all statistical analyses
6. 🔄 Generate all publication figures

The refactoring created a good foundation, but the experimental scope needs to be expanded significantly to match the original work.