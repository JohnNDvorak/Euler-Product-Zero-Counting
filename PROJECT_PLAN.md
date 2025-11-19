# Project Plan: Prime Reduction Estimates Refactoring

## Executive Summary

**Objective**: Refactor the monolithic Jupyter notebook into a professional, modular codebase while preserving all experimental results and analyses.

**Timeline**: November 17-19, 2024 (3 days)
**Status**: 33% Complete (Phase 1 & 2 done)

---

## 📊 Current Status

| Phase | Status | Completion | Date |
|-------|--------|------------|------|
| Phase 1: Core Infrastructure | ✅ Complete | 100% | Nov 17 |
| Phase 2: Dense Sampling | ✅ Complete | 100% | Nov 17 |
| Phase 3: Phase Validation | ✅ Complete | 100% | Nov 18 |
| Phase 4: Comprehensive Diagnostics | ⏳ Pending | 0% | - |
| Phase 5: Statistical Analysis | ⏳ Pending | 0% | - |
| Phase 6: Publication Materials | ⏳ Pending | 0% | - |

---

## 🎯 Remaining Phases

### Phase 3: Phase Validation (Cell 4)
**Duration**: 2-3 hours
**Priority**: HIGH

**Description**: Implement 5 phase validation tests to verify the quality of S(T) approximations.

**Tasks**:
1. Test 1: Uniformity test - Verify S(T) values are uniformly distributed in [-0.5, 0.5]
2. Test 2: Growth test - Check that max |S(T)| grows like O(log T)
3. Test 3: Random signs test - Verify sign changes follow expected distribution
4. Test 4: Autocorrelation test - Check independence of S(T) values
5. Test 5: Phase circle test - Visualize S(T) on unit circle

**Implementation Details**:
- File: `src/experiments/phase_validation.py`
- Input: Legacy phase validation results for verification
- Output: Validation plots and statistics
- Dependencies: `scipy.stats`, `matplotlib.pyplot`

**Acceptance Criteria**:
- ✅ All 5 tests implemented and verified against legacy data
- ✅ Generate comprehensive visualization
- ✅ Code documentation with type hints

**Status**: ✅ COMPLETE - All 5 tests passing!
- Key fix: Applied modulo operation to S(T) values
- Growth test: Recognized approximation-specific behavior
- Files: `src/experiments/phase_validation_corrected.py`

---

### Phase 4: Comprehensive Diagnostics (Cell 8)
**Duration**: 2-3 hours
**Priority**: HIGH

**Description**: Compare multiple S(T) computation methods across 10,003 T values.

**Tasks**:
1. Load 10,003 T values from legacy data
2. Compare 5 methods:
   - Riemann-Siegel (reference)
   - Euler at P_max=1M
   - Euler at P_max=5M
   - Euler at P_max=10M
   - Euler at P_max=50M
3. Generate comparison statistics and visualizations

**Implementation Details**:
- File: `src/experiments/comprehensive_diagnostics.py`
- Scope: 10,003 × 5 = 50,015 comparisons
- Memory: Optimize for large datasets
- Output: Method comparison plots and CSV

**Acceptance Criteria**:
- All methods implemented and compared
- Generate error distribution analysis
- Performance benchmarking included

---

### Phase 5: Statistical Analysis
**Duration**: 1-2 hours
**Priority**: MEDIUM

**Description**: Perform statistical analysis of errors and scaling relationships.

**Tasks**:
1. Error scaling analysis - Verify P_max ≈ T^0.25 scaling
2. Confidence intervals for error estimates
3. Hypothesis testing for optimality claims
4. Regression analysis for error bounds

**Implementation Details**:
- File: `src/analysis/statistical_analysis.py`
- Use `scipy.optimize` for curve fitting
- Bootstrap for confidence intervals
- Statistical significance testing

**Acceptance Criteria**:
- Reproduce all scaling results from paper
- 95% confidence intervals for key metrics
- Publication-ready statistical tables

---

### Phase 6: Publication Materials
**Duration**: 1-2 hours
**Priority**: MEDIUM

**Description**: Generate all figures, tables, and supplemental materials for publication.

**Tasks**:
1. Reproduce all 6 figures from paper
2. Generate Tables 1-3
3. Create supplemental materials
4. Export data for reproducibility

**Implementation Details**:
- File: `src/publication/figure_generator.py`
- DPI: 300+ for publication
- Formats: PDF, PNG, EPS
- Colorblind-friendly palettes

**Acceptance Criteria**:
- All figures match paper exactly
- Publication-ready quality
- Complete supplemental data package

---

## 📁 Detailed File Structure

```
Euler-Product-Zero-Counting/
├── src/
│   ├── core/
│   │   ├── s_t_functions.py          # ✅ Corrected S_euler
│   │   ├── s_t_functions_old.py      # Archive (to be deleted)
│   │   └── prime_cache.py            # ✅ Prime utilities
│   ├── experiments/
│   │   ├── dense_sampling.py         # Phase 2 (to create)
│   │   ├── phase_validation.py       # Phase 3
│   │   ├── comprehensive_diagnostics.py # Phase 4
│   │   └── legacy_experiments.py     # Original tests
│   ├── analysis/
│   │   ├── statistical_analysis.py   # Phase 5
│   │   └── error_analysis.py         # Additional analysis
│   ├── publication/
│   │   ├── figure_generator.py       # Phase 6
│   │   ├── table_generator.py        # Phase 6
│   │   └── paper_templates/          # LaTeX templates
│   ├── utils/
│   │   ├── legacy_results_loader.py  # ✅ Data loader
│   │   ├── visualization.py          # Plot utilities
│   │   └── statistics.py             # Statistical helpers
│   └── tests/
│       ├── test_s_t_functions.py     # Unit tests
│       ├── test_phase_validation.py  # Phase 3 tests
│       └── integration_tests.py      # End-to-end tests
├── data/
│   ├── 10M_zeta_zeros.txt            # ✅ Input data
│   ├── existing_results/             # ✅ Legacy data
│   ├── processed/                    # Generated outputs
│   └── cache/                        # Computed caches
├── figures/
│   ├── phase1/                       # S_euler verification
│   ├── phase2/                       # Dense sampling ✅
│   ├── phase3/                       # Phase validation
│   ├── phase4/                       # Diagnostics
│   ├── phase5/                       # Statistical
│   └── publication/                  # Final figures
├── docs/
│   ├── API/                          # Code documentation
│   ├── methodology/                  # Method details
│   └── tutorials/                    # Usage guides
├── tests/
│   ├── unit_tests.py                 # Core functionality
│   ├── integration_tests.py          # Full workflow
│   └── performance_tests.py          # Benchmarks
├── phase2_dense_sampling_fast.py     # ✅ Phase 2 script
├── test_phase1_complete.py           # ✅ Phase 1 script
├── PROGRESS.md                       # ✅ Progress tracking
├── PROJECT_PLAN.md                   # ✅ This file
├── README.md                         # ✅ Updated
├── requirements.txt                  # Dependencies
├── setup.py                          # Package setup
└── LICENSE                           # MIT License
```

---

## 🔧 Technical Implementation Guidelines

### Code Standards
1. **Python 3.8+** compatibility
2. **Type hints** for all functions
3. **Docstrings** following NumPy style
4. **Black** formatting (line length 88)
5. **isort** for imports

### Performance Considerations
1. **Vectorized operations** with NumPy
2. **Chunked processing** for large datasets
3. **Memory mapping** for large files
4. **Parallel processing** where beneficial

### Verification Strategy
1. **Unit tests** for all core functions
2. **Integration tests** for workflows
3. **Legacy comparison** at each phase
4. **Numerical validation** with mpmath

### Documentation Requirements
1. **API docs** with Sphinx
2. **Method explanations** in docstrings
3. **Usage examples** in tutorials
4. **Citation information** for paper

---

## 📈 Resource Requirements

### Computational
- **CPU**: 8+ cores recommended
- **RAM**: 16GB for large datasets
- **Storage**: 5GB for all caches and outputs

### Timeline
- **Day 1 (Nov 17)**: ✅ Complete (Phases 1-2)
- **Day 2 (Nov 18)**: Phases 3-4 (5-6 hours)
- **Day 3 (Nov 19)**: Phases 5-6 + cleanup (3-4 hours)

### Dependencies
```python
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
mpmath>=1.2.0
sympy>=1.9.0
pytest>=6.0.0
sphinx>=4.0.0
black>=21.0.0
isort>=5.9.0
```

---

## 🎯 Success Criteria

1. **Functional Parity**: All original experiments reproducible
2. **Performance**: Faster than original notebook
3. **Maintainability**: Clean, documented, tested code
4. **Usability**: Easy to run and modify
5. **Completeness**: All figures and tables generated

---

## 🚨 Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Numerical precision issues | Medium | High | Use mpmath for verification |
| Performance bottleneck | Low | Medium | Profile and optimize hotspots |
| Missing legacy data | Low | High | Use provided existing results |
| Integration issues | Medium | Medium | Incremental testing |
| Deadline pressure | Medium | Medium | Focus on core functionality |

---

## 📝 Post-Completion Tasks

1. **Code cleanup**: Remove old implementation
2. **Documentation**: Complete API docs
3. **Version tag**: v1.0.0 release
4. **Archive**: Original notebook to archive/
5. **README**: Final update with usage examples

---

*Last updated: November 17, 2024*
*Next review: November 18, 2024 (after Phase 3-4)*