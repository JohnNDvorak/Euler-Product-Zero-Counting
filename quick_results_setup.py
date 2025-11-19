#!/usr/bin/env python3
"""
Quick setup to load your existing results.

Run this script to set up templates for your data.
Then fill in your actual values.
"""

import os
import pandas as pd
import numpy as np

# Create directories
os.makedirs('data/existing_results', exist_ok=True)

print("="*60)
print("QUICK RESULTS SETUP")
print("="*60)

# 1. Create template for optimal P_max results
print("\n1. Creating optimal_results.csv template...")
optimal_template = pd.DataFrame({
    'T': [1000, 10000, 100000, 1000000, 10000000],
    'optimal_P_max': [0.0, 0.0, 0.0, 0.0, 0.0],  # <-- FILL THESE
    'min_error': [0.0, 0.0, 0.0, 0.0, 0.0],    # <-- FILL THESE
    'S_ref': [0.0, 0.0, 0.0, 0.0, 0.0]          # <-- FILL THESE
})

optimal_template.to_csv('data/existing_results/optimal_results.csv', index=False)
print("   ✓ Created: data/existing_results/optimal_results.csv")
print("   Please fill in your actual values.")

# 2. Create dense sampling template (showing first few rows)
print("\n2. Creating dense_sampling_results.csv template...")
dense_template = []

# Create first 6 rows as example (T=1000)
for P_max in [100, 1000, 10000, 100000, 1000000, 10000000]:
    dense_template.append({
        'T_idx': 0,
        'T': 1000.0,
        'P_max': P_max,
        'S_ref': 0.0,        # <-- FILL
        'S_euler': 0.0,      # <-- FILL
        'error': 0.0,        # <-- FILL
        'improvement': 0.0,  # <-- FILL
        'computation_time': 0.0  # <-- FILL
    })

# Add a few more examples
for T_idx, T in enumerate([1001, 1002], start=1):
    for P_max in [100, 1000]:
        dense_template.append({
            'T_idx': T_idx,
            'T': T,
            'P_max': P_max,
            'S_ref': 0.0,
            'S_euler': 0.0,
            'error': 0.0,
            'improvement': 0.0,
            'computation_time': 0.0
        })

dense_df = pd.DataFrame(dense_template)
dense_df.to_csv('data/existing_results/dense_sampling_results.csv', index=False)
print(f"   ✓ Created: data/existing_results/dense_sampling_results.csv")
print(f"   Template shows first {len(dense_df)} rows of 3,594 total")

# 3. Instructions
print("\n" + "="*60)
print("INSTRUCTIONS:")
print("="*60)
print("\nTo load your results:")
print()
print("Option 1 - From your notebook:")
print("   1. Add these cells to your original notebook:")
print("   ```python")
print("   # Export optimal results")
print("   optimal_df.to_csv('/path/to/Euler-Product-Zero-Counting/data/existing_results/optimal_results.csv', index=False)")
print("   ")
print("   # Export dense results")
print("   if 'results_df' in locals():")
print("       results_df.to_csv('/path/to/Euler-Product-Zero-Counting/data/existing_results/dense_sampling_results.csv', index=False)")
print("   ```")
print()
print("Option 2 - Manual entry:")
print("   1. Open the CSV files in data/existing_results/")
print("   2. Fill in your actual values")
print("   3. Save the files")
print()
print("Once you have the files:")
print("   Run: python scripts/verify_implementation.py")
print()
print("This will compare our implementation with your results!")

print("\n" + "="*60)
print("MOST IMPORTANT RESULTS:")
print("="*60)
print("\n1. DENSE SAMPLING (3,594 measurements):")
print("   - The complete 599 T values × 6 P_max values")
print("   - This is the core data for your paper")
print()
print("2. OPTIMAL P_MAX RESULTS:")
print("   - For each T, the P_max that minimizes error")
print("   - Shows the T^0.25 scaling relationship")
print()
print("3. S(T) VALUES:")
print("   - Reference S_RS values")
print("   - S_euler values at each P_max")