"""
Numerical utilities for stable and accurate computations.

This module provides numerically stable implementations of common operations
essential for computing S(T) and related quantities.
"""

import numpy as np


def kahan_sum(arr):
    """
    Kahan compensated summation for numerical stability.
    Essential for oscillatory sums with many cancellations.

    Parameters:
    -----------
    arr : array-like
        Array of numbers to sum

    Returns:
    --------
    float : Compensated sum
    """
    total = 0.0
    compensation = 0.0

    for x in arr:
        adjusted = x - compensation
        new_total = total + adjusted
        compensation = (new_total - total) - adjusted
        total = new_total

    return total


def kahan_sum_complex(arr):
    """
    Kahan summation for complex numbers.

    Parameters:
    -----------
    arr : array-like
        Array of complex numbers to sum

    Returns:
    --------
    complex : Compensated sum
    """
    total_real = 0.0
    total_imag = 0.0
    comp_real = 0.0
    comp_imag = 0.0

    for x in arr:
        # Real part
        adj_real = x.real - comp_real
        new_real = total_real + adj_real
        comp_real = (new_real - total_real) - adj_real
        total_real = new_real

        # Imaginary part
        adj_imag = x.imag - comp_imag
        new_imag = total_imag + adj_imag
        comp_imag = (new_imag - total_imag) - adj_imag
        total_imag = new_imag

    return total_real + 1j * total_imag


def pairwise_sum(arr):
    """
    Pairwise summation for improved numerical stability.

    Parameters:
    -----------
    arr : array-like
        Array of numbers to sum

    Returns:
    --------
    float : Pairwise sum
    """
    if len(arr) == 1:
        return arr[0]
    elif len(arr) == 0:
        return 0.0

    mid = len(arr) // 2
    return pairwise_sum(arr[:mid]) + pairwise_sum(arr[mid:])