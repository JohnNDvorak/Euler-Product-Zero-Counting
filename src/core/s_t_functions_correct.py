"""
Correct implementation of S(T) using Euler product approximation.

This matches the original notebook's formula:
    S(T) ≈ -(1/π) Σ_p p^{-1/2} sin(T log p)

NOT the full complex Euler product argument.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List
from functools import lru_cache


def S_euler_correct(T: float, P_max: float, primes: Union[np.ndarray, object],
                   k_max: int = 1, method: str = 'kahan') -> float:
    """
    Compute S(T) using Euler product approximation (original notebook formula).

    Formula: S(T) ≈ -(1/π) Σ_p Σ_{k=1}^K (1/k) p^{-k/2} sin(k T log p)

    Parameters:
    -----------
    T : float
        Height on critical line
    P_max : float
        Maximum prime/power to include in sum
    primes : np.ndarray or PrimeCache object
        Prime numbers or cache object
    k_max : int
        Maximum exponent for prime powers (1 for primes only)
    method : str
        Summation method ('kahan', 'direct')

    Returns:
    --------
    float : S(T) approximation
    """
    # Extract primes
    if hasattr(primes, 'get_primes_up_to'):
        # PrimeCache object
        prime_list = primes.get_primes_up_to(P_max)
    else:
        # Assume numpy array
        prime_list = primes[primes <= P_max]

    if len(prime_list) == 0:
        return 0.0

    total_sum = 0.0

    # Process each prime and its powers
    for p in prime_list:
        p_power = p

        for k in range(1, k_max + 1):
            if p_power > P_max:
                break

            # Term for p^k: (1/k) * p^(-k/2) * sin(k * T * log(p))
            log_p = np.log(p)
            term = (p_power ** (-0.5)) * np.sin(k * T * log_p) / k

            if method == 'kahan':
                total_sum = kahan_add(total_sum, term)
            else:
                total_sum += term

            p_power *= p

    result = -total_sum / np.pi
    return result


def kahan_add(total: float, value: float) -> float:
    """Kahan summation to reduce floating point error."""
    compensation = 0.0
    y = value - compensation
    t = total + y
    compensation = (t - total) - y
    total = t
    return total


class SEulerCorrect:
    """
    Correct S_euler implementation matching the original notebook.
    """

    def __init__(self):
        self.compensation = 0.0  # For Kahan summation

    def compute(self, T: float, P_max: float, primes: Union[np.ndarray, object],
                k_max: int = 1, use_kahan: bool = True) -> float:
        """
        Compute S(T) using the original notebook's formula.

        Parameters:
        -----------
        T : float
            Height on critical line
        P_max : float
            Maximum prime to include
        primes : np.ndarray or PrimeCache
            Prime numbers
        k_max : int
            Maximum exponent (for prime powers)
        use_kahan : bool
            Whether to use Kahan summation

        Returns:
        --------
        float : S(T) approximation
        """
        # Extract primes
        if hasattr(primes, 'get_primes_up_to'):
            prime_list = primes.get_primes_up_to(P_max)
        else:
            prime_list = primes[primes <= P_max]

        if len(prime_list) == 0:
            return 0.0

        # Precompute logs for efficiency
        log_primes = np.log(prime_list)

        total = 0.0

        if use_kahan:
            compensation = 0.0
            for i, p in enumerate(prime_list):
                p_k = p
                log_p = log_primes[i]

                for k in range(1, k_max + 1):
                    if p_k > P_max:
                        break

                    # Term: p^(-k/2) * sin(k * T * log(p)) / k
                    term = (p_k ** (-0.5)) * np.sin(k * T * log_p) / k

                    # Kahan summation
                    y = term - compensation
                    t = total + y
                    compensation = (t - total) - y
                    total = t

                    p_k *= p
        else:
            # Direct summation
            for i, p in enumerate(prime_list):
                p_k = p
                log_p = log_primes[i]

                for k in range(1, k_max + 1):
                    if p_k > P_max:
                        break

                    term = (p_k ** (-0.5)) * np.sin(k * T * log_p) / k
                    total += term

                    p_k *= p

        result = -total / np.pi
        return result


# For backward compatibility
def S_euler_simple(T: float, P_max: float, primes: Union[np.ndarray, object],
                   k_max: int = 1) -> float:
    """
    Simple implementation without numerical tricks.
    """
    calculator = SEulerCorrect()
    return calculator.compute(T, P_max, primes, k_max, use_kahan=False)