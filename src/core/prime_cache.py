"""
Efficient prime number generation and caching.

This module provides utilities for generating and storing prime numbers
with memory-efficient access patterns for S(T) computations.
"""

import numpy as np
import pickle
import os
from typing import List, Tuple, Optional


class PrimeCache:
    """
    Efficient prime number storage with lazy loading and memory management.
    """

    def __init__(self, max_prime: int = 1_000_000_000, cache_file: Optional[str] = None):
        """
        Initialize prime cache.

        Parameters:
        -----------
        max_prime : int
            Upper bound for prime generation
        cache_file : str, optional
            Path to cached prime data
        """
        self.max_prime = max_prime
        self.cache_file = cache_file
        self._primes = None
        self._prime_set = None
        self._loaded = False

    @property
    def primes(self) -> np.ndarray:
        """Get array of primes."""
        if not self._loaded:
            self._load()
        return self._primes

    @property
    def prime_set(self) -> set:
        """Get set of primes for fast membership testing."""
        if not self._loaded:
            self._load()
        return self._prime_set

    def _load(self):
        """Load primes from cache or generate them."""
        if self.cache_file and os.path.exists(self.cache_file):
            print(f"Loading primes from cache: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
                self._primes = data['primes']
                self._prime_set = set(self._primes)
            print(f"Loaded {len(self._primes):,} primes")
        else:
            print("Generating primes...")
            self._primes = self._generate_primes()
            self._prime_set = set(self._primes)
            print(f"Generated {len(self._primes):,} primes")

            if self.cache_file:
                self._save()

        self._loaded = True

    def _save(self):
        """Save primes to cache file."""
        print(f"Saving primes to cache: {self.cache_file}")
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump({
                'primes': self._primes,
                'max_prime': self.max_prime
            }, f)

    def _generate_primes(self) -> np.ndarray:
        """Generate primes up to max_prime using segmented sieve."""
        print(f"Generating primes up to {self.max_prime:,}...")
        primes_list = segmented_sieve(self.max_prime)
        return np.array(primes_list, dtype=np.int64)

    def is_prime(self, n: int) -> bool:
        """Check if n is prime."""
        return n in self.prime_set

    def get_primes_up_to(self, limit: int) -> np.ndarray:
        """Get all primes <= limit."""
        if not self._loaded:
            self._load()
        idx = np.searchsorted(self._primes, limit + 1)
        return self._primes[:idx]

    def count_primes_up_to(self, limit: int) -> int:
        """Count primes <= limit."""
        if not self._loaded:
            self._load()
        idx = np.searchsorted(self._primes, limit + 1)
        return idx


def segmented_sieve(limit: int, segment_size: int = 1_000_000) -> List[int]:
    """
    Generate primes up to limit using segmented sieve.

    Parameters:
    -----------
    limit : int
        Upper bound for prime generation
    segment_size : int
        Size of each segment for processing

    Returns:
    --------
    List[int] : List of primes up to limit
    """
    import math

    # Small primes up to sqrt(limit)
    sqrt_limit = int(math.sqrt(limit)) + 1
    small_primes = simple_sieve(sqrt_limit)

    # Segmented sieve for remaining numbers
    primes = small_primes.copy()
    low = sqrt_limit
    high = low + segment_size

    while low <= limit:
        if high > limit:
            high = limit + 1

        mark = np.ones(high - low, dtype=bool)

        for p in small_primes:
            # Find first multiple of p in [low, high)
            first_multiple = ((low + p - 1) // p) * p
            if first_multiple < p * p:
                first_multiple = p * p

            # Mark multiples of p
            for multiple in range(first_multiple, high, p):
                mark[multiple - low] = False

        # Collect primes
        for i in range(low, high):
            if mark[i - low]:
                primes.append(i)

        low += segment_size
        high += segment_size

    return primes


def simple_sieve(limit: int) -> List[int]:
    """Simple Sieve of Eratosthenes for small limits."""
    if limit < 2:
        return []

    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0:2] = False

    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False

    return list(np.where(sieve)[0])