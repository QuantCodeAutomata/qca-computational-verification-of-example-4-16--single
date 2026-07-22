"""
Partition generators for quadratic-variation approximations.

Provides refining dyadic partition sequences Pi_n of [0, T].
Custom — Context7 found no library equivalent (paper Section 2 partition setup).
"""

import numpy as np


def dyadic_partition(T: float, n: int) -> np.ndarray:
    """
    Generate the n-th dyadic partition of [0, T].

    Pi_n = {k * T / 2^n : k = 0, 1, ..., 2^n}

    Parameters
    ----------
    T : float
        Right endpoint of the interval.
    n : int
        Refinement level. Returns 2^n + 1 equally-spaced points.

    Returns
    -------
    np.ndarray
        Partition points of shape (2^n + 1,).
    """
    assert T > 0, "T must be positive"
    assert n >= 0, "Refinement level n must be non-negative"
    return np.linspace(0.0, T, 2 ** n + 1)


def dyadic_partition_sequence(T: float, n_levels: int) -> list[np.ndarray]:
    """
    Generate a list of successively refined dyadic partitions.

    Parameters
    ----------
    T : float
        Right endpoint.
    n_levels : int
        Number of refinement levels (0, 1, ..., n_levels - 1).

    Returns
    -------
    list of np.ndarray
        Partitions Pi_0, Pi_1, ..., Pi_{n_levels-1}.
    """
    return [dyadic_partition(T, n) for n in range(n_levels)]


def uniform_partition(T: float, n_steps: int) -> np.ndarray:
    """
    Generate a uniform partition of [0, T] with n_steps sub-intervals.

    Parameters
    ----------
    T : float
        Right endpoint.
    n_steps : int
        Number of sub-intervals.

    Returns
    -------
    np.ndarray
        Partition points of shape (n_steps + 1,).
    """
    assert n_steps >= 1, "n_steps must be at least 1"
    return np.linspace(0.0, T, n_steps + 1)
