"""
Discrete quadratic-variation approximations along partition sequences.

For C^1 paths, [R, R]_Pi = lim_{n->inf} C_n(t) = 0.
Verifies this convergence numerically and provides cumulative QV arrays.

Custom — Context7 found no library equivalent (paper Section 2 pathwise QV).
"""

import numpy as np


def discrete_qv_cumulative(
    path_func: callable,
    partition: np.ndarray,
    t_grid: np.ndarray,
    *path_args,
) -> np.ndarray:
    """
    Compute discrete quadratic-variation approximation C_n(t) along a partition.

    C_n(t) = sum_{t_k in Pi, t_k <= t} (R(t_k) - R(t_{k-1}))^2

    Parameters
    ----------
    path_func : callable
        R(t, *path_args) or R(t).
    partition : np.ndarray
        Partition points (Pi_n).
    t_grid : np.ndarray
        Times at which to evaluate C_n(t).
    *path_args :
        Extra arguments forwarded to path_func (e.g., theta for cubic path).

    Returns
    -------
    np.ndarray
        C_n evaluated at each t in t_grid, shape (len(t_grid),).
    """
    if path_args:
        R_vals = path_func(partition, *path_args)
    else:
        R_vals = path_func(partition)

    increments_sq = np.diff(R_vals) ** 2      # (R(t_k) - R(t_{k-1}))^2
    # For each t, sum increments where the right endpoint t_k <= t
    t_k_rights = partition[1:]
    C_n = np.zeros(len(t_grid))
    for i, t in enumerate(t_grid):
        mask = t_k_rights <= t + 1e-14
        C_n[i] = np.sum(increments_sq[mask])
    return C_n


def discrete_qv_total(
    path_func: callable,
    partition: np.ndarray,
    *path_args,
) -> float:
    """
    Compute total discrete quadratic variation C_n(T) along a partition.

    Parameters
    ----------
    path_func : callable
        R(t, *path_args).
    partition : np.ndarray
        Partition points.
    *path_args :
        Additional arguments for path_func.

    Returns
    -------
    float
        Total quadratic variation.
    """
    if path_args:
        R_vals = path_func(partition, *path_args)
    else:
        R_vals = path_func(partition)
    return float(np.sum(np.diff(R_vals) ** 2))


def qv_convergence_table(
    path_func: callable,
    T: float,
    n_levels: int,
    *path_args,
) -> dict[str, list]:
    """
    Build a convergence table of total QV under successive dyadic refinements.

    Parameters
    ----------
    path_func : callable
        R(t, *path_args).
    T : float
        Horizon.
    n_levels : int
        Number of dyadic refinement levels to test.
    *path_args :
        Extra arguments for path_func.

    Returns
    -------
    dict with keys 'n', 'n_steps', 'C_n_T' listing refinement levels and QV values.
    """
    ns, steps, qv_vals = [], [], []
    for n in range(1, n_levels + 1):
        partition = np.linspace(0.0, T, 2 ** n + 1)
        qv = discrete_qv_total(path_func, partition, *path_args)
        ns.append(n)
        steps.append(2 ** n)
        qv_vals.append(qv)
    return {"n": ns, "n_steps": steps, "C_n_T": qv_vals}
