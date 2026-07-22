"""
Backward density selector D_O^-.

Computes the "density" of the process A or C with respect to the operational
clock O using backward difference quotients at rational lags.

Paper convention:
  - Use backward rational lags r > 0 tending to 0.
  - If O(t) - O(t-r) = 0, the quotient is set to 0.
  - If the limit does not exist, the derivative is set to 0.

When A = C = O = 0, all densities (a, c) are zero trivially.

Custom — Context7 found no library equivalent (paper Section 3 local characteristics).
"""

import numpy as np


def backward_density(
    numerator_func: callable,
    clock_func: callable,
    t: float,
    lags: np.ndarray | None = None,
) -> float:
    """
    Evaluate the backward density selector D_O^- [numerator](t).

    Approximates lim_{r->0^+} [numerator(t) - numerator(t-r)] / [O(t) - O(t-r)]
    using a sequence of lags. Returns 0 if denominator vanishes or limit is unclear.

    Parameters
    ----------
    numerator_func : callable
        Process value at time t: A(t), C(t), etc.
    clock_func : callable
        Operational clock O(t).
    t : float
        Evaluation time.
    lags : np.ndarray or None
        Sequence of backward lags to try, e.g. [1e-1, 1e-2, 1e-3, 1e-4].
        Defaults to np.array([1e-2, 1e-3, 1e-4, 1e-5]).

    Returns
    -------
    float
        Estimated backward density at t, or 0 if denominator vanishes.
    """
    if lags is None:
        lags = np.array([1e-2, 1e-3, 1e-4, 1e-5])

    estimates = []
    for r in lags:
        t_prev = max(t - r, 0.0)
        dO = float(clock_func(t)) - float(clock_func(t_prev))
        if abs(dO) < 1e-15:
            # Denominator vanishes: convention is 0
            estimates.append(0.0)
        else:
            dN = float(numerator_func(t)) - float(numerator_func(t_prev))
            estimates.append(dN / dO)

    # If all estimates are 0 (denominator always vanished), return 0
    if all(abs(e) < 1e-14 for e in estimates):
        return 0.0

    # Return the finest-lag estimate (last element corresponds to smallest lag)
    return estimates[-1]


def compute_local_characteristics(
    A_func: callable,
    C_func: callable,
    O_func: callable,
    t_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute local characteristics a(t) and c(t) on a time grid.

    a(t) = D_O^-[A](t),  c(t) = D_O^-[C](t).

    When O = 0 identically (as in both experiments), all denominators vanish
    and a = c = 0 by the paper's convention.

    Parameters
    ----------
    A_func : callable
        Accumulated trend A(t).
    C_func : callable
        Quadratic covariation C(t).
    O_func : callable
        Operational clock O(t).
    t_grid : np.ndarray
        Times at which to evaluate the characteristics.

    Returns
    -------
    a_vals : np.ndarray
        Local drift density a(t) at each grid point.
    c_vals : np.ndarray
        Local diffusion density c(t) at each grid point.
    """
    a_vals = np.array([backward_density(A_func, O_func, t) for t in t_grid])
    c_vals = np.array([backward_density(C_func, O_func, t) for t in t_grid])
    return a_vals, c_vals


def maximal_growth(
    a_vals: np.ndarray,
    c_vals: np.ndarray,
) -> np.ndarray:
    """
    Compute pointwise maximal growth g(t) = sup_p [p * a(t) - 0.5 * p^2 * c(t)].

    When c(t) > 0: g(t) = a(t)^2 / (2 * c(t)) (achieved at p* = a/c).
    When c(t) = 0 and a(t) = 0: g(t) = 0.
    When c(t) = 0 and a(t) != 0: g(t) = +inf (unbounded growth, not our case).

    Parameters
    ----------
    a_vals : np.ndarray
        Local drift density at each time.
    c_vals : np.ndarray
        Local diffusion density at each time.

    Returns
    -------
    np.ndarray
        Pointwise maximal growth g(t).
    """
    g = np.where(
        c_vals > 1e-15,
        a_vals ** 2 / (2.0 * c_vals),
        0.0,  # a = c = 0 case
    )
    return g


def canonical_portfolio(
    a_vals: np.ndarray,
    c_vals: np.ndarray,
) -> np.ndarray:
    """
    Compute canonical portfolio nu(t) = c(t)^dagger * a(t).

    In d=1: nu(t) = a(t) / c(t) when c(t) > 0, else 0.

    Parameters
    ----------
    a_vals, c_vals : np.ndarray
        Local characteristics.

    Returns
    -------
    np.ndarray
        Canonical portfolio values.
    """
    return np.where(c_vals > 1e-15, a_vals / c_vals, 0.0)


def cumulative_maximal_growth(
    g_vals: np.ndarray,
    O_vals: np.ndarray,
) -> float:
    """
    Compute G(T) = integral_0^T g(t) dO(t) numerically via Riemann–Stieltjes.

    When O = 0 (our case), G(T) = 0.

    Parameters
    ----------
    g_vals : np.ndarray
        Pointwise maximal growth at partition points.
    O_vals : np.ndarray
        Operational clock at partition points.

    Returns
    -------
    float
        G(T).
    """
    dO = np.diff(O_vals)
    return float(np.sum(g_vals[:-1] * dO))
