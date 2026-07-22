"""
Non-anticipative extractors for local path characteristics.

Implements:
  - Faber–Schauder hat functions S_k^{(m)} on [0,1]
  - Faber–Schauder projection P_N x on [0,1]
  - Last-cell slope extractor D_N^{last}(x) = 2^N [(P_N x)(1) - (P_N x)(1 - 2^{-N})]
  - Lag extractor D_lag(x) = x(0)

Custom — Context7 found no library equivalent (paper Section 3 non-anticipative extractors).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Faber–Schauder basis
# ---------------------------------------------------------------------------

def faber_schauder_hat(m: int, k: int, u: float | np.ndarray) -> float | np.ndarray:
    """
    Standard Faber–Schauder hat function S_k^{(m)} at resolution level m.

    Defined on [0,1]:
        S_k^{(m)}(u) = max(0, 1 - |2^{m+1} * u - (2k + 1)|)

    The support is [k/2^m, (k+1)/2^m] and the peak at (2k+1)/2^{m+1} has height 1.

    Parameters
    ----------
    m : int
        Resolution level (m >= 0).
    k : int
        Position index (0 <= k < 2^m).
    u : float or np.ndarray
        Evaluation point(s) in [0,1].

    Returns
    -------
    float or np.ndarray
        Hat function value at u.
    """
    return np.maximum(0.0, 1.0 - np.abs(2.0 ** (m + 1) * u - (2 * k + 1)))


def faber_schauder_coefficient(
    x_func: callable, m: int, k: int
) -> float:
    """
    Compute Faber–Schauder coefficient theta_{m,k}^{(x)}.

    theta_{m,k}^{(x)} = x((2k+1)/2^{m+1}) - [x(k/2^m) + x((k+1)/2^m)] / 2

    Parameters
    ----------
    x_func : callable
        Function x : [0,1] -> R evaluated at dyadic nodes.
    m : int
        Resolution level.
    k : int
        Position index.

    Returns
    -------
    float
        Coefficient value.
    """
    node_mid = (2 * k + 1) / 2.0 ** (m + 1)
    node_left = k / 2.0 ** m
    node_right = (k + 1) / 2.0 ** m
    return float(x_func(node_mid) - 0.5 * (x_func(node_left) + x_func(node_right)))


def project_faber_schauder(
    x_func: callable, N: int, u: float | np.ndarray
) -> float | np.ndarray:
    """
    Compute the N-th Faber–Schauder projection of x on [0,1].

    P_N x(u) = x(0) + u*(x(1)-x(0))
               + sum_{m=0}^{N-1} sum_{k=0}^{2^m-1} theta_{m,k}^{(x)} * S_k^{(m)}(u)

    Because P_N interpolates x at every dyadic node j/2^N (j=0,...,2^N),
    in particular (P_N x)(1) = x(1) and (P_N x)(1-2^{-N}) = x(1-2^{-N}).

    Parameters
    ----------
    x_func : callable
        Function x : [0,1] -> R. Must support evaluation at dyadic points.
    N : int
        Projection depth (number of refinement levels).
    u : float or np.ndarray
        Evaluation point(s) in [0,1].

    Returns
    -------
    float or np.ndarray
        P_N x evaluated at u.
    """
    u = np.asarray(u, dtype=float)
    result = x_func(0.0) + u * (x_func(1.0) - x_func(0.0))
    for m in range(N):
        for k in range(2 ** m):
            theta = faber_schauder_coefficient(x_func, m, k)
            result = result + theta * faber_schauder_hat(m, k, u)
    return result


def last_cell_extractor_analytic(
    R_func: callable, t: float, h: float, N: int
) -> float:
    """
    Compute D_N^{last}(R^{t,h}) using direct analytic substitution.

    Since P_N interpolates at dyadic nodes, (P_N x)(1) = x(1) and
    (P_N x)(1 - 2^{-N}) = x(1 - 2^{-N}) exactly.

    D_N^{last}(R^{t,h}) = 2^N * [(P_N R^{t,h})(1) - (P_N R^{t,h})(1 - 2^{-N})]
                        = 2^N * [R(t) - R(t - delta)]

    where delta = h * 2^{-N}.

    For the sine path in Exp 1, R is delta-periodic so this equals 0.

    Parameters
    ----------
    R_func : callable
        Analytic path R(t) defined on [-h, T].
    t : float
        Current time (t >= 0).
    h : float
        Window length.
    N : int
        Resolution level.

    Returns
    -------
    float
        D_N^{last}(R^{t,h}).
    """
    delta = h * 2.0 ** (-N)
    val_at_1 = float(R_func(t))              # (P_N R^{t,h})(1) = R^{t,h}(1) = R(t)
    val_at_1_minus = float(R_func(t - delta))  # (P_N R^{t,h})(1-2^{-N}) = R(t-delta)
    return (2.0 ** N) * (val_at_1 - val_at_1_minus)


def alpha_sine(R_func: callable, t: float, h: float, N: int) -> float:
    """
    Trend signal alpha^{h,D}(t) = (1/h) * D_N^{last}(R^{t,h}).

    Parameters
    ----------
    R_func : callable
        Analytic sine path.
    t : float
        Current time.
    h : float
        Window length.
    N : int
        Resolution.

    Returns
    -------
    float
        alpha^{h,D}(t).
    """
    return last_cell_extractor_analytic(R_func, t, h, N) / h


# ---------------------------------------------------------------------------
# Lag extractor (Experiment 2)
# ---------------------------------------------------------------------------

def lag_extractor(x_func: callable, t: float, h: float) -> float:
    """
    Lag (left-endpoint) extractor D_lag(R^{t,h}) = R^{t,h}(0) = R(t - h).

    Used in Experiment 2 to produce alpha^{h,D_lag}(t) = (1/h) * R(t - h).

    Parameters
    ----------
    x_func : callable
        Analytic path R(t, ...) or R(t).
    t : float
        Current time.
    h : float
        Window length.

    Returns
    -------
    float
        D_lag(R^{t,h}) = R(t - h).
    """
    return float(x_func(t - h))


def alpha_cubic(
    R_func: callable, t: float, h: float
) -> float:
    """
    Trend signal alpha^{h,D_lag}(t) = (1/h) * D_lag(R^{t,h}) = (1/h) * R(t - h).

    For the cubic scenario family with tau > T - h, t - h < tau for all t in [0,T],
    so R(t - h) = 0 and alpha = 0 identically.

    Parameters
    ----------
    R_func : callable
        Scenario path R(t) for a fixed theta (single-argument).
    t : float
        Current time.
    h : float
        Window length.

    Returns
    -------
    float
        alpha^{h,D_lag}(t).
    """
    return float(R_func(t - h)) / h
