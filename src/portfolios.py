"""
Stress portfolio construction and pathwise Follmer-integral computation.

For Experiment 1: pi^m(t) = m * R_dot(t), using the sine path derivative.
For Experiment 2: pi^m(t, theta) = m * D^-R(t, theta), using the cubic path derivative.

The Follmer (pathwise Riemann–Stieltjes) integral is computed by:
  R_{pi^m}(T) = sum_{k} pi^m(t_{k-1}) * (R(t_k) - R(t_{k-1}))
which converges to the exact integral for fine partitions because pi is continuous.

Custom — Context7 found no library equivalent (paper Section 2 pathwise integrals).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Experiment 1 — stress portfolios on the sine path
# ---------------------------------------------------------------------------

def stress_portfolio_sine(m: float, R_dot: callable) -> callable:
    """
    Construct the stress portfolio pi^m for Experiment 1.

    pi^m(t) = m * R_dot(t)

    Parameters
    ----------
    m : float
        Stress index.
    R_dot : callable
        Exact derivative of the sine path.

    Returns
    -------
    callable
        pi^m(t) -> float or np.ndarray.
    """
    def pi_m(t: float | np.ndarray) -> float | np.ndarray:
        return m * R_dot(t)
    return pi_m


def follmer_integral_sine(
    m: float,
    R_func: callable,
    R_dot: callable,
    T: float,
    n_partition: int = 2 ** 16,
) -> float:
    """
    Compute Follmer return R_{pi^m}(T) for the sine path by Riemann–Stieltjes sum.

    R_{pi^m}(T) = sum pi^m(t_{k-1}) * (R(t_k) - R(t_{k-1}))
                -> m * integral_0^T |R_dot(t)|^2 dt  (classical smooth identity)

    Parameters
    ----------
    m : float
        Stress index.
    R_func : callable
        Sine path R(t).
    R_dot : callable
        Derivative R_dot(t).
    T : float
        Horizon.
    n_partition : int
        Number of partition sub-intervals.

    Returns
    -------
    float
        Follmer return.
    """
    ts = np.linspace(0.0, T, n_partition + 1)
    pi_vals = m * R_dot(ts[:-1])
    dR = np.diff(R_func(ts))
    return float(np.sum(pi_vals * dR))


def wealth_sine(follmer_return: float) -> float:
    """
    Compute pathwise wealth X_{pi^m}(T) = exp(R_{pi^m}(T)) for the sine case.

    Since C_{pi^m}(T) = 0 (path has zero quadratic variation), the wealth formula is:
      X_{pi^m}(T) = exp(R_{pi^m}(T) - 0.5 * C_{pi^m}(T)) = exp(R_{pi^m}(T))

    Parameters
    ----------
    follmer_return : float
        R_{pi^m}(T).

    Returns
    -------
    float
        Pathwise wealth X_{pi^m}(T).
    """
    return float(np.exp(follmer_return))


def gamma_sine(
    A_T: float,
    C_portfolio_T: float,
) -> float:
    """
    Compute Gamma_{pi^m}(T) = A_{pi^m}(T) - 0.5 * C_{pi^m}(T).

    For the sine path, A = C = 0, so Gamma = 0.

    Parameters
    ----------
    A_T : float
        Accumulated trend at T.
    C_portfolio_T : float
        Quadratic-covariation term at T.

    Returns
    -------
    float
        Gamma_{pi^m}(T).
    """
    return A_T - 0.5 * C_portfolio_T


# ---------------------------------------------------------------------------
# Experiment 2 — stress portfolios on the cubic path family
# ---------------------------------------------------------------------------

def stress_portfolio_cubic(m: float, D_minus_R: callable) -> callable:
    """
    Construct the stress portfolio pi^m for Experiment 2.

    pi^m(t, theta) = m * D^-R(t, theta) = 3 * m * theta * ((t-tau)^+)^2

    The portfolio is derived from the pathwise backward derivative, making it
    non-anticipative: it depends only on past path values.

    Parameters
    ----------
    m : float
        Stress index.
    D_minus_R : callable
        Backward derivative D^-R(t, theta).

    Returns
    -------
    callable
        pi^m(t, theta) -> float or np.ndarray.
    """
    def pi_m(t: float | np.ndarray, theta: float) -> float | np.ndarray:
        return m * D_minus_R(t, theta)
    return pi_m


def follmer_integral_cubic(
    m: float,
    R_func: callable,
    D_minus_R: callable,
    T: float,
    theta: float,
    n_partition: int = 2 ** 16,
) -> float:
    """
    Compute Follmer return R_{pi^m}(T, theta) for the cubic path by Riemann–Stieltjes.

    R_{pi^m}(T, theta) = m * integral_0^T |D^-R(t, theta)|^2 dt
                       = (9*m*theta^2/5) * (T-tau)^5

    Parameters
    ----------
    m : float
        Stress index.
    R_func : callable
        Cubic path R(t, theta).
    D_minus_R : callable
        Backward derivative D^-R(t, theta).
    T : float
        Horizon.
    theta : float
        Scenario amplitude.
    n_partition : int
        Partition sub-intervals.

    Returns
    -------
    float
        Follmer return.
    """
    ts = np.linspace(0.0, T, n_partition + 1)
    pi_vals = m * D_minus_R(ts[:-1], theta)
    dR = np.diff(R_func(ts, theta))
    return float(np.sum(pi_vals * dR))


def wealth_cubic(follmer_return: float) -> float:
    """
    Compute pathwise wealth X_{pi^m}(T, theta) = exp(R_{pi^m}(T, theta)).

    C_{pi^m} = 0 for the cubic scenario (C^1 path), so:
      X_{pi^m}(T, theta) = exp(R_{pi^m}(T, theta))

    Parameters
    ----------
    follmer_return : float
        R_{pi^m}(T, theta).

    Returns
    -------
    float
        Pathwise wealth.
    """
    return float(np.exp(follmer_return))


def compute_stress_table_sine(
    m_values: list[int],
    R_func: callable,
    R_dot: callable,
    T: float,
    integral_rdot_sq: float,
) -> dict[str, list]:
    """
    Compute a summary table of terminal wealth for all stress indices (Exp 1).

    Because integral_rdot_sq = (2π/δ)²·T/2 can be very large, X_{pi^m}(T) may
    overflow float64 for any m ≥ 1.  All wealth/capital quantities are therefore
    stored and compared in log-space.  Specifically:
      log_X_T  = m * integral_rdot_sq           (exact, no overflow)
      log_x_m  = -m * integral_rdot_sq
    and the viability identity  x_m * X_T = 1  is checked as
      log_x_m + log_X_T = 0.

    Parameters
    ----------
    m_values : list of int
        Stress indices.
    R_func : callable
        Sine path.
    R_dot : callable
        Derivative.
    T : float
        Horizon.
    integral_rdot_sq : float
        Exact integral_0^T |R_dot(t)|^2 dt.

    Returns
    -------
    dict with keys:
        m, follmer_return_exact, follmer_return_num,
        log_X_T, log_x_m, X_T (may be inf), x_m.
    """
    results = {
        "m": [], "follmer_return_exact": [], "follmer_return_num": [],
        "log_X_T": [], "log_x_m": [], "X_T": [], "x_m": [],
    }
    for m in m_values:
        exact_ret = float(m * integral_rdot_sq)     # = log X_T (exact formula)
        num_ret = follmer_integral_sine(m, R_func, R_dot, T)
        log_X_T = exact_ret                         # direct, no exp overflow
        log_x_m = -exact_ret
        # Store float values (may be inf/0 when exact_ret is large)
        X_T = float(np.exp(exact_ret)) if exact_ret < 700 else float("inf")
        x_m = float(np.exp(-exact_ret)) if exact_ret < 700 else 0.0
        results["m"].append(m)
        results["follmer_return_exact"].append(exact_ret)
        results["follmer_return_num"].append(num_ret)
        results["log_X_T"].append(log_X_T)
        results["log_x_m"].append(log_x_m)
        results["X_T"].append(X_T)
        results["x_m"].append(x_m)
    return results


def compute_stress_table_cubic(
    m_values: list[int],
    theta_values: list[float],
    R_func: callable,
    D_minus_R: callable,
    tau: float,
    T: float,
    theta_min: float,
) -> dict[str, list]:
    """
    Compute a summary table of terminal wealth for stress portfolios (Exp 2).

    Parameters
    ----------
    m_values : list of int
        Stress indices.
    theta_values : list of float
        Representative scenario amplitudes (sampled from Theta_*).
    R_func : callable
        Cubic path.
    D_minus_R : callable
        Backward derivative.
    tau, T, theta_min : float
        Parameters.

    Returns
    -------
    dict with keys: m, theta, X_T_exact, X_T_num, X_T_lower_bound, x_m.
    """
    results = {
        "m": [], "theta": [], "X_T_exact": [], "X_T_num": [],
        "X_T_lower_bound": [], "x_m": [],
    }
    for m in m_values:
        lower_bound_log = (9.0 * m * theta_min ** 2 / 5.0) * (T - tau) ** 5
        x_m = np.exp(-lower_bound_log)
        for theta in theta_values:
            exact_ret = (9.0 * m * theta ** 2 / 5.0) * (T - tau) ** 5
            num_ret = follmer_integral_cubic(m, R_func, D_minus_R, T, theta)
            X_T_exact = float(np.exp(exact_ret))
            X_T_num = float(np.exp(num_ret))
            results["m"].append(m)
            results["theta"].append(theta)
            results["X_T_exact"].append(X_T_exact)
            results["X_T_num"].append(X_T_num)
            results["X_T_lower_bound"].append(float(np.exp(lower_bound_log)))
            results["x_m"].append(float(x_m))
    return results
