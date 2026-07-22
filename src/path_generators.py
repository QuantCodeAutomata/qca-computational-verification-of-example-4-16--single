"""
Analytic path generators for both experiments.

Exp 1 (Example 4.16): R(t) = sin(2*pi*t/delta), a single smooth delta-periodic scenario.
Exp 2 (Example 4.17): R(t, theta) = theta * ((t - tau)^+)^3, a family of cubic scenarios.

Custom — Context7 found no library equivalent (paper Example 4.16/4.17).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Experiment 1: Singleton sine scenario
# ---------------------------------------------------------------------------

def make_sine_path(h: float, N: int) -> tuple[float, callable, callable, callable]:
    """
    Construct the analytic sine return path for Experiment 1.

    Parameters
    ----------
    h : float
        Observation window length > 0.
    N : int
        Dyadic resolution integer >= 1.

    Returns
    -------
    delta : float
        Cell width = h * 2^{-N}.
    R : callable
        R(t) = sin(2*pi*t / delta), defined on [-h, T].
    R_dot : callable
        R_dot(t) = (2*pi/delta) * cos(2*pi*t / delta).
    R_window : callable
        R^{t,h}(u) = R(t - h + h*u), u in [0,1].
    """
    delta = h * 2.0 ** (-N)
    omega = 2.0 * np.pi / delta

    def R(t: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the delta-periodic sine path at time t."""
        return np.sin(omega * t)

    def R_dot(t: float | np.ndarray) -> float | np.ndarray:
        """Exact derivative of R."""
        return omega * np.cos(omega * t)

    def R_window(t: float, u: float | np.ndarray) -> float | np.ndarray:
        """Window map R^{t,h}(u) = R(t - h + h*u)."""
        return R(t - h + h * u)

    return delta, R, R_dot, R_window


def sine_follmer_integral_exact(h: float, N: int, T: float, m: float) -> float:
    """
    Exact Follmer return for the stress portfolio pi^m in Experiment 1.

    R_{pi^m}(T) = m * integral_0^T |R_dot(t)|^2 dt
                = m * (2*pi/delta)^2 * T / 2

    Parameters
    ----------
    h, N : float, int
        Window length and dyadic resolution.
    T : float
        Time horizon.
    m : float
        Stress index.

    Returns
    -------
    float
        Exact Follmer return.
    """
    delta = h * 2.0 ** (-N)
    omega = 2.0 * np.pi / delta
    # integral_0^T cos^2(omega*t) dt = T/2 (when omega*T is an integer multiple of pi)
    # For a general T: exact value using antiderivative
    integral_rdot_sq = (omega ** 2) * (T / 2.0 + np.sin(2.0 * omega * T) / (4.0 * omega))
    return m * integral_rdot_sq


def sine_follmer_integral_numerical(
    h: float, N: int, T: float, m: float, n_points: int = 100_000
) -> float:
    """
    Numerical approximation of the Follmer return for the sine path (Exp 1).

    Uses a fine Riemann sum: sum pi^m(t_{k-1}) * [R(t_k) - R(t_{k-1})].

    Parameters
    ----------
    h, N : float, int
        Window and resolution.
    T : float
        Time horizon.
    m : float
        Stress index.
    n_points : int
        Number of quadrature sub-intervals.

    Returns
    -------
    float
        Numerical Follmer return.
    """
    delta, R, R_dot, _ = make_sine_path(h, N)
    ts = np.linspace(0.0, T, n_points + 1)
    pi_vals = m * R_dot(ts[:-1])       # pi^m evaluated at left endpoints
    dR = np.diff(R(ts))
    return float(np.sum(pi_vals * dR))


# ---------------------------------------------------------------------------
# Experiment 2: Multi-scenario cubic family
# ---------------------------------------------------------------------------

def make_cubic_path(tau: float, h: float) -> tuple[callable, callable, callable]:
    """
    Construct the cubic scenario path family for Experiment 2.

    R(t, theta) = theta * ((t - tau)^+)^3, where x^+ = max(x, 0).

    Parameters
    ----------
    tau : float
        Branch time in (T - h, T).
    h : float
        Observation window length > 0.

    Returns
    -------
    R : callable
        R(t, theta) = theta * max(t - tau, 0)^3.
    D_minus_R : callable
        Pathwise backward derivative: 3*theta*max(t-tau,0)^2.
    R_window : callable
        R^{t,h}(u, theta) = R(t - h + h*u, theta).
    """

    def R(t: float | np.ndarray, theta: float) -> float | np.ndarray:
        """Evaluate the cubic scenario path."""
        return theta * np.maximum(t - tau, 0.0) ** 3

    def D_minus_R(t: float | np.ndarray, theta: float) -> float | np.ndarray:
        """
        Pathwise backward derivative of R w.r.t. t.
        = 3 * theta * ((t - tau)^+)^2.
        """
        return 3.0 * theta * np.maximum(t - tau, 0.0) ** 2

    def R_window(t: float, u: float | np.ndarray, theta: float) -> float | np.ndarray:
        """Window map R^{t,h}(u, theta) = R(t - h + h*u, theta)."""
        return R(t - h + h * u, theta)

    return R, D_minus_R, R_window


def cubic_follmer_integral_exact(
    tau: float, T: float, theta: float, m: float
) -> float:
    """
    Exact Follmer return for the stress portfolio pi^m in Experiment 2.

    R_{pi^m}(T, theta) = m * integral_0^T |D^-R(t, theta)|^2 dt
                       = m * 9 * theta^2 * integral_tau^T (t-tau)^4 dt
                       = (9 * m * theta^2 / 5) * (T - tau)^5

    Parameters
    ----------
    tau : float
        Branch time.
    T : float
        Horizon.
    theta : float
        Scenario amplitude.
    m : float
        Stress index.

    Returns
    -------
    float
        Exact Follmer return.
    """
    return (9.0 * m * theta ** 2 / 5.0) * (T - tau) ** 5


def cubic_follmer_integral_numerical(
    tau: float, T: float, theta: float, m: float,
    h: float, n_points: int = 100_000
) -> float:
    """
    Numerical Follmer return using Riemann–Stieltjes sum for Experiment 2.

    Parameters
    ----------
    tau, T, theta, m, h : floats
        Path and stress parameters.
    n_points : int
        Sub-intervals.

    Returns
    -------
    float
        Numerical Follmer return.
    """
    R, D_minus_R, _ = make_cubic_path(tau, h)
    ts = np.linspace(0.0, T, n_points + 1)
    pi_vals = m * D_minus_R(ts[:-1], theta)
    dR = np.diff(R(ts, theta))
    return float(np.sum(pi_vals * dR))
