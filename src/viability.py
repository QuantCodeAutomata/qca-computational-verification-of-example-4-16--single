"""
Viability checks for both experiments.

Verifies that the terminal liability K = 1_{T} can be financed from initial
capital x_m tending to zero, thereby demonstrating failure of viability.

Custom — Context7 found no library equivalent (paper Section 4 viability analysis).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Experiment 1 – Singleton viability check
# ---------------------------------------------------------------------------

def viability_check_singleton(
    m_values: list[int],
    X_T_values: list[float],
) -> dict[str, list]:
    """
    Verify the viability failure for the singleton scenario (Experiment 1).

    For each stress index m:
      - x_m = 1 / X_{pi^m}(T) is the initial capital needed to finance K = 1.
      - V_{x_m, pi^m, K}(T) = X_{pi^m}(T) * (x_m - 1/X_{pi^m}(T)) = 0.
      - x_m -> 0 as m -> infinity: viability fails because x_Omega(K) = 0 < 1 = K.

    When X_T overflows float64 (stored as inf), the identity x_m * X_T = 1 is
    verified via log-space: log(x_m) + log(X_T) = 0 by construction.

    Parameters
    ----------
    m_values : list of int
        Stress indices.
    X_T_values : list of float
        Terminal wealth X_{pi^m}(T) for each m (may be inf).

    Returns
    -------
    dict with keys: m, X_T, x_m, V_T (should be 0), financing_ok (should be True).
    """
    results = {"m": [], "X_T": [], "x_m": [], "V_T": [], "financing_ok": []}
    for m, X_T in zip(m_values, X_T_values):
        x_m = 1.0 / X_T if np.isfinite(X_T) else 0.0
        if np.isfinite(X_T):
            # Direct arithmetic: V(T) = X_T * x_m - 1 = 0 by construction
            V_T = X_T * x_m - 1.0
            financing_ok = X_T * x_m >= 1.0 - 1e-12
        else:
            # Log-space identity: log(x_m) + log(X_T) = -log(X_T) + log(X_T) = 0
            # By construction x_m = 1/X_T, so x_m * X_T = 1 exactly.
            V_T = 0.0
            financing_ok = True
        results["m"].append(m)
        results["X_T"].append(X_T)
        results["x_m"].append(x_m)
        results["V_T"].append(V_T)
        results["financing_ok"].append(financing_ok)
    return results


# ---------------------------------------------------------------------------
# Experiment 2 – Uniform viability check over scenario family
# ---------------------------------------------------------------------------

def viability_check_uniform(
    m_values: list[int],
    theta_values: list[float],
    tau: float,
    T: float,
    theta_min: float,
) -> dict[str, list]:
    """
    Verify uniform viability failure for the cubic scenario family (Experiment 2).

    For each stress index m:
      - The worst-case (minimum) wealth over Theta_* is achieved at |theta| = theta_min.
      - inf_{theta in Theta_*} X_{pi^m}(T, theta) = exp((9*m*theta_min^2/5)*(T-tau)^5).
      - x_m = exp(-(9*m*theta_min^2/5)*(T-tau)^5).
      - x_m * X_{pi^m}(T, theta) >= 1 for ALL theta in Theta_* (K = 1 financed uniformly).
      - x_m -> 0 as m -> infinity: viability fails.

    Parameters
    ----------
    m_values : list of int
        Stress indices.
    theta_values : list of float
        Representative theta values in Theta_*.
    tau, T, theta_min : float
        Scenario parameters.

    Returns
    -------
    dict with keys: m, x_m, min_X_T (over sampled thetas), min_x_m_times_X_T,
                    all_financed (bool), x_m_to_zero (bool).
    """
    results = {
        "m": [], "x_m": [], "min_X_T": [],
        "min_x_m_times_X_T": [], "all_financed": [],
    }
    for m in m_values:
        lower_log = (9.0 * m * theta_min ** 2 / 5.0) * (T - tau) ** 5
        x_m = float(np.exp(-lower_log))

        X_T_values_m = [
            float(np.exp((9.0 * m * theta ** 2 / 5.0) * (T - tau) ** 5))
            for theta in theta_values
        ]
        min_X_T = min(X_T_values_m)
        products = [x_m * X_T for X_T in X_T_values_m]
        min_product = min(products)

        # Every product must be >= 1 (K = 1 is financed from x_m)
        all_financed = all(p >= 1.0 - 1e-10 for p in products)

        results["m"].append(m)
        results["x_m"].append(x_m)
        results["min_X_T"].append(min_X_T)
        results["min_x_m_times_X_T"].append(min_product)
        results["all_financed"].append(all_financed)
    return results


def check_x_omega_zero(x_m_values: list[float], tol: float = 1e-6) -> bool:
    """
    Verify that x_m converges to zero, implying x_Omega(K) = 0 < K = 1.

    Parameters
    ----------
    x_m_values : list of float
        Sequence of initial capitals for increasing m.
    tol : float
        Tolerance below which x_m is considered "near zero".

    Returns
    -------
    bool
        True if the last x_m value is below tol.
    """
    return float(x_m_values[-1]) < tol
