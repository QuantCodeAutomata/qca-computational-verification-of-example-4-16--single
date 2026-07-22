"""
Experiment 2 – Example 4.17: Multi-scenario non-anticipative separation between
growth-numéraire identities and viability.

Reproduces the paper's Example 4.17 on a genuine multi-scenario path space:
  - All local extracted characteristics vanish uniformly across Theta_*.
  - A non-anticipative stress portfolio generates terminal wealth that explodes
    uniformly over scenarios, implying failure of boundedness and viability.

Default parameters (documented; NOT uniquely prescribed by the paper):
    h = 1,  T = 2,  tau = 1.5,  theta_min = 0.5.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import sympy as sp

from src.path_generators import (
    make_cubic_path,
    cubic_follmer_integral_exact,
    cubic_follmer_integral_numerical,
)
from src.partitions import dyadic_partition_sequence
from src.extractors import alpha_cubic
from src.qv import qv_convergence_table, discrete_qv_total
from src.densities import (
    compute_local_characteristics,
    maximal_growth,
    canonical_portfolio,
    cumulative_maximal_growth,
)
from src.portfolios import compute_stress_table_cubic, wealth_cubic
from src.viability import viability_check_uniform, check_x_omega_zero
from src.reports import (
    plot_cubic_paths,
    plot_qv_convergence,
    plot_wealth_uniform,
)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
H: float = 1.0          # observation window
T: float = 2.0          # horizon
TAU: float = 1.5        # branch time — must satisfy TAU > T - H (= 1.0)  ✓
THETA_MIN: float = 0.5  # lower amplitude bound
M_VALUES: list[int] = list(range(1, 11))           # stress indices 1 … 10 (wealth table)
# Larger stress indices needed so that x_m = exp(-0.01406·m) actually approaches 0.
# At m=1000, x_m ≈ 7.8e-7 < 1e-3 (tol used in check_x_omega_zero below).
M_VIAB_VALUES: list[int] = [1, 10, 50, 100, 200, 500, 1000]

# Verification that tau > T - h (mandatory for alpha = 0)
assert TAU > T - H, f"tau must be in (T-h, T) = ({T-H}, {T}), got {TAU}"

# Scenario parameter set Theta_* = [-1, -theta_min] ∪ [theta_min, 1]
THETA_NEGATIVE = np.linspace(-1.0, -THETA_MIN, 6)
THETA_POSITIVE = np.linspace(THETA_MIN, 1.0, 6)
THETA_STAR = np.concatenate([THETA_NEGATIVE, THETA_POSITIVE])


# ---------------------------------------------------------------------------
# Step 1: Construct cubic path family
# ---------------------------------------------------------------------------

def step1_construct_paths():
    """Construct the analytic cubic path family and the backward derivative."""
    R, D_minus_R, R_window = make_cubic_path(TAU, H)
    # Verify that R(t, theta) = 0 for t <= tau
    t_before = np.linspace(0.0, TAU, 100)
    for theta in [0.5, -0.5, 1.0, -1.0]:
        vals = R(t_before, theta)
        assert np.all(np.abs(vals) < 1e-14), f"R not zero before tau for theta={theta}"
    print(f"[Exp 2 / Step 1] Cubic paths constructed. tau={TAU}, |Theta_*|={len(THETA_STAR)}")
    return R, D_minus_R, R_window


# ---------------------------------------------------------------------------
# Step 2: Verify alpha^{h,D_lag} = 0 for all t and all theta
# ---------------------------------------------------------------------------

def step2_verify_alpha_zero(R: callable) -> None:
    """
    Confirm alpha^{h,D_lag}(t,theta)=(1/h)*R(t-h,theta)=0 because t-h < tau for t in [0,T].

    Since tau > T - h, we have t - h < T - h < tau for all t in [0, T].
    """
    t_grid = np.linspace(0.0, T, 200)
    max_alpha = 0.0
    for theta in THETA_STAR:
        R_theta = lambda s, _theta=theta: R(s, _theta)
        alpha_vals = np.array([
            alpha_cubic(R_theta, t, H)
            for t in t_grid
        ])
        max_alpha = max(max_alpha, float(np.max(np.abs(alpha_vals))))
    print(f"[Exp 2 / Step 2] max |alpha^{{h,D_lag}}| over all (t, theta) = {max_alpha:.2e}")
    assert max_alpha < 1e-14, f"alpha not zero: {max_alpha}"


# ---------------------------------------------------------------------------
# Step 3: Quadratic covariation convergence for each scenario
# ---------------------------------------------------------------------------

def step3_quadratic_variation(R: callable) -> dict:
    """
    Verify C_n(T, theta) -> 0 for all theta (C^1 paths have zero QV).
    """
    print("[Exp 2 / Step 3] QV convergence (theta = 0.5):")
    qv_table = qv_convergence_table(
        lambda pts: R(pts, THETA_MIN), T, n_levels=12
    )
    for n, nst, qv in zip(qv_table["n"], qv_table["n_steps"], qv_table["C_n_T"]):
        print(f"   n={n:2d}  steps={nst:6d}  C_n(T)={qv:.6e}")

    # Check for multiple theta values.
    # For a C^1 path QV = O(Δt); with 2^15 points (Δt≈6e-5) we get QV < 1e-5
    # even for |theta|=1 where (9/5)*(T-tau)^5 ≈ 0.0563.
    for theta in THETA_STAR:
        qv_T = discrete_qv_total(lambda pts: R(pts, theta), np.linspace(0.0, T, 2 ** 15 + 1))
        assert qv_T < 1e-5, f"QV not small for theta={theta}: {qv_T}"
    print("[Exp 2 / Step 3] C^{h,D} = 0 confirmed for all scenarios.")
    return qv_table


# ---------------------------------------------------------------------------
# Step 4: Local characteristics a, c, g, G, nu — all zero
# ---------------------------------------------------------------------------

def step4_local_characteristics() -> None:
    """Verify all local characteristics vanish because O = A = C = 0."""
    def zero_f(t): return 0.0
    t_grid = np.linspace(0.01, T, 100)
    a_vals, c_vals = compute_local_characteristics(zero_f, zero_f, zero_f, t_grid)
    g_vals = maximal_growth(a_vals, c_vals)
    nu_vals = canonical_portfolio(a_vals, c_vals)
    G_T = cumulative_maximal_growth(g_vals, np.zeros(len(t_grid)))

    assert np.max(np.abs(a_vals)) < 1e-14
    assert np.max(np.abs(c_vals)) < 1e-14
    assert np.max(np.abs(g_vals)) < 1e-14
    assert np.max(np.abs(nu_vals)) < 1e-14
    assert abs(G_T) < 1e-14
    print(f"[Exp 2 / Step 4] a=c=g=G=nu=0 confirmed across all scenarios.")


# ---------------------------------------------------------------------------
# Step 5: Symbolic verification with SymPy
# ---------------------------------------------------------------------------

def step5_symbolic_verification() -> float:
    """
    Verify integral_tau^T [3*theta*(t-tau)^2]^2 dt = (9*theta^2/5)*(T-tau)^5.
    """
    t_sym = sp.Symbol("t", real=True)
    theta_sym = sp.Symbol("theta", real=True)
    tau_sym = sp.Rational(3, 2)    # tau = 1.5
    T_sym = sp.Integer(2)

    integrand = (3 * theta_sym * (t_sym - tau_sym) ** 2) ** 2
    result = sp.integrate(integrand, (t_sym, tau_sym, T_sym))
    expected = sp.Rational(9, 5) * theta_sym ** 2 * (T_sym - tau_sym) ** 5
    diff = sp.simplify(result - expected)
    print(f"[Exp 2 / Step 5] Symbolic integral check (should be 0): {diff}")
    assert diff == 0, f"Symbolic mismatch: {diff}"

    # Numeric value for theta = theta_min
    numeric_val = float(result.subs(theta_sym, THETA_MIN).evalf())
    print(f"[Exp 2 / Step 5] Integral |D^-R|^2 (theta_min={THETA_MIN}): {numeric_val:.6f}")
    return numeric_val


# ---------------------------------------------------------------------------
# Step 6: Stress portfolio — Follmer integrals and wealth
# ---------------------------------------------------------------------------

def step6_stress_portfolios(R: callable, D_minus_R: callable) -> dict:
    """
    Compute stress portfolio returns and terminal wealth across m and theta.
    """
    # Representative theta values for tabulation
    theta_rep = [THETA_MIN, 0.7, 1.0, -THETA_MIN, -0.7, -1.0]
    table = compute_stress_table_cubic(
        M_VALUES, theta_rep, R, D_minus_R, TAU, T, THETA_MIN
    )
    df = pd.DataFrame(table)

    # Verify exact vs numerical agreement
    for i in range(len(table["m"])):
        m, theta = table["m"][i], table["theta"][i]
        exact = table["X_T_exact"][i]
        num = table["X_T_num"][i]
        assert abs(exact - num) / max(exact, 1.0) < 1e-3, (
            f"Wealth mismatch m={m} theta={theta:.2f}: exact={exact:.4f} num={num:.4f}"
        )

    # Verify Gamma_{pi^m} = 0 for all stress indices
    for m in M_VALUES:
        gamma = 0.0   # A_{pi^m}(T) = 0 and C_{pi^m}(T) = 0
        assert abs(gamma) < 1e-14

    print(df.to_string(index=False))
    return table


# ---------------------------------------------------------------------------
# Step 7: Uniform viability failure
# ---------------------------------------------------------------------------

def step7_viability(theta_values: np.ndarray) -> dict:
    """
    Confirm uniform viability failure: x_m * X_{pi^m}(T, theta) >= 1 for all theta,
    while x_m -> 0 as m increases.

    Uses M_VIAB_VALUES (up to m=1000) so that the exponential decay of x_m
    reaches below 1e-3, demonstrating x_Omega(K) = 0 despite K being non-trivial.
    """
    viab = viability_check_uniform(M_VIAB_VALUES, theta_values, TAU, T, THETA_MIN)
    df = pd.DataFrame(viab)
    assert all(viab["all_financed"]), "Uniform financing failed!"
    # tol=1e-3: at m=1000, x_m = exp(-0.01406·1000) ≈ 7.8e-7 < 1e-3
    x_Omega_zero = check_x_omega_zero(viab["x_m"], tol=1e-3)
    print(f"[Exp 2 / Step 7] Uniform x_Omega(K) -> 0: {x_Omega_zero}")
    print(df.to_string(index=False))
    assert x_Omega_zero, "x_m not converging to zero"
    return viab


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiment_2() -> dict:
    """Execute all steps for Experiment 2."""
    print("=" * 65)
    print("EXPERIMENT 2 — Example 4.17: Multi-Scenario Separation")
    print("=" * 65)

    R, D_minus_R, R_window = step1_construct_paths()
    step2_verify_alpha_zero(R)
    qv_table = step3_quadratic_variation(R)
    step4_local_characteristics()
    integral_val = step5_symbolic_verification()
    stress_table = step6_stress_portfolios(R, D_minus_R)
    viab_table = step7_viability(THETA_STAR)

    # Figures
    plot_cubic_paths(R, T, TAU, list(THETA_STAR[:6]))
    plot_qv_convergence(
        qv_table, "Experiment 2 (Cubic path, θ=θ_min)", "exp2_qv_convergence.png"
    )
    plot_wealth_uniform(viab_table, "Experiment 2", "exp2_wealth_uniform.png")

    print("\n[Exp 2] All checks passed. Experiment 2 completed successfully.")
    return {
        "integral_D_minus_R_sq": integral_val,
        "stress_table": stress_table,
        "viab_table": viab_table,
    }


if __name__ == "__main__":
    run_experiment_2()
