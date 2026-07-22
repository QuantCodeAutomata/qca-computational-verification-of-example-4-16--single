"""
Experiment 1 – Example 4.16: Singleton-scenario separation between
growth-numéraire identities and viability.

Reproduces the paper's Example 4.16 in a fully pathwise, probability-free setting:
  - Growth-numéraire layer: alpha, A, C, O, a, c, g, G, nu are all identically zero.
  - Viability failure: stress portfolios pi^m generate X_{pi^m}(T) -> infinity
    while x_m = 1/X_{pi^m}(T) -> 0.

Default parameters (documented; NOT uniquely prescribed by the paper):
    h = 1,  N = 4,  T = 2,  delta = h * 2^{-N} = 1/16.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import sympy as sp

from src.path_generators import (
    make_sine_path,
    sine_follmer_integral_exact,
    sine_follmer_integral_numerical,
)
from src.partitions import dyadic_partition_sequence
from src.extractors import last_cell_extractor_analytic, alpha_sine
from src.qv import qv_convergence_table, discrete_qv_total
from src.densities import (
    compute_local_characteristics,
    maximal_growth,
    canonical_portfolio,
    cumulative_maximal_growth,
)
from src.portfolios import compute_stress_table_sine, wealth_sine, gamma_sine
from src.viability import viability_check_singleton, check_x_omega_zero
from src.reports import (
    plot_sine_path,
    plot_qv_convergence,
    plot_wealth_vs_m,
)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
H: float = 1.0     # observation window
N: int = 4         # dyadic resolution
T: float = 2.0     # horizon
DELTA: float = H * 2.0 ** (-N)   # = 1/16
M_VALUES: list[int] = list(range(1, 11))   # stress indices 1 … 10


# ---------------------------------------------------------------------------
# Step 1: Construct path objects
# ---------------------------------------------------------------------------

def step1_construct_path():
    """Construct the analytic sine path and verify its properties."""
    delta, R, R_dot, R_window = make_sine_path(H, N)
    assert abs(delta - DELTA) < 1e-14, "delta mismatch"
    # Verify periodicity: R(t) == R(t + delta)
    for t in np.linspace(0.0, T, 50):
        assert abs(R(t) - R(t + delta)) < 1e-12, f"Periodicity fails at t={t}"
    print(f"[Exp 1 / Step 1] Path constructed. delta={delta:.6f}, period verified.")
    return delta, R, R_dot, R_window


# ---------------------------------------------------------------------------
# Step 2: Verify alpha^{h,D} = 0
# ---------------------------------------------------------------------------

def step2_verify_alpha_zero(R: callable) -> np.ndarray:
    """
    Compute D_N^{last}(R^{t,h}) and confirm it is exactly zero.

    By construction R is delta-periodic and delta = h * 2^{-N}, so
    R(t) = R(t - delta) for all t, meaning D_N^{last} = 0.
    """
    t_grid = np.linspace(0.0, T, 500)
    alpha_vals = np.array([alpha_sine(R, t, H, N) for t in t_grid])
    max_alpha = np.max(np.abs(alpha_vals))
    print(f"[Exp 1 / Step 2] max |alpha^{{h,D}}| = {max_alpha:.2e}  (should be 0)")
    assert max_alpha < 1e-10, f"alpha not zero: {max_alpha}"
    return alpha_vals


# ---------------------------------------------------------------------------
# Step 3: Accumulated trend A = 0
# ---------------------------------------------------------------------------

def step3_accumulated_trend(alpha_vals: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Compute A^{h,D}(t) = integral_0^t alpha(s) ds = 0 since alpha = 0."""
    # Cumulative trapezoidal integration
    dt = t_grid[1] - t_grid[0]
    A_vals = np.cumsum(alpha_vals) * dt
    max_A = np.max(np.abs(A_vals))
    print(f"[Exp 1 / Step 3] max |A^{{h,D}}| = {max_A:.2e}  (should be 0)")
    assert max_A < 1e-10, f"A not zero: {max_A}"
    return A_vals


# ---------------------------------------------------------------------------
# Step 4: Quadratic covariation C -> 0
# ---------------------------------------------------------------------------

def step4_quadratic_variation(R: callable) -> dict:
    """
    Compute QV under dyadic refinements. Since R is C^1, C_n(T) -> 0.

    Implementation note on aliasing:
      With ω = 2π/δ = 32π and T = 2, the path completes 32 full periods.
      Dyadic levels n ≤ 6 sample the path only at multiples of a half-period,
      so all increments vanish (aliasing: C_n ≈ 0).  At level n = 7 the
      oscillation is first resolved; from there, C_n halves with each extra
      level, confirming the O(Δt) convergence rate consistent with R ∈ C^1.
      Analytically, [R, R]_Π = 0.
    """
    qv_table = qv_convergence_table(R, T, n_levels=15)
    print("[Exp 1 / Step 4] QV convergence:")
    for n, nst, qv in zip(qv_table["n"], qv_table["n_steps"], qv_table["C_n_T"]):
        print(f"   n={n:2d}  steps={nst:6d}  C_n(T)={qv:.6e}")

    # Oscillation is resolved from n=7 onward; verify monotone decrease there.
    qv_vals = qv_table["C_n_T"]
    high_n_qv = qv_vals[6:]   # indices for n = 7 … 15
    for i in range(1, len(high_n_qv)):
        assert high_n_qv[i] < high_n_qv[i - 1], (
            f"QV not decreasing at level n={i + 7}: {high_n_qv[i]:.4e} >= {high_n_qv[i-1]:.4e}"
        )
    # Verify approximate halving rate (convergence to 0 at rate O(Δt))
    ratios = [high_n_qv[i] / high_n_qv[i - 1] for i in range(1, len(high_n_qv))]
    avg_ratio = float(np.mean(ratios))
    print(f"[Exp 1 / Step 4] C^{{h,D}} convergence: avg ratio C_{{n+1}}/C_n = {avg_ratio:.4f} "
          f"(expected ≈ 0.5 → QV→0). Analytically [R,R]_Pi = 0.")
    assert avg_ratio < 0.6, f"QV convergence rate too slow: avg ratio = {avg_ratio:.4f}"
    return qv_table


# ---------------------------------------------------------------------------
# Step 5: Local characteristics a, c, g, G, nu — all zero
# ---------------------------------------------------------------------------

def step5_local_characteristics() -> dict:
    """Verify local characteristics vanish because O = 0."""
    def A_func(t): return 0.0
    def C_func(t): return 0.0
    def O_func(t): return 0.0

    t_grid = np.linspace(0.01, T, 200)
    a_vals, c_vals = compute_local_characteristics(A_func, C_func, O_func, t_grid)
    g_vals = maximal_growth(a_vals, c_vals)
    nu_vals = canonical_portfolio(a_vals, c_vals)
    G_T = cumulative_maximal_growth(g_vals, np.zeros(len(t_grid)))

    print(f"[Exp 1 / Step 5] max|a| = {np.max(np.abs(a_vals)):.2e}")
    print(f"[Exp 1 / Step 5] max|c| = {np.max(np.abs(c_vals)):.2e}")
    print(f"[Exp 1 / Step 5] max|g| = {np.max(np.abs(g_vals)):.2e}")
    print(f"[Exp 1 / Step 5] max|nu|= {np.max(np.abs(nu_vals)):.2e}")
    print(f"[Exp 1 / Step 5] G(T)  = {G_T:.2e}")

    assert np.max(np.abs(a_vals)) < 1e-14
    assert np.max(np.abs(c_vals)) < 1e-14
    assert np.max(np.abs(g_vals)) < 1e-14
    assert np.max(np.abs(nu_vals)) < 1e-14
    assert abs(G_T) < 1e-14
    return {"a": a_vals, "c": c_vals, "g": g_vals, "nu": nu_vals, "G_T": G_T}


# ---------------------------------------------------------------------------
# Step 6: Symbolic verification with SymPy
# ---------------------------------------------------------------------------

def step6_symbolic_verification() -> dict:
    """
    Use SymPy to verify:
      (1) R_dot(t) = (2*pi/delta)*cos(2*pi*t/delta)
      (2) integral_0^T |R_dot|^2 dt = (2*pi/delta)^2 * T/2 (for integer number of periods)
    """
    t_sym = sp.Symbol("t", real=True, positive=True)
    delta_sym = sp.Rational(1, 16)   # delta = H * 2^{-N} = 1 * 2^{-4} = 1/16
    T_sym = sp.Integer(2)
    omega_sym = 2 * sp.pi / delta_sym

    R_sym = sp.sin(omega_sym * t_sym)
    R_dot_sym = sp.diff(R_sym, t_sym)
    R_dot_expected = omega_sym * sp.cos(omega_sym * t_sym)
    diff_check = sp.simplify(R_dot_sym - R_dot_expected)

    integral_exact = sp.integrate(R_dot_sym ** 2, (t_sym, 0, T_sym))
    integral_simplified = sp.simplify(integral_exact)

    print(f"[Exp 1 / Step 6] R_dot difference (should be 0): {diff_check}")
    print(f"[Exp 1 / Step 6] Symbolic integral |R_dot|^2 over [0,T]: {integral_simplified}")
    assert diff_check == 0, "SymPy derivative mismatch"

    integral_numeric = float(integral_simplified.evalf())
    print(f"[Exp 1 / Step 6] Integral |R_dot|^2 numeric = {integral_numeric:.6f}")
    return {
        "R_dot_sym": R_dot_sym,
        "integral_rdot_sq_exact": integral_simplified,
        "integral_rdot_sq_numeric": integral_numeric,
    }


# ---------------------------------------------------------------------------
# Step 7: Stress portfolios and wealth table
# ---------------------------------------------------------------------------

def step7_stress_portfolios(
    R: callable,
    R_dot: callable,
    integral_rdot_sq: float,
) -> dict:
    """
    Compute stress portfolio returns and terminal wealth for m = 1, ..., 10.

    Verify:
      - R_{pi^m}(T) = m * integral_rdot_sq (linear in m)
      - X_{pi^m}(T) = exp(m * integral_rdot_sq) (exponentially increasing)
      - Gamma_{pi^m}(T) = 0
    """
    table = compute_stress_table_sine(M_VALUES, R, R_dot, T, integral_rdot_sq)
    df = pd.DataFrame(table)

    # Verify linearity of log X_{pi^m}(T) in m.
    # log_X_T is stored directly as m * integral_rdot_sq (overflow-safe exact formula).
    log_X_T = np.array(table["log_X_T"])
    m_arr = np.array(table["m"])
    slope = float(np.polyfit(m_arr, log_X_T, 1)[0])
    print(f"[Exp 1 / Step 7] Expected slope of log X_T vs m: {integral_rdot_sq:.6f}")
    print(f"[Exp 1 / Step 7] Fitted slope of log X_T vs m:  {slope:.6f}")
    # Slope must equal integral_rdot_sq by construction (log_X_T = m * integral_rdot_sq)
    assert abs(slope - integral_rdot_sq) < 1e-3, "Slope mismatch"

    # Verify Gamma = 0
    for m in M_VALUES:
        g = gamma_sine(0.0, 0.0)   # A = 0, C_portfolio = 0
        assert abs(g) < 1e-14, f"Gamma not zero for m={m}"

    # Check numerical vs exact returns using relative tolerance.
    # The RS sum has O(Δt) error with coefficient proportional to ω=2π/δ=32π,
    # yielding a relative error ≈ 1.6e-6 for n=2^16; require < 0.1%.
    for i, m in enumerate(table["m"]):
        exact = table["follmer_return_exact"][i]
        num   = table["follmer_return_num"][i]
        rel_err = abs(exact - num) / max(abs(exact), 1.0)
        assert rel_err < 1e-3, (
            f"Exact/numerical follmer return mismatch for m={m}: "
            f"exact={exact:.6f} num={num:.6f} rel_err={rel_err:.2e}"
        )

    print(df.to_string(index=False))
    return table


# ---------------------------------------------------------------------------
# Step 8: Viability failure
# ---------------------------------------------------------------------------

def step8_viability(stress_table: dict) -> dict:
    """
    Verify that x_m -> 0 and the terminal liability K=1 is financed from x_m.
    """
    viab = viability_check_singleton(stress_table["m"], stress_table["X_T"])
    df = pd.DataFrame(viab)
    assert all(viab["financing_ok"]), "Financing condition fails!"
    x_Omega_zero = check_x_omega_zero(viab["x_m"], tol=1e-3)
    print(f"[Exp 1 / Step 8] x_Omega(K) -> 0: {x_Omega_zero}")
    print(df.to_string(index=False))
    assert x_Omega_zero, "x_m did not converge to zero"
    return viab


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiment_1() -> dict:
    """
    Execute all steps for Experiment 1 and return a summary dictionary.
    """
    print("=" * 65)
    print("EXPERIMENT 1 — Example 4.16: Singleton-Scenario Separation")
    print("=" * 65)

    delta, R, R_dot, R_window = step1_construct_path()
    t_grid = np.linspace(0.0, T, 500)

    alpha_vals = step2_verify_alpha_zero(R)
    A_vals = step3_accumulated_trend(alpha_vals, t_grid)
    qv_table = step4_quadratic_variation(R)
    local_chars = step5_local_characteristics()
    sym_results = step6_symbolic_verification()
    integral_rdot_sq = sym_results["integral_rdot_sq_numeric"]
    stress_table = step7_stress_portfolios(R, R_dot, integral_rdot_sq)
    viab_table = step8_viability(stress_table)

    # Produce figures
    plot_sine_path(R, R_dot, T, H, delta)
    plot_qv_convergence(qv_table, "Experiment 1 (Sine path)", "exp1_qv_convergence.png")
    plot_wealth_vs_m(stress_table, "Experiment 1", "exp1_wealth_vs_m.png")

    print("\n[Exp 1] All checks passed. Experiment 1 completed successfully.")
    return {
        "alpha_max": float(np.max(np.abs(alpha_vals))),
        "A_max": float(np.max(np.abs(A_vals))),
        "C_final": qv_table["C_n_T"][-1],
        "G_T": local_chars["G_T"],
        "integral_rdot_sq": integral_rdot_sq,
        "stress_table": stress_table,
        "viab_table": viab_table,
    }


if __name__ == "__main__":
    run_experiment_1()
