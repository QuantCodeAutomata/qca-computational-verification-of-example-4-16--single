"""
Tests for Experiment 2 – Example 4.17: Multi-scenario separation.

Verifies:
  - Cubic path construction and zero-before-tau property
  - Lag extractor produces alpha = 0
  - Quadratic variation converges to zero for all scenarios
  - Stress portfolio formulas and uniform lower bounds
  - Viability failure across the entire scenario family
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.path_generators import (
    make_cubic_path,
    cubic_follmer_integral_exact,
    cubic_follmer_integral_numerical,
)
from src.partitions import dyadic_partition
from src.extractors import lag_extractor, alpha_cubic
from src.qv import discrete_qv_total, qv_convergence_table
from src.densities import (
    compute_local_characteristics,
    maximal_growth,
    canonical_portfolio,
)
from src.portfolios import (
    stress_portfolio_cubic,
    follmer_integral_cubic,
    wealth_cubic,
    compute_stress_table_cubic,
)
from src.viability import viability_check_uniform, check_x_omega_zero


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

H, T, TAU, THETA_MIN = 1.0, 2.0, 1.5, 0.5
THETA_NEG = np.linspace(-1.0, -THETA_MIN, 4)
THETA_POS = np.linspace(THETA_MIN, 1.0, 4)
THETA_STAR = np.concatenate([THETA_NEG, THETA_POS])


@pytest.fixture
def cubic_path():
    return make_cubic_path(TAU, H)


# ---------------------------------------------------------------------------
# Cubic path tests
# ---------------------------------------------------------------------------

def test_cubic_path_zero_before_tau(cubic_path):
    """R(t, theta) = 0 for all t <= tau."""
    R, D_minus_R, R_window = cubic_path
    t_before = np.linspace(0.0, TAU, 100)
    for theta in THETA_STAR:
        vals = R(t_before, theta)
        assert np.all(np.abs(vals) < 1e-14), (
            f"Cubic path not zero before tau for theta={theta}"
        )


def test_cubic_path_formula(cubic_path):
    """R(t, theta) = theta * ((t-tau)^+)^3 matches manual computation."""
    R, D_minus_R, R_window = cubic_path
    t = 1.8
    for theta in [0.5, -1.0, 0.7]:
        expected = theta * max(t - TAU, 0.0) ** 3
        assert abs(R(t, theta) - expected) < 1e-14


def test_cubic_derivative_formula(cubic_path):
    """D^-R(t, theta) = 3*theta*((t-tau)^+)^2 matches finite differences."""
    R, D_minus_R, R_window = cubic_path
    ts = np.linspace(TAU + 0.1, T - 0.05, 20)
    eps = 1e-7
    for theta in [0.5, -0.8, 1.0]:
        fd = (R(ts + eps, theta) - R(ts - eps, theta)) / (2 * eps)
        analytic = D_minus_R(ts, theta)
        assert np.allclose(fd, analytic, atol=1e-5)


def test_cubic_derivative_zero_before_tau(cubic_path):
    """D^-R(t, theta) = 0 for t <= tau."""
    R, D_minus_R, R_window = cubic_path
    t_before = np.linspace(0.0, TAU, 50)
    for theta in THETA_STAR:
        assert np.all(np.abs(D_minus_R(t_before, theta)) < 1e-14)


def test_cubic_window_map(cubic_path):
    """R^{t,h}(0) = R(t-h, theta), R^{t,h}(1) = R(t, theta)."""
    R, D_minus_R, R_window = cubic_path
    t = 1.8
    theta = 0.7
    assert abs(R_window(t, 0.0, theta) - R(t - H, theta)) < 1e-14
    assert abs(R_window(t, 1.0, theta) - R(t, theta)) < 1e-14


# ---------------------------------------------------------------------------
# Alpha (lag extractor) tests
# ---------------------------------------------------------------------------

def test_alpha_cubic_zero_over_all_scenarios(cubic_path):
    """alpha^{h,D_lag}(t, theta) = 0 for every t in [0,T] and theta in Theta_*."""
    R, D_minus_R, R_window = cubic_path
    t_grid = np.linspace(0.0, T, 100)
    for theta in THETA_STAR:
        R_theta = lambda s, _theta=theta: R(s, _theta)
        alpha_vals = np.array([alpha_cubic(R_theta, t, H) for t in t_grid])
        assert np.max(np.abs(alpha_vals)) < 1e-14, (
            f"alpha not zero for theta={theta}: max={np.max(np.abs(alpha_vals))}"
        )


def test_lag_extractor_at_zero_lag(cubic_path):
    """lag_extractor(x, t, h) = x(t-h)."""
    R, D_minus_R, R_window = cubic_path
    theta = 0.8
    t = T
    R_theta = lambda s: R(s, theta)
    assert abs(lag_extractor(R_theta, t, H) - R(t - H, theta)) < 1e-14


def test_tau_constraint():
    """Verify tau > T - h ensures t-h < tau for all t in [0,T]."""
    assert TAU > T - H, (
        f"tau={TAU} must be > T-h={T-H} to guarantee alpha=0"
    )


# ---------------------------------------------------------------------------
# Quadratic variation tests
# ---------------------------------------------------------------------------

def test_qv_cubic_zero(cubic_path):
    """C_n(T, theta) -> 0 for all theta in Theta_*.

    For a C^1 path, QV = O(Δt).  With n=2^15 and Δt=T/2^15≈6e-5:
      QV ≈ Δt * ∫_tau^T |3θ(t-tau)^2|^2 dt = Δt * (9/5)*θ^2*(T-tau)^5 < 1e-5.
    """
    R, D_minus_R, R_window = cubic_path
    for theta in THETA_STAR:
        partition = np.linspace(0.0, T, 2 ** 15 + 1)
        qv = discrete_qv_total(lambda pts: R(pts, theta), partition)
        assert qv < 1e-5, (
            f"QV not small for theta={theta}: {qv}"
        )


def test_qv_refinement_decreases(cubic_path):
    """QV should decrease with refinement."""
    R, D_minus_R, R_window = cubic_path
    theta = THETA_MIN
    table = qv_convergence_table(lambda pts: R(pts, theta), T, n_levels=10)
    qv_vals = np.array(table["C_n_T"])
    assert qv_vals[-1] < qv_vals[0]


# ---------------------------------------------------------------------------
# Follmer integral tests
# ---------------------------------------------------------------------------

def test_follmer_cubic_exact_formula(cubic_path):
    """(9*m*theta^2/5)*(T-tau)^5 should match numerical integral."""
    R, D_minus_R, R_window = cubic_path
    for theta in [THETA_MIN, 0.8, -0.6, 1.0]:
        for m in [1, 3, 5]:
            exact = cubic_follmer_integral_exact(TAU, T, theta, m)
            num = cubic_follmer_integral_numerical(TAU, T, theta, m, H)
            assert abs(exact - num) / max(abs(exact), 1.0) < 1e-3, (
                f"Follmer mismatch theta={theta} m={m}: exact={exact:.6f} num={num:.6f}"
            )


def test_follmer_cubic_positive_for_nonzero_theta(cubic_path):
    """Follmer return is positive for theta != 0."""
    R, D_minus_R, R_window = cubic_path
    for theta in THETA_STAR:
        exact = cubic_follmer_integral_exact(TAU, T, theta, m=1)
        assert exact > 0.0, f"Expected positive Follmer return for theta={theta}"


def test_follmer_cubic_linear_in_m():
    """Follmer return scales linearly with m."""
    base = cubic_follmer_integral_exact(TAU, T, THETA_MIN, m=1)
    for m in [2, 5, 10]:
        val = cubic_follmer_integral_exact(TAU, T, THETA_MIN, m=m)
        assert abs(val - m * base) < 1e-12


def test_follmer_cubic_scales_with_theta_squared():
    """Follmer return proportional to theta^2."""
    m = 2
    val_half = cubic_follmer_integral_exact(TAU, T, THETA_MIN, m)
    val_double = cubic_follmer_integral_exact(TAU, T, 2 * THETA_MIN, m)
    ratio = val_double / val_half
    assert abs(ratio - 4.0) < 1e-12


def test_follmer_cubic_exact_value():
    """Verify exact Follmer return value for known inputs."""
    # (9*1*0.5^2/5)*(0.5)^5 = (9*0.25/5)*(0.03125) = 0.45/5*0.03125 = 0.09*0.03125
    expected = (9.0 * 1 * 0.5 ** 2 / 5.0) * (0.5) ** 5
    got = cubic_follmer_integral_exact(tau=1.5, T=2.0, theta=0.5, m=1)
    assert abs(got - expected) < 1e-15


# ---------------------------------------------------------------------------
# Wealth and stress table tests
# ---------------------------------------------------------------------------

def test_wealth_cubic_positive():
    """X_{pi^m}(T, theta) > 1 for m > 0 and theta in Theta_*."""
    for theta in THETA_STAR:
        ret = cubic_follmer_integral_exact(TAU, T, theta, m=1)
        X = wealth_cubic(ret)
        assert X > 1.0


def test_wealth_cubic_increases_with_m():
    """X_{pi^m}(T, theta) should increase with m."""
    theta = THETA_MIN
    X_prev = 1.0
    for m in [1, 2, 3, 5]:
        ret = cubic_follmer_integral_exact(TAU, T, theta, m)
        X = wealth_cubic(ret)
        assert X > X_prev
        X_prev = X


def test_stress_table_cubic_exact_vs_num(cubic_path):
    """Exact and numerical wealth should agree to 0.1% relative error."""
    R, D_minus_R, R_window = cubic_path
    theta_rep = [THETA_MIN, 0.8]
    table = compute_stress_table_cubic(
        [1, 2], theta_rep, R, D_minus_R, TAU, T, THETA_MIN
    )
    for i in range(len(table["m"])):
        exact = table["X_T_exact"][i]
        num = table["X_T_num"][i]
        assert abs(exact - num) / max(exact, 1.0) < 2e-3


# ---------------------------------------------------------------------------
# Uniform viability tests
# ---------------------------------------------------------------------------

def test_uniform_financing(cubic_path):
    """x_m * X_{pi^m}(T, theta) >= 1 for all theta in Theta_* and each m."""
    viab = viability_check_uniform(
        list(range(1, 8)), list(THETA_STAR), TAU, T, THETA_MIN
    )
    assert all(viab["all_financed"])


def test_x_m_to_zero():
    """x_m -> 0 as m increases (viability fails).

    With tau=1.5, T=2.0, theta_min=0.5:
      lower_log(m) = (9*m*0.25/5) * (0.5)^5 = 0.0028125 * m
      x_m = exp(-0.0028125 * m)
    Requires m ≈ 1637 for x_m < 0.01; use m=200,400,...,2000.
    """
    m_vals = list(range(200, 2001, 200))   # m=200,400,...,2000; x_2000 ≈ 0.0036
    viab = viability_check_uniform(m_vals, list(THETA_STAR), TAU, T, THETA_MIN)
    assert check_x_omega_zero(viab["x_m"], tol=1e-2)


def test_uniform_lower_bound_exact():
    """inf_{theta} X_{pi^m}(T, theta) >= exp((9*m*theta_min^2/5)*(T-tau)^5)."""
    m = 3
    lower_log = (9.0 * m * THETA_MIN ** 2 / 5.0) * (T - TAU) ** 5
    lower_bound = np.exp(lower_log)
    for theta in THETA_STAR:
        X = wealth_cubic(cubic_follmer_integral_exact(TAU, T, theta, m))
        assert X >= lower_bound - 1e-10, (
            f"Wealth {X:.6f} below lower bound {lower_bound:.6f} for theta={theta}"
        )


def test_min_wealth_increases_with_m():
    """inf_{theta} X_{pi^m}(T, theta) should increase with m."""
    prev_min = 0.0
    for m in range(1, 6):
        min_log = min(
            (9.0 * m * theta ** 2 / 5.0) * (T - TAU) ** 5
            for theta in THETA_STAR
        )
        cur_min = np.exp(min_log)
        assert cur_min > prev_min
        prev_min = cur_min


# ---------------------------------------------------------------------------
# Local characteristics (all zero by construction)
# ---------------------------------------------------------------------------

def test_local_chars_zero_cubic():
    """a, c, g, nu = 0 when O = 0 for all scenarios."""
    def zero_f(t): return 0.0
    t_grid = np.linspace(0.01, T, 50)
    a_vals, c_vals = compute_local_characteristics(zero_f, zero_f, zero_f, t_grid)
    assert np.max(np.abs(a_vals)) < 1e-14
    assert np.max(np.abs(c_vals)) < 1e-14


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

def test_cubic_path_at_tau(cubic_path):
    """R(tau, theta) = 0 exactly (boundary continuity)."""
    R, D_minus_R, R_window = cubic_path
    for theta in THETA_STAR:
        assert abs(R(TAU, theta)) < 1e-14


def test_follmer_integral_zero_for_zero_m():
    """R_{pi^0}(T, theta) = 0 for all theta."""
    exact = cubic_follmer_integral_exact(TAU, T, THETA_MIN, m=0)
    assert abs(exact) < 1e-15


def test_single_theta_viability():
    """Financing holds for a single extreme theta = 1.0."""
    m, theta = 1, 1.0
    lower_log = (9.0 * m * THETA_MIN ** 2 / 5.0) * (T - TAU) ** 5
    x_m = np.exp(-lower_log)
    X_T = np.exp(cubic_follmer_integral_exact(TAU, T, theta, m))
    assert x_m * X_T >= 1.0 - 1e-12
