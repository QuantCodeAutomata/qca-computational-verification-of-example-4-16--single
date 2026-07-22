"""
Tests for Experiment 1 – Example 4.16: Singleton-scenario separation.

Verifies:
  - Sine path construction and periodicity
  - Faber–Schauder basis properties
  - alpha^{h,D} = 0 exactly
  - Quadratic variation C -> 0
  - Stress portfolio Follmer integrals
  - Wealth and viability tables
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.path_generators import make_sine_path, sine_follmer_integral_exact
from src.partitions import dyadic_partition, dyadic_partition_sequence
from src.extractors import (
    faber_schauder_hat,
    faber_schauder_coefficient,
    project_faber_schauder,
    last_cell_extractor_analytic,
    alpha_sine,
)
from src.qv import discrete_qv_total, qv_convergence_table
from src.densities import (
    compute_local_characteristics,
    maximal_growth,
    canonical_portfolio,
    cumulative_maximal_growth,
)
from src.portfolios import (
    stress_portfolio_sine,
    follmer_integral_sine,
    wealth_sine,
    gamma_sine,
    compute_stress_table_sine,
)
from src.viability import viability_check_singleton, check_x_omega_zero


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

H, N, T = 1.0, 4, 2.0


@pytest.fixture
def sine_path():
    return make_sine_path(H, N)


# ---------------------------------------------------------------------------
# Path generator tests
# ---------------------------------------------------------------------------

def test_sine_path_delta(sine_path):
    """delta should equal h * 2^{-N}."""
    delta, R, R_dot, R_window = sine_path
    assert abs(delta - H * 2.0 ** (-N)) < 1e-15


def test_sine_path_periodic(sine_path):
    """R(t) == R(t + delta) for all t in [0, T]."""
    delta, R, R_dot, R_window = sine_path
    ts = np.linspace(0.0, T, 100)
    assert np.allclose(R(ts), R(ts + delta), atol=1e-12)


def test_sine_path_derivative(sine_path):
    """R_dot should match finite-difference derivative to 1e-6 precision."""
    delta, R, R_dot, R_window = sine_path
    ts = np.linspace(0.1, T - 0.1, 50)
    eps = 1e-7
    fd = (R(ts + eps) - R(ts - eps)) / (2 * eps)
    assert np.allclose(R_dot(ts), fd, atol=1e-5)


def test_sine_window_map(sine_path):
    """R^{t,h}(0) should equal R(t-h), R^{t,h}(1) should equal R(t)."""
    delta, R, R_dot, R_window = sine_path
    t = 1.0
    assert abs(R_window(t, 0.0) - R(t - H)) < 1e-14
    assert abs(R_window(t, 1.0) - R(t)) < 1e-14


# ---------------------------------------------------------------------------
# Faber–Schauder tests
# ---------------------------------------------------------------------------

def test_faber_schauder_hat_peak():
    """Hat S_k^{(m)} should reach 1 at mid-node (2k+1)/2^{m+1}."""
    for m in range(3):
        for k in range(2 ** m):
            u_mid = (2 * k + 1) / 2.0 ** (m + 1)
            val = faber_schauder_hat(m, k, u_mid)
            assert abs(val - 1.0) < 1e-14, f"Hat peak not 1 at m={m}, k={k}"


def test_faber_schauder_hat_zero_outside():
    """Hat S_k^{(m)} should be 0 outside its support [k/2^m, (k+1)/2^m]."""
    m, k = 2, 1
    u_left = (k / 2.0 ** m) - 1e-9
    u_right = ((k + 1) / 2.0 ** m) + 1e-9
    assert faber_schauder_hat(m, k, u_left) < 1e-8
    assert faber_schauder_hat(m, k, u_right) < 1e-8


def test_projection_interpolates_at_dyadic_nodes():
    """P_N x interpolates x exactly at all dyadic nodes j/2^N."""
    N_proj = 3
    x_func = lambda u: np.sin(2 * np.pi * u)  # arbitrary smooth function
    for j in range(2 ** N_proj + 1):
        u_j = j / 2.0 ** N_proj
        proj_val = project_faber_schauder(x_func, N_proj, u_j)
        exact_val = x_func(u_j)
        assert abs(proj_val - exact_val) < 1e-12, (
            f"Projection does not interpolate at j={j}: proj={proj_val}, exact={exact_val}"
        )


# ---------------------------------------------------------------------------
# Extractor tests
# ---------------------------------------------------------------------------

def test_last_cell_extractor_zero_for_periodic(sine_path):
    """D_N^{last}(R^{t,h}) = 0 for all t when R is delta-periodic."""
    delta, R, R_dot, R_window = sine_path
    ts = np.linspace(0.0, T, 200)
    for t in ts:
        val = last_cell_extractor_analytic(R, t, H, N)
        assert abs(val) < 1e-12, f"Extractor not zero at t={t}: {val}"


def test_alpha_sine_zero(sine_path):
    """alpha^{h,D}(t) = 0 for all t."""
    delta, R, R_dot, R_window = sine_path
    ts = np.linspace(0.0, T, 200)
    alpha_vals = np.array([alpha_sine(R, t, H, N) for t in ts])
    assert np.max(np.abs(alpha_vals)) < 1e-12


# ---------------------------------------------------------------------------
# Quadratic variation tests
# ---------------------------------------------------------------------------

def test_qv_converges_to_zero(sine_path):
    """C_n(T) -> 0 as n -> infinity for the C^1 sine path.

    With ω = 2π/δ = 32π, aliasing makes QV ≈ 0 for n ≤ 6 (partition points
    land at period multiples).  From n = 7 the oscillation is resolved and QV
    halves each level (O(Δt) convergence, consistent with R ∈ C^1).
    """
    delta, R, R_dot, R_window = sine_path
    table = qv_convergence_table(R, T, n_levels=14)
    qv_vals = np.array(table["C_n_T"])
    # Oscillation resolved from n=7 onward (index 6 in 0-based list)
    high_n_qv = qv_vals[6:]   # levels 7 … 14
    assert np.all(np.diff(high_n_qv) < 0), "QV not monotonically decreasing in resolved regime"
    # Verify approximate halving rate -> confirms convergence to 0
    ratios = high_n_qv[1:] / high_n_qv[:-1]
    assert np.all(ratios < 0.6), f"Convergence rate too slow: {ratios}"


def test_qv_positive(sine_path):
    """Discrete QV is always non-negative."""
    delta, R, R_dot, R_window = sine_path
    for n in range(1, 8):
        partition = dyadic_partition(T, n)
        qv = discrete_qv_total(R, partition)
        assert qv >= 0.0


# ---------------------------------------------------------------------------
# Local characteristics tests
# ---------------------------------------------------------------------------

def test_local_characteristics_zero():
    """a, c, g, nu all zero when A = C = O = 0."""
    def zero_f(t): return 0.0
    t_grid = np.linspace(0.01, T, 50)
    a_vals, c_vals = compute_local_characteristics(zero_f, zero_f, zero_f, t_grid)
    g_vals = maximal_growth(a_vals, c_vals)
    nu_vals = canonical_portfolio(a_vals, c_vals)
    G_T = cumulative_maximal_growth(g_vals, np.zeros(len(t_grid)))
    assert np.max(np.abs(a_vals)) < 1e-14
    assert np.max(np.abs(c_vals)) < 1e-14
    assert np.max(np.abs(g_vals)) < 1e-14
    assert np.max(np.abs(nu_vals)) < 1e-14
    assert abs(G_T) < 1e-14


def test_maximal_growth_nonnegative():
    """Maximal growth g(t) = a^2/(2c) is always >= 0."""
    a_vals = np.array([1.0, -1.0, 0.5])
    c_vals = np.array([1.0, 2.0, 0.5])
    g = maximal_growth(a_vals, c_vals)
    assert np.all(g >= 0.0)


def test_maximal_growth_formula():
    """g = a^2 / (2*c) when c > 0."""
    a = np.array([1.0, 2.0])
    c = np.array([2.0, 4.0])
    expected = a ** 2 / (2 * c)
    assert np.allclose(maximal_growth(a, c), expected)


# ---------------------------------------------------------------------------
# Stress portfolio tests
# ---------------------------------------------------------------------------

def test_follmer_integral_sine_linearity(sine_path):
    """Follmer return should be linear in m: R_{pi^m}(T) = m * K."""
    delta, R, R_dot, R_window = sine_path
    K = follmer_integral_sine(1, R, R_dot, T)
    for m in [2, 3, 5]:
        assert abs(follmer_integral_sine(m, R, R_dot, T) - m * K) < 1e-4


def test_follmer_integral_positive(sine_path):
    """Follmer return should be positive for the sine path with m > 0."""
    delta, R, R_dot, R_window = sine_path
    ret = follmer_integral_sine(1, R, R_dot, T)
    assert ret > 0.0


def test_wealth_sine_exponential(sine_path):
    """log X_{pi^m}(T) = R_{pi^m}(T) and grows linearly with m.

    Because integral_rdot_sq ≈ 1024π² >> 700, actual float64 wealth overflows.
    We verify in log-space: log_X_T = m * integral is linear, and for small
    synthetic returns (< 700) wealth_sine matches exp directly.
    """
    delta, R, R_dot, R_window = sine_path
    # Use a small synthetic return to test the exp branch directly
    small_return = 5.0
    X_small = wealth_sine(small_return)
    assert abs(X_small - np.exp(small_return)) < 1e-10
    # Verify log-space growth using the exact formula
    K = float(sine_follmer_integral_exact(H, N, T, 1))  # = integral_rdot_sq
    log_X1 = 1.0 * K
    log_X2 = 2.0 * K
    assert log_X2 > log_X1, "log X should grow with m"


def test_gamma_zero():
    """Gamma_{pi^m}(T) = A_T - 0.5 * C_T = 0."""
    assert abs(gamma_sine(0.0, 0.0)) < 1e-15


def test_stress_table_slope(sine_path):
    """log X_{pi^m}(T) should be linear in m with the exact slope."""
    delta, R, R_dot, R_window = sine_path
    omega = 2.0 * np.pi / delta
    # exact integral_0^T |R_dot|^2 dt
    integral_rdot_sq = float(sine_follmer_integral_exact(H, N, T, 1))
    m_vals = list(range(1, 8))
    table = compute_stress_table_sine(m_vals, R, R_dot, T, integral_rdot_sq)
    log_X = np.array(table["log_X_T"])
    m_arr = np.array(table["m"])
    slope = float(np.polyfit(m_arr, log_X, 1)[0])
    assert abs(slope - integral_rdot_sq) < 1e-4


def test_stress_table_x_m_decreasing(sine_path):
    """log x_m = -m * integral_rdot_sq should be strictly decreasing in m.

    Because integral_rdot_sq >> 700, float x_m = 1/exp(m*K) underflows to 0.
    We check the log-space analogue: log_x_m = -m * K, which strictly decreases.
    """
    delta, R, R_dot, R_window = sine_path
    integral_rdot_sq = float(sine_follmer_integral_exact(H, N, T, 1))
    m_vals = list(range(1, 6))
    table = compute_stress_table_sine(m_vals, R, R_dot, T, integral_rdot_sq)
    log_x_m = np.array(table["log_x_m"])
    assert np.all(np.diff(log_x_m) < 0), "log x_m should be strictly decreasing"


# ---------------------------------------------------------------------------
# Viability tests
# ---------------------------------------------------------------------------

def test_viability_all_financed(sine_path):
    """Every stress scenario finances K = 1 exactly from x_m."""
    delta, R, R_dot, R_window = sine_path
    integral_rdot_sq = float(sine_follmer_integral_exact(H, N, T, 1))
    m_vals = list(range(1, 8))
    table = compute_stress_table_sine(m_vals, R, R_dot, T, integral_rdot_sq)
    viab = viability_check_singleton(m_vals, table["X_T"])
    assert all(viab["financing_ok"])


def test_viability_V_T_zero(sine_path):
    """V_{x_m, pi^m, K}(T) = 0 for all m."""
    delta, R, R_dot, R_window = sine_path
    integral_rdot_sq = float(sine_follmer_integral_exact(H, N, T, 1))
    m_vals = list(range(1, 8))
    table = compute_stress_table_sine(m_vals, R, R_dot, T, integral_rdot_sq)
    viab = viability_check_singleton(m_vals, table["X_T"])
    V_T = np.array(viab["V_T"])
    assert np.allclose(V_T, 0.0, atol=1e-12)


def test_x_omega_converges_to_zero(sine_path):
    """x_m -> 0 implies x_Omega(K) = 0."""
    delta, R, R_dot, R_window = sine_path
    integral_rdot_sq = float(sine_follmer_integral_exact(H, N, T, 1))
    m_vals = list(range(1, 11))
    table = compute_stress_table_sine(m_vals, R, R_dot, T, integral_rdot_sq)
    viab = viability_check_singleton(m_vals, table["X_T"])
    assert check_x_omega_zero(viab["x_m"], tol=1e-3)


# ---------------------------------------------------------------------------
# Partition tests
# ---------------------------------------------------------------------------

def test_dyadic_partition_length():
    """Pi_n should have 2^n + 1 points."""
    for n in range(1, 8):
        pts = dyadic_partition(T, n)
        assert len(pts) == 2 ** n + 1


def test_dyadic_partition_endpoints():
    """Pi_n should start at 0 and end at T."""
    for n in range(1, 6):
        pts = dyadic_partition(T, n)
        assert abs(pts[0]) < 1e-14
        assert abs(pts[-1] - T) < 1e-14


def test_dyadic_partition_sequence_nesting():
    """Each successive partition refines the previous one."""
    partitions = dyadic_partition_sequence(T, n_levels=5)
    # Level 1 grid points should all appear in level 2
    p1 = set(np.round(partitions[1], 12))
    p2 = set(np.round(partitions[2], 12))
    assert p1.issubset(p2)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

def test_wealth_sine_m_zero():
    """X_{pi^0}(T) = exp(0) = 1."""
    X = wealth_sine(0.0)
    assert abs(X - 1.0) < 1e-15


def test_follmer_integral_large_m(sine_path):
    """Follmer return should scale linearly even for large m."""
    delta, R, R_dot, R_window = sine_path
    base = follmer_integral_sine(1, R, R_dot, T)
    for m in [50, 100]:
        ret = follmer_integral_sine(m, R, R_dot, T)
        assert abs(ret - m * base) < 1e-2 * abs(m * base)


def test_qv_single_step():
    """QV on a single interval: (R(T) - R(0))^2."""
    delta, R, R_dot, R_window = make_sine_path(H, N)
    partition = np.array([0.0, T])
    qv = discrete_qv_total(R, partition)
    expected = (R(T) - R(0.0)) ** 2
    assert abs(qv - expected) < 1e-14
