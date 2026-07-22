"""
Report generation: plots and tables for both experiments.

Produces:
  - Path plots (sine and cubic scenarios)
  - QV convergence plots
  - Wealth vs. stress-index tables/plots
  - Viability initial-capital plots

All figures are saved to results/figures/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def _ensure_dirs() -> None:
    """Create results directories if they do not exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Exp 1 – Sine path plots
# ---------------------------------------------------------------------------

def plot_sine_path(
    R_func: callable,
    R_dot: callable,
    T: float,
    h: float,
    delta: float,
    n_plot: int = 2000,
) -> str:
    """
    Plot the sine return path R(t) and derivative R_dot(t) over [0, T].

    Parameters
    ----------
    R_func : callable
        Sine path.
    R_dot : callable
        Derivative.
    T : float
        Horizon.
    h, delta : float
        Window and cell width (for annotation).
    n_plot : int
        Number of plot points.

    Returns
    -------
    str
        Path to saved figure.
    """
    _ensure_dirs()
    ts = np.linspace(0.0, T, n_plot)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(ts, R_func(ts), color="steelblue", lw=1.5)
    axes[0].set_ylabel("R(t)")
    axes[0].set_title(f"Experiment 1 – Sine Path  (h={h}, δ={delta:.4f})")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(ts, R_dot(ts), color="coral", lw=1.5)
    axes[1].set_ylabel(r"$\dot{R}(t)$")
    axes[1].set_xlabel("t")
    axes[1].grid(True, alpha=0.4)

    fig.tight_layout()
    fpath = os.path.join(FIGURES_DIR, "exp1_sine_path.png")
    fig.savefig(fpath, dpi=120)
    plt.close(fig)
    return fpath


def plot_qv_convergence(
    qv_table: dict[str, list],
    experiment_label: str,
    filename: str,
) -> str:
    """
    Plot quadratic variation versus partition refinement level.

    Parameters
    ----------
    qv_table : dict
        Output of qv.qv_convergence_table.
    experiment_label : str
        Label for the plot title.
    filename : str
        Filename in results/figures/.

    Returns
    -------
    str
        Path to saved figure.
    """
    _ensure_dirs()
    ns = qv_table["n"]
    qv_vals = qv_table["C_n_T"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(ns, np.maximum(qv_vals, 1e-300), "o-", color="darkorange", lw=2)
    ax.set_xlabel("Dyadic refinement level n")
    ax.set_ylabel(r"$C_n(T)$ (log scale)")
    ax.set_title(f"{experiment_label} — QV convergence to 0")
    ax.grid(True, alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fpath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(fpath, dpi=120)
    plt.close(fig)
    return fpath


def plot_wealth_vs_m(
    stress_table: dict[str, list],
    experiment_label: str,
    filename: str,
) -> str:
    """
    Plot log terminal wealth and log initial capital versus stress index m.

    When X_{pi^m}(T) overflows float64 (stored as inf), the plot uses
    log_X_T = m * integral_rdot_sq directly, which is always finite.

    Parameters
    ----------
    stress_table : dict
        Must contain keys 'm', 'X_T' (or 'log_X_T'), 'x_m' (or 'log_x_m').
    experiment_label : str
        Plot label.
    filename : str
        Output filename.

    Returns
    -------
    str
        Path to saved figure.
    """
    _ensure_dirs()
    m_vals = np.asarray(stress_table["m"])

    # Prefer precomputed log values (overflow-safe) when present
    if "log_X_T" in stress_table:
        log_X_T = np.asarray(stress_table["log_X_T"])
    else:
        log_X_T = np.log(np.maximum(stress_table["X_T"], 1e-300))

    if "log_x_m" in stress_table:
        log_x_m = np.asarray(stress_table["log_x_m"])
    else:
        raw = np.asarray(stress_table["x_m"])
        log_x_m = np.where(raw > 0, np.log(np.maximum(raw, 1e-300)), -log_X_T)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(m_vals, log_X_T, "s-", color="mediumseagreen", lw=2)
    axes[0].set_xlabel("Stress index m")
    axes[0].set_ylabel(r"$\log X_{\pi^m}(T)$")
    axes[0].set_title(f"{experiment_label} — Log terminal wealth (linear in m)")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(m_vals, log_x_m, "D-", color="tomato", lw=2)
    axes[1].set_xlabel("Stress index m")
    axes[1].set_ylabel(r"$\log x_m$")
    axes[1].set_title(f"{experiment_label} — Log initial capital (→ −∞)")
    axes[1].grid(True, alpha=0.4)

    fig.tight_layout()
    fpath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(fpath, dpi=120)
    plt.close(fig)
    return fpath


# ---------------------------------------------------------------------------
# Exp 2 – Cubic path and uniform wealth plots
# ---------------------------------------------------------------------------

def plot_cubic_paths(
    R_func: callable,
    T: float,
    tau: float,
    theta_values: list[float],
    n_plot: int = 1000,
) -> str:
    """
    Plot cubic scenario paths for representative theta values.

    Parameters
    ----------
    R_func : callable
        Cubic path R(t, theta).
    T, tau : float
        Horizon and branch time.
    theta_values : list of float
        Scenario amplitudes to plot.
    n_plot : int
        Number of time points.

    Returns
    -------
    str
        Path to saved figure.
    """
    _ensure_dirs()
    ts = np.linspace(0.0, T, n_plot)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlBu(np.linspace(0, 1, len(theta_values)))
    for theta, color in zip(theta_values, colors):
        ax.plot(ts, R_func(ts, theta), color=color, lw=1.5,
                label=f"θ = {theta:.2f}")
    ax.axvline(tau, color="gray", ls="--", lw=1, label=f"τ = {tau}")
    ax.set_xlabel("t")
    ax.set_ylabel("R(t, θ)")
    ax.set_title(f"Experiment 2 – Cubic Scenario Paths  (τ={tau}, T={T})")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fpath = os.path.join(FIGURES_DIR, "exp2_cubic_paths.png")
    fig.savefig(fpath, dpi=120)
    plt.close(fig)
    return fpath


def plot_wealth_uniform(
    viab_table: dict[str, list],
    experiment_label: str,
    filename: str,
) -> str:
    """
    Plot minimum wealth and initial capital for the uniform viability check.

    Parameters
    ----------
    viab_table : dict
        Output of viability.viability_check_uniform.
    experiment_label : str
        Plot label.
    filename : str
        Output filename.

    Returns
    -------
    str
        Path to saved figure.
    """
    _ensure_dirs()
    m_vals = viab_table["m"]
    min_X_T = viab_table["min_X_T"]
    x_m_vals = viab_table["x_m"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].semilogy(m_vals, min_X_T, "s-", color="mediumseagreen", lw=2)
    axes[0].set_xlabel("Stress index m")
    axes[0].set_ylabel(r"$\inf_\theta X_{\pi^m}(T, \theta)$ (log scale)")
    axes[0].set_title(f"{experiment_label} — Uniform minimum wealth")
    axes[0].grid(True, alpha=0.4)

    axes[1].semilogy(m_vals, np.maximum(x_m_vals, 1e-300), "D-", color="tomato", lw=2)
    axes[1].set_xlabel("Stress index m")
    axes[1].set_ylabel(r"$x_m$ (log scale)")
    axes[1].set_title(f"{experiment_label} — Initial capital (→ 0)")
    axes[1].grid(True, alpha=0.4)

    fig.tight_layout()
    fpath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(fpath, dpi=120)
    plt.close(fig)
    return fpath
