# Experimental Results

## Overview

This repository reproduces two pathwise, probability-free examples from the
mathematical finance literature that illustrate the **singleton- and multi-scenario
separation** between the growth–numéraire layer and viability.

- **Experiment 1** (Example 4.16) — singleton scenario: a single analytic sine path.
- **Experiment 2** (Example 4.17) — multi-scenario: a one-parameter family of cubic
  paths indexed by amplitude θ.

In both cases the extracted local characteristics all vanish and the canonical
portfolio is zero (*benign growth–numéraire layer*), yet a sequence of admissible
portfolios generates unbounded terminal wealth (*viability failure*).

---

## Experiment 1 — Example 4.16: Singleton-Scenario Separation

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| h | 1.0 | Observation window |
| N | 4 | Faber–Schauder depth |
| T | 2.0 | Horizon |
| δ = h·2⁻ᴺ | 0.0625 | Sine period |
| ω = 2π/δ | ≈ 100.53 | Angular frequency |

### Path

R(t) = sin(2πt/δ)  with  Ṙ(t) = (2π/δ)·cos(2πt/δ)

The path is δ-periodic and C¹ (smooth), so its quadratic variation along any
nested partition sequence converges to zero.

### Identity Checks (all numerically zero)

| Quantity | Analytic value | Numerical max-abs |
|----------|---------------|-------------------|
| α^{h,D}(t) — trend signal | 0 | 3.45 × 10⁻¹³ |
| A^{h,D}(t) — accumulated trend | 0 | 7.95 × 10⁻¹⁵ |
| C^{h,D}(t) — quadratic covariation | 0 (analytic) | converges → 0 |
| O^{h,D}(t) — operational clock | 0 | 0 |
| a^{h,D}(t) — local drift density | 0 | 0 |
| c^{h,D}(t) — local diffusion density | 0 | 0 |
| g^{h,D}(t) — maximal growth | 0 | 0 |
| G^{h,D}(T) — cumulative maximal growth | 0 | 0 |
| ν^{h,D}(t) — canonical portfolio | 0 | 0 |

The last-cell Faber–Schauder extractor returns zero because R is δ-periodic,
so R(t) = R(t − δ) exactly, and the slope 2ᴺ[R(t) − R(t − δ)] = 0.

### Quadratic Variation Convergence

The QV converges to zero at rate O(Δt), consistent with a C¹ path.  Average
half-step ratio C_{n+1}/C_n ≈ 0.514 (expected 0.5).

```
n= 7  steps=   128  C_n(T)=1.28e+02
n= 8  steps=   256  C_n(T)=7.50e+01
n= 9  steps=   512  C_n(T)=3.90e+01
n=10  steps=  1024  C_n(T)=1.97e+01
n=11  steps=  2048  C_n(T)=9.86e+00
n=12  steps=  4096  C_n(T)=4.93e+00
n=13  steps=  8192  C_n(T)=2.47e+00
n=14  steps= 16384  C_n(T)=1.23e+00
n=15  steps= 32768  C_n(T)=6.17e-01
```

### Stress Portfolios (Föllmer Integral)

Integral of |Ṙ|² over [0,T]:

  ∫₀ᵀ |Ṙ(t)|² dt = (2π/δ)²·T/2 = 2π²T/δ² = 1024π² ≈ 10,106.47

| m | Föllmer return (exact) | Föllmer return (numeric) | log X_T |
|---|------------------------|--------------------------|---------|
| 1 | 10,106.47 | 10,106.46 | 10,106.47 |
| 2 | 20,212.95 | 20,212.92 | 20,212.95 |
| 5 | 50,532.37 | 50,532.30 | 50,532.37 |
| 10 | 101,064.75 | 101,064.59 | 101,064.75 |

- **Gamma_{π^m}(T) = 0** for all m (confirmed numerically).
- **X_{π^m}(T) = exp(m · 1024π²) → ∞** (overflows float64 for all m ≥ 1).
- **x_m = 1/X_{π^m}(T) → 0** machine-exactly for all m ≥ 1.

### Viability Failure

A terminal lump-sum withdrawal K = 1_{{T}} is financed from x_m → 0:

| m | x_m | X_T | V_T | Financed |
|---|-----|-----|-----|----------|
| 1 | 0.0 (underflow) | ∞ (overflow) | 0 | ✓ |
| 2 | 0.0 | ∞ | 0 | ✓ |
| … | … | … | … | ✓ |

**Conclusion:** The growth–numéraire layer holds exactly (all characteristics
zero, canonical portfolio zero), while viability fails because the stress
portfolio sequence produces unbounded terminal wealth on the single smooth
scenario, and the nonzero terminal liability K is financed from vanishing
initial capital x_m → 0.

### Figures

| File | Description |
|------|-------------|
| `figures/exp1_sine_path.png` | Analytic sine path R(t) on [0, T] |
| `figures/exp1_qv_convergence.png` | Discrete QV C_n(T) vs. partition levels n |
| `figures/exp1_wealth_vs_m.png` | log X_{π^m}(T) linear in m (exact vs. numeric) |

---

## Experiment 2 — Example 4.17: Multi-Scenario Separation

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| h | 1.0 | Observation window |
| T | 2.0 | Horizon |
| τ | 1.5 | Branch time (must satisfy τ > T − h = 1.0 ✓) |
| θ_min | 0.5 | Lower amplitude bound |
| Θ_* | [−1,−0.5] ∪ [0.5,1] | Scenario parameter set (12 sampled points) |

### Scenario Paths

R(t, θ) = θ · ((t − τ)⁺)³,  t ∈ [−h, T]

D⁻R(t, θ) = 3θ · ((t − τ)⁺)²   (backward derivative)

Each path is C¹ on [0, T] (zero for t ≤ τ, then a cubic branch).

### Identity Checks (all zero for every θ ∈ Θ_*)

| Quantity | Value |
|----------|-------|
| α^{h,D_lag}(t, θ) | 0 (exact, because t − h < τ ∀t ∈ [0,T]) |
| A^{h,D}(t, θ) | 0 |
| C^{h,D}(t, θ) | 0 (analytic; numeric ≤ 1e-5 at 2^15 points) |
| O^{h,D}(t, θ) | 0 |
| a, c, g, G, ν | all 0 |

The lag extractor D_lag(x) = x(0) applied to the window path R^{t,h}
gives α = (1/h)·R(t − h, θ) = 0, because t − h < τ ensures the
cubic is still in its zero phase.

### Quadratic Variation Convergence (θ = θ_min = 0.5)

```
n= 6  steps=    64  C_n(T)=4.39e-04
n= 8  steps=   256  C_n(T)=1.10e-04
n=10  steps=  1024  C_n(T)=2.75e-05
n=12  steps=  4096  C_n(T)=6.87e-06
```
Confirmed < 1e-5 for all θ ∈ Θ_* at 2^15 partition points.

### Symbolic Verification

Using SymPy:

  ∫_τ^T [3θ(t−τ)²]² dt = 9θ²(T−τ)⁵/5

For θ_min = 0.5, T − τ = 0.5:  9·0.25·0.03125/5 = 0.014063

### Stress Portfolios and Terminal Wealth

π^m(t, θ) = m · D⁻R(t, θ) = 3mθ·((t−τ)⁺)²

Föllmer return: R_{π^m}(T, θ) = (9mθ²/5)·(T−τ)⁵

Terminal wealth: X_{π^m}(T, θ) = exp((9mθ²/5)·(T−τ)⁵)

Selected results (exact vs. numeric, relative error < 1e-3):

| m | θ | X_T (exact) | X_T (numeric) | X_T lower bound |
|---|---|------------|---------------|-----------------|
| 1 | 0.5 | 1.0142 | 1.0142 | 1.0142 |
| 1 | 1.0 | 1.0579 | 1.0579 | 1.0142 |
| 5 | 0.5 | 1.0728 | 1.0728 | 1.0728 |
| 5 | 1.0 | 1.3248 | 1.3248 | 1.0728 |
| 10 | 0.5 | 1.1510 | 1.1510 | 1.1510 |
| 10 | 1.0 | 1.7551 | 1.7551 | 1.1510 |

- **Gamma_{π^m}(T, θ) = 0** for all m and all θ (C = 0 ⇒ C_{π^m} = 0).

### Viability Failure (Uniform across Θ_*)

Uniform lower bound: inf_{θ ∈ Θ_*} X_{π^m}(T, θ) ≥ exp((9m θ_min²/5)(T−τ)⁵)

| m | x_m | min X_T | x_m · min X_T | All financed |
|---|-----|---------|---------------|--------------|
| 1 | 9.86e-01 | 1.014 | 1.0 | ✓ |
| 10 | 8.69e-01 | 1.151 | 1.0 | ✓ |
| 50 | 4.95e-01 | 2.020 | 1.0 | ✓ |
| 100 | 2.45e-01 | 4.081 | 1.0 | ✓ |
| 200 | 6.01e-02 | 16.65 | 1.0 | ✓ |
| 500 | 8.84e-04 | 1131 | 1.0 | ✓ |
| 1000 | 7.81e-07 | 1.28e6 | 1.0 | ✓ |

x_m → 0 (confirmed: last entry ≈ 7.8e-7 < 1e-3 tolerance).

**Conclusion:** All local growth–numéraire identities hold with zero canonical
portfolio uniformly across all scenarios θ ∈ Θ_*. The non-anticipative stress
portfolio π^m (based purely on the realized backward derivative) generates
terminal wealth that uniformly diverges to +∞, and the nonzero terminal
liability K = 1_{{T}} is financed from initial capital x_m → 0, proving
uniform viability failure.

### Figures

| File | Description |
|------|-------------|
| `figures/exp2_cubic_paths.png` | Cubic scenario paths for representative θ values |
| `figures/exp2_qv_convergence.png` | QV convergence C_n(T) for θ = θ_min |
| `figures/exp2_wealth_uniform.png` | x_m and min X_T vs. m (viability failure) |

---

## Test Suite Summary

All **55 tests** pass (`pytest tests/ -v`):

- `tests/test_exp1.py` — 29 tests covering: sine path, Faber–Schauder basis,
  last-cell extractor, quadratic variation, local characteristics, Föllmer
  integrals, wealth, viability.
- `tests/test_exp2.py` — 26 tests covering: cubic paths, lag extractor, QV
  convergence, Föllmer return formula (exact + numeric), uniform financing,
  viability.

---

## Reproduction Instructions

```bash
# Install dependencies (already present in container)
pip install numpy scipy sympy matplotlib pandas

# Run experiments
python exp/exp_1.py
python exp/exp_2.py

# Run full test suite
pytest tests/ -v
```

Figures are saved automatically to `results/figures/`.
