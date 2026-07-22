# Computational Verification of Examples 4.16 & 4.17: Singleton and Multi-Scenario Separation

This repository reproduces the **singleton-scenario** (Example 4.16) and **multi-scenario**
(Example 4.17) separation results between growth–numéraire identities and viability in a
fully pathwise, probability-free setting.

## Experiments

### Experiment 1 — Example 4.16 (Singleton Scenario)

Constructs a single smooth return path `R(t) = sin(2πt/δ)` on `[-h, T]` with
`δ = h · 2^{-N}` and verifies:

* The last-cell Faber–Schauder extractor `D_N^{last}` produces **α = 0** everywhere because
  `R` is δ-periodic.
* All local characteristics `A, C, O, a, c, g, G, ν` are **identically zero**.
* Stress portfolios `π^m(t) = m · Ṙ(t)` generate Föllmer return
  `R_{π^m}(T) = m · ∫|Ṙ|² dt > 0` with terminal wealth
  `X_{π^m}(T) = exp(m · ∫|Ṙ|² dt) → ∞`.
* Viability fails: `K = 1_{T}` is financed from `x_m → 0`.

### Experiment 2 — Example 4.17 (Multi-Scenario)

Constructs the cubic scenario family `R(t, θ) = θ · ((t-τ)⁺)³` on a genuine
multi-scenario path space and verifies:

* The lag extractor `D_lag(x) = x(0)` produces **α = 0** uniformly across all `θ ∈ Θ_*`
  because `τ > T - h`.
* All local characteristics vanish uniformly across scenarios.
* Non-anticipative stress portfolios `π^m(t, θ) = m · D⁻R(t, θ)` generate
  `X_{π^m}(T, θ) = exp((9mθ²/5)(T-τ)⁵)` with
  `inf_{θ} X_{π^m}(T, θ) → ∞` as `m → ∞`.
* Viability fails uniformly: `x_m = exp(-(9mθ_min²/5)(T-τ)⁵) → 0`.

## Repository Structure

```
.
├── data/               # placeholder (no external data required)
├── src/                # core modules
│   ├── path_generators.py   # analytic sine and cubic paths
│   ├── partitions.py        # dyadic partition generators
│   ├── extractors.py        # Faber–Schauder basis, D_N^last, D_lag
│   ├── qv.py                # discrete quadratic-variation approximations
│   ├── densities.py         # backward density selector D_O^-
│   ├── portfolios.py        # stress portfolios and Föllmer integrals
│   ├── viability.py         # viability failure checks
│   └── reports.py           # figures and tables
├── exp/
│   ├── exp_1.py             # Experiment 1 runner
│   └── exp_2.py             # Experiment 2 runner
├── tests/
│   ├── test_exp1.py         # unit + integration tests for Exp 1
│   └── test_exp2.py         # unit + integration tests for Exp 2
├── results/
│   ├── figures/             # saved plots (PNG)
│   └── RESULTS.md           # summary of findings
└── README.md
```

## Parameters (defaults; not uniquely prescribed by the paper)

| Parameter | Exp 1 | Exp 2 |
|-----------|-------|-------|
| `h`       | 1.0   | 1.0   |
| `T`       | 2.0   | 2.0   |
| `N`       | 4     | —     |
| `δ`       | 1/16  | —     |
| `τ`       | —     | 1.5   |
| `θ_min`   | —     | 0.5   |
| `m` range | 1–10  | 1–10  |

## Installation

```bash
pip install numpy scipy matplotlib pandas sympy pytest
```

## Running

```bash
# Run both experiments
python exp/exp_1.py
python exp/exp_2.py

# Run tests
pytest tests/ -v
```

## Key Mathematical Objects

| Symbol | Meaning |
|--------|---------|
| `D_N^{last}` | Last-cell Faber–Schauder slope extractor |
| `D_lag` | Lag (left-endpoint) extractor |
| `α^{h,D}` | Non-anticipative trend signal |
| `[R,R]_Π` | Pathwise quadratic variation along partition Π |
| `D_O^-` | Backward density selector w.r.t. operational clock O |
| `ν^{h,D}` | Canonical portfolio |
| `π^m` | Stress portfolio (proportional to path derivative) |
| `X_{π^m}(T)` | Pathwise Doléans exponential (terminal wealth) |
