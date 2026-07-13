# Diagnostic: the `sim_sarimax` exogenous discrepancy (B4) — RESOLVED

Date: 2026-06-27. Scripts: `scripts/diagnose_exog.py`, raw: `results/raw/validation/exog_diagnostic.jsonl`.

## Symptom

On the simulated SARIMAX series, the original validation table showed:

| impl | φ | β=(b1,b2) | RSS | σ² |
|------|----|-----------|-----|----|
| SARIMAX.jl | 0.520 | (1.534, −0.684) | 295.6 | 0.995 |
| statsmodels | 0.659 | (1.168, −0.505) | 451.5 | 1.51 |
| R forecast | 0.658 | (1.170, −0.506) | 451.3 | — |

The true DGP (from `gen_sim_data.py`) is `y_t = φ y_{t-1} + b1·x1_t + b2·x2_t + e_t` with
φ=0.5, β=(1.5, −0.8), σ²=1.0.

## Root cause: two different model families (not a bug)

The discrepancy is a **specification mismatch**, not a defect.

- **SARIMAX.jl** builds (verified in `src/models/sarima.jl:707-716` and the constraint
  `y_t = ŷ_t + ε_t` at `:884`) a **dynamic-regression / ARX** model — the AR term acts on the
  *observed* series:
  `y_t = c + Σφ_i y_{t-i} + Σβ_j x_{j,t} + Σθ_j ε_{t-j} + ε_t`.
- **statsmodels `SARIMAX(exog=…)`** and **R `Arima(xreg=…)`** fit **regression with ARIMA errors** —
  the AR term acts on the regression *residual*:
  `y_t = c + Σβ_j x_{j,t} + u_t`, `u_t = Σφ_i u_{t-i} + ε_t`.

Expanding the second form introduces a cross term `−φ·X_{t-1}·β` absent from the first. The two are
genuinely different models. The shared DGP is ARX, so SARIMAX.jl matches it and recovers the true
parameters; statsmodels/R fit a model that is misspecified *for this DGP*, attenuating β and inflating σ².

## Confirmation (like-for-like ARX in every tool)

Emulating ARX inside statsmodels/OLS (lagged `y` as an ordinary regressor, no AR-error structure)
reproduces SARIMAX.jl exactly:

| model | family | φ | b1 | b2 | RSS | σ² |
|-------|--------|----|----|----|-----|----|
| TRUE DGP | ARX | 0.500 | 1.500 | −0.800 | — | 1.00 |
| statsmodels reg+ARIMA-errors | reg-w-ARIMA-errors | 0.659 | 1.168 | −0.505 | 451.55 | 1.51 |
| statsmodels ARX (lagged-y) | ARX | 0.520 | 1.534 | −0.684 | 295.61 | 0.99 |
| OLS ARX (closed form) | ARX | 0.520 | 1.534 | −0.684 | 295.61 | 1.00 |
| **SARIMAX.jl (native)** | ARX | 0.520 | 1.534 | −0.684 | 295.61 | 1.00 |

SARIMAX.jl, statsmodels-ARX, and closed-form OLS agree to printed precision. SARIMAX.jl's σ²≈1.0 and
β≈(1.53,−0.68) recover the true DGP.

## Resolution applied

The exogenous validation row now compares **like-for-like ARX** across all three tools
(`run_validation_python.py` / `run_validation_r.R` fit lagged-y-as-regressor ARX for `sim_sarimax`;
SARIMAX.jl is ARX natively). The regenerated `table_validation` shows φ=0.5203, RSS=295.61,
logLik=−422.56 for all three.

## Manuscript implication

- **Supported:** SARIMAX.jl's exogenous estimation is correct for the dynamic-regression (ARX) model
  it implements, and agrees with statsmodels/OLS when the same model is fit. The earlier divergence
  was an apples-to-oranges comparison.
- **Must state explicitly:** SARIMAX.jl's exogenous specification is ARX (AR on the observed series),
  **not** regression-with-ARIMA-errors. Do not compare its exogenous coefficients directly against
  default statsmodels/R `xreg` output without matching the model family.
- AIC/BIC remain non-comparable across tools (different constants; B3); coefficients, RSS, and
  log-likelihood agree under the matched ARX specification.
