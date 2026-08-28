[build-img]: https://github.com/LAMPSPUC/Sarimax.jl/actions/workflows/ci.yml/badge.svg?branch=master
[build-url]: https://github.com/LAMPSPUC/Sarimax.jl/actions/workflows/ci.yml

[codecov-img]: https://codecov.io/gh/LAMPSPUC/Sarimax.jl/branch/master/graph/badge.svg?token=6Zhd8Jiub3
[codecov-url]: https://codecov.io/github/LAMPSPUC/Sarimax.jl

[docs-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-url]: https://lampspuc.github.io/Sarimax.jl/dev/

# Sarimax.jl

| **Build Status** | **Coverage** | **Documentation** |
|:-----------------:|:-----------------:|:-----------------:|
| [![Build Status][build-img]][build-url] | [![codecov][codecov-img]][codecov-url]| [![docs][docs-img]][docs-url] |

Sarimax.jl estimates SARIMA/SARIMAX models by formulating them as **explicit
JuMP optimization problems**: coefficients and residuals are decision
variables, the model dynamics are constraints, and the objective function is
yours to choose. This design makes things that are hard-coded in classical
implementations — the loss function, coefficient constraints, regularization,
the solver itself — into swappable arguments of `fit!`, up to and including
**certified globally optimal estimates** via the SCIP solver.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/LAMPSPUC/Sarimax.jl")
```

## Quickstart

```julia
using Sarimax

airp = load_dataset(AIR_PASSENGERS)          # monthly TimeArray, 204 obs

model = auto(airp; seasonality = 12)         # Hyndman-Khandakar stepwise search
print(model)
```

```
SARIMA(4,1,1)(0,1,1)[12]
Seasonal form: multiplicative | Estimation: CSS (innovations) | Deterministic: none
───────────────────────────────────────
coefficient      estimate    std. error
───────────────────────────────────────
ar_1              -0.2466        0.1753
ar_2               0.0934        0.1658
ar_3               0.1033        0.1214
ar_4              -0.2527        0.0917
ma_1              -0.6844        0.1684
sma_1             -0.4761        0.0653
───────────────────────────────────────
σ² = 0.832211 | n = 191 | loglik = -250.429 | AIC = 520.045 | AICc = 520.657 | BIC = 542.811
```

```julia
predict!(model; stepsAhead = 12, displayConfidenceIntervals = true)
model.forecast                               # TimeArray: forecast, lower, upper

using Plots
plot(model)                                  # observed + fitted + forecast ribbon
```

Manual specification and estimation options:

```julia
airp_log = log.(airp)
m = SARIMA(airp_log, 0, 1, 1; seasonality = 12, P = 0, D = 1, Q = 1, allowMean = false)

fit!(m)                                            # CSS, multiplicative form (defaults)
fit!(m; initialization = :warmup)                  # R-compatible: matches arima(method="CSS")
fit!(m; objectiveFunction = "mae")                 # robust L1 loss
fit!(m; objectiveFunction = "elastic_net",         # penalized: λ[α‖·‖₁ + (1−α)/2‖·‖₂²]
     alpha = 0.5, lambda = 1.0)
fit!(m; invertible = true, stationary = true)      # constrained-by-construction estimates
fit!(m; optimizer = Sarimax.SCIP.Optimizer)        # certified global optimum

coef(m), coefnames(m), stderror(m)                 # StatsAPI accessors
residuals(m), nobs(m), loglike(m), aicc(m)
```

Diagnostics, Box-Cox and temporal cross-validation:

```julia
ljung_box_test(m)                            # residual autocorrelation (χ² test)
jarque_bera_test(m)                          # residual normality

λ = boxcox_lambda(airp; seasonality = 12)    # Guerrero's method (λ ≈ 0.13 here)
airp_bc = boxcox_transform(airp, λ)

cv = cross_validation(airp; initialTrainSize = 150, stepsAhead = 12,
                      fitFunction = train -> auto(train; seasonality = 12))
cv.rmse                                      # RMSE by forecast horizon
```

## Model formulation and comparability

Sarimax.jl differs from `forecast` (R) and `statsmodels` (Python) in ways you
should know before comparing outputs:

1. **Seasonal form.** The default is the **multiplicative** Box-Jenkins SARIMA,
   `φ(B)Φ(B^s)y'_t = θ(B)Θ(B^s)ε_t` — coefficients are directly comparable with
   R/statsmodels given the same estimation method (item 3). The pre-v0.3
   **additive** form (no cross terms) remains available via
   `seasonalForm = :additive`.
2. **Exogenous variables (ARX by default).** Regressors enter a
   **dynamic-regression/ARX** model — the AR terms act on the *observed* series,
   so the coefficient is an impact multiplier conditional on past `y`. R's
   `Arima(xreg=)` and statsmodels' `SARIMAX(exog=)` fit *regression with ARIMA
   errors* instead, where the coefficient is the usual marginal effect. These are
   different model families and the two coincide only when the autoregressive
   polynomial is unitary and there is no differencing. Both are available on
   `fit!`: `exogDynamics = :armax` (default) and
   `exogDynamics = :regression_errors`, the latter matching the reference
   implementations. Order selection through `auto` always uses the ARX form.
3. **Estimation (CSS, no Kalman filter).** Estimation is conditional least
   squares / concentrated conditional Gaussian ML as a JuMP problem. Log-likelihood
   and AIC/AICc/BIC follow the CSS convention with full Gaussian constants —
   comparable to R's `arima(..., method = "CSS")`, not to exact-ML defaults.
   The default conditioning convention is `initialization = :innovations`, which
   leaves the pre-sample block free and penalizes it in the objective so that the
   error is summed from `t = 1`. `:zeroed` (fix the block at zero) and `:warmup`
   (R-compatible) are also available.

**Verified against R** (`arima(log(y), order, seasonal, method = "CSS")`, this
repository's AirPassengers data, `initialization = :warmup` — pinned in CI):

| Specification | Coefficient | Sarimax.jl | R |
|---|---|---|---|
| (0,1,1)(0,1,1)[12] | θ | −0.787298 | −0.787298 |
| (0,1,1)(0,1,1)[12] | Θ | −0.714078 | −0.714076 |
| (1,1,1) | φ | 0.322774 | 0.322779 |
| (1,1,1) | θ | −0.824214 | −0.824212 |

## What the optimization formulation buys you

- **Swappable objectives**: `"mse"`, `"mae"` (L1), `"huber"`, `"ml"`
  (concentrated Gaussian CSS), `"ml_exact"` (exact treatment of the initial
  observations), `"ridge"`, `"elastic_net"` (penalized,
  `L(ε) + λ[α‖·‖₁ + (1−α)/2‖·‖₂²]`, with `α = 0` giving ridge and `α = 1` lasso),
  and `"stable"` (a tail-oriented criterion: the conditional value at risk of the
  squared errors, in the spirit of Bertsimas & Paskov's sample-robust regression).
  `"bilevel"` is deprecated as of v1.0 and will be removed in v2.0.
- **Constraints by construction**: `invertible = true` and `stationary = true`
  reparameterize the MA/AR coefficients through bounded reflection coefficients
  (Durbin-Levinson), guaranteeing invertibility/stationarity instead of
  checking it after the fact; `invertibilityMargin`/`stationarityMargin` keep
  roots away from the unit circle.
- **Solver choice**: Ipopt (default, local), SCIP (certified global optimum on
  small/moderate samples), Alpine, or any JuMP-compatible optimizer — same
  `fit!` call.
- **Outlier dummies inside `auto`** (`outlierDetection = true`), objective
  choice during order selection, and opt-in parallel candidate fitting
  (`parallel = true` for the grid search).

## Exogenous variables (SARIMAX)

```julia
gdp  = load_dataset(GDPC1)     # US real GDP, quarterly
nrou = load_dataset(NROU)      # noncyclical rate of unemployment (with projections)

y = gdp[1:300]                 # exog must extend beyond the training window
m = auto(y; exog = nrou, seasonality = 4)
predict!(m; stepsAhead = 8)    # uses the future NROU values
```

Remember item 2 above: this is an ARX specification, not regression with ARIMA
errors. To fit the regression-with-ARIMA-errors form, pass
`exogDynamics = :regression_errors` to `fit!` on a model of known order; see the
documentation for both model equations.

## Ecosystem

- **Tables.jl**: `load_dataset(table; timestampColumn = :date)` accepts any
  Tables.jl-compatible source.
- **StatsAPI**: `coef`, `coefnames`, `residuals`, `fitted`, `nobs`, `vcov`,
  `stderror`, `loglikelihood`.
- **Plots.jl**: `plot(model)` (RecipesBase recipe).
- **MLJ**: `SARIMAForecaster` (deterministic wrapper; exogenous variables via
  MLJ not yet supported).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs are very welcome; for non-trivial
changes please open an issue first. The changelog lives in
[CHANGELOG.md](CHANGELOG.md).

## References

- Hyndman, R.J., & Khandakar, Y. (2008). Automatic time series forecasting:
  The forecast package for R. *Journal of Statistical Software*, 26(3).
- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice* (3rd ed.). OTexts.
- Guerrero, V.M. (1993). Time-series analysis supported by power
  transformations. *Journal of Forecasting*, 12, 37–48.
