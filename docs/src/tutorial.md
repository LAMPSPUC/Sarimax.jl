# Tutorial: a complete forecasting workflow

This walkthrough follows the classical Box-Jenkins cycle in the style of
*Forecasting: Principles and Practice*: **identify → estimate → diagnose →
forecast → validate**, using the monthly air-passengers series shipped with
the package.

```julia
using Sarimax

airp = load_dataset(AIR_PASSENGERS)   # TimeArray, 204 monthly observations
airp_log = log.(airp)                 # variance grows with the level → log
```

If you prefer a data-driven transformation, Guerrero's method selects a
Box-Cox λ (here close to the log):

```julia
λ = boxcox_lambda(airp; seasonality = 12)   # ≈ 0.13
airp_bc = boxcox_transform(airp, λ)
# ... after forecasting, map back with inverse_boxcox(z, λ)
```

## 1. Identification: how much differencing?

The package chooses the seasonal differencing order `D` with the OCSB test
(the pmdarima convention) and the regular order `d` by repeated KPSS tests:

```julia
kpss_test(values(airp_log))            # level stationarity? (small p ⇒ difference)
ocsb_test(values(airp_log); m = 12)    # "seasonal_difference" ⇒ D

diff_series = differentiate(airp_log, 1, 1, 12)   # (1-B)(1-B¹²) y
kpss_test(values(diff_series))                    # now stationary
```

You rarely need to do this by hand — `auto` runs both tests internally.

## 2. Estimation

Fit a specific model, or let the Hyndman-Khandakar stepwise search choose:

```julia
airline = SARIMA(airp_log, 0, 1, 1; seasonality = 12, P = 0, D = 1, Q = 1,
                 allowMean = false)
fit!(airline)
print(airline)          # coefficient table with CSS standard errors

best = auto(airp_log; seasonality = 12)   # automatic order selection (AICc)
```

Estimation is conditional least squares formulated as a JuMP optimization
problem. That formulation is the extension point:

```julia
fit!(airline; initialization = :warmup)          # match R's arima(method = "CSS")
fit!(airline; objectiveFunction = "mae")         # robust L1 objective
fit!(airline; invertible = true)                 # invertible MA by construction
fit!(airline; stationary = true)                 # stationary AR by construction
fit!(airline; optimizer = Sarimax.SCIP.Optimizer)  # certified global optimum
```

Model quantities follow the StatsAPI conventions:

```julia
coef(airline)        # estimates (order: mean, drift, ar, ma, sar, sma, exog)
coefnames(airline)
stderror(airline)    # CSS asymptotics via a numerical Hessian
residuals(airline)
aicc(airline)        # CSS-convention information criteria
loglike(airline)
```

## 3. Diagnostics

A well-specified model leaves white, approximately normal residuals:

```julia
lb = ljung_box_test(airline)     # H₀: no residual autocorrelation
lb["p_value"] > 0.05             # want: true

jb = jarque_bera_test(airline)   # H₀: residual normality
jb["p_value"]
```

If the Ljung-Box test rejects, revisit the specification (often a missing
seasonal term).

## 4. Forecasting

```julia
predict!(airline; stepsAhead = 12, displayConfidenceIntervals = true,
         confidenceLevel = 0.95)
airline.forecast     # TimeArray with columns: forecast, lower, upper
```

The interval variances propagate the uncertainty through re-integration —
for an ARIMA(0,1,0) they grow as σ²·h, as theory requires. To visualize:

```julia
using Plots
plot(airline)        # observed + in-sample fit + forecast with its band
```

For scenario analysis, `simulate` draws full sample paths instead of a point
forecast:

```julia
scenarios = simulate(airline, 12, 500)   # 500 paths, 12 steps each
```

## 5. Validation: temporal cross-validation

Rolling-origin evaluation (expanding window) gives an honest out-of-sample
error profile by horizon:

```julia
cv = cross_validation(airp_log; initialTrainSize = 150, stepsAhead = 12,
                      fitFunction = train -> auto(train; seasonality = 12))
cv.rmse    # 12-vector: RMSE at horizons 1…12
cv.mae
```

## 6. Exogenous variables (SARIMAX)

Exogenous regressors must cover the forecast horizon at fit time:

```julia
gdp  = load_dataset(GDPC1)
nrou = load_dataset(NROU)     # includes FRED projections beyond the GDP sample

y = gdp[1:300]
m = auto(y; exog = nrou, seasonality = 4)
predict!(m; stepsAhead = 8)
```

!!! warning "ARX by default, not regression with ARIMA errors"
    Sarimax.jl's default exogenous specification is a dynamic regression (ARX):
    the AR terms act on the observed series, so the coefficient is an impact
    multiplier conditional on past `y`. R's `Arima(xreg=)` and statsmodels'
    `SARIMAX(exog=)` fit regression-with-ARIMA-errors, where the coefficient is
    the usual marginal effect. To match them, pass
    `exogDynamics = :regression_errors` to `fit!`:

    ```julia
    m = SARIMA(y, 1, 1, 1; exog = nrou)
    fit!(m; exogDynamics = :regression_errors)
    ```

    The two forms coincide only when the autoregressive polynomial is unitary and
    there is no differencing. The keyword is currently available on `fit!` only:
    order selection through `auto` always uses the ARX form.

## Where to go next

- [API Reference](@ref) for every exported function.
- The README's "Model formulation and comparability" section for the exact
  estimation conventions and the verified-against-R coefficient table.
