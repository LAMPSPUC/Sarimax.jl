# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased

### Changed
- **BREAKING — multiplicative seasonal form is the new default.** `fit!` and `auto`
  now estimate the Box-Jenkins multiplicative SARIMA
  `φ(B)Φ(B^s)y' = θ(B)Θ(B^s)ε` (cross terms included). Models with `p·P > 0` or
  `q·Q > 0` produce different (now R/statsmodels-comparable) coefficients than
  v0.2. The previous additive behavior is available via `seasonalForm = :additive`.
  `toMA`, `forecastErrors`, prediction and the stationarity/invertibility checks
  are all form-aware. Notably, the airline-model θ no longer piles up at the
  unit-root boundary — that was an artifact of the additive form.
- `predict` applies the multiplicative cross terms in the forecast recursion.

### Added
- `seasonalForm::Symbol` keyword (`:multiplicative` default, `:additive`) on `fit!`
  and `auto`; `:free` reserved for a future release.
- **`initialization::Symbol` keyword** on `fit!`/`auto`: `:zeroed` (default —
  pre-sample residuals fixed at zero, warm-up observations dropped) or `:warmup`
  (R-compatible: conditions only on the AR-side lags and warm-starts the MA
  recursion from the beginning of the differenced sample). Under `:warmup`,
  coefficients match R's `arima(..., method = "CSS")` to ~1e-5 on the airline and
  ARIMA(1,1,1) specifications (pinned in `test/reference_values.jl`). Exact
  (Kalman) initialization remains out of scope by design.
- **Cross-implementation reference tests**: coefficient fixtures generated from R
  `arima(method = "CSS")` on the AirPassengers dataset, asserted in CI.
- **Real drift term**: `allowDrift` now adds the differentiated deterministic-time
  regressor (a genuine linear trend for `d+D = 0`, the classic constant-in-differences
  for `d = 1`, with a warning for `d+D > 1` where it is not identifiable). `allowMean`
  and `allowDrift` are now mutually exclusive — they were perfectly collinear before.
- **`stationary::Bool` keyword** on `fit!`/`auto` (with `stationarityMargin`): AR
  coefficients generated from bounded reflection coefficients via the Levinson
  recursion (`reflectionToAR`) — stationarity by construction (exact under
  `:multiplicative`).
- **Residual diagnostics**: `ljung_box_test` and `jarque_bera_test` (vector and
  fitted-model methods).
- **Box-Cox**: `boxcox_transform`, `inverse_boxcox`, and `boxcox_lambda` (Guerrero's
  method, as in `forecast::BoxCox.lambda`).
- **Temporal cross-validation**: `cross_validation` — rolling-origin/expanding-window
  evaluation with per-horizon MAE/RMSE.
- **Readable model display**: `show`/`print` now render a summary with the
  specification, seasonal form, estimation convention, a coefficient table with CSS
  standard errors, and fit statistics.
- **`parallel::Bool` keyword** on `auto` (experimental): fits candidate models
  across Julia threads in the "grid" and "stepwiseNaive" searches.
- **Tables.jl input**: `load_dataset(table; timestampColumn = :date)` accepts any
  Tables.jl-compatible source.
- **Plots.jl recipe** (via RecipesBase): `plot(model)` draws the observed series,
  in-sample fit and forecast with its confidence band.
- **MLJ interface**: `SARIMAForecaster` (MLJModelInterface deterministic wrapper;
  exogenous variables via MLJ not yet supported).
- CONTRIBUTING.md; CI now also runs on macOS (Apple Silicon).
- **Aqua.jl quality checks** in the test suite.
- `auto` discards candidates whose solver did not terminate successfully.
- **StatsAPI interface**: `coef`, `coefnames`, `residuals`, `nobs`, `fitted`,
  `vcov`, `stderror`; `loglikelihood` now extends the StatsAPI generic.
- **Standard errors**: CSS asymptotics via a numerical Hessian of the residual sum
  of squares over a pure-Julia replica of the fit recursion (`Sarimax.cssResiduals`),
  `Var(θ̂) ≈ 2σ̂²H⁻¹` — validated against the AR(1) theory value.

### Internal
- `stepwiseSearch`'s 16 unrolled neighbour blocks (~700 lines of copy-paste)
  replaced by a `tryCandidate!` closure and an explicit move table — verified
  selection-equivalent (same model, same AICc) on the AirPassengers benchmark.
- `gridSearch` restructured (candidate list + fit + selection passes) to support
  parallel fitting.

### Deprecated
- Public API renamed to snake_case: `load_dataset`, `split_train_test`,
  `has_fit_methods`, `has_hyperparameters_methods`, `get_hyperparameters_number`,
  `automatic_differentiation`, `identify_granularity`, `build_datetimes`,
  `copy_time_array`, `deepcopy_time_array`, `to_ma`, `differentiated_coefficients`.
  The camelCase names keep working with a deprecation warning until v1.0.
  Keyword-argument names are unchanged in this release.

## [0.2.0] - Unreleased

### Fixed
- **Exogenous forecasting**: `predict` used the exogenous row of the *last* forecast
  horizon for every step; each step now uses its own row. Multi-step SARIMAX
  forecasts with non-constant exogenous variables were wrong before this fix.
- **Forecast variances**: prediction-interval variances are now propagated through
  re-integration — the ψ-weights include the differencing operator
  `(1-B)^d (1-B^s)^D`. Intervals for models with `d + D ≥ 1` were previously too
  narrow (e.g. ARIMA(0,1,0) now correctly yields `Var[h] = σ²·h`).
- **Stepwise search**: a no-op statement (`constant != constant`) prevented the
  constant-toggle step from ever updating its state; `stepWiseSearchNaive` no longer
  crashes when no stationary/invertible initial candidate exists.
- **Short-series prediction**: the seasonal AR term in `predict` is now bounds-guarded
  (was a `BoundsError`).
- **OCSB test**: internal cleanup and debug-logging; the lag-selection information
  criterion call is kept in the argument order that reproduces pmdarima's selection
  on all test fixtures (verified — see comment in `src/statistical_tests.jl`).

### Changed
- **Log-likelihood / information criteria (CSS convention, declared)**: `loglike` /
  `loglikelihood` now return the conditional (CSS) Gaussian log-likelihood with full
  constants, `ℓ = -n/2·(log 2π + 1 + log(RSS/n))`, over the `n` effective residuals.
  `aic = 2K - 2ℓ`, `bic = K·log(n) - 2ℓ`, `aicc` accordingly. These are comparable to
  R's `arima(..., method = "CSS")`, not to exact-likelihood defaults of R/statsmodels.
- **Parameter counting**: `getHyperparametersNumber` counts every declared parameter
  (+ σ²) for regular objectives; the active-coefficient (|coef| > 1e-5) count is now
  used only for elastic-net fits, where it estimates effective degrees of freedom.
- **`auto` search comparability**: all candidate models are conditioned on the same
  pre-sample length (`minConditioningObs`), so their information criteria are computed
  on the same effective sample. The `icOffset` machinery was removed.
- **`"ml"` objective**: rewritten as the CONCENTRATED conditional Gaussian likelihood —
  σ is profiled out analytically (σ̂² = RSS/n), making the coefficient estimation
  equivalent to least squares over the effective sample. The previous free-σ
  formulation was degenerate on (near-)noise-free data (RSS → 0 drives σ → 0 and the
  objective to +∞ — Ipopt returned `OTHER_ERROR` and garbage coefficients; this was
  the root cause of the historically failing AR-recovery test). The `μ` constraint
  hack and the pre-sample residual inflation were removed.
- **Test harness**: `runtests.jl` now wraps all files in one outer `@testset`. Before,
  the first failing top-level testset aborted the run, so every test file after
  `sarima_fit.jl` had never actually executed in a full run.
- Solver termination status is checked after optimization (warning on non-success)
  and stored in `model.metadata["solverStatus"]`.

- **`auto` default search**: `searchMethod` now defaults to `"stepwise"` (what the
  documentation always stated). The previous silent default `"sarimax"` performs no
  search — it fits a single dense specification and is now documented as intended
  for regularized estimation.
- Internal `predict` and `predict!` now agree on `isSimulation = false` as default.

### Removed
- `icOffset` computation in `auto` (a base-model fit used only to shift IC levels; it
  cancelled in rankings).
- Unused dependencies `Requires`, `Combinatorics` and `Pkg`; `Revise` removed from the
  test target. Julia compat corrected from the impossible `"1.0 - 1.11"` to `"1.10"`.

### Documentation
- New "Model formulation and comparability" section (README and docs): additive
  seasonal form, ARX exogenous family, CSS estimation/IC conventions, and what the
  JuMP formulation buys. Factually wrong docstrings fixed (`differentiate` d/D
  restriction note, `kpss_test` default lag method, NROU dataset description,
  `fit!` estimation-method description).
