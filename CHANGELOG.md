# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-27

First stable release. It accompanies the paper describing the package and freezes
the API and the estimation defaults that the reported results were produced under.

### Changed (breaking)
- **`elastic_net` is now the conventional penalized estimator.** The objective is
  the selected residual loss plus `lambda * [alpha * L1 + (1 - alpha)/2 * L2]`
  over the autoregressive, moving-average and exogenous coefficient blocks, solved
  in a single stage. The intercept and the drift are excluded. `alpha = 0` gives a
  ridge-type penalty and `alpha = 1` a lasso-type one; `lambda` defaults to the
  square root of the effective sample size, matching the scale of the sum-form
  objective.

  This replaces a two-stage construction that solved conditional least squares and
  then minimized an adaptively weighted coefficient norm subject to a tolerance on
  the residual sum of squares. That construction had no penalty multiplier, so a
  caller-supplied `lambda` skipped the second stage entirely and silently returned
  an unregularized fit. `regularizationObjective` is removed.
- **`exogDynamics` defaults to `:armax` again**, the dynamic-regression/ARX form in
  which the exogenous coefficient is an impact multiplier conditional on past `y`.
  `:regression_errors` remains available on `fit!` and is the form
  `forecast::Arima(xreg=)` estimates, in which the coefficient is the usual
  marginal effect; on a discriminator over three data generating processes it
  closes a log-RMSE gap of 0.4262 against R to +0.0020 under an ARIMA-errors
  process, with the coefficient agreeing to three decimals. The default is only
  observable when regressors are present: at `nExog == 0` both modes take the same
  branch.

### Deprecated
- **`objectiveFunction = "bilevel"`** warns once per session and is scheduled for
  removal in v2.0. It optimizes the moving-average coefficients in an outer loop
  rather than as decision variables, at orders of magnitude more solver time, and
  covers no case the other objectives do not.

### Documentation
- Source comments and docstrings are in technical English throughout, with
  development narrative removed. The README quickstart output is regenerated under
  the shipped defaults, the exogenous section states which form is the default and
  where the alternative is reachable, and the documentation now carries the
  elastic-net equation and a "Known limitations" section covering the exact-likelihood
  fallback ([#15]), the determinant exponent ([#14]) and the scope of `cssResiduals`.

### Internal
- GitHub Actions bumped to current major versions (checkout v4, cache v4,
  setup-julia v2, codecov-action v5).

[#14]: https://github.com/LAMPSPUC/Sarimax.jl/issues/14
[#15]: https://github.com/LAMPSPUC/Sarimax.jl/issues/15

## [0.3.0] - 2026-07-15

### Fixed
- **`identifyOutliers` no longer flags every value that is not bit-identical to the
  quartiles.** When the interquartile range is zero the IQR fences collapse onto `q1` and
  `q3`, so the rule degenerates into an equality test against a float. `detectOutliers`
  feeds it residuals that are JuMP variables tied by an equality constraint and satisfied
  only up to the solver's tolerance, so which residuals count as "identical" is decided by
  the last bit and therefore by the machine the solver ran on. Measured on the constant
  fixture the old `detectOutliers` test used (`ones(31)` with one spike): 30 of the 31
  residuals came out bit-identical here and exactly one outlier was reported, while shifting
  three of them by 1-2 ULP — the difference between one runner and another — turns the
  answer into four. That is the whole flake; the same two assertions failed with the same
  values on unrelated branches, on whichever CI job happened to land on the other side.

  `identifyOutliers` now returns no outliers when the IQR is below `DEGENERATE_IQR_RTOL`
  (1e-8) times `max(|q1|, |q3|)`. The tolerance is **relative** to the data's own scale, and
  the comparison is `<=` so that `q1 == q3 == 0` is covered too. Zero dispersion is zero
  evidence of atypicality.

  This inverts the contract for constant-plus-spike inputs — they now yield nothing — which
  is why the `detectOutliers` fixture had to change with it: it now carries real dispersion
  (IQR = 2 on a 10..14 pattern) plus an unambiguous spike, leaving a measured margin of
  ~2.0 in data units between the outermost inlier and the fence, against solver noise of
  order 1e-8. The degenerate case itself is covered directly, as a unit test over
  constructed vectors, in `identifyOutliers dispersao degenerada`.
- **The exact likelihood removed the wrong deterministic term.** Two conversions were
  missing in `exactLoglike`, both verified against `stats::arima(method = "ML")`:
  - `model.c` is the regression **constant**, not the mean. The level to remove is
    `mu = c / (1 - sum(ar))`. Measured on M4 monthly series 44895: subtracting `c` gave
    a log-likelihood of -2305.891, subtracting `mu` gave -2304.253, and R gives
    -2304.253 — an error of 1.64, i.e. 3.3 AICc units, enough to flip a selection.
  - `model.trend` **multiplies** the differenced time regressor (`trend * driftValues[t]`),
    and that regressor is not 1 in general: it is 1 for `d = 1, D = 0` (where the scalar
    happened to be right), but **12** for `d = 0, D = 1` at monthly frequency, so the
    scalar was off by a factor of `s` for that whole class.

  Incidence on M4 monthly: **15.7% of the 48,000 series** (10.7% seasonal-differenced
  drift + 5.0% mean models) were scored on a wrong likelihood. Within those 7,529 series
  the fix improves OWA from 0.8551 to **0.8464 (-0.0087)**; over the full 48k it moves the
  benchmark from 0.9080 to **0.9065** and changes the selected order for 5.6% of series.

  The existing acceptance test (`dbg_valida_exata.jl`) could not catch this: it centres the
  series and calls R with `include.mean = FALSE`, exercising precisely the path where no
  deterministic term exists. `test/deterministic_term.jl` covers that gap.
- **A `lambda` the estimation ignores no longer moves the information criteria.** The sparse
  parameter count was triggered by the *presence* of `lambda`/`alpha` on the model rather
  than by the objective that actually fitted it. Measured: with fixed coefficients
  `[0.5, 0.0, 0.0]` and `objectiveFunction = "mse"`, passing `lambda = 1.0` left the
  coefficients bit-identical but took `K` from 4 to 2 and the AICc from 190.9058 to
  186.3890 — 4.5 units, on a scale where decisions turn at ~2. The count now keys off
  `model.metadata["objectiveFunction"]`. `elastic_net` keeps its sparse count at every
  `alpha`; restricting it to the lasso case (`alpha = 1`), the only one with a
  degrees-of-freedom result behind it, is a policy question and is left open.
- **`objectiveFunction = "ridge"` now warns that it ignores `lambda`.** The shrinkage is
  fixed at `sqrt(effective sample size)` in the objective; the argument was accepted,
  stored and silently discarded (verified: `lambda` from 0.01 to 100 gives identical
  coefficients to six decimals). Whether to honour it or reject it outright is left open —
  accepting it silently was the one indefensible option.
- **`objectiveFunction = "ml_exact"` now warns when it degrades to plain CSS**, i.e. when
  the reflection parameterization is off (`stationary = false`) or there is no non-seasonal
  AR part (`p = 0`). In those cases the user asked for an exact likelihood and got the
  conditional one verbatim (verified: identical coefficients to `"mse"`). The partial
  coverage on ARMA/seasonal models is documented scope and does not warn.

### Documentation
- `auto`'s `maxOrder` now documents that it applies to `searchMethod = "grid"` only. At the
  monthly defaults the grid therefore reaches 96 of the 324 order combinations in the box
  while the stepwise search reaches all 324 — the exhaustive method searches a smaller space
  than the heuristic one and can lose to it. The stepwise behaviour is deliberate parity with
  `forecast`; the surprise was that it went unstated.
- `criterionLoglike` documents that the criteria are a **quasi-AIC**: the likelihood is
  evaluated at the coefficients the user's objective produced, not at the Gaussian maximum,
  with the measured size of the deficit and why a one-step Newton refinement does not close
  it (the optimum sits on the invertibility boundary, where Le Cam equivalence fails).

- **Criterion fallback no longer rewards boundary candidates.** When the exact
  Gaussian likelihood is not computable (roots at the boundary), the criterion falls
  back to the CSS likelihood — which is evaluated on the conditioned sample
  (`T - lb + 1` observations vs `T` for the exact one) and is therefore less
  negative, granting tens of AICc units of advantage precisely to near-nonstationary
  candidates. The search criterion (`getInformationCriteriaFunction`) now adds a
  fixed penalty to fallback-scored candidates, imposing a two-tier order analogous
  to `forecast::myarima`'s `Inf` on non-finite likelihoods: an exact-scored
  candidate always outranks a fallback-scored one, while fallback-scored candidates
  remain comparable among themselves (same conditioning sample). The public
  `aic`/`aicc`/`bic` accessors are unaffected. The fallback is recorded in
  `model.metadata["criterionFallback"]` so its rate is measurable.
- **AICc/BIC sample size matches the likelihood actually scored.** The small-sample
  correction and the BIC `log(n)` factor used `n = length(observedResiduals)`
  (the CSS-conditioned count) even when the likelihood was the exact one evaluated
  on the full differenced sample — an undocumented extra size penalty, growing
  with `K`, that `forecast::Arima` does not have. Both now use the sample size of
  whichever likelihood was used.
- **`criterionLoglike` no longer swallows exceptions.** `exactLoglike`'s contract is
  `nothing`-on-refusal, so the blanket `try/catch` around it could only mask bugs
  (`MethodError`, `UndefVarError`) and made the fallback rate uninterpretable;
  model types without `exactLoglike` fall back via `applicable`.
- **Internal data scaling (numerical conditioning).** `fit!` now solves in units of
  the differenced series' standard deviation and maps the scale-dependent estimates
  (constant, trend, exogenous coefficients, residuals, σ², fitted and imputed values)
  back to the original units; AR/MA coefficients are scale-invariant and unaffected.
  Without this, series in the 1e4+ range pushed Ipopt into its restoration phase where
  a single iteration could exceed any time or iteration cap (measured: 208s without
  convergence on an M4 daily series vs 1.1s converged after scaling). Where the solver
  already converged, results are unchanged up to solver tolerance — verified by
  fitting `y` and `y*1000`: identical AR/MA coefficients, forecasts equal to a
  relative 1.35e-16, σ² ratio exactly 1e6. Provided (fixed) coefficients are
  converted on entry and back on exit, and are preserved to 1 ulp.
- **`kpssShort` now reproduces `forecast::ndiffs` exactly.** The mode's KPSS
  bandwidth was urca's `lags = "short"` (`trunc(4(n/100)^0.25)`), but
  `forecast::ndiffs` fixes `use.lag = trunc(3*sqrt(n)/13)` (verified in the
  forecast 8.23.0 source); the two disagree on real data — on Box-Jenkins' classic
  AirPassengers the wrong bandwidth selected `d = 0` where `auto.arima` selects
  `d = 1`. The corrected bandwidth is exposed as `nlags = :ndiffs` on `kpss_test`
  and is what `integrationTest = "kpssShort"` now uses. The KPSS statistic itself
  matches `urca::ur.kpss` to ~1e-6 at both bandwidths (pinned in the test suite).
- `test/test_statistical_tests.jl` referenced `seasonalStrengthTest` without the
  module qualifier, erroring the suite since the STL internalization.

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
- **`warmStartFromBox::Bool` keyword** on `fit!`/`auto`: solves the cheap
  unconstrained (box) problem first and seeds the stationarity/invertibility-by-
  construction fit from it — the O(T) residual vector makes the constrained problem
  start near-feasible. Falls back through three tiers (full constraints →
  invertibility only → the unconstrained seed), recording the tier reached in
  `model.metadata["warmStartTier"]`. On the longest M4 weekly series the constrained
  fit went from not converging in 48 minutes to 12.7s. `arToReflection` /
  `maToReflection` (step-down Levinson-Durbin inverses) provide the reflection-space
  starting points.
- **`maxTimeSeconds` keyword** on `fit!`/`auto`: bounds each solve with both a
  wall-clock limit and an Ipopt iteration ceiling (a wall-clock limit alone is only
  checked between iterations and cannot bound a pathological fit). Inside `auto`'s
  search, candidate fits are capped tighter (≤10s) than the final refit — a candidate
  that cannot be solved quickly is not going to win.
- **`optimizer` accepts `MOI.OptimizerWithAttributes`** (e.g.
  `optimizer_with_attributes(SCIP.Optimizer, "limits/gap" => 0.01)`) in addition to a
  bare constructor, so solver tolerances and limits are finally configurable from
  `fit!`/`auto`.
- `kpss_test(nlags = :ndiffs)`: the exact `forecast::ndiffs` KPSS bandwidth (see
  Fixed).
- Regression suites for the above: scale invariance and large-magnitude convergence
  (`test/numerical_conditioning.jl`), reflection round-trips, warm-start tiers and
  budget caps (`test/warm_start.jl`), configured-optimizer acceptance and
  fit/auto reproducibility (`test/solver_interface.jl`).
- **Missing-data estimation** (stationary models): `NaN` entries in the endogenous
  series are treated as free decision variables of the same JuMP problem. Retaining
  their residuals in the objective yields the two-sided conditional smoother (matching
  the exact AR(1) interpolation `ϕ(y_{t-1}+y_{t+1})/(1+ϕ²)`); σ², the log-likelihood,
  the effective sample size, and the Ljung-Box/Jarque-Bera diagnostics exclude the
  imputed indices. Supported for `d = D = 0`, `mse`/`ml` objectives, without exogenous
  regressors; other combinations raise a clear error. Imputed values are written back
  into `model.y` and the gap count is stored in `model.metadata["nMissing"]`.
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
  The camelCase names keep working with a deprecation warning; they are scheduled
  for removal in v2.0. Keyword-argument names are unchanged in this release.

## [0.2.0] - 2026-07-11

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
