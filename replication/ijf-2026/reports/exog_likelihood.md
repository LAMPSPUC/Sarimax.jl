# AIC, AICc and BIC of models with regressors

**Finding.** In v1.0.0 the exact likelihood behind `aic`, `aicc`, `bic` and every `auto` search
did not score the model `fit!` estimates when the model has regressors. Confirmed by an
independent computation, fixed in the package (`src/exact_likelihood.jl`, commit "Score models
with regressors by the likelihood of the model fit! estimated"), pinned by
`test/exog_likelihood.jl`. Univariate models, and therefore every M4 result of the manuscript,
are not affected.

## 1. The two likelihoods

Let `z_t` be the differenced series (`d`, `D`), `X_t` the regressors as `fit!` uses them,
`phi*(B) = 1 - sum_j a_j B^j` the expanded AR polynomial (regular times seasonal) and `theta*(B)`
the expanded MA polynomial. `fit!` estimates (`src/models/sarima.jl`, the `ŷ` expression and
`arAcc`):

* `:armax` (default): `phi*(B) z_t = c + X_t'b + theta*(B) e_t`. Write `z_t = c/phi*(1) + mu_t +
  w_t` with `phi*(B) mu_t = X_t'b`; then `w_t` is ARMA(`phi*`, `theta*`) and the exact Gaussian
  log-likelihood of the model is that of `w = z - c/phi*(1) - mu` under that ARMA.
  `mu` needs values before the sample; the package now takes the steady state of the first
  observation, `mu_t = X_1'b / phi*(1)` for `t <= 0`. That is the convention that already turns
  the constant into the level `c / phi*(1)`: a constant regressor and `c` give the same
  likelihood. With no AR terms, `mu_t = X_t'b`.
* `:regression_errors`: `z_t - X_t'b = c + (ARMA)`, so `w = z - X b - c/phi*(1)`.

`exactGaussianLogLikelihood` then evaluates the ARMA likelihood of `w` (Durbin-Levinson,
`sigma^2` concentrated), unchanged.

## 2. What v1.0.0 computed

`exactLoglike` removed `differentiate(X, d, D) * b` from `z` in all cases:

1. under `:armax` this is the regression-with-errors likelihood, the likelihood of a different
   model whenever there are AR terms;
2. `fit!` and `predict!` do NOT difference the regressors (unless
   `automaticExogDifferentiation = true`): with `d + D > 0` the likelihood used other
   regressors than the fit, under both semantics;
3. when `model.exog` had future rows (needed by `predict!`), the lengths differed and the
   regressors were silently not removed at all;
4. a single regressor given as a one-dimensional `TimeArray` (the form `load_dataset` returns)
   made `X * b` a `Vector * Vector` `MethodError`, caught and turned into "likelihood
   unavailable": the criteria fell back to the CSS plug-in, with `criterionFallback = true` and
   no reason given.

## 3. Deterministic harness

`test/exog_likelihood.jl` (in `Pkg.test()`) simulates, with fixed seeds, two regressors with
non-zero means (AR(1) around 5 and -2), T = 240, and `z` from either equation with
`phi = 0.6`, `theta = 0.3`, `b = [2.0, -1.5]`, then `y = z` (d = 0) or `y = cumsum(z)` (d = 1).
The reference is the dense multivariate normal density of `w`, with autocovariances from psi
weights computed in the test, sharing no code with the package.

| check | v1.0.0 | fixed |
|---|---|---|
| `exactLoglike` equals the reference of the fitted semantics, d in {0, 1}, both DGPs, both semantics (8 checks) | 2 pass (regression with errors at d = 0) | 8 pass (to 1e-8) |
| the semantics that generated the data has the higher exact likelihood (by more than 10) | 1 of 2 | 2 of 2 |
| a single one-dimensional regressor is scored, not sent to the CSS fallback | error | pass |
| without AR terms both semantics give the same likelihood | pass | pass |
| future regressor rows do not change the likelihood | fail | pass |
| `armaxRegressionPath`: constant input gives the constant's level; unit root refused | error | pass |
| `auto` with a regressor ranks by the exact likelihood (no fallback) | fail | pass |
| total | 6 of 21 | 21 of 21 |

On the ARMAX process fitted under `:armax`, for example, v1.0.0 gave -908.27, the fixed code
-450.91 (reference -450.91); the regression-with-errors fit of the same data scores -596.16. So
v1.0.0 ranked the wrong semantics first, and the fixed criterion ranks the true one first.

The archived battery's ARX(1) row with two regressors (`sim_sarimax`, `:zeroed`, fixed order):
AIC 1025.87 in v1.0.0, 861.19 fixed; CSS, `statsmodels` and `forecast::Arima` on the same
model (`y_(t-1)` as a regressor, conditional on the first observation) give 855.12. The
remaining 6.07 is the first observation's density, which the exact likelihood includes.

## 4. Effects on criteria, orders and forecasts

Harness `reports/exog_likelihood/effects_exog.jl`, run once with v1.0.0 and once with the fix
(identical inputs; outputs `effects_v1.0.0.csv`, `effects_fixed.csv`, summaries
`effects_summary.csv`, `order_recovery.csv`): 80 simulated series (2 DGPs x 1 or 2 regressors x
d in {0, 1} x 10 seeds), each searched by `auto` under both semantics (160 searches), with 12
future observations for the forecast error.

* **AIC, AICc, BIC** change for every model with regressors and AR terms, and for every model
  with regressors and `d + D > 0`.
* **Selected orders** change in 140 of 160 searches.
* **True order recovered** (semantics equal to the DGP's and `d` selected correctly, 60 cases):
  0 of 60 with v1.0.0, 37 of 60 fixed; 12-step RMSE median 3.155 -> 2.920, mean 6.835 -> 5.457.
* **All 160 searches:** true order 15 -> 40; RMSE median 4.512 -> 4.923, mean 12.301 -> 14.941.
  The cases that get worse are those where `auto` selects a `d` other than the DGP's: the model
  `fit!` estimates then enters the regressors undifferenced into a differenced equation, which is
  misspecified, and the corrected criterion selects the best fit within that misspecified class
  (often long MA polynomials). v1.0.0's criterion, scoring differenced regressors, happened to
  favour short models there.
* **Fixed-order fits** do not change (the likelihood is not part of estimation), except under
  `multistart = true`, whose choice between its two starts uses AICc.
* In this harness the single-regressor cases were two-dimensional `TimeArray`s; the
  one-dimensional case (defect 4) is covered by the test.

## 5. What is not affected

* Univariate models: the changed code runs only when `model.exog` is set; `fit!` gains one
  metadata key (`automaticExogDifferentiation`). The package test suite passes (1,439 pass, 0
  fail, 5 pre-existing broken), including the exact-likelihood checks against
  `stats::arima(method = "ML")`.
* Every M4 table of the manuscript (no regressors).
* Table S2 and the Section 5.2 comparison (fixed orders, no ranking).

## 6. Not changed, for the authors

* `fit!` uses the regressors undifferenced in the differenced equation when `d + D > 0` (unless
  `automaticExogDifferentiation`), whereas `forecast::Arima(xreg=)` differences them with the
  series. The description of LAMPSPUC/Sarimax.jl#18 records this as a separately reported
  defect (both semantics "attenuate identically" under `D = 1`). Changing it changes estimates
  and forecasts and needs its own decision; the likelihood now scores what `fit!` estimates.
* The drift level in the ranking likelihood under AR terms (`LIMITATIONS.md` item 17, C2), which
  does affect univariate M4 selections, is untouched.

## 7. Reproduce

```bash
julia --project=. -e 'using Pkg; Pkg.test()'          # the package's suite, at the repository root
julia --project=<environment with v1.0.0> reports/exog_likelihood/effects_exog.jl effects_v1.0.0.csv <sim_sarimax.csv>
julia --project=replication/ijf-2026/environments/julia reports/exog_likelihood/effects_exog.jl effects_fixed.csv <sim_sarimax.csv>
python3 reports/exog_likelihood/compare_effects.py effects_v1.0.0.csv effects_fixed.csv
```

`sim_sarimax.csv` is regenerated by `experiments/runners/battery/setup/gen_sim_data.py`. Each
harness run takes about 5 minutes. For this report the suite ran with the replication
environment's pinned Manifest plus Aqua 0.8.16 (Julia 1.10.12, Windows), and the v1.0.0 harness
run used the same Manifest with `src/` at `14dd581`.
