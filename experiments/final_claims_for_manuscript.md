# ADDENDUM 2026-07-13 — battery re-run on Sarimax.jl v0.3.0

The full battery was re-run on package commit `3f68017` (v0.3.0: multiplicative
Box-Jenkins seasonal form as default, `initialization=:warmup` R-CSS convention,
real drift term, CSS log-likelihood with full Gaussian constants). Tables in
`tables/` and the manuscript were regenerated/updated accordingly. Key deltas
relative to the claims below (which documented the v0.1.3 run at `144fb6e`):

- **Validation upgraded from proximity to equality.** Under the matched CSS
  convention (R `method="CSS"` vs `initialization=:warmup`) coefficients coincide
  to 4 decimals on every simulated spec (AR 0.7381; MA 0.7872; ARMA 0.5203/0.5561,
  identical RSS). The multiplicative seasonal spec is now estimable in matched
  form: SARIMAX.jl (0.9933, -0.8075, RSS 167.93) sits inside the reference family
  (statsmodels exact-ML RSS 167.59); the residual gap to R-CSS traces to R's
  unbounded CSS optimizer accepting an explosive point (phi=1.0011, RSS 139.63).
- **§4 (AIC/BIC caveat) SOFTENED**: with full Gaussian constants the package's
  loglik/ICs are on R's CSS scale (still not exact-ML defaults' scale).
- **§5 (invertibility) REWRITTEN**: the unit-root pile-up (theta=-1.0) was an
  artifact of the additive seasonal form. Under the multiplicative default the
  box fit is interior (theta=-0.813) and `invertible=true` returns the identical
  solution (constraint inactive). The parameterization is a guarantee, not a fix.
- **PJME forecasting IMPROVED ~8% RMSE** (3131.95 -> 2869.13): now ahead of
  statsmodels (3177.50), ~1% behind R (2837.04). The (1,1,1)(1,0,1)_7 spec has
  both seasonal interactions — exactly where the additive form was misspecified.
- **PJME global-value certificate now CLOSES** (OPTIMAL, 900 s; was gap 1.1e-7 at
  600 s). All three global-value instances certified; conclusions unchanged
  (guarantee changes nothing statistically; it buys verification).
- **AirPassengers/GDPC1 rolling-origin essentially unchanged** (mid-pack /
  tied with statsmodels); warm per-fit ~11 ms on AirPassengers (multiplicative).
- **Package tests: ~650 assertions, all passing, 91.5% coverage** (was
  "42 passed, 1 failed" — and the suite silently never ran past sarima_fit.jl).
- Scripts changed for the re-run: run_validation_{julia,r} gained CSS-matched
  records; run_forecasting_julia sets allowMean=false with drift (now mutually
  exclusive); run_architecture_extensions adds the additive-form invertibility row.

The sections below are retained as the record of the v0.1.3 run.

---

# Final Claims for the Manuscript — SARIMAX.jl

Final empirical battery. Git commit tested: `144fb6e86c2743ff726c9716364407e6f2db12ba`.
Julia 1.11.9 · Python 3.9.6 (statsmodels 0.14.6) · R 4.6.1 (forecast). No fabricated values.

Central bounded claim (supported):
> SARIMAX.jl exposes SARIMA/SARIMAX estimation as an explicit JuMP optimization model, allowing the
> same dynamic SARIMAX specification to be reused while objective functions, constraints,
> regularization, initialization handling, and solvers are varied.

Package tests: 42 passed, 1 failed — the single failure is a pre-existing strict `atol=1e-3`
AR-coefficient recovery assertion (`test/models/sarima_fit.jl`); the new `invertible_fit` testset
passes 7/7. SCIP is in deps/Manifest; HiGHS has 0 mentions in the Manifest (removed as a dependency).

## 1. Claims supported

- **Optimization-based estimation.** SARIMA/SARIMAX is built as an explicit JuMP model (coefficients
  and residuals as variables; dynamics as constraints). [Block 1–4]
- **Estimation correctness vs established tools.** On simulated ARMA/ARIMA, coefficients and RSS match
  statsmodels and R under comparable specifications (e.g. ARIMA(1,0,1): φ≈0.52, θ≈0.556, RSS≈307 for
  all three). [`table_validation`]
- **Exogenous (ARX) correctness.** Under a matched ARX specification, SARIMAX.jl, statsmodels-ARX and
  R-ARX agree exactly: φ=0.5203, RSS=295.61, logLik=−422.56. [`table_validation`]
- **Bounded forecasting adequacy.** Rolling-origin (expanding window, 5 origins) on AirPassengers
  (s=12, H=12) and GDPC1 (H=8): SARIMAX.jl is competitive — RMSE 1.48 (vs statsmodels 1.30, R 1.59) on
  AirPassengers and 324.9 (vs statsmodels 324.6, R 367.4) on GDPC1 — and beats naive/seasonal-naive
  baselines (2.50 / 498.2). [`table_forecasting`]
- **Architecture/extensibility from one specification.** The same dynamic spec is reused under:
  objective swap (MSE vs MAE), regularization (ridge/lasso/elastic-net; coefficient norm shrinks
  1.90→1.73/1.77/1.76 at small RSS cost), and an invertibility parameterization. [`table_architecture_checks`]
- **Invertibility parameterization (new fix).** `fit!(invertible=true)` generates MA coefficients from
  bounded reflection coefficients, keeping the estimate strictly inside the invertibility region by
  construction: on the airline model the default box fit piles θ on the unit-root boundary (θ=−1.0)
  while `invertible=true` returns θ=−0.95. [`table_architecture_checks`, see §5]
- **Solver modularity.** Solver choice is a swappable argument of `fit!`. Ipopt (local) is the default;
  **direct SCIP runs through the same `fit!` call** and returns the certified global optimum
  (mse, 19.3241 = Ipopt's local optimum, proven global). At the JuMP level SCIP certifies
  T=8/10/20/40 exactly (gap 0). [`table_solver_comparison`]
- **Transparent solver diagnostics.** Genuine randomized multistart (25 random starts, JuMP level) on a
  small MA(1) finds a single optimum (objective spread 1.4e-14), i.e. empirically unimodal here. The
  package emits a clear warning when HiGHS is requested for a quadratic objective (see §6).

## 2. Claims NOT supported (do not make)

- No forecasting **superiority** — SARIMAX.jl is competitive, not best. (Note on speed: reported
  runtimes are warm/steady-state with Julia's one-time JIT compilation excluded; warm per-fit is
  competitive — faster than statsmodels on the monthly/quarterly tasks, somewhat slower on the larger
  daily series. The earlier "slower" impression was dominated by one-time compilation, not by the
  optimization approach.)
- No **scalable** global-optimality claim. Nuance (see §7): direct **SCIP certifies EXACT global
  optimality (gap 0)** up to **T=120** (brute-force-verified); at T=150/200 it reaches a ~10^-5 gap
  but does not close it within 600 s. The bounded statement is that exact open-source certification
  is real but limited to small/moderate T (frontier ≈120–150), not a routine estimation strategy.
- No claim that `fit!` (default) enforces stationarity or invertibility — it imposes only box bounds;
  invertibility is enforced only under `invertible=true` (see §5).
- No claim that SARIMAX.jl's exogenous coefficients equal default statsmodels/R `xreg` output (different
  model family — see §3).
- No robust-objective superiority claim from the MSE/MAE swap. (Update 2026-07-07: the swap now uses
  an AR(1)-only, convex design — MSE attains min RSS 603.97, MAE attains min resid-MAE 0.973 — so the
  demonstration is that the criterion changes the fitted solution, not that either is superior.)
- No cross-implementation AIC/BIC quality comparison (see §4).
- No claim about HiGHS as an Alpine sub-solver for quadratic objectives (it cannot — see §6).

## 3. Caveat — ARX vs regression-with-ARIMA-errors (exogenous)

SARIMAX.jl models exogenous regressors as a **dynamic-regression / ARX** model: the AR term acts on the
**observed** series, `y_t = c + Σφ_i y_{t-i} + Σβ_j x_{j,t} + Σθ_j ε_{t-j} + ε_t`. statsmodels
`SARIMAX(exog=…)` and R `Arima(xreg=…)` default to **regression-with-ARIMA-errors** (AR on the
regression residual). These are different models. The exogenous validation therefore compares
**like-for-like ARX** (lagged-y regressor) across all tools; under that match they agree exactly. The
manuscript must state SARIMAX.jl's exogenous specification is ARX and must not compare it against default
`xreg` output. (Diagnostic: `exog_discrepancy.md`, `results/raw/validation/exog_diagnostic.jsonl`.)

## 4. Caveat — AIC/BIC comparability

AIC/BIC are NOT comparable across implementations because of differing likelihood-constant conventions
(e.g. for the ARX `sim_sarimax` fit, SARIMAX.jl AIC = 8.60 vs statsmodels/R = 855.12 for the **same**
fit and the **same** log-likelihood −422.56). Use coefficients, RSS, and matched log-likelihood for
cross-implementation comparison; do not tabulate cross-tool AIC/BIC differences as quality gaps.

## 5. Caveat — stationarity/invertibility after the fix

Exactly what is enforced:
- **`fit!` default:** per-coefficient **box bounds** `−1 ≤ φ,θ,Φ,Θ ≤ 1` only. This equals the
  stationarity/invertibility region only for order 1; for higher orders the box is a strict superset,
  and the box **boundary is a unit root** (non-stationary/non-invertible). Evidence: `fit!` ARIMA(1,0,1)
  on AirPassengers returns φ=1.0 → **non-stationary**; the airline model returns θ=−1.0 → **unit MA
  root**. So `fit!` does **not** enforce stationarity/invertibility.
- **`fit!(invertible=true)`:** MA coefficients are generated from bounded reflection coefficients
  (`|κ| ≤ 1−ρ`) via a Durbin–Levinson-type recursion, which **guarantees an invertible MA polynomial by
  construction**. Demonstrated: airline θ moves from −1.0 (box) to −0.95 (`invertible=true`, ρ=0.05).
  This enforces **invertibility only** (the MA side); it does **not** enforce AR stationarity.
- **`auto`:** can **post-filter** candidate models by stationarity/invertibility via root checks
  (`assertStationarity`/`assertInvertibility`); this is a search filter, not an estimation constraint.
- **Known limitation (separate, pre-existing bug):** the post-fit checker
  `checkModelStationarityInvertibility` uses an **additive** expansion of the seasonal MA/AR polynomial
  (omits cross terms), so its invertibility flag is **unreliable for seasonal models** — it reports the
  `invertible=true` airline fit (θ=−0.95) as non-invertible. The reflection-coefficient guarantee is
  by construction and does not depend on this buggy checker. (Flagged for a separate fix.)

Manuscript wording: say `fit!` imposes coefficient box bounds; `invertible=true` enforces MA
invertibility through a bounded reflection-coefficient parameterization; admissibility can also be used
as a search filter in `auto`. Do not claim stationarity enforcement.

## 6. Caveat — global solvers (SCIP direct; Alpine retired; HiGHS guard)

- **SCIP is the global backend used in the paper (2026-07-07).** It solves the nonconvex (bilinear)
  MA model **directly** — raw model, no decomposition, no warm start, no tuned initialization — and
  returns **exact certificates (rel. gap 0)**: T=8 in 1.7 s, T=10 in 1.2 s, T=20 in 4.3 s, T=40 in
  14.3 s. Every certified optimum was verified by a brute-force profile search over (θ,c)
  (`run_solver_scip_cert.jl`, `brute_force_agrees=true`). Package level:
  `fit!(optimizer=SCIP.Optimizer)` works through the generic path (RSS 19.324 = global, ~6.4 s warm).
- **Alpine retired from the headline experiment (audit finding B8):** the brute-force audit showed the
  earlier Alpine "1%-certified" incumbents (7.3447 at T=8; 8.506 at T=10) lie **below the true global
  optima** (7.36853; 8.52977) — i.e. they were infeasible beyond tolerance, so those certificates were
  invalid. Alpine's wiring remains a package feature (`optimizer=Alpine.Optimizer`, `mipSolver=` kwarg),
  and the earlier raw records are preserved as `alpine_cert_results.superseded.json`.
- **HiGHS guard (unchanged, verified):** requesting `mipSolver=HiGHS.Optimizer` with a non-`mae`
  objective emits a warning (HiGHS cannot solve the MIQP relaxations); no warning for `mae`.
- **Gurobi (optional, commercial):** the package auto-sets `NonConvex=2` when Gurobi is selected, and
  `run_solver_gurobi.jl` is provided (license-gated, not in the default pipeline). Gurobi's global
  optimality for nonconvex QCQPs is documented capability; **no performance numbers are claimed**
  because no license was available on the test machine.
- `fit!` does not surface termination status or optimality gap for any backend; those are observable
  at the JuMP level.

## 7. Does the study support a global-optimality claim?

**Yes — exact, up to a measured frontier, via direct SCIP (open-source).**
- **Global-value experiment (2026-07-08, `run_solver_global_value.jl`):** on T=80 instances the
  certified solutions coincide with Ipopt's to ~2e-4 in objective; OOS forecast errors identical to
  reported precision (certified marginally worse on sim ARMA(1,0,1)). PJME: gap 1.1e-7 at time limit
  — the dual bound itself certifies near-optimality of the local fit. CLAIM TO MAKE: certification =
  verification/audit value on these instances, NOT forecast-accuracy value; the architecture makes
  the trade-off measurable. Do NOT claim global solves improve forecasts.
- SCIP returns `OPTIMAL` with **relative gap 0** on the diagnostic instances (T=8/10/20/40) in
  1.2–14.3 s, on the raw model. Each optimum matches an exhaustive profile search.
- The T=40 certificate additionally **proves the multistart/Ipopt optimum (34.83) is global**.
- **Scaling study (2026-07-07, `run_solver_scip_scaling.jl`, 600 s/instance):** exact certificates
  (gap 0, brute-force-agreed) extend to **T=120** (297 s); at **T=150 and T=200** SCIP reaches a
  residual ~10^-5 relative gap but does **not** close it to a certificate within 600 s. Wall time is
  instance-dependent and non-monotone (T=100 = 400 s > T=120 = 297 s), as expected for spatial
  branch-and-bound. So the frontier of exact open-source certification for this problem is
  small/moderate T (order 10^2), not the full sample sizes used in estimation.
- **Bounds of the claim:** exact certification is real but does not scale; do not claim scalable
  global solving. Defensible statement: *exact global-optimality certificates are attainable on
  small-to-moderate nonconvex MA instances (up to about a hundred observations here) entirely within
  the open-source stack, with cost that grows steeply and a frontier around T≈120–150.*

## 8. Recommended manuscript wording (experiment section)

- "Estimates from SARIMAX.jl match mature implementations (statsmodels, R `forecast`) under comparable
  specifications; for exogenous models we compare the matched dynamic-regression (ARX) form, which all
  three implement identically."
- "Under rolling-origin evaluation on a monthly seasonal series (AirPassengers) and a quarterly macro
  series (GDPC1), SARIMAX.jl delivers forecasting accuracy competitive with statsmodels and R and
  superior to naive baselines; we make no claim of forecasting superiority."
- "The same dynamic specification is reused under alternative objectives (squared vs absolute error),
  regularization (ridge/lasso/elastic-net), and a bounded reflection-coefficient parameterization that
  enforces moving-average invertibility by construction. Estimation under `fit!` imposes coefficient box
  bounds; invertibility is enforced when requested; stationarity/invertibility can additionally be used
  as a model-search filter."
- "Solver choice is an explicit, interchangeable argument. Local estimation uses Ipopt; the same model
  can be passed to the Alpine global optimizer, whose open-source MIP sub-solver (SCIP) handles the
  mixed-integer quadratic relaxation. We report solver behaviour transparently and do not claim global
  optimality: within the time budget Alpine returned the same objective as the local solver without a
  certified optimality gap."
- "Reported diagnostics include randomized multistart, which found a single optimum on the evaluated
  MA(1) instance."

---

## 9. External validation — PJME electricity demand (added 2026-06-29)

**Dataset.** PJM Hourly Energy Consumption, PJME (PJM East) zone — public ISO operational load,
distributed as the Kaggle "Hourly Energy Consumption" dataset (CC0 1.0, R. Mulla); downloaded from a
public GitHub mirror for unauthenticated reproducibility. 145,366 hourly rows (2002–2018), cleaned
programmatically (4 duplicate DST timestamps dropped, 30 missing hours time-interpolated) and
aggregated to **daily mean load** (6,059 days, weekly seasonality s=7). Provenance/stats:
`results/raw/energy/data/pjme_preprocessing.json`. Hourly full-series rolling-origin is not
computationally reasonable for the optimization formulation (one residual variable per observation,
145k points), so the daily aggregation — consistent with the monthly/quarterly framing of the other
datasets — is used; a small recent hourly slice is used only for the reduced solver experiment.

**Why this strengthens the manuscript.** It adds an *independent, real-world, high-volume* domain
(electricity demand) to the existing AirPassengers (transport) and GDPC1 (macro) series, plus a
controlled simulation. It demonstrates the optimization-based architecture **generalizes** beyond the
original benchmark datasets. It is **external validation only** — it does not change or add any claim.

**Forecasting (rolling-origin, daily, s=7, H=14, 6 origins; `table_energy_forecasting`).**
SARIMAX.jl RMSE 3132 / MASE 0.830; statsmodels 3177 / 0.842; R `forecast` 2837 / 0.743; seasonal-naive
3466 / 0.900. SARIMAX.jl is **competitive** (between R and statsmodels) and beats the seasonal-naive
baseline — same bounded-adequacy conclusion as the other datasets; **no superiority claim**.

**Architecture/extensibility.** The same SARIMA(1,1,1)(1,0,1)_7 specification was estimated under MSE,
MAE, Ridge and Elastic-Net by changing only the objective/penalty (RSS, residual-MAE and coefficient
norm vary as expected; e.g. coefficient norm 1.08/1.57/1.22/1.27). Confirms the "same dynamic
specification, varied objective" claim on real data.

**Solver (updated 2026-07-07, Alpine→SCIP).** On a reduced MA(1) instance, Ipopt (local) reaches
obj 296.4645 (θ=0.883) and direct SCIP (global, JuMP level) **certifies the same optimum globally**
(`OPTIMAL`, rel. gap 0, 8.1 s) — consistent with the main solver block: exact certificates on small
instances within the open-source stack; no scalability claim.

**Conclusions changed: none.** All §1 claims hold; all §2 non-claims still apply. The energy dataset is
additional external validation, not a new or strengthened claim.
