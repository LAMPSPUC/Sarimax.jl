# Benchmark Status

Living log of the SARIMAX.jl empirical benchmark pipeline. Updated continuously.
Do not invent results. Failures are recorded, never silently dropped.

## Machine / OS

- Platform: macOS (Darwin 25.5.0), arm64 (Apple Silicon)
- Working dir: `Sarimax.jl/` (package); manuscript lives in parent `IJF Special Issue/`
- Replication package location (decided): **inside `Sarimax.jl/experiments/`**

## Environment

| Tool | Version | Status | Notes |
|------|---------|--------|-------|
| Julia | 1.11.9 (via `julia +1.11`) | OK | See blocker B1: default Julia 1.12.2 is NOT compatible with package compat `julia = "1.0 - 1.11"`. Use `julia +1.11`. |
| Python | 3.9.6 | OK | venv at `experiments/.venv-benchmarks` |
| numpy / pandas / scipy / statsmodels | 2.0.2 / 2.3.3 / 1.13.1 / 0.14.6 | OK | primary Python baseline = `statsmodels.tsa.statespace.sarimax.SARIMAX` |
| R | 4.6.1 (Homebrew) | OK | `/opt/homebrew/bin/Rscript` |
| R packages (forecast, jsonlite, readr, dplyr, tibble) | — | IN PROGRESS | installing; baseline = `forecast::Arima` / `stats::arima` |

Reproducibility metadata captured in `results/raw/environment/`:
`julia_version.txt`, `python_versions.txt`, `python_versions.json`, `python_freeze.txt`,
`r_version.txt`, `os_uname.txt`, `git_commit.txt`.

## Package API (verified from source, not assumed)

- Constructors: `SARIMA(y, p, d, q; seasonality, P, D, Q, allowMean, allowDrift, lambda, alpha)`;
  `SARIMA(y, exog, p, d, q; ...)`; coefficient-initialized `SARIMA(y; arCoefficients=..., ...)`.
- `fit!(model; silent, optimizer=Ipopt.Optimizer, objectiveFunction="mse")`.
  - `objectiveFunction ∈ {"mae","mse","ml","bilevel","elastic_net","stable"}`.
  - Regularization via `lambda` (strength) + `alpha` (1=lasso, 0=ridge) on the SARIMA model.
- `auto(y; seasonality, informationCriteria, objectiveFunction, ...)` — Hyndman-Khandakar style search.
- `predict!(model; stepsAhead, ...)`; `simulate(model, stepsAhead, numScenarios, seed)`.
- `loadDataset(AIR_PASSENGERS | GDPC1 | NROU)`; `splitTrainTest(data; trainPercentage=0.8)`.
- `aic/aicc/bic(model)`, `loglikelihood`/`loglike`.
- Bundled datasets: `datasets/airpassengers.csv`, `GDPC1.csv`, `NROU.csv`.
- Solvers in deps (final run, commit 144fb6e): **Ipopt, SCIP, Alpine**. HiGHS was REMOVED as a
  package dependency; SCIP is Alpine's default MIP sub-solver. EAGO is NOT a dependency.

## How to run (benchmark Julia env)

The package's own `Project.toml` lacks JSON/CSV needed by scripts, so a dedicated env was created:

```bash
julia +1.11 --project=experiments/env experiments/scripts/run_validation_julia.jl
julia +1.11 --project=experiments/env experiments/scripts/run_forecasting_julia.jl
julia +1.11 --project=experiments/env experiments/scripts/run_architecture_extensions.jl
julia +1.11 --project=experiments/env experiments/scripts/run_solver_diagnostics.jl
source experiments/.venv-benchmarks/bin/activate
python experiments/scripts/run_validation_python.py     # + run_forecasting_python.py
/opt/homebrew/bin/Rscript experiments/scripts/run_validation_r.R   # + run_forecasting_r.R
python experiments/scripts/combine_results.py           # builds all CSV + .tex tables
```

`experiments/env` does `Pkg.develop(path="..")` + adds JSON, CSV, DataFrames, TimeSeries,
Distributions, Ipopt, HiGHS, Alpine. The package source is NOT modified.

## Blockers

- **B1 (resolved via workaround):** Default Julia 1.12.2 fails `Pkg.instantiate()` because package
  `compat` declares `julia = "1.0 - 1.11"`. Workaround: run everything with `julia +1.11` (1.11.9).
- **B2 (recorded):** `Pkg.test()` = **35 passed, 1 failed**. The single failure is a strict
  `atol=1e-3` AR-coefficient recovery assertion in `test/models/sarima_fit.jl:55` ("fit (p=1 P=0)
  and (p=2 P=0) without white noise"). Ridge/lasso/bilevel/stable/auto testsets all pass. Treated as
  a tolerance-sensitive recovery check, not a functional defect — benchmarks remain meaningful.
- **B3 (cross-impl comparability):** AIC/BIC are NOT directly comparable across SARIMAX.jl,
  statsmodels, and R (different likelihood-constant conventions). Coefficients, RSS, and (under a
  matched specification) log-likelihood ARE comparable.
- **B4 (sim_sarimax discrepancy) — RESOLVED (2026-06-27):** NOT a bug. SARIMAX.jl models exog as a
  dynamic-regression / **ARX** model (AR acts on observed `y`), whereas statsmodels `SARIMAX(exog)`
  and R `Arima(xreg)` default to **regression-with-ARIMA-errors** (AR acts on the regression
  residual) — different models. The DGP is ARX, so SARIMAX.jl recovers the true parameters
  (φ=0.520, β=(1.534,−0.684), σ²≈1.0). Emulating ARX in statsmodels/OLS (lagged-y regressor)
  reproduces SARIMAX.jl exactly (RSS 295.61 for all three). Validation table now compares like-for-like
  ARX. Full analysis in `exog_discrepancy.md`; raw in `results/raw/validation/exog_diagnostic.jsonl`.
- **B5 (solver API) — characterized (2026-06-27):** high-level `fit!(optimizer::DataType)` cannot pass
  sub-solver attributes (so Alpine/EAGO cannot be configured through it) and uses deterministic start
  values (no randomized multistart through the API). At the **JuMP level** (see
  `run_solver_jump_diagnostics.jl`): (a) genuine randomized multistart IS implementable — 25 random
  starts on a small MA(1) all reach the same optimum (spread 1e-14; empirically unimodal); (b) Alpine
  CAN be wired with explicit Ipopt+HiGHS sub-solvers and runs local search (matches Ipopt = 34.83) +
  OBBT, but its global lower-bounding MIP step fails with HiGHS (`OTHER_ERROR`) at every tested size
  (T=12/20/40) — no global certificate with the available open-source MIP solver (a commercial MIP
  solver would likely be required). EAGO is not a dependency and was not installed/tested.
- **B5 — SUPERSEDED (2026-06-28):** the package now adds SCIP and uses it as Alpine's default MIP
  sub-solver. The missing `using HiGHS` bug was fixed by removing HiGHS entirely; `fit!(optimizer=
  Alpine.Optimizer)` now runs end-to-end for `mse` (SCIP solves the MIQP relaxation; obj 19.3241 =
  Ipopt). HiGHS is optional and warned-against for non-`mae` objectives. Global certificate still NOT
  obtained (Alpine OTHER_LIMIT at the JuMP level) — see `final_claims_for_manuscript.md` §6–7.

## Run log

- 2026-06-27 — Environment setup; package instantiate OK and `Pkg.test()` run (B2).
- 2026-06-27 — Block 1 validation: Julia 6/6 ok, Python 6/6 ok, R 5/6 ok (seasonal airpassengers
  CSS-ML failed). Table built.
- 2026-06-27 — Block 2 forecasting (airpassengers, 80/20, s=12): all ok. SARIMAX.jl competitive
  (RMSE 1.90) with statsmodels (1.81) and R (2.01); all beat seasonal-naive (5.19).
- 2026-06-27 — Block 3 architecture: objective swap (mse/mae) + regularization (ridge/lasso/
  elastic_net) all ok. Table built.
- 2026-06-27 — Block 4 solver diagnostics: Ipopt ok (deterministic, obj matches validation RSS);
  HiGHS unsupported (quadratic constraints); Alpine config-required (B5). Table built.
- 2026-06-27 — All four tables generated (CSV + escaped .tex); `experiment_report.md` written.
  Minimal completion criteria met.

### Continuation run (2026-06-27, session 2)

- **B4 resolved**: `diagnose_exog.py` shows the exog discrepancy is an ARX vs reg-w-ARIMA-errors
  specification mismatch (not a bug). Validation baselines (Python/R) now fit comparable ARX for
  `sim_sarimax`; all three agree (φ=0.5203, RSS=295.61, logLik=−422.56). `table_validation` regenerated.
- **R switched to `method="ML"`** (CSS-ML gave "non-stationary seasonal AR" on the seasonal model);
  R now fits all validation + forecasting models.
- **Rolling-origin forecasting added** (expanding window) on AirPassengers (s=12, H=12, 5 origins) and
  GDPC1 (quarterly, H=8, 5 origins). All implementations 0 failures. `table_forecasting` regenerated.
- **Admissibility experiment added** (Block 3): direct `fit!` checks show `fit!` only imposes box
  bounds `[-1,1]` and CAN return a non-stationary fit (airpassengers ARIMA(1,0,1): φ→1.0, stat=N);
  `auto` can filter via `assert*` flags (root-based `StateSpaceModels` checks). `table_architecture_checks`
  regenerated.
- **JuMP-level solver diagnostics added** (Block 4): genuine multistart (25 starts, single optimum)
  and Alpine-with-subsolvers (local search ok, global MIP fails). `table_solver_comparison` regenerated.

## Completed tasks

- Repository inspected (combined workspace; package here, manuscript in parent).
- Environment provisioned (Julia 1.11, Python venv + statsmodels, R 4.6.1); metadata captured.
- `Pkg.test()` run and recorded (B2).
- All four blocks executed and regenerated; tables `table_validation`, `table_forecasting`,
  `table_architecture_checks`, `table_solver_comparison` (CSV + ASCII .tex, column-checked).
- B4 resolved with focused diagnostic (`exog_discrepancy.md`).

## Failed / blocked tasks

- Alpine global solve: local search + OBBT run, but global MIP lower-bounding fails with HiGHS
  (`OTHER_ERROR`) at all sizes — no global certificate (B5). EAGO not installed.
- HiGHS cannot fit the nonlinear MA model via `fit!` (quadratic-equality constraints unsupported).
- See B2, B5 above.

## Additional scripts (session 2)

```bash
python experiments/scripts/diagnose_exog.py                              # B4 diagnostic
julia +1.11 --project=experiments/env experiments/scripts/run_solver_jump_diagnostics.jl  # multistart + Alpine
```

## Next actions

1. (Optional) Obtain a commercial MIP solver (Gurobi/CPLEX) to attempt an Alpine global certificate;
   or install and test EAGO.
2. (Optional) Add a second seasonal real dataset and longer rolling-origin horizons.
3. Insert regenerated `.tex` tables into `../chapters/experiments.tex` (trim columns to page width).

## Generated-file → manuscript-table mapping

| Manuscript table (label) | Source `.tex` |
|--------------------------|---------------|
| `tab:validation_implementations` | `experiments/tables/table_validation.tex` |
| `tab:forecast_oos` | `experiments/tables/table_forecasting.tex` |
| `tab:architecture_extensibility` | `experiments/tables/table_architecture_checks.tex` |
| `tab:solver_comparison` | `experiments/tables/table_solver_comparison.tex` |

## Final paper run (2026-06-28)

- **Git commit tested:** `144fb6e86c2743ff726c9716364407e6f2db12ba`.
- **Package state:** SCIP in deps + Manifest; **HiGHS = 0 mentions in Manifest** (removed as a
  dependency). `fit!` has `mipSolver::DataType = SCIP.Optimizer`. `using SCIP` in module.
- **Tests:** `Pkg.test()` = **42 passed, 1 failed** (only the pre-existing B2 tolerance test;
  `invertible_fit` 7/7). No new failures from the SCIP/Alpine/invertibility changes.
- **Block 1 validation:** Julia 6/6, statsmodels 6/6, R 6/6 (R uses `method="ML"`). Exogenous compared
  like-for-like ARX; all three agree (φ=0.5203, RSS=295.61, logLik=−422.56).
- **Block 2 forecasting:** rolling-origin (5 origins) on AirPassengers (s=12, H=12) and GDPC1 (H=8);
  all implementations 0 failures. SARIMAX.jl competitive, beats naive baselines.
- **Block 3 architecture:** objective swap, regularization, admissibility, and the **new invertibility
  parameterization** (`invertible=true`: airline θ −1.0 → −0.95).
- **Block 4 solver:** Ipopt baseline; **Alpine+SCIP via `fit!` works** (mse, obj 19.3241 = Ipopt,
  ~165 s). Alpine+SCIP `mae` (global MILP) **did not finish within budget** (package sets no Alpine
  time limit) → recorded as blocked, not run unbounded. HiGHS warning verified (mse→emitted,
  mae→none). JuMP-level: multistart 25→single optimum (spread 1.4e-14); Alpine+SCIP (300 s) →
  `OTHER_LIMIT`, obj 19.3241, **no global certificate**.
- **Incident:** the first attempt ran Alpine+SCIP `mae` via `fit!` unbounded and hung (~1 h); it was
  killed and the script changed to bound all global runs (package mse only; mae documented; HiGHS
  warning tested at config level; JuMP-level global with explicit 300 s limit).
- **All four tables regenerated** (CSV + ASCII `.tex`, column-count verified).
- **Final claims:** see `experiments/final_claims_for_manuscript.md`.

## External-validation dataset added (2026-06-29)

- **PJME electricity demand** (PJM East hourly, CC0 via Kaggle/GitHub mirror; daily-aggregated, s=7).
  Download + deterministic preprocessing: `scripts/setup/gen_energy_data.py` (145,366 hourly rows; 4
  DST duplicates dropped; 30 missing hours interpolated; 6,059 daily points). Raw under
  `results/raw/energy/`.
- Forecasting rolling-origin (daily, s=7, H=14, 6 origins): SARIMAX.jl competitive (RMSE 3132, MASE
  0.830), beats seasonal-naive (3466); statsmodels 3177, R 2837. `table_energy_forecasting.{csv,tex}`.
- Architecture: same (1,1,1)(1,0,1)_7 spec under MSE/MAE/Ridge/Elastic-Net (objective-only variation).
- Solver: Ipopt obj 296.4645 on a reduced MA(1); direct SCIP certifies the same optimum globally
  (OPTIMAL, gap 0, 8.1 s) — updated 2026-07-07 with the Alpine→SCIP migration.
- Purpose: external validation of generalization. No claim weakened or added.

## Solver experiment migrated to direct SCIP (2026-07-07)

- **Finding (B8):** a brute-force profile-search audit over (θ,c) — feasible because ε follows a
  deterministic recursion for fixed coefficients — showed the true global optima are 7.36853 (T=8),
  8.52977 (T=10), 19.32405 (T=20), 34.83005 (T=40). The Alpine tuned-run incumbents (7.3447 at T=8,
  8.506 at T=10) lie BELOW these values, i.e. they were infeasible beyond tolerance; the previous
  "certified within 1%" Alpine rows were therefore invalid and have been retired
  (`alpine_cert_results.superseded.json` kept for provenance).
- **Replacement:** SCIP solves the nonconvex MA(1) SSE model DIRECTLY (no decomposition, raw model,
  no warm start needed) and returns EXACT certificates (rel. gap 0): T=8 1.7 s, T=10 1.2 s,
  T=20 4.3 s, T=40 14.3 s — every value confirmed by the brute-force audit
  (`run_solver_scip_cert.jl`, `brute_force_agrees=true`). T=40 also proves the multistart optimum
  (34.83) is global. Package-level `fit!(optimizer=SCIP.Optimizer)` works via the generic path
  (RSS 19.324 = global, ~6.4 s warm).
- **Pipeline:** `run_solver_alpine_cert.jl` removed; `run_solver_scip_cert.jl` added;
  `run_solver_diagnostics.jl` now runs Ipopt + SCIP-via-fit! + the Alpine/HiGHS warning checks
  (config-level; the Alpine wiring and guard remain package features). Optional
  `run_solver_gurobi.jl` added (gated on a Gurobi license; NonConvex=2 is set automatically by the
  package) — not in the default pipeline, no performance claims without a license.
- Manuscript updated accordingly (solver table/caption/text, architecture, conclusion, framework
  solver paragraph + SCIP reference).

## SCIP scaling frontier (2026-07-07)

Ran direct SCIP without the small-instance restriction (`run_solver_scip_scaling.jl`, raw model,
600 s/instance). Exact certificates (gap 0, brute-force-verified) up to **T=120** (297 s); **T=150
and T=200 hit the 600 s limit** at a residual ~10^-5 gap without certifying. Non-monotone wall time
(T=100=400 s > T=120=297 s) is normal for spatial B&B. Fixed two script issues found in the run: NaN
JSON serialization on time-limited rows (`naclean`) and a soft-scope `consecutive_fail` bug
(`global`). New table: `table_solver_scaling`. Frontier ⇒ manuscript claim stays bounded: exact
open-source certification is attainable but limited to small/moderate T.

## Global-value experiment + strategic SCIP revision (2026-07-08)

- **New experiment (Block 4d):** `run_solver_global_value.jl` — Ipopt vs direct SCIP on the SAME
  specification (identical fit! conventions), OOS forecasts over 12-step held-out window, T=80.
  Finding: certified and local solutions essentially coincide (obj ~2e-4 apart, coefs differ 3rd
  decimal, OOS errors identical; certified fit marginally worse OOS on sim ARMA(1,0,1)). PJME row:
  TIME_LIMIT at 600 s with residual gap 1.1e-7 — dual bound proves local solution within 1.1e-7 of
  global. New table `table_global_value`; added to reproduce.sh.
- **SoPlex segfault note:** batch run crashed (signal 11 inside libscip 8.0 SoPlex) on the PJME
  instance; isolated rerun completed. Raw record carries a `note`; treat as solver flakiness.
- **SCIP description verified (Task 0):** SCIP 8.0.0 via SCIP.jl 0.11.6, Ipopt_jll linked. Problem
  class: nonconvex QCQP (continuous). Wording used in the manuscript: global optimization framework =
  spatial branch-and-bound on linear outer-approximations via convex over/underestimation +
  presolve/propagation/cuts/heuristics; certificate = incumbent vs global dual bound within
  tolerance (numerical, not exact arithmetic). References: vigerske2018scip, bestuzheva2021scip,
  bestuzheva2023global (all added to references.bib).
- Manuscript strategically revised (abstract, intro, framework, architecture incl. API panel,
  experiments incl. new §"Does global optimality matter for forecasting?", design insights incl.
  new §"When does certified global optimality matter?", conclusions fully rewritten).
