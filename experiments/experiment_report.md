# Experiment Report — SARIMAX.jl Empirical Battery

Date: 2026-06-27 (updated, session 2). No fabricated values.

Central bounded claim supported:

> SARIMAX.jl exposes SARIMA/SARIMAX estimation as an explicit JuMP optimization model, allowing the
> dynamic SARIMAX specification to be reused while objective functions, constraints, regularization,
> initialization handling, and solvers are varied.

## Environment

- macOS (Darwin 25.5.0), arm64. Julia **1.11.9** (`julia +1.11`; default 1.12.2 is incompatible with
  the package's `julia = "1.0 - 1.11"` compat — B1). Python **3.9.6** (numpy 2.0.2, pandas 2.3.3,
  scipy 1.13.1, statsmodels 0.14.6). R **4.6.1** (forecast, jsonlite); R fits use `method="ML"`
  (CSS-ML gives "non-stationary seasonal AR" on the seasonal model).
- Package tests: **35 passed, 1 failed** (B2) — a strict `atol=1e-3` AR-coefficient recovery
  assertion; not a functional defect.
- Metadata in `results/raw/environment/`.

## Commands

```bash
python experiments/scripts/setup/gen_sim_data.py
julia +1.11 --project=experiments/env experiments/scripts/run_validation_julia.jl
python experiments/scripts/run_validation_python.py
/opt/homebrew/bin/Rscript experiments/scripts/run_validation_r.R
python experiments/scripts/diagnose_exog.py                       # B4 diagnostic
julia +1.11 --project=experiments/env experiments/scripts/run_forecasting_julia.jl
python experiments/scripts/run_forecasting_python.py
/opt/homebrew/bin/Rscript experiments/scripts/run_forecasting_r.R
julia +1.11 --project=experiments/env experiments/scripts/run_architecture_extensions.jl
julia +1.11 --project=experiments/env experiments/scripts/run_solver_diagnostics.jl        # package-level
julia +1.11 --project=experiments/env experiments/scripts/run_solver_jump_diagnostics.jl   # JuMP-level
python experiments/scripts/combine_results.py                     # builds CSV + .tex tables
```

## Block 1 — Validation (`table_validation` → `tab:validation_implementations`)

ARIMA(1,0,0)/(0,0,1)/(1,0,1) on simulated ARMA, airpassengers, and an ARX exogenous case. Julia 6/6,
statsmodels 6/6, R 6/6 (with `method="ML"`).

- **Pure ARMA/ARIMA:** AR/MA coefficients agree across all three (ARIMA(1,0,1): φ≈0.52, θ≈0.556; RSS≈307).
- **Exogenous case (B4 — RESOLVED):** now compared like-for-like as **ARX**; all three agree:
  φ=0.5203, RSS=295.61, logLik=−422.56. See "B4 resolution" below and `exog_discrepancy.md`.
- **Bound:** AIC/BIC remain non-comparable across tools (constant conventions, B3); coefficients, RSS,
  and matched-spec log-likelihood are comparable.

### B4 resolution — exogenous discrepancy explained (not a bug)

SARIMAX.jl models exog as a **dynamic-regression / ARX** model — the AR term acts on the *observed*
series (`src/models/sarima.jl:707-716`, constraint `y_t = ŷ_t + ε_t`):
`y_t = c + Σφ_i y_{t-i} + Σβ_j x_{j,t} + Σθ_j ε_{t-j} + ε_t`.
statsmodels `SARIMAX(exog)` and R `Arima(xreg)` default to **regression-with-ARIMA-errors** (AR on the
regression residual). These are different models. The DGP is ARX, so SARIMAX.jl recovers the truth
(φ→0.5, β→(1.5,−0.8), σ²→1.0). Emulating ARX in statsmodels/OLS (lagged-y regressor) reproduces
SARIMAX.jl exactly (φ=0.520, β=(1.534,−0.684), RSS=295.61). **Manuscript must state** SARIMAX.jl's
exogenous form is ARX, not regression-with-ARIMA-errors.

## Block 2 — Rolling-origin forecasting (`table_forecasting` → `tab:forecast_oos`)

Expanding-window rolling origins; mean over origins. AirPassengers (monthly, s=12, H=12, 5 origins) and
GDPC1 (quarterly, H=8, 5 origins). 0 failures across all implementations.

| Task | Implementation | RMSE | MASE |
|------|----------------|------|------|
| AirPassengers | statsmodels (1,0,1)(1,0,1)₁₂ | 1.30 | 1.09 |
| AirPassengers | **SARIMAX.jl** (1,0,1)(1,0,1)₁₂ | **1.48** | **1.25** |
| AirPassengers | R forecast | 1.59 | 1.34 |
| AirPassengers | seasonal-naive | 2.50 | 2.22 |
| GDPC1 | statsmodels (1,1,1) | 324.6 | 2.97 |
| GDPC1 | **SARIMAX.jl** (1,1,1)+drift | **324.9** | **2.97** |
| GDPC1 | R forecast | 367.4 | 3.44 |
| GDPC1 | naive | 498.2 | 4.99 |

**Does rolling-origin change the forecasting claim?** No — it *strengthens* it. Across two datasets and
multiple origins, SARIMAX.jl is **competitive**: essentially tied with statsmodels on GDPC1, mid-pack on
AirPassengers (between statsmodels and R), and always beating the naive baseline. The claim remains
bounded predictive adequacy, **not** superiority. Runtime note: tables report the **warm mean seconds
per refit** (Julia's one-time JIT compilation excluded via an untimed warm-up fit). Per-fit: AirPassengers
SARIMAX.jl ~4 ms vs statsmodels ~104 ms / R ~100 ms (SARIMAX.jl fastest); GDPC1 ~9 ms vs statsmodels
~27 ms / R ~3 ms (competitive); PJME daily ~0.28 s vs ~0.07–0.08 s (SARIMAX.jl slower — heavier
seasonal optimization on the larger series, one residual variable per observation). The earlier
"slower" impression was dominated by one-time compilation. Note: an R timing bug (a `<<-` scoping
error that made `run_forecasting_r.R` report runtime 0) was fixed; R runtimes are now real.

## Block 3 — Architecture & extensibility (`table_architecture_checks` → `tab:architecture_extensibility`)

Same SARIMAX specification reused under varied settings.

- **Objective swap** (outlier-contaminated series) — REDESIGNED (2026-07-07) to an AR(1)-only
  specification so both problems are convex (QP vs LP over the same constraints) and each objective
  attains its own global optimum: `mse` → RSS 603.97 (min RSS), resid-MAE 0.988, φ̂=0.30;
  `mae` → RSS 616.35, resid-MAE 0.973 (min resid-MAE), φ̂=0.40. Clean, reviewer-proof demonstration;
  the earlier MA-containing swap (MSE RSS 603.96 vs MAE 437.22) confounded objective choice with
  local optima on the nonconvex surface and was replaced.
- **Regularization** (8 regressors): coef-norm shrinks (unreg 1.90 → ridge 1.73 / lasso 1.77 /
  elastic-net 1.76) at small RSS cost (283→307); reuses `objectiveFunction="elastic_net"`, `alpha∈{0,0.5,1}`.
- **Constraint / admissibility (new):** `fit!` imposes only per-coefficient box bounds `[-1,1]` and
  CAN return a **non-stationary** estimate — airpassengers ARIMA(1,0,1) drives φ→1.0 (stat=**N**,
  inv=Y); sim_arma ARIMA(1,0,1) is admissible (stat=Y, inv=Y). `auto` can filter candidates via
  `assertStationarity`/`assertInvertibility` (root-based `StateSpaceModels` checks); on airpassengers
  the best-IC model (2,1,2)(1,1,1)₁₂ was already admissible so the filter did not change the selection.
  **Manuscript must NOT claim `fit!` enforces stationarity/invertibility** — it enforces box bounds;
  admissibility is a post-fit check / search filter.

## Block 4 — Solver modularity & nonconvexity (`table_solver_comparison` → `tab:solver_comparison`)

Two levels.

**Package-level (`fit!`):** ARIMA(0,0,1)/(1,0,1) on `sim_arma`.
- Ipopt: succeeds; objective 373.85 / 306.99 (= validation RSS, cross-confirming the pipeline);
  repeated runs spread 0.0 (deterministic starts).
- HiGHS: fails — quadratic-equality constraints unsupported (cannot represent the nonlinear MA model).
- Alpine via `fit!`: fails (`config-required`) — `fit!(optimizer::DataType)` cannot pass sub-solvers.

**JuMP-level (`run_solver_jump_diagnostics.jl`):** small MA(1), T=40, reconstructing SARIMAX.jl's SSE model.
- **Multistart (genuine):** 25 random starts, all 25 converge to the *same* optimum (34.830, spread
  1.4e-14, 1 distinct optimum) — the conditional-SSE surface is empirically unimodal here. Randomized
  multistart IS implementable at the JuMP level; it is **not** exposed through `fit!` (deterministic starts).
- **Alpine global (explicit Ipopt+HiGHS sub-solvers):** wireable and runs local search (reaches 34.83,
  matching Ipopt) + OBBT bound-tightening, but the global lower-bounding **MIP step fails with HiGHS
  (`OTHER_ERROR`)** at every tested size (T=12/20/40). No global certificate with the available
  open-source MIP solver; a commercial MIP solver (Gurobi/CPLEX) would likely be required.

**Are global solver / multistart usable?**
- Multistart: yes, at the JuMP level (demonstrated); not via the high-level API.
- Global (Alpine): partially — configurable and runs locally, but cannot complete the global solve with
  the open-source MIP back-end. EAGO: not a dependency; not installed/tested.
- **Do not claim** routine global-solver superiority.
- **Certificate (update, `run_solver_alpine_cert.jl`):** with a tuned initialization — Ipopt warm start;
  the *valid* bounds `|ε_t| ≤ √f̄` and objective cutoff `Σε² ≤ f̄` (f̄ = incumbent); minimum-vertex-cover
  partitioning (only θ discretized); and a per-solve MIP time cap — **Alpine+SCIP certifies global
  optimality within a 1% gap** on T=8 (0.60%, 13.7 s) and T=10 (0.52%, 7.4 s). It does not scale
  (T=15 ~8%, T=20 unclosed). So: a certified global optimum (≤1%) IS attainable on small instances in
  the open-source stack; exact/scalable global optimality is not claimed.

## Manuscript mapping

| Manuscript table | Generated source |
|---|---|
| `tab:validation_implementations` | `experiments/tables/table_validation.tex` |
| `tab:forecast_oos` | `experiments/tables/table_forecasting.tex` |
| `tab:architecture_extensibility` | `experiments/tables/table_architecture_checks.tex` |
| `tab:solver_comparison` | `experiments/tables/table_solver_comparison.tex` |

Tables are NOT auto-inserted. Each `.tex` is a standalone `table` with the matching `\label`, ASCII-only
and column-count-checked. The generated column sets are wider than the manuscript placeholders — trim to
the paper's `p{}` widths.

## Exact manuscript claims now supported

- **Estimation correctness (incl. exogenous):** SARIMAX.jl agrees with statsmodels/R/OLS on classical
  ARMA/ARIMA and — under the matched ARX specification — on the exogenous case (coefficients, RSS,
  log-likelihood). B4 is resolved as a specification difference, not a defect.
- **Bounded forecasting adequacy:** under rolling-origin evaluation on two datasets, SARIMAX.jl is
  competitive with mature implementations and beats naive baselines. (Not superiority.)
- **Architecture/extensibility:** one dynamic specification reused across objective (mse/mae), penalty
  (ridge/lasso/elastic-net), and an admissibility search filter.
- **Solver modularity & transparency:** the optimizer is a swappable choice (`fit!(optimizer=…)` for
  Ipopt; JuMP-level for Alpine/multistart); diagnostics are reported honestly.

## Claims NOT supported (do not make)

- That SARIMAX.jl's exogenous coefficients equal default statsmodels/R `xreg` output (different model family).
- Forecasting superiority; robust-objective superiority.
- That `fit!` enforces stationarity/invertibility (it enforces box bounds; admissibility is checked separately).
- Global optimality / routine global-solver superiority (no certificate obtained; Alpine global MIP failed).

## Recommended next steps

1. Try a commercial MIP solver for an Alpine global certificate, or install/test EAGO.
2. Add a second seasonal real dataset and longer rolling-origin horizons.
3. Insert the regenerated tables into `../chapters/experiments.tex` (trim columns to page width).

---

## Final paper run update (2026-06-28, commit 144fb6e)

Package changes since the previous run: HiGHS removed as a dependency; SCIP added and set as Alpine's
default MIP sub-solver; `fit!` gained `mipSolver::DataType = SCIP.Optimizer`; MA invertibility
parameterization added (`invertible=true`). Tests: 42 passed, 1 failed (pre-existing B2 only).

**Block 1 (validation)** — unchanged conclusions; all three implementations agree under comparable
specs; exogenous compared like-for-like ARX (φ=0.5203, RSS=295.61, logLik=−422.56).

**Block 2 (forecasting)** — rolling-origin on AirPassengers (s=12, H=12, 5 origins) and GDPC1
(H=8, 5 origins); SARIMAX.jl competitive (AirPassengers RMSE 1.48 vs statsmodels 1.30, R 1.59;
GDPC1 324.9 vs 324.6, 367.4), beats naive baselines. Bounded adequacy, not superiority.

**Block 3 (architecture)** — objective swap and regularization as before; **new invertibility result**:
`fit!` default box bounds let the airline MA coefficient pile up at the unit root (θ=−1.0), while
`fit!(invertible=true, invertibilityMargin=0.05)` returns θ=−0.95, invertible by construction.
Caveat: the package's seasonal invertibility *checker* uses an additive expansion (separate bug) and
mislabels the seasonal fit; the reflection guarantee does not depend on it. `fit!` does not enforce
stationarity (e.g. ARIMA(1,0,1) on AirPassengers returns φ=1.0).

**Block 4 (solver)** — superseded findings:
- Ipopt baseline deterministic (ARIMA(0,0,1) 19.3241; ARIMA(1,0,1) 15.4668).
- **Alpine+SCIP via `fit!` now works** for `mse`: obj 19.3241 (= Ipopt); time-limited global solve (≈3 min, varies run-to-run) on a 20-point MA(1).
- Alpine+SCIP `mae` (global MILP) did NOT finish within budget via `fit!` (the package sets no Alpine
  time limit) — recorded as blocked, not run unbounded.
- HiGHS (optional, not a dependency): warning emitted for `mse`, none for `mae` (verified).
- JuMP-level: multistart 25 → single optimum (spread 1.4e-14); Alpine+SCIP (300 s) → `OTHER_LIMIT`,
  obj 19.3241, **no global certificate**.

**Global-optimality:** not claimed — Alpine+SCIP returned the local optimum's objective without a
certified gap. Solver modularity and transparency ARE supported.

See `final_claims_for_manuscript.md` for the exact supported/unsupported claims, all caveats
(ARX vs reg-w-ARIMA-errors; AIC/BIC; stationarity/invertibility; Alpine+SCIP), and recommended wording.

---

## External-validation dataset — PJME electricity demand (2026-06-29)

Added one real-world electricity-demand series to test generalization beyond the original benchmarks.
External validation only — no claim changed or added.

- **Dataset:** PJM Hourly Energy Consumption, PJME (PJM East). Public ISO load; Kaggle "Hourly Energy
  Consumption" (CC0 1.0, R. Mulla); GitHub mirror for unauthenticated download. 145,366 hourly rows
  (2002–2018) → cleaned (4 DST duplicates dropped, 30 missing hours interpolated) → **daily mean load**
  (6,059 days, s=7). Hourly full rolling-origin is not tractable for the O(n)-variable formulation, so
  daily aggregation is used (consistent with the paper). Provenance: `results/raw/energy/data/pjme_preprocessing.json`.
- **Forecasting (daily, s=7, H=14, 6 origins; `table_energy_forecasting`):** SARIMAX.jl RMSE 3132 /
  MASE 0.830; statsmodels 3177 / 0.842; R 2837 / 0.743; seasonal-naive 3466 / 0.900. Competitive,
  beats naive. Bounded adequacy, not superiority.
- **Architecture:** same SARIMA(1,1,1)(1,0,1)_7 under MSE/MAE/Ridge/Elastic-Net (only objective
  changes); coefficient norm 1.08/1.57/1.22/1.27, RSS/residual-MAE vary as expected.
- **Solver (reduced MA(1), updated 2026-07-07):** Ipopt reaches obj 296.4645 (θ=0.883); direct SCIP
  certifies the same optimum globally (`OPTIMAL`, gap 0, 8.1 s) — replaces the earlier Alpine run
  (247 s, `OTHER_LIMIT`, no certificate).
- **Conclusions changed:** none.

---

## Solver experiment update — direct SCIP replaces Alpine (2026-07-07)

Motivated by the question "SCIP/Gurobi direto não seria melhor que o Alpine?", we tested SCIP
directly on the nonconvex MA(1) SSE model (it is a QCQP; SCIP handles bilinear terms natively).

**Answer: yes, decisively.** Raw model, no warm start, no tuned initialization:

| T | SCIP term. | gap | time | brute-force check |
|---|---|---|---|---|
| 8 | OPTIMAL | 0 | 1.7 s | 7.36853 ✓ |
| 10 | OPTIMAL | 0 | 1.2 s | 8.52977 ✓ |
| 20 | OPTIMAL | 0 | 4.3 s | 19.32405 ✓ |
| 40 | OPTIMAL | 0 | 14.3 s | 34.83005 ✓ (proves multistart optimum global) |

Package level: `fit!(optimizer=SCIP.Optimizer)` works via the generic path (~6.4 s warm, global RSS).

**Audit finding (B8):** a brute-force profile search over (θ,c) — exact up to grid refinement because
ε follows a deterministic recursion for fixed coefficients — showed the earlier Alpine tuned-run
incumbents (7.3447 at T=8; 8.506 at T=10) lie BELOW the true global optima: they were infeasible
beyond tolerance and the previous "1% certificates" were invalid. Those rows were removed from the
manuscript; raw records preserved (`alpine_cert_results.superseded.json`). Alpine remains wired in the
package (with the HiGHS/MIQP warning guard, still exercised at config level).

**Gurobi:** documented global optimality for nonconvex QCQP via `NonConvex=2` (auto-set by the
package); optional license-gated script `run_solver_gurobi.jl` added; no empirical numbers claimed
(no license on the test machine).

Manuscript synchronized: solver table now shows Ipopt (local), Ipopt multistart, SCIP via `fit!`,
SCIP JuMP-level certificates (gap 0, T=8/10/20/40), and the HiGHS warning checks.

---

## SCIP scaling study — where does exact certification stop? (2026-07-07)

Following "run the SCIP tests without the size limitation", direct SCIP was pushed on increasing
sample sizes of the simulated ARMA series (raw MA(1) SSE model, no warm start, 600 s budget per
instance, `run_solver_scip_scaling.jl`). Frontier found:

| T | Termination | Certified | rel. gap | time |
|---|---|---|---|---|
| 40 | OPTIMAL | yes | 0 | 15.4 s |
| 60 | OPTIMAL | yes | 0 | 31.7 s |
| 80 | OPTIMAL | yes | 0 | 68.0 s |
| 100 | OPTIMAL | yes | 0 | 400.0 s |
| 120 | OPTIMAL | yes | 0 | 297.0 s |
| 150 | TIME_LIMIT | no | 1.0e-5 | 600.8 s |
| 200 | TIME_LIMIT | no | 1.2e-5 | 600.8 s |

**Reading:** exact open-source certificates (rel. gap 0, brute-force-agreed) hold up to **T=120**;
at **T=150 and T=200** SCIP gets within a ~10^-5 relative gap but does **not** close it to a
certificate within 600 s. Wall time is instance-dependent and non-monotone (T=100 took longer than
T=120), as expected for spatial branch-and-bound. This is the honest scalability frontier: the
exact-certification capability is real but limited to small/moderate instances, which is exactly the
bounded claim the manuscript should make. (`table_solver_scaling`.)

---

## Global-value experiment — does certified optimality matter for forecasting? (2026-07-08)

New experiment (`run_solver_global_value.jl`, `table_global_value`): the SAME specification estimated
with Ipopt (local) and direct SCIP (global), identical fit! conventions, forecasts over a 12-step
held-out window (T=80 train).

| Instance | Solver | Certified | Obj | RMSE_oos | MAE_oos | Time |
|---|---|---|---|---|---|---|
| sim MA(1) | Ipopt | no | 73.6533 | 0.50670 | 0.40863 | 0.6 s |
| sim MA(1) | SCIP | yes (gap 0) | 73.6531 | 0.50667 | 0.40862 | 71.6 s |
| sim ARMA(1,0,1) | Ipopt | no | 60.5257 | 0.54986 | 0.44641 | 0.04 s |
| sim ARMA(1,0,1) | SCIP | yes (gap 0) | 60.5256 | 0.54989 | 0.44634 | 393.1 s |
| PJME ARMA(1,0,1) | Ipopt | no | 246.3799 | 3.59617 | 2.84674 | 0.02 s |
| PJME ARMA(1,0,1) | SCIP | gap 1.1e-7 (time limit) | 246.3796 | 3.59619 | 2.84677 | 601.8 s |

**Finding:** the guarantee changes essentially nothing statistically — objectives agree to ~2e-4,
coefficients differ at the 3rd decimal, OOS errors identical to reported precision (certified fit
marginally WORSE OOS on sim ARMA(1,0,1)). On PJME the certificate did not close in 600 s but the dual
bound proves the local solution is within 1.1e-7 of the global optimum. Interpretation for the paper:
certification = verification/audit value, not forecast-accuracy value, on these benign surfaces; the
architecture makes the guarantee-vs-cost trade-off measurable.

**Operational notes:** (i) the batch run segfaulted inside SoPlex (SCIP 8.0 LP solver) on the PJME
instance; the isolated rerun completed normally — recorded as a solver-level flakiness note, raw
record carries `note`. (ii) `bench_common.jl` defines `rmse`/`mae` helpers; scripts must not shadow
them with scalars.
