# Experiment Protocol

Empirical battery supporting the SARIMAX.jl manuscript. The central, bounded claim is:

> SARIMAX.jl exposes SARIMA/SARIMAX estimation as an explicit JuMP optimization model, allowing the
> dynamic SARIMAX specification to be reused while objective functions, constraints, regularization,
> initialization handling, and solvers are varied.

Evidence layers are kept separate and claims stay bounded to what each block demonstrates.

## How to run

All Julia commands use the package-compatible interpreter:

```bash
cd Sarimax.jl
julia +1.11 --project=. experiments/scripts/<script>.jl
source experiments/.venv-benchmarks/bin/activate
python experiments/scripts/<script>.py
/opt/homebrew/bin/Rscript experiments/scripts/<script>.R
```

Raw run records: JSON Lines in `results/raw/<block>/`. Processed tables: CSV in `tables/` and
LaTeX `tabular` in `tables/*.tex`. Every record carries: block, dataset id, model order,
implementation, objective, solver, seed, split, status, error (if any), runtime, estimates, metrics.

Fixed seeds for all simulations and multistart diagnostics.

## Block 1 — Validation against established implementations

Goal: estimation credibility in classical least-squares / Gaussian settings vs mature implementations
(statsmodels SARIMAX, R `forecast::Arima`). Directly comparable metrics (coefficients, loglik/AIC/BIC
where definitions align) are distinguished from implementation-specific diagnostics.

Models: ARIMA(1,0,0), ARIMA(0,0,1), ARIMA(1,0,1), SARIMA(1,0,1)(1,0,1)_s (s=4 sim / s=12 monthly),
SARIMAX with 1–2 exogenous regressors. Data: simulated series with known parameters + a bundled real
series (`airpassengers`). Outputs → `table_validation.{csv,tex}`.

## Block 2 — Out-of-sample forecasting

Goal: forecasting adequacy on representative tasks (NOT a superiority claim). Identical train/test
splits across implementations where possible; rolling-origin preferred, fixed split documented if used.
Metrics: MAE, RMSE, sMAPE, MASE (only when scaling benchmark well-defined). Baselines: seasonal naive,
statsmodels SARIMAX, `forecast::Arima`. Outputs → `table_forecasting.{csv,tex}`.

## Block 3 — Architecture and extensibility

Goal: the main software claim — same SARIMAX specification reused under varied estimation settings.

1. Objective swap: same spec under `objectiveFunction="mse"` vs `"mae"` on a clean simulated series
   with documented injected outliers. Compare coefficient stability, residual summaries, objective,
   convergence, runtime.
2. Regularization: SARIMAX with several exogenous regressors, unregularized vs `lambda`/`alpha`
   (ridge α=0, lasso α=1, elastic net via `objectiveFunction="elastic_net"`). Compare coefficient
   norms/stability, forecast error, convergence, runtime.
3. Constraints: unconstrained vs `assertStationarity`/`assertInvertibility` admissibility. No
   stationarity/invertibility-enforcement claims beyond what the code/theory support.

Outputs → `table_architecture_checks.{csv,tex}`.

## Block 4 — Solver modularity & nonconvexity diagnostics

Goal: solver choice exposed as a modular numerical decision; inspect sensitivity in MA-containing
models. NOT a claim that global solvers routinely replace local ones.

Models: ARIMA(0,0,1), ARIMA(1,0,1), small seasonal MA spec if feasible.
Solvers/settings: Ipopt (local); multistart Ipopt with fixed seeds; Alpine on small diagnostic
instances if it works (tested first). EAGO not available (not a dependency). Metrics: objective value,
convergence/termination, runtime, estimates, variation across initializations, optimality gap if
provided. Outputs → `table_solver_comparison.{csv,tex}`.
