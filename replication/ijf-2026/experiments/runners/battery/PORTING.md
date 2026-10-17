# Porting the July 2026 battery to the release API

The scripts in this directory are ports of the archived battery scripts
(`artifacts/original-scripts/battery-2026-07/scripts/`, identical to `74eb882^:experiments/scripts`
and to the scripts of the v0.3.0 re-run, commit `8ffc512`, Sarimax `3f68017`) to the release
v1.0.0 (`14dd581`). They are run by `experiments/configs/battery_<block>.toml`
(`./reproduce.sh full --experiment battery_<block>`). Orders, objectives, solver options, time
limits, seeds, rolling origins, brute-force audits and record layouts are those of the originals;
this file lists every deviation and why. Each script repeats its own list at the top.

## Common to every script

- **Locations.** Records go to `$IJF_RUN_DIR/raw/battery-2026-07/<block>/<archived file name>`
  (the originals wrote to `<script dir>/../results/raw/`). Simulated inputs are regenerated into the
  run directory; PJME is read from `$IJF_PJME_DIR`; the package's bundled `datasets/` is found five
  levels above this directory (the originals used `../../datasets`). Nothing is written elsewhere.
- **v0.3.0 defaults made explicit.** Two `fit!` defaults changed: `initialization`
  (`:zeroed` -> `:innovations`) and `stationary` (`false` -> `true`, reflection parameterisation).
  Every `fit!` the originals called without them now passes `initialization = :zeroed,
  stationary = false` (`V030_INITIALIZATION`, `V030_STATIONARY` in `bench_common.jl`). The other
  changed defaults (`invertibilityMargin`, `stationarityMargin` 0 -> 1e-6) are read only when
  `invertible` / `stationary` is true, which no call relies on implicitly.
- **Release changes no keyword undoes.** `fit!` solves on the differenced series divided by its
  standard deviation (commit `1831718`, after v0.3.0) and maps the estimates back; order-2
  coefficients get the box `[-binomial(p,i), binomial(p,i)]` instead of `[-1, 1]` (only `auto`
  reaches order 2 here); `aic`/`bic`/`aicc` score the exact Gaussian likelihood when computable
  (v0.3.0: CSS); `"elastic_net"` is a single-stage penalised estimator; the `"mae"` residual sign
  was fixed. Their effects are listed per script below.
- **Status strings.** Every `termination` is what the solver returned (the release's
  `metadata["solverStatus"]` for `fit!`, `termination_status` for JuMP models), never a label.
- **SCIP records** also carry, from MOI: raw status, primal status, result count, solver objective
  value, dual bound (`objective_bound`), `relative_gap`, solver time, node count and the SCIP
  version (`MOI.SolverVersion`). The archived records had none of these.
- **JSON.** `append_record` writes NaN/Inf as `null` (the original threw, which cost the archived
  scaling run a record) and adds `sarimax_version`. `loadDataset` -> `load_dataset` (same function).

## Per script

| Script | Deviations from the original (beyond the common ones) |
|---|---|
| `bench_common.jl` | `RAW` from `$IJF_RUN_DIR`; `outfile`/`datafile` helpers; NaN-safe `append_record`; new `css_criteria` (v0.3.0 AIC/BIC/AICc), `moi_diagnostics`, `fit_status`, `status_summary`. Metric, origin and naive helpers unchanged. |
| `setup/gen_sim_data.py` | Writes `\n` line endings on every platform (pandas defaults to `os.linesep`). Compares each file with the bundle's copy (and `sim_arma.csv` with `$IJF_FIXTURES`): SHA-256, then values; prints MATCH / MISMATCH / NOT COMPARED, writes `<block>/data/input_check.json`, and on a mismatch continues with the regenerated data, loudly. |
| `setup/gen_energy_data.py` | Reads and verifies `$IJF_PJME_DIR/PJME_hourly.csv`; `\n` line endings; `pjme_preprocessing.json` also records the input path/digest. The derived files are not in the bundle, so they are compared with the SHA-256 digests and statistics of the battery's committed copies (`experiments/results/raw/energy/data/` at `8ffc512`), embedded in the script. |
| `check_pjme.py` | New (the original `reproduce.sh` ran `shasum -c` inline). Exit 4 with "run ./reproduce.sh fetch pjme" when the file is missing or its digest differs; a CRLF copy is accepted on the CR-stripped digest, the convention of `./reproduce.sh fetch`. |
| `run_validation_julia.jl` | `stationary = false` added (initialization was already explicit). AIC/BIC: the release's exact-likelihood values; the v0.3.0 CSS values are recorded as `aic_css`/`bic_css`. Extra fields `solver_status`, `initialization`. |
| `run_validation_python.py`, `run_forecasting_python.py`, `run_energy_forecasting_python.py` | Paths only. |
| `run_validation_r.R`, `run_forecasting_r.R`, `run_energy_forecasting_r.R` | Paths only (`--file=` still locates `datasets/`). The pinned stack is R 4.4.1 / forecast 8.23.0 / jsonlite 1.8.9; the archive used R 4.6.1 / forecast 9.0.2 / jsonlite 2.0.0. |
| `run_forecasting_julia.jl` | v0.3.0 `fit!` defaults; extra `solver_status_counts`. |
| `run_architecture_extensions.jl` | v0.3.0 `fit!` defaults. `"elastic_net"` rows call the release estimator with the same `alpha` and `penaltyTarget = :all` (written out); v0.3.0's two-stage tolerance-constrained adaptive norm no longer exists. Admissibility flags use `checkModelStationarityInvertibility(...; rootMargin = 0.0)` (v0.3.0's strict unit-circle test; release default 1%). `auto` gets every v0.3.0 keyword value explicitly (`integrationTest = "kpss"`, `seasonalIntegrationTest = "ocsb"`, `initialization = :zeroed`, `stationary = false`, ...) and the release-only keywords at neutral values (`rootMargin = 0`, `constrainedRefit = false`, `multistart = false`, ...). `ic` is the release AICc; `ic_css` the v0.3.0 one. |
| `run_solver_diagnostics.jl` | v0.3.0 `fit!` defaults. Ipopt `termination` = statuses of the three repeats (was the label `locally_solved`); SCIP `termination` = MOI status (was "global optimum (matches certified JuMP-level solve)", never checked). The SCIP instance created inside `fit!` is read back through an `optimizer_with_attributes` constructor that keeps a handle on it; bound/objective are recorded in `fit!`'s internal units and converted (`objective_scale` = s^2). Alpine comes from `Sarimax.Alpine` (not in the replication environment). `warning_text` cut at 400 characters (was 400 bytes, which can split a UTF-8 character). |
| `run_solver_jump_diagnostics.jl` | `termination` = Ipopt statuses over the 25 starts plus `termination_counts` (was "locally_solved (optimum certified global by SCIP)"). |
| `run_solver_scip_cert.jl` | SCIP diagnostics; `objective_value`/`value` only with a primal solution. |
| `run_solver_scip_scaling.jl` | Budget and sizes as arguments (`--budget 600 --sizes 40,...,300`, defaults = archived protocol) instead of environment variables; no append mode (one fresh invocation; the archived file was appended over three, two of which crashed). SCIP diagnostics; `theta` only with a primal solution. |
| `run_solver_global_value.jl` | SCIP diagnostics on every record; duplicate `rmse`/`mae` definitions dropped (identical to `bench_common`'s). PJME instance read from the run's `energy/data/pjme_daily.csv`. |
| `run_energy_forecasting_julia.jl` | v0.3.0 `fit!` defaults; `global` on two top-level assignments: the original's `jit = @elapsed ...` inside a top-level `try` created a local (soft scope), so the archive records `jit_warmup_s = 0.0` although the warm-up ran; `nfail += 1` would have thrown on a failed origin. Extra `solver_status_counts`. |
| `run_energy_architecture.jl`, `run_energy_solver.jl` | Not run by any configuration (no table reads their records); run them by hand after the first two steps of `battery_energy.toml`. Same adaptations as the architecture and solver scripts. |

`diagnose_exog.py`, `combine_results.py` and `run_solver_gurobi.jl` are not ported: no table reads
their output (`ijftables/battery.py` replaces `combine_results.py`).

## What differs, and why (Windows check, October 2026)

Measured by running the Julia and Python steps natively (R and SCIP unavailable on that machine)
and rebuilding the tables. The regenerated simulated inputs and PJME-derived files were
byte-identical to the archived ones.

- **AIC/BIC of SARIMAX.jl (validation, 14 cells), IC(aicc) (architecture).** Definition change:
  the release scores the exact Gaussian likelihood. In the validation records the CSS values
  recorded alongside (`aic_css`, `bic_css`) reproduce the archived AIC/BIC. For the ARX row (`:armax` with regressors) the release's exact likelihood removes
  `X*beta` from `y` as for regression with ARMA errors, which is not the fitted model: AIC 1025.87
  against 855.12 (CSS, statsmodels and R all agree on 855.12).
- **auto() order (architecture, 8 cells).** The release search selects (2,1,2)(0,1,1)_12 (exact
  AICc 526.017) over (1,1,1)(0,1,1)_12 (526.073); v0.3.0 selected the latter by CSS AICc.
- **elastic_net rows (architecture, 9 cells).** Different estimator (see above).
- **PJME SARIMAX.jl row (energy, 3 cells).** On 3 of 6 origins the internally rescaled solve
  reaches a lower-SSE local optimum than v0.3.0's unscaled solve; re-solving the unscaled v0.3.0
  problem from the same start reproduces the archived MAE/RMSE exactly.
- **statsmodels (validation 5 cells, forecasting 6 cells).** scipy 1.16.3 vs 1.13.1 (and platform)
  move the optimiser's end point; the AirPassengers ARMA(1,0,1) fit sits on the MA unit root.
- **Termination (solver comparison, 3 cells generated here; the SCIP `fit!` row will be a fourth).**
  Labels replaced by solver statuses, by design.
- Timing columns differ by nature.
