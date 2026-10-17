# Reproducibility status

State on 2026-10-17, branch `feature/ijf-replication` (package source = v1.0.0, `14dd581`).
Two dimensions per experiment: table reconstruction from archived results (Level B) and
re-execution from the original data (Level C).

| Status | Meaning |
|---|---|
| **VERIFIED** | raw results are archived and the tables regenerate from them: every reference cell agrees at its printed precision |
| **EXECUTABLE** | the full runner works on the release environment (exercised on a subset); the complete re-run has not been performed |
| **PARTIAL** | some artifacts exist, but inputs, runners or logs needed for the other dimension are missing |
| **BLOCKED** | cannot be supported until a specific missing item is supplied |
| **PENDING** | waiting for results that are still being produced |

The reference values of Level B are the values printed in the historical tables generated at
the time; **none has been compared with the manuscript**, whose source was not available
(LIMITATIONS.md, item 1).

## Summary

| Experiment | Tables (B) | Cells | Full re-run (C) | Notes |
|---|---|---|---|---|
| battery_validation (`tab:validation_implementations`) | VERIFIED | 224 | EXECUTABLE | re-run (Julia, Python): 100/119 non-timing cells agree; AIC/BIC now exact-likelihood by definition; R steps untested |
| battery_forecasting (`tab:forecast_oos`) | VERIFIED | 72 | EXECUTABLE | re-run (Julia, Python): 42/48 agree (statsmodels 4th decimal); R untested |
| battery_architecture (`tab:architecture_extensibility`) | VERIFIED | 117 | EXECUTABLE | re-run end to end: 87/104 agree; elastic net is a new estimator, `auto` picks another order (0.055 AICc apart) |
| battery_solver_comparison (`tab:solver_comparison`) | VERIFIED | 70 | EXECUTABLE | Ipopt rows 27/30 (labels replaced by solver statuses); SCIP rows untested (Linux only) |
| battery_solver_scaling (`tab:solver_scaling`) | VERIFIED | 42 | EXECUTABLE | SCIP only; up to about 4 h; untested |
| battery_global_value (`tab:global_value`) | VERIFIED | 42 | EXECUTABLE | Ipopt rows 18/18; SCIP rows untested; needs `fetch pjme` |
| battery_energy (`tab:energy_forecasting`) | VERIFIED | 16 | EXECUTABLE | 6/9 agree; the release reaches lower-SSE optima on 3 of 6 origins; R untested; needs `fetch pjme` |
| m4_benchmark (headline M4 vs auto.arima) | VERIFIED | 192 | EXECUTABLE | subset re-run reproduces archived series (6/6 yearly) |
| m4_objectives (controlled objective comparison) | PENDING | – | EXECUTABLE (provisional protocol) | final results from a separate workstream |
| m4_objectives_icarai_zeroed (historical) | VERIFIED | in 880 | EXECUTABLE | archived commit not in a release |
| m4_objectives_icarai_innovations (historical) | VERIFIED | in 880 | EXECUTABLE | mixed commits; superseded |
| m4_objective_distributions (Table 7 evidence) | VERIFIED | 2,872 | not applicable | analysis of archived results |
| objectives_monthly_sample (recreio B) | VERIFIED | 86 | PARTIAL | runner needs the ForecastTester harness; commit unknown |
| multistart_monthly_sample (recreio C) | VERIFIED | 21 | PARTIAL | as above |
| isolation_weekly_yearly (recreio D) | VERIFIED | 42 | BLOCKED | the v0_6 wrapper is a reconstruction of an uncommitted tree |
| axes_weekly (recreio E) | VERIFIED | 86 | PARTIAL | runner needs the harness; two commits |
| stable_weekly_yearly (recreio F) | VERIFIED | 64 | EXECUTABLE | restored 0.9 arm |
| robust_cvar_contamination | VERIFIED (internal reference) | 19 | PARTIAL | reference values transcribed from internal notes |
| scip_ipopt_m4_yearly | VERIFIED | 23 | PARTIAL | SCIP gap limit 1%; no dual bounds |
| stable_admissibility_gavea | BLOCKED | – | BLOCKED | raw files not available |
| admissibility_margins | PARTIAL | – | PARTIAL | raw results only; no archived table |
| initialization_order_search | PARTIAL | – | PARTIAL | raw results only; no archived table |

Level B in total: **4,868 reference cells in seven table groups, all agreeing**
(`./reproduce.sh tables`, 31 s on a laptop, run natively on 2026-10-17; the container run is
pending, see "What was tested").

"Historical" experiments ran on feature-branch commits; their tables are verified from the
archive, but they are superseded by `m4_objectives` and must not be combined with it.

## What was tested

| Test | How | Result |
|---|---|---|
| Table reconstruction (all groups) from the full bundle | natively, Python 3.11 + pinned packages | 4,868/4,868 cells agree |
| Table reconstruction from the committed fixture subset | natively | battery, robust_cvar, scip_ipopt: PASS |
| Missing and corrupted archived inputs | natively (removed files; one edited byte) | reported MISSING, exit code 4 |
| Bundle build, extraction and verification | `build_artifact_bundle.py`, `fetch artifacts --from` | deterministic tarball, 535/535 checksums |
| M4 acquisition | `fetch m4 --frequency yearly` from the pinned commit | verified SHA-256 |
| Smoke, Julia (Ipopt fits, non-default objectives, admissibility constraints, forecast evaluation) | natively, Julia 1.10.12 + pinned Manifest | all quantities PASS |
| Smoke, Python (statsmodels, metrics, OWA definitions) | natively | all PASS |
| Smoke, SCIP | natively on Windows | SCIP 8.0.0 crashes on Windows (known upstream); runs in the Linux container |
| Smoke, R | – | R not installed on the development host; runs in the container |
| `check` environment verification | natively | Julia environment matches its Manifest; solvers load |
| Full runner: subset, checkpoint/resume, deadlines | natively, `m4_benchmark --frequency yearly --arm sarimax --limit 4/6` and `--timeout 0.01` | 6/6 series reproduce the archive; resume fitted only the new series; TIMEOUT rows recorded, workers replaced, exit 1 |
| Full runner: interrupted checkpoint, changed source | natively: last row of a task file cut in half, then `--resume`; resume after editing a fingerprinted file | the cut row was dropped (logged, original kept as `.bak`) and its series refitted, 4/4 reproduce the archive; resuming after a source change was refused |
| Battery re-runs (Level C) | natively: every Julia (Ipopt) and Python step of the seven battery configurations, through `run_script_experiment`; `battery_architecture` via `./reproduce.sh full` | all steps exit 0; regenerated simulated and PJME-derived inputs byte-identical to the archived ones; rebuilt cells: validation 100/119, forecasting 42/48, architecture 87/104, solver comparison 27/30, global value 18/18, energy 6/9 (differences explained in experiments/runners/battery/PORTING.md); R steps and SCIP solves not run (no R on the host; SCIP 8.0.0 crashes on Windows) |
| Runner metrics against the archive | `m4data.block_metrics` on the archived forecasts of `cel_mse_zeroed_yearly` and the official M4 yearly test set | 23,000 series, all eight block metrics within 5e-7 relative (the archive stores six decimals) |
| Host script | Git Bash without Docker | exit 3 with message; usage errors exit 2 |
| Docker image build, `check`/`smoke`/`tables`/`full` in the container | GitHub Actions (`ijf-replication.yml`) | **PENDING**: the workflow could not be pushed with the available credential (missing `workflow` scope) |
| Published image pulled by digest | – | not applicable before the release |

## Missing artifacts

1. The manuscript source, to map tables to LaTeX labels and to compare printed values.
2. The final results and frozen protocol of the controlled `m4_objectives` experiment.
3. The raw campaign files of the September 2026 GAVEA campaigns (stable objective: hourly,
   quarterly, daily, monthly; 2x2x2x2 census), the census Manifest and the `fator=1` patch.
4. Dual bounds, gaps, tolerances and SCIP logs of every archived SCIP run, and the log of the
   OPTIMAL re-run of the PJME global-value instance.
5. The package commits behind the contamination/CVaR, admissibility, initialization and
   order-search studies, and runners for them that do not need the ForecastTester harness.
6. The ForecastTester harness repository state used by the recreio campaigns B–E, and the
   committed source of the v0_6 production wrapper.
7. A recorded R/forecast version for the July 2026 auto.arima/Naive2 baselines (inferred
   from logs: R 4.4.1, forecast 8.23.0).
8. Licence confirmation for M4 and PJME redistribution (currently fetched, not redistributed).
