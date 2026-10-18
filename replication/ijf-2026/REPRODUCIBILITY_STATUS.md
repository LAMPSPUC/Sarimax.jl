# Reproducibility status

State on 2026-10-18, branch `feature/ijf-replication` (package source = v1.0.0, `14dd581`),
manuscript revision of 2026-10-16. Two dimensions per experiment: reconstruction of its tables
from archived results (Level B) and re-execution from the original data (Level C).

| Status | Meaning |
|---|---|
| **VERIFIED** | raw results are archived and the tables regenerate from them: every reference cell agrees at its printed precision |
| **EXECUTABLE** | the full runner works on the release environment (exercised on a subset); the complete re-run has not been performed |
| **PARTIAL** | some artifacts exist, but inputs, runners or logs needed for the other dimension are missing |
| **BLOCKED** | cannot be supported until a specific missing item is supplied |
| **PENDING** | waiting for results that are still being produced |

## The manuscript, number by number

`results/manuscript/manuscript_values.csv` lists the 672 numbers printed in the tables, figures
and experimental sections of the 2026-10-16 revision. "Historical" compares each with the
archived historical table it comes from (`check_manuscript_mapping.py`, repository only);
"Regenerated" with the value rebuilt from archived per-series results on a second machine
(macOS arm64, 2026-10-18; inputs described under "What was tested").

| Item | Label | Values | Historical | Regenerated | Source |
|---|---|---|---|---|---|
| Table 2 | `tab:matched_validation` | 20 | 20 agree | 20 agree | battery_validation |
| Table 3 | `tab:m4_forecasting` | 12 | 12 agree | 8 agree; daily and hourly need the full bundle | m4_benchmark |
| Table 4 | `tab:m4_paired_quantiles` | 84 | 84 agree | 56 agree; daily and hourly need the full bundle | m4_benchmark (m4_paired) |
| Table 5 | `tab:m4_objectives` | 44 | 44 agree | 44 agree | m4_objectives_icarai_innovations, m4_objective_distributions |
| Table 6 | `tab:objective_paired_quantiles` | 60 | 60 agree | 60 agree | m4_objective_distributions |
| Table 7 | `tab:local_global_forecasts` | 40 | 38 agree, **2 differ** | 38 agree, **2 differ** | battery_global_value |
| Table S2 | `tab:exog_validation` | 9 | no archived source | – | exog_convention_simulation (BLOCKED) |
| Table S3 | `tab:m4_baseline` | 2 | configuration, see LIMITATIONS item 1 | – | – |
| Table S4 | (added on 2026-10-16) | 2 | 2 agree | 2 agree | m4_benchmark, m4_objective_distributions |
| Table S5 | `tab:global_experiment` | 12 | no archived source | – | global_optimization_grid (BLOCKED) |
| Table S6 | `tab:quantile_demo` | 9 | agree with a derived table only | – | quantile_demo (PARTIAL) |
| Table S7 | `tab:multistart` | 10 | agree with a derived table only | – | multistart_kstart (PARTIAL) |
| Table S8 | `tab:supp_m4_marginal` | 115 | 115 agree | 75 agree; daily and hourly need the full bundle | m4_benchmark (m4_paired) |
| Table S9 | `tab:supp_obj_mase` | 84 | 84 agree | 84 agree | m4_objective_distributions |
| Table S10 | `tab:supp_obj_smape` | 84 | 84 agree | 84 agree | m4_objective_distributions |
| Figure 2 | `fig:m4_pair_ecdf` | 36 | 36 agree | 24 agree; daily and hourly need the full bundle | m4_benchmark (m4_paired) |
| Figure 3 | `fig:objective_ecdf` | 12 | 12 agree | 12 agree | m4_objective_distributions |
| Section 5.3 | `sec:m4_validation` | 4 | 2 agree, 2 checked at run time only | 2 agree; the 41–78% range needs all six frequencies | m4_benchmark (m4_paired) |
| Section 5.4 | `sec:m4_objectives` | 4 | 4 agree | 4 agree | m4_objective_distributions |
| Section 5.5 | `sec:estimation_sensitivity` | 15 | 4 agree with a derived table only; 11 no archived source | – | stable_admissibility_gavea; initialization_experiment, ranking_likelihood_diagnostic (BLOCKED) |
| Section 5.6 | `sec:global_diagnostics` | 14 | 2 agree; 12 no archived source | 2 agree | battery_solver_comparison; global_optimization_grid (BLOCKED) |
| Figure 1, Figure S1 | `fig:ar3_cross_sections`, `fig:weighted_demo` | – | figures without printed numbers; Figure S1's data agree with a derived table | – | ar3_stationarity_figure, weighted_regularization_demo (PARTIAL) |

In all: 599 printed values agree with the historical tables and 515 were regenerated on the
second machine (86 more need the daily and hourly baselines, which only the full bundle
holds); **2 differ** (Table 7, Ipopt times: printed 0.04 s and 0.02 s, archived 0.0018 s and
0.0020 s); 23 agree only with derived tables whose raw inputs are not archived; 46 have no
archived source. Details and what each gap needs: `LIMITATIONS.md`, item 1.

## Experiments

| Experiment | Manuscript | Tables (B) | Full re-run (C) | Notes |
|---|---|---|---|---|
| battery_validation | Table 2 | VERIFIED | EXECUTABLE | re-run (Julia, Python): 100/119 non-timing cells agree; AIC/BIC now exact-likelihood by definition |
| battery_global_value | Table 7 | VERIFIED | EXECUTABLE | Ipopt rows 18/18 re-run; needs `fetch pjme`; two printed times differ from the archive |
| battery_solver_comparison | Section 5.6 | VERIFIED | EXECUTABLE | Ipopt rows 27/30 re-run (labels replaced by solver statuses) |
| battery_forecasting, battery_architecture, battery_solver_scaling, battery_energy | not reported | VERIFIED | EXECUTABLE | July 2026 battery; kept for the record |
| m4_benchmark | Tables 3, 4, S8, S4; Figure 2; Section 5.3 | VERIFIED | EXECUTABLE | subset re-runs reproduce archived series; Section 5.1 describes another monthly protocol |
| m4_objectives (controlled) | to replace the Table 5–6 sources | PENDING | EXECUTABLE (provisional protocol) | final results from a separate workstream; left out of `full --all` until final |
| m4_objectives_icarai_innovations | Table 5 | VERIFIED | EXECUTABLE | feature-branch commits; a re-run executes the protocol on the release |
| m4_objective_distributions | Tables 5, 6, S9, S10; Figure 3; Section 5.4 | VERIFIED | NOT APPLICABLE | analysis of archived campaign files |
| m4_objectives_icarai_zeroed | not reported | VERIFIED | EXECUTABLE | historical |
| objectives_monthly_sample, multistart_monthly_sample, axes_weekly | not reported | VERIFIED | PARTIAL | runners need the ForecastTester harness |
| isolation_weekly_yearly | not reported | VERIFIED | BLOCKED | the v0_6 wrapper is a reconstruction of an uncommitted tree |
| stable_weekly_yearly | not reported (yearly/weekly rows of the 2026-10-17 stable table) | VERIFIED | EXECUTABLE | |
| robust_cvar_contamination | not reported | VERIFIED (internal reference) | PARTIAL | reference values transcribed from internal notes |
| scip_ipopt_m4_yearly | not reported | VERIFIED | PARTIAL | SCIP gap limit 1%; no dual bounds |
| stable_admissibility_gavea | Section 5.5 (admissibility) | BLOCKED | BLOCKED | derived tables of 2026-10-17 archived and consistent; raw `brutos_gavea.zip` not in the bundle |
| admissibility_margins, initialization_order_search | not reported | PARTIAL | PARTIAL | raw results only |
| exog_convention_simulation | Table S2 | BLOCKED | BLOCKED | no runner, outputs or table archived |
| global_optimization_grid | Table S5; Section 5.6 certificates | BLOCKED | BLOCKED | August 2026 certification batch not archived |
| quantile_demo, multistart_kstart, weighted_regularization_demo | Tables S6, S7; Figure S1 | PARTIAL | BLOCKED | derived tables archived; simulation scripts not |
| initialization_experiment, ranking_likelihood_diagnostic | Section 5.5 | BLOCKED | BLOCKED | records not identified |
| ar3_stationarity_figure | Figure 1 | PARTIAL | NOT APPLICABLE | analytic figure; plotting script not archived |

Level B in total: 5,288 reference cells in eight table groups. On the maintainer host that
built the bundle (2026-10-17) the seven groups then present agreed on all 4,868 of their
cells; on the second machine (2026-10-18) every cell whose inputs could be rebuilt there
agreed (4,796 cells; see below). "Historical" experiments ran on feature-branch commits; their
tables are verified from the archive, but they are superseded by `m4_objectives` and must not
be combined with it.

## What was tested

### Second machine (macOS arm64, native, 2026-10-18; this pull request)

| Test | How | Result |
|---|---|---|
| Bundle rebuilt from its sources | `build_artifact_bundle.py` logic for every section whose source exists outside the maintainer host (git: battery, ICARAI, recreio, GAVEA; files: the Table 7 evidence), Python 3.14 | 299 of 535 files byte-identical to `artifacts.sha256` (after making the redaction sum independent of the Python version; before: 291, the 8 redacted recreio files differed in last bits) |
| Table groups on the rebuilt files | `./reproduce.sh tables --native --artifacts <partial bundle>` | battery 583, m4_objectives 880, arms 299, robust_cvar 19, scip_ipopt 23 cells: all agree; groups with missing inputs reported MISSING, exit 4 |
| m4_objective_distributions | generator on the rebuilt per-series and raw files, `archived` set (the published totals taken from `results/expected/` because `tabelas_paper.txt` comes from the maintainer host) | 2,584/2,584 cells agree |
| m4_benchmark, m4_paired (new) | generators on the rebuilt campaign A files, monthly/quarterly/yearly/weekly, with icarai's copy of the stored auto.arima/Naive2 values (same values, another format) | 128/128 and 280/280 cells agree |
| Manuscript reconciliation | all of the above, then `manuscript.reconcile` | 515 printed values regenerated and agree; 2 differ (Table 7 times); see the table above |
| Static checks | `check_package.py`, `check_manuscript_mapping.py`, shellcheck 0.11 | consistent; 599 agree, 2 differ; shellcheck clean |
| Exit codes on the host | `./reproduce.sh` without the Docker daemon, without Docker, with bad usage, with a missing `--from` file | 3, 3, 2, 4 |
| Smoke scripts outside the container | `smoke.R` (R 4.6.1, forecast 9.0.2) and `smoke_python.py` (statsmodels 0.15) | every compared quantity within its tolerance |
| Container on this machine | – | not run: Docker Desktop's virtual disk is limited to 8 GiB here and the image needs more; the container is tested in CI |

### Continuous integration (GitHub-hosted runners, container; pull request 34)

| Test | linux/amd64 | linux/arm64 (experimental) |
|---|---|---|
| Host-side checks (shellcheck, exit codes, checksums, consistency) | PASS | – |
| First image build (`685ffa3`) | FAIL at the R step: the renv.lock check compared `packageVersion()` ("2.1.1") with DESCRIPTION versions ("2.1-1"); fixed in `1c0d0f6` | not reached |
| Second image build (`1c0d0f6`) | R step PASS (binary packages from the snapshot, every version as locked) | FAIL at the R step: the snapshot has no arm64 binaries, packages compile, and the R package curl needs the libcurl headers; fixed by installing them on arm64 |
| Image build, default run without the bundle, fixture tables, missing/corrupted inputs, foreign image, M4 fetch, 2→3-series run with `--resume` | see the latest run of pull request 34 | see the latest run of pull request 34 |

### Maintainer host that assembled the package (Windows, native, 2026-10-17)

| Test | How | Result |
|---|---|---|
| Table reconstruction (seven groups) from the full bundle | natively, Python 3.11 + pinned packages | 4,868/4,868 cells agree |
| Bundle build, extraction and verification | `build_artifact_bundle.py`, `fetch artifacts --from` | deterministic tarball, 535/535 checksums |
| M4 acquisition | `fetch m4 --frequency yearly` from the pinned commit | verified SHA-256 |
| Smoke, Julia (Ipopt fits, non-default objectives, admissibility constraints, forecast evaluation) | natively, Julia 1.10.12 + pinned Manifest | all quantities PASS |
| Smoke, SCIP | natively on Windows | SCIP 8.0.0 crashes on Windows (known upstream); runs in the Linux container |
| Full runner: subset, checkpoint/resume, deadlines | `m4_benchmark --frequency yearly --arm sarimax --limit 4/6` and `--timeout 0.01` | 6/6 series reproduce the archive; resume fitted only the new series; TIMEOUT rows recorded, workers replaced, exit 1 |
| Full runner: interrupted checkpoint, changed source | last row of a task file cut in half, then `--resume`; resume after editing a fingerprinted file | the cut row was dropped (original kept as `.bak`) and refitted, 4/4 reproduce the archive; resuming after a source change was refused |
| Battery re-runs (Level C) | every Julia (Ipopt) and Python step of the seven battery configurations | all steps exit 0; rebuilt cells: validation 100/119, forecasting 42/48, architecture 87/104, solver comparison 27/30, global value 18/18, energy 6/9 (differences in experiments/runners/battery/PORTING.md); R steps and SCIP solves not run there |
| Runner metrics against the archive | `m4data.block_metrics` on archived forecasts of `cel_mse_zeroed_yearly` and the official M4 yearly test set | 23,000 series, all eight block metrics within 5e-7 relative |

Not applicable before the release: the published image pulled by digest (the publish workflow
does it), the Zenodo records.

## Missing artifacts

1. The records behind the printed values without an archived source: the exogenous-convention
   simulation (Table S2); the global-optimization grid and the certified instances of Section
   5.6 (the August 2026 certification batch: instances, SCIP logs, primal and dual bounds,
   tolerances); the initialization and likelihood-ranking diagnostics of Section 5.5.
2. The raw GAVEA archive `brutos_gavea.zip` in the artifact bundle (the 2026-10-17 derived
   tables are archived; the census Manifest and the `fator=1` patch are lost).
3. The simulation scripts of the September 2026 revision analysis (`efg_julia_experiments.jl`,
   `efg_figures.py`, the `a_` stationarity figure script) and their outputs, for Tables S6 and
   S7 and Figures 1 and S1.
4. The final results and frozen protocol of the controlled `m4_objectives` experiment.
5. Dual bounds, gaps, tolerances and SCIP logs of every archived SCIP run, and the log of the
   OPTIMAL re-run of the PJME global-value instance.
6. The package commits behind the contamination/CVaR, admissibility, initialization and
   order-search studies, and runners for them that do not need the ForecastTester harness.
7. The ForecastTester harness repository state used by the recreio campaigns B–E, and the
   committed source of the v0_6 production wrapper.
8. A recorded R/forecast version for the July 2026 auto.arima/Naive2 baselines (inferred from
   logs: R 4.4.1, forecast 8.23.0).
9. Licence confirmation for M4 and PJME redistribution (currently fetched, not redistributed),
   and a decision on redistributing SCIP 8.0.0 (ZIB Academic License) in the public image.
10. The published bundle (Zenodo), image (GHCR) and the DOIs and digest that go into
    `artifacts.lock`, `image.lock`, `CITATION.cff` and the availability statement.
