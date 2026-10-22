# Reproducibility status

State on 2026-10-20, branch `feature/ijf-replication` (package source = v1.0.0, `14dd581`),
manuscript revision of 2026-10-16. Each experiment has two dimensions: reconstruction of its
tables from archived results (Level B) and re-execution from the original data (Level C).

| Status | Meaning |
|---|---|
| **VERIFIED** | raw results are archived and the tables regenerate from them: every reference cell agrees at its printed precision |
| **EXECUTABLE** | the full runner works on the release environment (exercised on a subset); the complete re-run with this runner has not been performed |
| **PARTIAL** | some artifacts exist, but inputs, runners or logs needed for the other dimension are missing |
| **BLOCKED** | cannot be supported until a specific missing item is supplied |

## The manuscript, number by number

**Canonical revision not transcribed.** The canonical manuscript is the revision of 2026-10-19
(controlled objective comparison), which is not available here; the values below are those of
the 2026-10-16 revision. The release check reports this as missing evidence, maps the items the
2026-10-19 revision takes from the controlled campaign to `m4_objectives` in a template
awaiting transcription, and sorts every value into five categories (raw-reproduced 603,
derived-only 56, differing 6, no evidence 5, 2 checked at run time only; 13 textual claims to
verify). See `reports/manuscript_traceability.md` and `FINAL_SCIENTIFIC_AUDIT.md`.

`results/manuscript/manuscript_values.csv` lists the 672 numbers printed in the tables,
figures and experimental sections of the 2026-10-16 revision. The table compares each number
in two ways:

* **Historical**: against the archived historical table it comes from. This is done by
  `check_manuscript_mapping.py`, which needs only the repository.
* **Regenerated**: against the value rebuilt from archived per-series results, in two runs.
  * The container run of 2026-10-20 (linux/amd64, Docker Desktop on Windows 10), with 446 of
    the 535 part-1 files rebuilt from their sources and all of part 2.
  * A native macOS arm64 run of 2026-10-18, with the inputs described under "What was
    tested".

| Item | Label | Values | Historical | Regenerated (container, 2026-10-20) | Regenerated (native macOS, 2026-10-18) | Source |
|---|---|---|---|---|---|---|
| Table 2 | `tab:matched_validation` | 20 | 20 agree | 20 agree | 20 agree | battery_validation |
| Table 3 | `tab:m4_forecasting` | 12 | 12 agree | 12 agree (all six frequencies) | 8 agree | m4_benchmark |
| Table 4 | `tab:m4_paired_quantiles` | 84 | 84 agree | 84 agree | 56 agree | m4_benchmark (m4_paired) |
| Table 5 | `tab:m4_objectives` | 44 | 44 agree | 20 agree (`m4_objectives`); 24 need part-1 inputs not rebuildable from the repository | 44 agree | m4_objectives_icarai_innovations, m4_objective_distributions |
| Table 6 | `tab:objective_paired_quantiles` | 60 | 60 agree | needs part-1 inputs not rebuildable from the repository | 60 agree | m4_objective_distributions |
| Table 7 | `tab:local_global_forecasts` | 40 | 38 agree, **2 differ** (registered errata) | 38 agree, **2 differ** | 38 agree, **2 differ** | battery_global_value |
| Table S2 | `tab:exog_validation` | 9 | 9 agree with the recovered aggregate (reconstructed record, `results/historical/certification-2026-08/`) | – | – | exog_convention_simulation (PARTIAL) |
| Table S3 | `tab:m4_baseline` | 2 | configuration, see LIMITATIONS item 1 | – | – | – |
| Table S4 | (added on 2026-10-16) | 2 | 2 agree | 1 agree | 2 agree | m4_benchmark, m4_objective_distributions |
| Table S5 | `tab:global_experiment` | 12 | 12 agree with the reconstructed record of the August 2026 batch | – | – | global_optimization_grid (PARTIAL) |
| Table S6 | `tab:quantile_demo` | 9 | agree with a derived table, recomputed in CI from the recovered outputs | – | – | quantile_demo |
| Table S7 | `tab:multistart` | 10 | agree with a derived table, recomputed in CI from the recovered per-start records | – | – | multistart_kstart |
| Table S8 | `tab:supp_m4_marginal` | 115 | 115 agree | 115 agree | 75 agree | m4_benchmark (m4_paired) |
| Table S9 | `tab:supp_obj_mase` | 84 | 84 agree | needs part-1 inputs not rebuildable from the repository | 84 agree | m4_objective_distributions |
| Table S10 | `tab:supp_obj_smape` | 84 | 84 agree | needs part-1 inputs not rebuildable from the repository | 84 agree | m4_objective_distributions |
| Figure 2 | `fig:m4_pair_ecdf` | 36 | 36 agree | 36 agree | 24 agree | m4_benchmark (m4_paired) |
| Figure 3 | `fig:objective_ecdf` | 12 | 12 agree | needs part-1 inputs not rebuildable from the repository | 12 agree | m4_objective_distributions |
| Section 5.3 | `sec:m4_validation` | 4 | 2 agree, 2 checked at run time only | 4 agree (the 41–78% range needs all six frequencies) | 2 agree | m4_benchmark (m4_paired) |
| Section 5.4 | `sec:m4_objectives` | 4 | 4 agree | needs part-1 inputs not rebuildable from the repository | 4 agree | m4_objective_distributions |
| Section 5.5 | `sec:estimation_sensitivity` | 15 | 4 agree (admissibility); 8 agree with recovered aggregates (initialization 5, ranking 3); 3 no archived source (ranking: 12.1%, 16.5%, +0.0097) | 4 agree (regenerated from the raw archive) | – | stable_admissibility_gavea; initialization_experiment, ranking_likelihood_diagnostic (PARTIAL) |
| Section 5.6 | `sec:global_diagnostics` | 14 | 2 agree; 8 agree with the reconstructed record; **4 differ** (seasonal grid, registered errata) | 2 agree | 2 agree | battery_solver_comparison; global_optimization_grid (PARTIAL) |
| Figure 1, Figure S1 | `fig:ar3_cross_sections`, `fig:weighted_demo` | – | figures without printed numbers; Figure S1's data agree with a derived table | – | – | ar3_stationarity_figure, weighted_regularization_demo (PARTIAL) |

In all, against the historical tables:

| Outcome | Values |
|---|---:|
| agree | 603 |
| **differ** (Table 7, Ipopt times: printed 0.04 s and 0.02 s, archived 0.0018 s and 0.0020 s) | **2** |
| checked at run time only | 2 |
| agree only with derived tables whose raw inputs are not archived | 19 |
| agree with records reconstructed from the session that produced them (2026-10-21) | 37 |
| **differ** from those records (Section 5.6 seasonal grid: the exponents lost) | **4** |
| no archived source | 5 |

Every printed value that has a regenerable source was regenerated in at least one of the two
runs: 605 values, of which 603 agree and 2 differ. Details, and what each gap needs, are in
`LIMITATIONS.md`, item 1.

**The replacement objective tables.** Tables 5, 6, S9 and S10, Figure 3 and Section 5.4 print
the historical objective campaigns in the 2026-10-16 revision. The controlled experiment
`m4_objectives` regenerates its replacement tables from its own archived records (760 cells,
including `controlled.paired` for S9 and S10, all agree); the replacements are proposed in
`derived/m4-objectives-controlled-2026-10/manuscript_update_proposal.md`, and
`results/manuscript/pending/manuscript_values_2026-10-19_controlled.csv` points every value of
those items at its `m4_objectives_controlled` cell, awaiting the printed values of the
2026-10-19 revision.

## Experiments

| Experiment | Manuscript | Tables (B) | Full re-run (C) | Notes |
|---|---|---|---|---|
| battery_validation | Table 2 | VERIFIED | EXECUTABLE | re-run (Julia, Python): 100/119 non-timing cells agree; AIC/BIC now exact-likelihood by definition |
| battery_global_value | Table 7 | VERIFIED | EXECUTABLE | Ipopt rows 18/18 re-run; needs `fetch pjme`; two printed times differ from the archive (registered errata). Re-run 2026-10-20 with dual bounds and logs (`results/reruns/`): MA(1) certified, both ARMA(1,1) instances at the 900 s limit; PJME certificate classified as insufficient evidence (`reports/global_certification_audit.md`) |
| battery_solver_comparison | Section 5.6 | VERIFIED | EXECUTABLE | Ipopt rows 27/30 re-run (labels replaced by solver statuses); the MA(1) certificates T = 8-40 re-run with dual bounds (certified, 6.2-61.2 s) |
| battery_forecasting, battery_architecture, battery_solver_scaling, battery_energy | not reported | VERIFIED | EXECUTABLE | July 2026 battery; kept for the record |
| m4_benchmark | Tables 3, 4, S8, S4; Figure 2; Section 5.3 | VERIFIED | EXECUTABLE | subset re-runs reproduce archived series; Section 5.1 describes another monthly protocol |
| **m4_objectives** (controlled) | proposed replacement for Tables 5, 6, S9, S10, Figure 3, Section 5.4 | **VERIFIED** (544 cells) | **EXECUTABLE** | complete archived run with the release: 381,436 fits, every series under every objective, none failed (bundle part 2); the package's runner re-runs it and compares series by series: every weekly series (359/359 per objective) and samples of quarterly (40), yearly (40) and monthly (12) reproduce the archive in the image; included in `full --all` |
| m4_objectives_icarai_innovations | Table 5 (2026-10-16 revision) | VERIFIED | EXECUTABLE | feature-branch commits, guard on; **superseded by m4_objectives** (left out of `full --all`) |
| m4_objective_distributions | Tables 5, 6, S9, S10; Figure 3; Section 5.4 (2026-10-16 revision) | VERIFIED | NOT APPLICABLE | analysis of the historical campaign files; superseded by m4_objectives |
| m4_objectives_icarai_zeroed | not reported | VERIFIED | EXECUTABLE | historical |
| objectives_monthly_sample, multistart_monthly_sample, axes_weekly | not reported | VERIFIED | PARTIAL | runners need the ForecastTester harness |
| isolation_weekly_yearly | not reported | VERIFIED | BLOCKED | the v0_6 wrapper is a reconstruction of an uncommitted tree |
| stable_weekly_yearly | not reported (yearly/weekly rows of the 2026-10-17 stable table) | VERIFIED | EXECUTABLE | |
| robust_cvar_contamination | not reported | VERIFIED (internal reference) | PARTIAL | reference values transcribed from internal notes |
| scip_ipopt_m4_yearly | not reported | VERIFIED | PARTIAL | SCIP gap limit 1%; no dual bounds |
| stable_admissibility_gavea | Section 5.5 (admissibility) | **VERIFIED** (1,028 cells) | BLOCKED | the raw archive `brutos_gavea.zip` is in bundle part 2; the stable, admissibility-grid and guard tables of 2026-10-17 regenerate from it; the GAVEA runners need the GAVEA harness (Julia 1.12) and the census Manifest and one patch are lost |
| admissibility_margins, initialization_order_search | not reported | PARTIAL | PARTIAL | raw results only |
| exog_convention_simulation | Table S2 | PARTIAL | BLOCKED | per-DGP aggregate, generator, fitters and reader recovered (2026-10-21); per-replication outputs lost; not ported to a runner |
| global_optimization_grid | Table S5; Section 5.6 certificates | PARTIAL | EXECUTABLE | August 2026 batch reconstructed from its session record (raw outputs lost); 4 Section 5.6 figures differ; the batch's harness and 46 instances re-run by `run_certification_2026_08.jl` (Ipopt half validated against the record; not run in the container). Re-run by the batch's author on 2026-10-21 in the batch's environment, with SCIP logs (`results/reruns/certification-2026-08-rerun-2026-10-21/`): same series, the 23 certificates and every Section 5.6 instance value again, Table S5 counts again except T = 960 (id 38 did not return) |
| quantile_demo, multistart_kstart, weighted_regularization_demo | Tables S6, S7; Figure S1 | VERIFIED (recovered outputs) | PARTIAL | script and outputs recovered (`results/historical/revision-2026-09/recovered/`); CI recomputes the derived tables from them; re-running the script is a new execution |
| initialization_experiment, ranking_likelihood_diagnostic | Section 5.5 | PARTIAL | BLOCKED | scripts and aggregates recovered (2026-10-21); 3 ranking values not recovered; per-fit outputs lost; not ported to runners |
| ar3_stationarity_figure | Figure 1 | PARTIAL | NOT APPLICABLE | analytic figure; its script recovered with the others |

**Level B in total:** 6,860 reference cells in ten table groups.

| Run | Date | Groups | Cells that agree |
|---|---|---|---|
| bundle assembly, the full part 1 | 2026-10-17 | 7 then present | 4,868 of 4,868 |
| native macOS arm64 | 2026-10-18 | every group whose inputs could be rebuilt there | 4,796 |
| container (linux/amd64) | 2026-10-20 | 9 of the 10 groups | 3,988 of 3,988 |
| container (linux/amd64), final audit, commit `8dc4290` | 2026-10-20 | 9 of the 10 groups | 4,204 of 4,204 (m4_objectives_controlled now 760) |

In the container run, `m4_objective_distributions` needed part-1 inputs that cannot be rebuilt from the
repository (the published part 1 supplies them).

The historical objective campaigns ran on feature-branch commits. Their tables are verified
from the archive, but they are superseded by `m4_objectives` and must not be combined with it.

`full --all` runs `battery_validation`, `battery_solver_comparison`, `battery_global_value`,
`m4_benchmark` and `m4_objectives`. It leaves out the experiments the manuscript does not
report and the superseded historical campaign.

## What was tested

### Final scientific audit (2026-10-20; this pull request)

| Test | Command / where | Result |
|---|---|---|
| Package test suite, both fixes (`src/exact_likelihood.jl`, `src/penalty.jl`) | `test/runtests.jl`, Julia 1.10.12 on Windows, the replication Manifest plus Aqua 0.8.16 | 1,443 pass, 0 fail, 0 error, 5 broken (pre-existing `@test_broken`), 6 min |
| Information criteria with regressors | `test/exog_likelihood.jl` against an independent dense likelihood | 21/21; 6/21 on v1.0.0 (`reports/exog_likelihood.md`) |
| Effects of that fix on `auto` | `reports/exog_likelihood/effects_exog.jl`, v1.0.0 and fixed, 160 searches | orders change in 140; true order 0/60 -> 37/60 where the semantics and `d` are right |
| Elastic net with an L1 weight of zero | `test/objective_functions.jl`, new testset, OpenBLAS kernels default, Haswell, Zen, Sandybridge, Prescott | every fit LOCALLY_SOLVED, `lambda = 0` equal to least squares; before the fix `lambda = 10, alpha = 0` ended OTHER_ERROR on every kernel |
| Dataset identity | `test/datasets.jl` | 40/40 (11 new checks of `fpp2::a10`) |
| Image rebuilt for the changed `src/` | `./reproduce.sh build`, commit `8dc4290` (fingerprint `429b13e3…`) | 445 s; build checks pass (Ipopt 3.14.19, SCIP 8.0.0, 148 packages) |
| Smoke | `./reproduce.sh smoke --no-build`, clean worktree at `8dc4290` | SUCCESS, 3 min |
| Tables | `./reproduce.sh tables --no-build --artifacts <local bundle>` | 9 groups PASS, 4,204 of 4,204 cells (m4_objectives_controlled 760); m4_objective_distributions MISSING with the local bundle, exit 4 as before; manuscript: 2 DIFFER (the registered errata), the rest as in the table above |
| **M4 univariate results under the fixed code** | `full --experiment m4_objectives --frequency weekly --jobs 10` | 5 x 359 series OK in 22 min; every objective reproduces the archive on 359/359 series (same order; total sMAPE within 1.0e-11 MSE, 1.1e-12 MAE, 5.3e-12 Huber, 1.7e-13 Ridge; forecasts bit-identical for 230/222/230/286 series and otherwise within 6.4e-13 relative), the same as with v1.0.0; OWA MSE 0.9002, MAE 1.2735, Huber 1.0840, Ridge 0.9436 |
| Samples of quarterly and yearly | `full ... --frequency quarterly --frequency yearly --sample 20` | every objective reproduces the archive on 20/20 series of each frequency |
| Global-certificate inventory | `build_global_certificates.py --bundle <local bundle> --pjme-daily <re-run data>` | 255 rows; 22 certified, 190 feasible with open gap, 8 improved feasible without certification, 35 insufficient evidence |
| Reconstructed records (2026-10-21) | `check_certification_2026_08.py` | 45 files verified; the consolidated per-instance table agrees with the per-batch tables on 44 rows; 42 derived values follow from the files |
| Certification re-run (2026-10-21) | `check_certification_rerun_2026_10.py` | 154 files verified against the archived and the received lists (49 logs with the host name redacted, 2 CSV files with CRLF kept as LF); 43 instances identical to the package's; records, comparison and summary follow from the files |
| Recovered outputs | `check_recovered_2026_09.py` | 14 files verified; Tables S6 (30 values), S7 (96 values, recomputed from 360 per-start records) and the Figure S1 table (60 values) agree |
| Release check | `check_manuscript_mapping.py` / `--release` | pull-request mode PASS (6 registered errata, 0 unregistered, 0 stale); release mode FAIL on the 6 errata, as designed; a changed printed value and a stale erratum each make it fail |
| CI at `8dc4290` | GitHub Actions | IJF replication: host, manuscript release check, container amd64 and arm64 all PASS; Documentation PASS; package CI PASS on ubuntu (Julia 1, 1.10) and macOS, FAIL on Windows in the elastic-net test fixed by the `src/penalty.jl` change |

### Container run (linux/amd64, Docker Desktop 27.2.0 on Windows 10, 2026-10-20; this pull request)

The container was given 12 CPU threads and 31 GB of memory.

**The image.** It was built with `./reproduce.sh build` from the checkout, under Git Bash.
* Building every layer took 10.5 min; rebuilding the last layer took 3–5 min.
* The build ran `julia_env.jl` and passed: 148 pinned packages, Ipopt 3.14.19, SCIP 8.0.0.
* The tests below ran on the image of commit `a873ab9` (fingerprint `a3c290ac…`).
* After the wording-only changes that followed (fingerprint `80db5fee…`), `check`, `smoke` and
  `tables` were repeated, plus a sampled full run of `m4_objectives` (weekly and yearly, 5 series
  per objective, all reproducing the archive), with the same results.

**The bundle.** It was built in two parts and checked against both inventories.
* Part 1: 446 of its 535 files were rebuilt from their sources by the part-1 builder's code.
  Each matched `artifacts.sha256` byte for byte before use. The other 89 come from workspaces
  that are not in the repository; the published part 1 holds them.
* Part 2: built twice, with byte-identical tarballs.

| Test | Command | Result |
|---|---|---|
| Environment, identity, inputs | `./reproduce.sh check --artifacts <local bundle>` | exit 4, as required (part 1 incomplete). Python 17/17, Julia 148/148 and R 39/39 locked packages match; forecast 8.23.0; image content and checkout fingerprints match; 176 committed files intact; part 2 368/368 verified; part 1 MISSING (446/535 present) |
| Smoke (Level A) | `./reproduce.sh smoke` | PASS: 61 of 61 compared quantities (Julia/Ipopt fits, non-default objectives, admissibility, SCIP global solve with dual bound, R `forecast`, Python `statsmodels`), 2 min 50 s |
| Tables (Level B), every group | `./reproduce.sh tables --artifacts <local bundle>` | 9 groups PASS, 3,988 of 3,988 cells (battery 583, m4_benchmark 192, m4_paired 420, m4_objectives 880, m4_objectives_controlled 544, arms 299, robust_cvar 19, scip_ipopt 23, stable_admissibility_gavea 1,028); m4_objective_distributions MISSING (part-1 inputs not rebuildable from the repository), so exit 4, as required |
| The two new groups | `... tables --group m4_objectives_controlled --group stable_admissibility_gavea` | exit 0: 544/544 and 1,028/1,028 |
| Manuscript values | (part of `tables`) | see "The manuscript, number by number": Tables 2, 3, 4, S8, Figure 2, Section 5.3 complete; Section 5.5's admissibility values regenerated; Table 7's two Ipopt times differ |
| Fetch a bundle part from a local tarball | `./reproduce.sh fetch artifacts --from sarimax-ijf-2026-artifacts-part2-v1.tar.gz --artifacts <empty dir>` | part 2 extracted, 368/368 verified, its marker written; part 1 has no location yet, so exit 4 |
| An unknown tarball | `... fetch artifacts --from not-a-bundle.tar.gz` | refused (unknown name or SHA-256), exit 4 |
| A corrupted part-2 input | one byte appended to `per_series/mse__weekly.csv.gz`, then `tables --group m4_objectives_controlled` | "checksum mismatch", every cell MISSING, exit 4; file restored and re-verified |
| Default run without the bundle | `./reproduce.sh --offline --artifacts <empty dir>` | check and tables MISSING INPUT, smoke PASS, final verdict FAILURE, exit 4 |
| `list` | `./reproduce.sh list` | 30 experiments with both statuses; `full --all` targets as listed above |
| M4 acquisition | `./reproduce.sh fetch m4 --frequency weekly --frequency quarterly --frequency yearly --frequency monthly` | 8 files from the pinned commit, SHA-256 verified (the campaign's local copies are the same files up to line endings) |
| **Full run, controlled campaign, all weekly series** | `./reproduce.sh full --experiment m4_objectives --frequency weekly --jobs 10 --artifacts <local bundle>` | 5 × 359 series OK in 20 min. **Every objective reproduces the archived campaign on 359/359 series**: same selected order, total sMAPE within 1.1e-11. Forecasts are bit-identical for 230 (MSE), 222 (MAE), 230 (Huber) and 286 (Ridge) series; elsewhere they differ by at most 6.4e-13 relative. OWA with the run's own Naive2: MSE 0.9002, MAE 1.2735, Huber 1.0840, Ridge 0.9436 (the archived Table 5 row) |
| Full run, samples of the other frequencies, with `--resume` | `full --experiment m4_objectives --frequency quarterly --frequency yearly --sample 25`, then the same with `--sample 40 --resume`; `--frequency monthly --sample 12` | all OK. The resumed run fitted only the 15 new series of each of its 10 tasks (25 already done). **Every objective reproduces the archive: quarterly 40/40, yearly 40/40, monthly 12/12** (same order, same total sMAPE) |
| `full --all`, one series per task | `./reproduce.sh full --all --limit 1 --jobs 4` (fetches the M4 daily/hourly files and PJME first) | overall PASS in 73 min. It ran `battery_validation`, `battery_solver_comparison` and `battery_global_value` (every Julia, R and Python step; the three SCIP global solves took about 40 min). Their rebuilt tables against the historical ones: 177/224, 58/70 and 27/42 cells agree. That includes timing cells and the definition changes from v0.3.0 to the release listed in `experiments/runners/battery/PORTING.md`; reported as WARN, as designed. It then ran `m4_benchmark` (6 frequencies × Sarimax.jl, auto.arima and Naive2: every series reproduces the archive, the R baselines included) and `m4_objectives` (4 frequencies × 4 objectives: every series reproduces the archive) |
| Worker start-up | events of the runs above | the 10 workers of a task were ready 1 min 50 s after it started (each loads the package for about a minute); before this change they started one after another, 11 min per task |

### Native run on macOS arm64 (2026-10-18; this pull request)

| Test | How | Result |
|---|---|---|
| Bundle rebuilt from its sources | `build_artifact_bundle.py` logic for every section whose source is in git (the battery and the three staging branches) or in the Table 7 evidence, Python 3.14 | 299 of 535 files byte-identical to `artifacts.sha256` (after making the redaction sum independent of the Python version; before: 291, the 8 redacted campaign A–F files differed in last bits) |
| Table groups on the rebuilt files | `./reproduce.sh tables --native --artifacts <partial bundle>` | battery 583, m4_objectives 880, arms 299, robust_cvar 19, scip_ipopt 23 cells: all agree; groups with missing inputs reported MISSING, exit 4 |
| m4_objective_distributions | generator on the rebuilt per-series and raw files, `archived` set (the published totals taken from `results/expected/` because `tabelas_paper.txt` is not in git) | 2,584/2,584 cells agree |
| m4_benchmark, m4_paired (new) | generators on the rebuilt campaign A files, monthly/quarterly/yearly/weekly, with the August 2026 objective campaigns' copy of the stored auto.arima/Naive2 values (same values, another format) | 128/128 and 280/280 cells agree |
| Manuscript reconciliation | all of the above, then `manuscript.reconcile` | 515 printed values regenerated and agree; 2 differ (Table 7 times); see the table above |
| Static checks | `check_package.py`, `check_manuscript_mapping.py`, shellcheck 0.11 | consistent; 599 agree, 2 differ; shellcheck clean |
| Exit codes outside the container | `./reproduce.sh` without the Docker daemon, without Docker, with bad usage, with a missing `--from` file | 3, 3, 2, 4 |
| Smoke scripts outside the container | `smoke.R` (R 4.6.1, forecast 9.0.2) and `smoke_python.py` (statsmodels 0.15) | every compared quantity within its tolerance |
| SCIP smoke test after the fix | `smoke.jl scip_global`, Julia 1.12.6, SCIP.jl 0.11.14 (SCIP 8.0.0) | 11/11 checks pass; `Sarimax.fit!` with SCIP, `:warmup`: θ = 0.67573 against the grid's 0.67584 (`:zeroed` gives 0.77839, the optimum of its own objective, 0.77859) |
| R runners outside the container | `m4_worker.R` (both methods, JSON protocol) and the battery's `run_validation_r.R` after `gen_sim_data.py`, R 4.6.1 + forecast 9.0.2 | the worker answers both methods; the battery R step reproduces the 11 archived records exactly (runtime excluded) |
| Container | – | not run natively: Docker Desktop's virtual disk was limited to 8 GiB and the image needs more; the container is tested in CI and in the container run above |

### Continuous integration (GitHub-hosted runners, container; pull request 34)

| Test | linux/amd64 | linux/arm64 (experimental) |
|---|---|---|
| Host-side checks (shellcheck, exit codes, checksums, consistency) | PASS | – |
| First image build (`685ffa3`) | FAIL at the R step: the renv.lock check compared `packageVersion()` ("2.1.1") with DESCRIPTION versions ("2.1-1"); fixed in `1c0d0f6` | not reached |
| Second image build (`1c0d0f6`) | R step PASS (binary packages from the snapshot, every version as locked) | FAIL at the R step: the snapshot has no arm64 binaries, packages compile, and the R package curl needs the libcurl headers; fixed by installing them on arm64 (`09f98a2`) |
| Third run (`09f98a2`): image build | PASS | PASS |
| `check` in the container | PASS: Python, Julia and R environments match their locks, Ipopt and SCIP load, image identity verified; the unpublished bundle is reported MISSING, as it must be | same |
| `smoke` in the container | 60 of 61 compared quantities PASS (Julia/Ipopt, SCIP, R, Python). FAIL: `package_scip_theta`, a defect of the test: the package fit used `initialization = :zeroed` (which drops the first observation), the reference is the optimum of the objective that keeps it, `:warmup`. Fixed; checked in the native macOS run (11/11 SCIP checks pass) | same |
| Fixture tables, missing/corrupted inputs, foreign image, M4 fetch, sampled full runs (m4_benchmark 2→3 series with `--resume`, the objective campaign, the battery validation experiment) | not reached in the third run (the job stops at the first failing step); see the latest run of pull request 34 | same |
| Fourth run (`f415b60`): every step | PASS (fixture tables, missing/corrupted inputs, foreign image, M4 fetch, sampled full runs) | PASS |
| Fifth run (`a873ab9`, this update: rebased Julia environment, two-part bundle, final `m4_objectives`, parallel worker start-up): every step, including the controlled objective campaign on 3 weekly series per arm compared with its archive | PASS (job 13 min) | PASS (job 15 min) |
| Package CI (`ci.yml`) at `a873ab9` | Julia 1 and 1.10 on ubuntu, Julia 1 on windows and macos: PASS (the windows job had failed at `f415b60`, a commit that changes no package file; it passed at `a873ab9`, so that failure is not reproducible; its log needs a GitHub sign-in) | – |

### Native run on Windows that assembled bundle part 1 (2026-10-17)

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

Two items of the previous list were resolved on 2026-10-20:

* the final results and frozen protocol of the controlled `m4_objectives` experiment, now in
  bundle part 2;
* the raw archive of the September 2026 stable and admissibility campaigns, also in bundle part 2.

What remains:

0. **The canonical manuscript** (2026-10-19 revision: PDF, supplement and source zip).
1. **Records behind printed values with no archived source** (`reports/provenance_gaps.md`):
   the source of the mixed-seasonal-order shares and the OWA of the Section 5.5 ranking
   diagnostic. The August 2026 certification batch, Table S2 and the other Section 5.5
   diagnostics were supplied on 2026-10-21 as reconstructions (raw outputs lost;
   `reports/certification_batch_2026_08.md`); the certification batch was re-run with logs on
   2026-10-21 by its author (`results/reruns/certification-2026-08-rerun-2026-10-21/`).
2. **The census Manifest and the `fator=1` patch** behind its normalization-off cells
   (lost), needed to re-execute those arms.
3. ~~The simulation scripts of the September 2026 revision analysis and their outputs~~:
   recovered on 2026-10-20 (`results/historical/revision-2026-09/recovered/`).
4. **SCIP evidence:** the tolerances and SCIP logs of the archived SCIP runs, and the log of the
   OPTIMAL re-run of the PJME global-value instance (not recoverable). Dual bounds and logs now
   exist for a new identified re-run of the Table 7 and MA(1) certificate experiments
   (`results/reruns/full-all-limit1-2026-10-20/`).
5. **Package commits** behind the contamination/CVaR, admissibility, initialization and
   order-search studies, and runners for them that do not need the ForecastTester harness.
6. **The ForecastTester harness** repository state used by campaigns B–E, and the
   committed source of the v0_6 production wrapper.
7. **The R/forecast version** of the July 2026 auto.arima/Naive2 baselines, which was not
   recorded (inferred from logs: R 4.4.1, forecast 8.23.0).
8. **Licences and redistribution:**
   * confirmation for M4 and PJME, which are currently fetched, not redistributed;
   * a decision on redistributing SCIP 8.0.0 (ZIB Academic License) in the public image;
   * its authors' agreement to publish the raw archive `brutos_gavea.zip` in the data record.
9. **Publication identifiers:** the published bundle (Zenodo, both parts), the image (GHCR),
   and the DOIs and digest that go into `artifacts.lock`, `image.lock`, `CITATION.cff` and the
   availability statement.
10. **Part-1 inputs not rebuildable from the repository:** a run of the `m4_objective_distributions`
    group needs part-1 files that cannot be rebuilt from the repository
    (the a1ebc0d re-run and the derived objective distributions). The published part 1
    supplies them.
