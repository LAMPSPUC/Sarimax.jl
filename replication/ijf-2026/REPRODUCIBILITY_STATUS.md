# Reproducibility status

State at the commit that contains this file, branch `feature/ijf-replication` (package source:
v1.0.0 with the fixes listed in the CHANGELOG), for the submitted manuscript identified in
`results/manuscript/REVISION.toml`. Each experiment has two dimensions: reconstruction of its
tables from archived results (Level B) and re-execution from the original data (Level C).

| Status | Meaning |
|---|---|
| **VERIFIED** | raw results are archived and the tables regenerate from them: every reference cell agrees at its printed precision |
| **EXECUTABLE** | the full runner works on the release environment (exercised on a subset); the complete re-run with this runner has not been performed |
| **PARTIAL** | some artifacts exist, but inputs, runners or logs needed for the other dimension are missing |
| **BLOCKED** | cannot be supported until a specific missing item is supplied |

## The manuscript, number by number

`results/manuscript/manuscript_values.csv` lists the 836 numbers printed in the tables, figures
and text of the submitted manuscript and its Supplementary Material, each with the archived
record it is compared with. `check_manuscript_mapping.py` (CI job "Manuscript release check")
compares every value with the historical value of its table-group cell or with its derived
record, without the artifact bundle; `./reproduce.sh tables` compares the regenerated cells
with the bundle (`verification/VERIFICATION_REPORT.md`, section 2).

Categories: **raw-reproduced**, regenerated from archived per-series or per-instance records by a
table group; **derived-only**, agrees with an archived record that no table group rebuilds (a
value recomputed by a check from the records of an identified re-run, a recovered script
output, a derived quantity, or a summary record whose individual outputs are lost; never
counted as reproduced; the breakdown is in `verification/VERIFICATION_REPORT.md`, section 2);
**not checked**, compared only at run time.

| item | label | raw-reproduced | derived-only | differing | no evidence | not checked |
|---|---|---|---|---|---|---|
| Table 2 | tab:matched_validation | 20 |  |  |  |  |
| Table 3 | tab:m4_forecasting | 12 |  |  |  |  |
| Table 4 | tab:m4_paired_quantiles | 84 |  |  |  |  |
| Table 7 | tab:local_global_forecasts | 33 | 3 |  |  |  |
| Table S2 | tab:exog_validation |  | 9 |  |  |  |
| Table S10 | tab:quantile_demo |  | 9 |  |  |  |
| Table S11 | tab:multistart |  | 10 |  |  |  |
| Table S12 | tab:supp_m4_marginal | 115 |  |  |  |  |
| Table 5 | tab:m4_objectives | 24 |  |  |  |  |
| Table 6 | tab:objective_paired_quantiles | 60 |  |  |  |  |
| Table S13 | tab:supp_obj_mase | 84 |  |  |  |  |
| Table S14 | tab:supp_obj_smape | 84 |  |  |  |  |
| Section 5.3 | sec:m4_validation | 3 |  |  |  | 2 |
| Section 5.5 | sec:estimation_sensitivity | 4 | 14 |  |  |  |
| Section 5.6 | sec:global_diagnostics | 2 | 26 |  |  |  |
| Section 5.1 | sec:experimental_setup | 3 |  |  |  |  |
| Section 5.4 | sec:m4_objectives | 7 | 3 |  |  |  |
| Table S5 | tab:global_experiment |  | 49 |  |  |  |
| Table S9 | tab:initialization_gate |  | 18 |  |  |  |
| Figure 2 | fig:m4_pair_ecdf | 36 |  |  |  |  |
| Figure 3 | fig:objective_ecdf | 4 |  |  |  |  |
| Table S4 | tab:weekly_mse_protocols | 4 |  |  |  |  |
| Table S6 | tab:global_certificates |  | 14 |  |  |  |
| Table S7 | tab:global_isolated |  | 6 |  |  |  |
| Table S8 | tab:ten_seed_initialization |  | 4 |  |  |  |
| Table S15 | tab:stable_m4 | 36 | 9 |  |  |  |
| Table S16 | tab:supp_stable_mase | 45 |  |  |  |  |
| **total** | | **660** | **174** | **0** | **0** | **2** |

The 23 statements of the text checked against the archive are in
`results/manuscript/textual_claims.csv`: all 23 supported, including the moving-average
normalization of Section 3.5 and Supplementary Section S2 (C8, C9), the configuration of
Table S3 (C3) and the archived records behind Tables S2, S8 and S9 (C18-C20).

## Experiments

| Experiment | Manuscript | Tables (B) | Full re-run (C) | Notes |
|---|---|---|---|---|
| battery_validation | Table 2 | VERIFIED | EXECUTABLE | re-run (Julia, Python): 100/119 non-timing cells agree; AIC/BIC now exact-likelihood by definition |
| battery_global_value | Table 7 | VERIFIED | EXECUTABLE | Ipopt rows 18/18 re-run; needs `fetch pjme`; the Ipopt times are printed at the precision of the archived record. Re-run 2026-10-20 with dual bounds and logs (`results/reruns/`): MA(1) certified, both ARMA(1,1) instances at the 900 s limit; PJME certificate classified as insufficient evidence (`reports/global_certification_audit.md`) |
| battery_solver_comparison | Section 5.6 | VERIFIED | EXECUTABLE | Ipopt rows 27/30 re-run (labels replaced by solver statuses); the MA(1) certificates T = 8-40 re-run with dual bounds (certified, 6.2-61.2 s) |
| m4_benchmark | Tables 3, 4, S12, S4; Figure 2; Section 5.3 | VERIFIED | EXECUTABLE | subset re-runs reproduce archived series |
| **m4_objectives** (controlled) | Tables 5, 6, S13, S14, S4; Figure 3; Section 5.4 | **VERIFIED** (760 cells) | **EXECUTABLE** | complete archived run with the release: 381,436 fits, every series under every objective, none failed; the package's runner re-runs it and compares series by series: every weekly series (359/359 per objective) and samples of quarterly (40), yearly (40) and monthly (12) reproduce the archive in the image; included in `full --all` |
| stable_weekly_yearly | weekly and yearly rows of Tables S15, S16 (through stable_admissibility) | VERIFIED | EXECUTABLE | |
| stable_admissibility | Tables S15, S16; Sections 5.4 (Stable), 5.5 (admissibility) | **VERIFIED** (824 cells) | BLOCKED | the stable and admissibility-grid tables of 2026-10-17 regenerate from the campaigns' raw files; their runners need their original harness (Julia 1.12), and the grid's Manifest and one patch are lost |
| exog_convention_simulation | Table S2 | PARTIAL | BLOCKED | per-DGP aggregate, generator, fitters and reader recovered (2026-10-21); per-replication outputs lost; not ported to a runner |
| global_optimization_grid | Tables S5, S6, S7; Section 5.6 | PARTIAL | EXECUTABLE | August 2026 batch reconstructed from its session record (raw outputs lost); the batch's harness and 46 instances re-run by `run_certification_2026_08.jl` (Ipopt half validated against the record; not run in the container). Re-run by the batch's author on 2026-10-21 in the batch's environment, with SCIP logs (`results/reruns/certification-2026-08-rerun-2026-10-21/`): same series, the 23 certificates and every Section 5.6 instance value again, Table S5 counts again except T = 960 in the queue (id 38 did not return; alone on 2026-10-22 it stops at the time limit, as printed); id 35 does not return (SCIP's presolve, `completesol`) |
| quantile_demo, multistart_kstart, weighted_regularization_demo | Tables S10, S11; Figure S1 | VERIFIED (recovered outputs) | PARTIAL | script and outputs recovered (`results/historical/revision-2026-09/recovered/`); CI recomputes the derived tables from them; re-running the script is a new execution |
| initialization_experiment | Tables S8, S9; Section 5.5 | PARTIAL | PARTIAL | ten-seed per-seed rows recovered; the 1,500-series comparison re-run at f90e554 with per-series records (`results/reruns/gate-1500-f90e554-2026-10-22/`) |
| ranking_likelihood_diagnostic | Section 5.5 | PARTIAL | BLOCKED | aggregates recovered (2026-10-21); per-fit outputs lost; not ported to runners |
| ar3_stationarity_figure | Figure 1 | PARTIAL | NOT APPLICABLE | analytic figure; its script recovered with the others |

**Level B in total:** 2,532 reference cells in five table groups (battery 336, m4_benchmark 192,
m4_paired 420, m4_objectives_controlled 760, stable_admissibility 824); every one agrees with the
bundle (section "Artifact bundle and tables" below).

`full --all` runs `battery_validation`, `battery_solver_comparison`, `battery_global_value`,
`m4_benchmark` and `m4_objectives`; the certification grid and the 1,500-series comparison have
their own runners (`global_optimization_grid`; the gate's archived bundle).

## What was tested

The verification of the package for the submitted manuscript is
`verification/VERIFICATION_REPORT.md`. The dated records below are the tests run while the
package was assembled; where they count manuscript values, they count the inventory of that
date.

### Artifact bundle and tables (2026-10-29; this pull request)

The bundle holds only the experiments behind printed values (`ARTIFACT_GUIDE.md`, section 4).

| Test | Command / where | Result |
|---|---|---|
| Bundle build | `build_artifact_bundle.py`, Python 3.12, from the repository and the ForecastTester workspace, twice | the same tarball both times: 385 files, SHA-256 `0d42bcf2…` (`artifacts.lock`); every file built from the same original bytes as the earlier internal build of these campaigns is byte-identical to it, except the seven campaign A and F files, which now carry two comment lines about the removed column (their data rows are identical), and the row selection of `table3_weekly_reconciliation.csv` |
| Staging | `stage_release.py --bundle FILE`, then `--check` | 385/385 files verified against `artifacts.sha256` |
| Tables, every group | `ijf.py tables`, natively (Python 3.12), on the extracted bundle | 5 groups PASS, 2,532 of 2,532 cells; manuscript: 836 values, 660 regenerated, 174 consistent with a derived record, 0 differing |
| Fixture subset | `ijf.py tables --group battery` on `data/fixtures/bundle-subset` | PASS |
| Host checks | `check_package.py`, `check_manuscript_mapping.py --release`, `check_recovered_2026_09.py`, `check_certification_2026_08.py`, `check_certification_rerun_2026_10.py`, `check_gate_rerun_2026_10.py`, `derive_manuscript_quantities.py --bundle` | all PASS |
| Global-certificate inventory | `build_global_certificates.py --bundle <bundle> --pjme-daily <derived by gen_energy_data.py>` | 21 rows (17 certified, 3 feasible with open gap, 1 insufficient evidence), each identical to its row in the inventory of 2026-10-20 |
| Container (linux/amd64), clean worktree of `7110fd8` | `./reproduce.sh build`, then `./reproduce.sh --no-build --artifacts <bundle>` | build 4 min; **SUCCESS**, exit 0, in 4.7 min: check PASS (locks, identity, the bundle's 385 files, SCIP 8.0.0), smoke 61/61, tables 2,532 of 2,532 cells, manuscript 662 regenerated + 174 consistent, 0 differing |

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
| **M4 univariate results under the fixed code** | `full --experiment m4_objectives --frequency weekly --jobs 10` | 5 x 359 series OK in 22 min; every objective reproduces the archive on 359/359 series (same order; total sMAPE within 1.0e-11 MSE, 1.1e-12 MAE, 5.3e-12 Huber, 1.7e-13 Ridge; forecasts bit-identical for 230/222/230/286 series and otherwise within 6.4e-13 relative), the same as with v1.0.0; OWA MSE 0.9002, MAE 1.2735, Huber 1.0840, Ridge 0.9436 |
| Samples of quarterly and yearly | `full ... --frequency quarterly --frequency yearly --sample 20` | every objective reproduces the archive on 20/20 series of each frequency |
| Reconstructed records (2026-10-21) | `check_certification_2026_08.py` | 45 files verified; the consolidated per-instance table agrees with the per-batch tables on 44 rows; 42 derived values follow from the files |
| Initialization gate re-run (2026-10-22) | `results/reruns/gate-1500-f90e554-2026-10-22/` (`f90e554` from a git bundle, the August Manifest, Julia 1.12.3) | 1,500 series per arm, all OK; fits at the iteration limit 48.3% (`:penalized`, August 47.4%) and 0.2% (`:innovations`, as in August); `:innovations` 1,494/1,500 series equal to the August per-series record; wall times machine-dependent |
| Certification re-run (2026-10-21) | `check_certification_rerun_2026_10.py` | 154 files verified against the archived and the received lists (49 logs with the host name redacted, 2 CSV files with CRLF kept as LF); 43 instances identical to the package's; records, comparison and summary follow from the files |
| Recovered outputs | `check_recovered_2026_09.py` | 14 files verified; Tables S6 (30 values), S7 (96 values, recomputed from 360 per-start records) and the Figure S1 table (60 values) agree |
| Release check | `check_manuscript_mapping.py` / `--release` | a changed printed value and a stale erratum each make it fail, as designed |
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

These runs used the bundle layout of that date; the tables of the current bundle are in the
first section.

| Test | Command | Result |
|---|---|---|
| Environment, identity | `./reproduce.sh check --artifacts <local bundle>` | Python 17/17, Julia 148/148 and R 39/39 locked packages match; forecast 8.23.0; image content and checkout fingerprints match; committed files intact |
| Smoke (Level A) | `./reproduce.sh smoke` | PASS: 61 of 61 compared quantities (Julia/Ipopt fits, non-default objectives, admissibility, SCIP global solve with dual bound, R `forecast`, Python `statsmodels`), 2 min 50 s |
| An unknown tarball | `... fetch artifacts --from not-a-bundle.tar.gz` | refused (unknown name or SHA-256), exit 4 |
| A corrupted input | one byte appended to `per_series/mse__weekly.csv.gz`, then `tables --group m4_objectives_controlled` | "checksum mismatch", every cell MISSING, exit 4; file restored and re-verified |
| Default run without the bundle | `./reproduce.sh --offline --artifacts <empty dir>` | check and tables MISSING INPUT, smoke PASS, final verdict FAILURE, exit 4 |
| `list` | `./reproduce.sh list` | every experiment with both statuses; `full --all` targets as listed above |
| M4 acquisition | `./reproduce.sh fetch m4 --frequency weekly --frequency quarterly --frequency yearly --frequency monthly` | 8 files from the pinned commit, SHA-256 verified (the campaign's local copies are the same files up to line endings) |
| **Full run, controlled campaign, all weekly series** | `./reproduce.sh full --experiment m4_objectives --frequency weekly --jobs 10 --artifacts <local bundle>` | 5 × 359 series OK in 20 min. **Every objective reproduces the archived campaign on 359/359 series**: same selected order, total sMAPE within 1.1e-11. Forecasts are bit-identical for 230 (MSE), 222 (MAE), 230 (Huber) and 286 (Ridge) series; elsewhere they differ by at most 6.4e-13 relative. OWA with the run's own Naive2: MSE 0.9002, MAE 1.2735, Huber 1.0840, Ridge 0.9436 (the archived Table 5 row) |
| Full run, samples of the other frequencies, with `--resume` | `full --experiment m4_objectives --frequency quarterly --frequency yearly --sample 25`, then the same with `--sample 40 --resume`; `--frequency monthly --sample 12` | all OK. The resumed run fitted only the 15 new series of each of its 10 tasks (25 already done). **Every objective reproduces the archive: quarterly 40/40, yearly 40/40, monthly 12/12** (same order, same total sMAPE) |
| `full --all`, one series per task | `./reproduce.sh full --all --limit 1 --jobs 4` (fetches the M4 daily/hourly files and PJME first) | overall PASS in 73 min. It ran `battery_validation`, `battery_solver_comparison` and `battery_global_value` (every Julia, R and Python step; the three SCIP global solves took about 40 min). Their rebuilt tables against the historical ones: 177/224, 58/70 and 27/42 cells agree. That includes timing cells and the definition changes from v0.3.0 to the release listed in `experiments/runners/battery/PORTING.md`; reported as WARN, as designed. It then ran `m4_benchmark` (6 frequencies × Sarimax.jl, auto.arima and Naive2: every series reproduces the archive, the R baselines included) and `m4_objectives` (4 frequencies × 4 objectives: every series reproduces the archive) |
| Worker start-up | events of the runs above | the 10 workers of a task were ready 1 min 50 s after it started (each loads the package for about a minute); before this change they started one after another, 11 min per task |

### Native run on macOS arm64 (2026-10-18; this pull request)

| Test | How | Result |
|---|---|---|
| Campaign A and F files rebuilt from git | the builder's metric code, Python 3.14 | byte-identical after making the per-block sum independent of the Python version (before, the eight files with removed test values differed in last bits) |
| Static checks | `check_package.py`, `check_manuscript_mapping.py`, shellcheck 0.11 | consistent; shellcheck clean |
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
| Fifth run (`a873ab9`: rebased Julia environment, final `m4_objectives`, parallel worker start-up): every step, including the controlled objective campaign on 3 weekly series per arm compared with its archive | PASS (job 13 min) | PASS (job 15 min) |
| Package CI (`ci.yml`) at `a873ab9` | Julia 1 and 1.10 on ubuntu, Julia 1 on windows and macos: PASS (the windows job had failed at `f415b60`, a commit that changes no package file; it passed at `a873ab9`, so that failure is not reproducible; its log needs a GitHub sign-in) | – |

### Native run on Windows (2026-10-17)

| Test | How | Result |
|---|---|---|
| M4 acquisition | `fetch m4 --frequency yearly` from the pinned commit | verified SHA-256 |
| Smoke, Julia (Ipopt fits, non-default objectives, admissibility constraints, forecast evaluation) | natively, Julia 1.10.12 + pinned Manifest | all quantities PASS |
| Smoke, SCIP | natively on Windows | SCIP 8.0.0 crashes on Windows (known upstream); runs in the Linux container |
| Full runner: subset, checkpoint/resume, deadlines | `m4_benchmark --frequency yearly --arm sarimax --limit 4/6` and `--timeout 0.01` | 6/6 series reproduce the archive; resume fitted only the new series; TIMEOUT rows recorded, workers replaced, exit 1 |
| Full runner: interrupted checkpoint, changed source | last row of a task file cut in half, then `--resume`; resume after editing a fingerprinted file | the cut row was dropped (original kept as `.bak`) and refitted, 4/4 reproduce the archive; resuming after a source change was refused |
| Battery re-runs (Level C) | every Julia (Ipopt) and Python step of the battery configurations | all steps exit 0; rebuilt cells: validation 100/119, solver comparison 27/30, global value 18/18 (differences in experiments/runners/battery/PORTING.md); R steps and SCIP solves not run there |

Not applicable before the release: the published image pulled by digest (the publish workflow
does it), the Zenodo records.

## Missing artifacts

The final results and frozen protocol of the controlled `m4_objectives` experiment and the raw
files of the September 2026 stable and admissibility campaigns are in the bundle. What remains:

1. **Individual records behind derived-only printed values** (`reports/provenance_gaps.md`):
   every printed value has an archived source, but 73 agree only with records received after
   the original outputs were lost: reconstructions of the August 2026 certification batch, of
   Table S2 and of the Section 5.5 diagnostics, the August aggregate of the 1,500-series
   comparison, and the per-seed table of the ten-seed study of Table S8
   (`reports/certification_batch_2026_08.md`). The
   certification batch and the 1,500-series initialization comparison were re-executed as
   identified runs (`results/reruns/`). Regenerated runs of Table S2, the ten-seed study and the
   AICc pairing are not deposited, and the text cites only the archived records
   (`results/manuscript/textual_claims.csv`, C18-C20).
2. **The 2x2x2x2 grid's Manifest and the `fator=1` patch** behind its normalization-off cells
   (lost), needed to re-execute those arms.
3. **SCIP evidence:** the tolerances and SCIP logs of the archived SCIP runs, and the log of the
   OPTIMAL re-run of the PJME global-value instance (not recoverable). Dual bounds and logs now
   exist for a new identified re-run of the Table 7 and MA(1) certificate experiments
   (`results/reruns/full-all-limit1-2026-10-20/`).
4. **The R/forecast version** of the July 2026 auto.arima/Naive2 baselines, which was not
   recorded (inferred from logs: R 4.4.1, forecast 8.23.0).
5. **Licences:** confirmation for M4 and PJME, which are fetched, not redistributed. SCIP 8.0.0
   (ZIB Academic License) is fetched on the user's machine, not shipped in the image
   (`reports/scip_licence.md`).
6. **Publication identifiers:** the published bundle (Zenodo) and the DOIs that go into
   `artifacts.lock`, `CITATION.cff` and the availability statement. The image is published and
   its digest is in `image.lock`.
