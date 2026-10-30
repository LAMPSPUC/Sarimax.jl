# Reproducing "SARIMAX.jl: Open-Source Optimization-Based SARIMAX Modeling in Julia"

Replication package of the article submitted to the *International Journal of Forecasting*
(special issue *Advances in Open Source Forecasting Software*). It lives in the Sarimax.jl
repository itself and runs the package source of the same checkout.

## Quickstart

You need `git`, `bash` and [Docker](https://docs.docker.com/get-docker/) (Docker Desktop on
macOS and Windows). Nothing else: Julia, R, Python and Ipopt are inside the container. The
global solver SCIP 8.0.0 is under the ZIB Academic License (free for academic, non-commercial
use), which does not let us ship it in a public image. Sarimax.jl loads SCIP, so the first run
downloads it through Julia's package manager, verified by its content hash, into the Docker
volume `sarimax-ijf-scip` (about 90 MB, once).

```bash
git clone --branch ijf-2026-replication https://github.com/LAMPSPUC/Sarimax.jl.git
cd Sarimax.jl
./reproduce.sh
```

The replication package is on the branch `ijf-2026-replication` of the repository (and, once
published, on its release tag `ijf-2026-replication-v1.0.0`); the default branch holds the
library only.

`./reproduce.sh` runs three stages and ends with a one-line verdict:

1. **check**: verifies the environment inside the container, that the image was built from
   this exact checkout, and every input against its SHA-256;
2. **smoke**: real fits in Julia (Ipopt, and a SCIP global solve with its dual bound), R
   (`forecast::Arima`) and Python (`statsmodels`), compared with archived and analytic values;
3. **tables**: rebuilds the tables from the archived per-series results, compares every
   cell with the historical value at the precision it was printed, and then compares every
   number printed in the manuscript with the regenerated value it comes from.

The report is in `replication/ijf-2026/output/reports/summary.md` (one Markdown, CSV and
JSON report per stage next to it). The command exits with 0 only if everything passed.

On Windows, run the commands in Git Bash or WSL.

> **Before the release.** The container image and the archived artifacts are published at
> release time (see [Status](#status)). Until then `./reproduce.sh` builds the image from
> the checkout (10 to 40 minutes, once). Pass the artifact bundle (one tarball of the Zenodo
> record) explicitly with
> `./reproduce.sh fetch artifacts --from sarimax-ijf-2026-artifacts-v1.tar.gz`,
> or point to an extracted copy with `--artifacts DIR`. Without the bundle, `check`
> and `tables` report it as MISSING and the run fails, as it should.

## What this reproduces

| Level | Command | What it does | Time |
|---|---|---|---|
| A | `./reproduce.sh smoke` | Real computations on small problems with every engine and solver | 5–10 min |
| B | `./reproduce.sh tables` | Every archived table rebuilt from per-series results, cell by cell, and the manuscript's printed numbers checked against them | 2–5 min |
| C | `./reproduce.sh full --experiment ID` | Re-runs an experiment from the original input data | minutes to hundreds of CPU-hours |

The experiments and their status are listed by `./reproduce.sh list` and documented in
[`experiment_manifest.csv`](experiment_manifest.csv) (protocol, revision, environment,
archived files, readiness) and [`REPRODUCIBILITY_STATUS.md`](REPRODUCIBILITY_STATUS.md).

## Commands

```bash
./reproduce.sh                       # check + smoke + tables
./reproduce.sh check                 # environment, image identity, inputs, checksums
./reproduce.sh smoke                 # Level A
./reproduce.sh tables                # Level B
./reproduce.sh list                  # experiments and their readiness
./reproduce.sh full --experiment m4_benchmark --jobs 4 --resume
./reproduce.sh full --all --jobs 4 --resume   # every re-runnable experiment the manuscript reports
./reproduce.sh fetch artifacts|m4|pjme|scip   # download and verify inputs
./reproduce.sh build                 # build the image from this checkout
```

Useful options: `--jobs N` (worker processes), `--resume` (continue an interrupted run from
its checkpoints), `--sample N` (N series drawn at random per frequency, the same for every arm;
a larger N extends a smaller one, so `--resume` can grow it) and `--limit N` (the first N series
per task) — both check the execution path on a few series and are not results, `--frequency F` and `--arm A` (part of an M4 experiment), `--timeout S`
(per-series limit), `--offline`, `--artifacts DIR`, `--data-dir DIR`, `--output DIR`,
`--build`. `./reproduce.sh help` lists them all.

Exit codes: 0 success; 1 a check, test, comparison or run failed; 2 usage error;
3 environment problem (Docker missing or unreachable, image unavailable or not built from
this checkout); 4 an input is missing or corrupted.

## Requirements

| | Quick reproduction (A + B) | Full re-execution (C) |
|---|---|---|
| Disk | about 8 GB (image 4.3–6 GB, artifact bundle 0.6 GB compressed and about 0.7 GB extracted) | plus a few GB of run outputs and 0.3 GB of M4 data |
| Memory | 4 GB available to Docker | about 2 GB per worker |
| CPU | any | as many cores as `--jobs` |
| Network | to pull the image and fetch the artifacts and SCIP once | to fetch M4 (and PJME) once |

The reference platform is `linux/amd64`. On Apple Silicon, Docker runs the amd64 image under
emulation, which is slow and has not been tested; a native `linux/arm64` image
(`--platform linux/arm64 --build`) is built and exercised in CI as an experimental platform
(results in `REPRODUCIBILITY_STATUS.md`).

Computations run without network access (`--network none`); only `fetch` downloads, and
every download is pinned to a version (a commit, a DOI record, or the content hash of a Julia
artifact) and verified. With `--offline` nothing is downloaded: run `./reproduce.sh fetch scip`
once beforehand, or SCIP is reported as a missing input.

## Where results go

```
replication/ijf-2026/output/
  reports/   summary.md, check.*, smoke.*, tables.*, tables_cells.csv, full.*
  tables/    rebuilt tables, one directory per table group
  figures/   rebuilt figures
  smoke/     logs and measured values of the smoke tests
  runs/      new executions: runs/<experiment>/<run id>/ with per-series CSVs,
             run.json, events.jsonl, logs/ and summary.txt
```

Archived results (the artifact bundle, read-only) and new executions are never mixed: a run
writes only to its own run directory and compares itself with the archive afterwards.

## How to read the outcome

* `reports/summary.md` gives the verdict of each stage.
* `reports/tables.md` lists each table group with the number of cells that agree, then each
  manuscript table, figure and section with the number of printed values that agree with
  their regenerated source; the cell-level comparisons are in `tables_cells.csv` and
  `manuscript_cells.csv`. A printed value that can only be checked against a derived table,
  or that has no archived source, is reported as such (WARN) and never counted as reproduced.
* `reports/smoke.md` lists each measured quantity with its reference, tolerance and source.
* A full run's report (`reports/full.md`) states how many series completed, how many ended
  in ERROR, TIMEOUT or CRASH (kept as rows, never dropped), and how many reproduce the
  archived per-series results.

## The paper's results

Every number printed in the submitted manuscript and its Supplementary Material (836 values) is
mapped to the archived record it comes from in
[`results/manuscript/manuscript_values.csv`](results/manuscript/manuscript_values.csv); the
mapping is bound to the submitted files by SHA-256
([`results/manuscript/REVISION.toml`](results/manuscript/REVISION.toml)). 660 values are
regenerated by `./reproduce.sh tables` from archived per-series or per-instance records; 174
agree with archived records that the table groups do not rebuild (the per-instance records of
identified re-runs, recomputed by the package's checks; recovered script outputs; summary
records of runs whose individual outputs are lost), and are not counted as reproduced; none
differs. The verification and what it does not
cover are in [`verification/VERIFICATION_REPORT.md`](verification/VERIFICATION_REPORT.md).

| Paper | Experiment (`./reproduce.sh list`) | Evidence |
|---|---|---|
| Table 2 | `battery_validation` | regenerated from records; re-runnable |
| Tables 3, 4, S12; Figure 2; Section 5.3 | `m4_benchmark` | regenerated from per-series records; re-runnable |
| Tables 5, 6, S13, S14; Figure 3; Section 5.4 | `m4_objectives` (controlled campaign) | regenerated from per-series records; re-runnable; re-run checked series by series |
| Table S4 | `m4_benchmark`, `m4_objectives` | regenerated, including the weekly re-execution under the six-frequency protocol |
| Tables S15, S16; Section 5.5 (admissibility grid) | `stable_admissibility`, `stable_weekly_yearly` | regenerated from per-series records |
| Table 7; Section 5.6 (MA(1) certificates) | `battery_global_value`, `battery_solver_comparison` | regenerated; logged re-run with dual bounds |
| Tables S5, S6, S7; Section 5.6 | `global_optimization_grid` | August summary record; identified re-run with per-instance records and SCIP logs |
| Tables S8, S9; Section 5.5 (initialization) | `initialization_experiment` | ten-seed summary record; 1,500-series comparison re-run with per-series records |
| Table S2; Table S9 and Section 5.5 (ranking diagnostics) | `exog_convention_simulation`, `ranking_likelihood_diagnostic` | summary records only |
| Tables S10, S11; Figure S1 | `quantile_demo`, `multistart_kstart`, `weighted_regularization_demo` | recovered outputs of the scripts |

Which source revision, environment and runner each experiment used is in
[`reports/experiment_provenance.md`](reports/experiment_provenance.md) and
[`experiment_manifest.csv`](experiment_manifest.csv). Per-experiment status:
[`REPRODUCIBILITY_STATUS.md`](REPRODUCIBILITY_STATUS.md); caveats:
[`LIMITATIONS.md`](LIMITATIONS.md).

## Software and archived artifacts

* Source code: https://github.com/LAMPSPUC/Sarimax.jl (this repository). The controlled
  objective comparison ran on the release v1.0.0 (commit `14dd581`); the other experiments ran
  on the development revisions recorded per experiment, which are not interchangeable with
  the release.
* Container image: `ghcr.io/lampspuc/sarimax-ijf-replication` (immutable digest recorded in
  [`image.lock`](image.lock) once published).
* Archived experimental artifacts: a Zenodo record (version DOI recorded in
  [`artifacts.lock`](artifacts.lock) once published); its content is pinned file by file in
  [`artifacts.sha256`](artifacts.sha256).
* Software release: a Zenodo record of the tagged GitHub release (DOI added at release).

Technical details: [`ARTIFACT_GUIDE.md`](ARTIFACT_GUIDE.md). What "reproduced" means here:
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Citation

Please cite the article and the archived software release; to point at the exact results,
cite the version DOIs of the software and artifact records (see `CITATION.cff` at the
repository root, which GitHub renders as "Cite this repository").
