# Reproducing "SARIMAX.jl: Open-Source Optimization-Based SARIMAX Modeling in Julia"

Replication package of the article submitted to the *International Journal of Forecasting*
(special issue *Advances in Open Source Forecasting Software*). It lives in the Sarimax.jl
repository itself and runs the package source of the same checkout.

## Quickstart

You need `git`, `bash` and [Docker](https://docs.docker.com/get-docker/) (Docker Desktop on
macOS and Windows). Nothing else: Julia, R, Python and the solvers are inside the container.

```bash
git clone https://github.com/LAMPSPUC/Sarimax.jl.git
cd Sarimax.jl
./reproduce.sh
```

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
> the checkout (20 to 40 minutes, once). Pass the artifact bundle explicitly with
> `./reproduce.sh fetch artifacts --from sarimax-ijf-2026-artifacts-v1.tar.gz`, or point to
> an extracted copy with `--artifacts DIR`. Without the bundle, `check` and `tables` report
> it as MISSING and the run fails, as it should.

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
./reproduce.sh fetch artifacts|m4|pjme   # download and verify inputs
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
| Disk | about 8 GB (image about 6 GB, artifacts 0.6 GB) | plus a few GB of run outputs and 0.3 GB of M4 data |
| Memory | 4 GB available to Docker | about 2 GB per worker |
| CPU | any | as many cores as `--jobs` |
| Network | to pull the image and fetch the artifacts once | to fetch M4 (and PJME) once |

The reference platform is `linux/amd64`. On Apple Silicon, Docker runs the amd64 image under
emulation, which is slow and has not been tested; a native `linux/arm64` image
(`--platform linux/arm64 --build`) is built and exercised in CI as an experimental platform
(results in `REPRODUCIBILITY_STATUS.md`).

Computations run without network access (`--network none`); only `fetch` downloads, and
every download is pinned to a version (a commit or a DOI record) and checked against its
SHA-256.

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

## Status

What is verified, what is only executable and what is missing is listed per experiment in
[`REPRODUCIBILITY_STATUS.md`](REPRODUCIBILITY_STATUS.md); known caveats are in
[`LIMITATIONS.md`](LIMITATIONS.md). In short: every archived table regenerates from
per-series data; the experiments' runners are re-executable on the release code; most of the
numbers the manuscript prints are regenerated from archived per-series results, while some
agree only with derived tables or have no archived source, and two printed timings differ from
the archive (all listed in LIMITATIONS.md, item 1); and the controlled comparison of
estimation objectives (`m4_objectives`) is still running, so its results are not part of this
package yet.

## Software and archived artifacts

* Source code: https://github.com/LAMPSPUC/Sarimax.jl (this repository; the paper's results
  use the package exactly as released in v1.0.0, commit `14dd581`).
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
