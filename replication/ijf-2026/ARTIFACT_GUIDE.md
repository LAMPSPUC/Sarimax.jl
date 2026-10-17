# Artifact guide (technical)

Reviewers only need [README.md](README.md). This guide documents how the package works, how
its inputs were assembled, how to extend it and how it is released.

## 1. Layout

```
reproduce.sh                       host entry point (bash + Docker)
replication/ijf-2026/
  README.md, ARTIFACT_GUIDE.md, REPRODUCIBILITY.md, LIMITATIONS.md, REPRODUCIBILITY_STATUS.md
  CODE_AND_DATA_AVAILABILITY.txt   proposed statement for the manuscript
  experiment_manifest.csv          one row per experiment family: protocol, revision, status
  image.lock                       published image: repository, tag, digest (set at release)
  artifacts.lock                   artifact bundle: file, SHA-256, DOI/URL (set at release)
  artifacts.sha256                 SHA-256 of every file inside the artifact bundle
  checksums.sha256                 SHA-256 of every committed file a computation reads
  compose.yaml                     Docker Compose equivalent of reproduce.sh
  docker/Dockerfile                the single replication image
  environments/                    julia/{Project,Manifest}.toml, r/renv.lock, python/requirements.lock
  experiments/configs/             protocol of each re-runnable experiment (TOML)
  experiments/runners/             M4 workers (Julia, R) and the ported battery runners
  experiments/validation/          smoke tests and their expected values
  scripts/ijf.py, scripts/ijflib/  orchestrator (check, smoke, tables, full, fetch, list)
  scripts/ijftables/               table generators (one module per table group)
  scripts/checks/                  environment checks run inside the image
  scripts/maintainers/             bundle builder, expected-value extraction, lock generators
  scripts/fingerprint.sh           source fingerprint (host and image)
  data/external_files.csv          pinned locations and SHA-256 of M4 and PJME
  data/fixtures/                   small committed inputs: smoke data, a 0.7 MB subset of the bundle
  results/expected/                reference cell values, parsed from the historical tables
  artifacts/   (not committed)     the extracted artifact bundle
  data/external/ (not committed)   fetched M4 and PJME files
  output/      (not committed)     everything a reproduction writes
```

## 2. How a command runs

`reproduce.sh` (host) checks Docker, resolves the image (pull by the digest in `image.lock`;
before the release, a local build of the checkout), compares the image's source fingerprint
label with the checkout's, and runs `scripts/ijf.py <command>` in the container with:

| Host | Container | Mode |
|---|---|---|
| `replication/ijf-2026/output` (`--output`) | `/ijf/output` | read-write |
| `replication/ijf-2026/artifacts` (`--artifacts`) | `/ijf/artifacts` | read-only (read-write for `fetch`) |
| `replication/ijf-2026/data/external` (`--data-dir`) | `/ijf/data` | read-only (read-write for `fetch`) |

Computations run with `--network none`; only `fetch` has network access. On Linux the
container runs as the calling user, so outputs are owned by you. The code that runs is the
image's copy of the checkout (`/opt/sarimax`), never a download.

`--native` runs `ijf.py` with the host's toolchains instead (developers and CI only; every
report is labelled NATIVE).

## 3. The image

| Component | Version | Pinned by |
|---|---|---|
| Base | rocker/r-ver 4.4.1 on Ubuntu 22.04 | image index digest `sha256:78cb94ce…1312` |
| R | 4.4.1; CRAN frozen at the Posit Package Manager snapshot of 2024-10-30 | base image; packages checked against `renv.lock` at build time |
| forecast | 8.23.0 (the version of the archived auto.arima baselines) | snapshot + `renv.lock` |
| Julia | 1.10.12 | official tarball SHA-256 |
| Julia packages | Sarimax.jl from the checkout; JuMP 1.31.1, MathOptInterface 1.52.0, Ipopt.jl 1.15.0 (Ipopt 3.14, MUMPS 5.9.1), SCIP.jl 0.11.6 (SCIP 8.0.0), HiGHS 1.26.0, … | `environments/julia/Manifest.toml` (170 packages) |
| Python | CPython 3.12.14 (python-build-standalone 20260901) | release tarball SHA-256 |
| Python packages | numpy 2.4.6, pandas 2.3.3, scipy 1.16.3, statsmodels 0.14.6, matplotlib 3.11.2 + dependencies | wheel hashes in `requirements.lock` |

The Julia versions of JuMP, MathOptInterface and Ipopt.jl equal those recorded by the
August 2026 ICARAI campaigns. The build verifies the R library against `renv.lock`, the
Julia environment against its Manifest, and the copied source against the fingerprint
passed as a build argument; any mismatch fails the build.

**Identity.** `scripts/fingerprint.sh` hashes (carriage returns stripped) `Project.toml`,
`src/`, and `docker/`, `environments/`, `experiments/`, `scripts/`, `data/` (without
`data/external/`), `results/`, `experiment_manifest.csv`, `artifacts.sha256` and
`checksums.sha256` of this directory, excluding Markdown. The image stores it as the label
`io.github.lampspuc.sarimax.ijf.fingerprint` and as `IJF_IMAGE_FINGERPRINT`; the commit it
was built from is the label `org.opencontainers.image.revision`. `image.lock` and
`artifacts.lock` are outside the fingerprint, so recording a published digest or DOI does
not invalidate the image.

**Licences.** `/opt/licenses` in the image collects the licence files of every binary
dependency, and the licence fields of the R and Python packages. Notable:

| Software | Licence |
|---|---|
| Sarimax.jl | MIT |
| Julia, most Julia packages | MIT; JuMP and MathOptInterface: MPL-2.0 |
| Ipopt | EPL-2.0; MUMPS: CeCILL-C; METIS: Apache-2.0; OpenBLAS: BSD-3-Clause |
| HiGHS | MIT |
| **SCIP 8.0.0 and SoPlex (bundled by SCIP_jll 0.2.1, used by SCIP.jl 0.11)** | **ZIB Academic License** (academic, non-commercial use); PaPILO: GPL-3.0 |
| R, forecast | GPL-2/GPL-3 |
| Python, numpy, scipy, pandas, statsmodels, matplotlib | PSF, BSD-3-Clause and compatible |

The ZIB Academic License of SCIP 8.0.0 is not an open-source licence. SCIP is Apache-2.0
from version 8.0.3 on, but Sarimax.jl 1.0.0 is compatible only with SCIP.jl 0.11
(`SCIP = "^0.11.6"`), which bundles 8.0.0. This affects how the paper can describe SCIP and
whether and how the image may be redistributed; see LIMITATIONS.md and the release
checklist below.

## 4. The artifact bundle

Built by `scripts/maintainers/build_artifact_bundle.py` from the repository history, the
three `paper/replication-staging-*` branches, the ForecastTester experiment workspace and the
`sarimax-replication` folder (paths are arguments). The output is deterministic (gzip and
tar without timestamps or owners): `sarimax-ijf-2026-artifacts-v1.tar.gz` (296 MB), with
`artifacts.sha256` (535 files) and `bundle.json`.

| Directory | Content |
|---|---|
| `raw/battery-2026-07/` | July 2026 battery: validation, forecasting, architecture, solver and energy records (jsonl), logs, simulated inputs; PJME data omitted |
| `raw/m4-objectives-icarai-2026-08/` | 37 objective campaigns (cel_*, obj_*, req_*, censo_*, smoke_*), auto.arima/Naive2 baselines for M/Q/Y/W, campaign index, validity and probe files |
| `raw/m4-objectives-rerun-a1ebc0d-2026-08/` | re-run of the huber and ridge cells at a1ebc0d, with logs |
| `raw/m4-recreio-2026-08/` | recreio campaigns A–F (`actual` redacted, per-block metrics added) and verification files |
| `raw/m4-baselines-r-2026-07/` | auto.arima and Naive2 per-series metrics, all six M4 frequencies, four horizon blocks |
| `raw/robust-cvar-monthly-2026-08/`, `raw/admissibility-2026-08/`, `raw/initialization-order-search-2026-08/`, `raw/scip-ipopt-yearly-2026-07/`, `raw/stable-admissibility-gavea-2026-09/` | the smaller studies (see the manifest) |
| `derived/m4-objective-distributions-2026-10/` | the per-series sets of the October 2026 Table 7 evidence |
| `historical-tables/` | the tables as generated at the time (source of `results/expected/`) |
| `original-scripts/` | verbatim runners, table scripts, environment files and notes of each campaign |
| `PROVENANCE.csv` | for each file: origin (git object or workspace file), original SHA-256, transform |

Only one transform changes content: in the recreio files, the M4 test values (`actual`) were
removed and per-block sMAPE/MASE computed from them were added (definitions of recreio
`make_table_m4.jl`). The `m4_benchmark` table group then regenerates recreio's table from
the redacted files to the last printed digit, which also validates the transform.

`./reproduce.sh fetch artifacts` downloads the tarball from the location in
`artifacts.lock`, checks its SHA-256, extracts it safely and verifies every file against
`artifacts.sha256`; `--from FILE` uses a local tarball.

## 5. Third-party data

`data/external_files.csv` pins each file to a URL at a fixed commit and its SHA-256 over LF
content (`fetch` verifies with carriage returns stripped, so CRLF copies of the same content
are accepted):

* M4: the twelve train/test files of Mcompetitions/M4-methods at commit
  `b8f01984fde0b562aa6f1a9181a242ce076ba73e` (the files the experiments used are byte-identical
  to these, up to line endings; also recorded as git blob SHA-1);
* PJME: `PJME_hourly.csv` from this repository at commit `308461e` (SHA-256 `4eb2b16d…`).

## 6. Level B internals

A table group is a module in `scripts/ijftables/` with `TITLE`, `INPUTS` (bundle paths) and
`generate(ctx) -> list[Cell]`; a `Cell` is (table, row, column, value, printed decimals).
Its reference values are `results/expected/<group>.csv`, produced by
`scripts/maintainers/extract_expected.py`, which PARSES the historical tables with one
module per group in `scripts/maintainers/expected/`. To add a group: write both modules,
run the extractor, add the group to `GROUPS` in `scripts/ijflib/tables.py`, list the group
in the manifest, and regenerate `checksums.sha256`
(`scripts/maintainers/update_checksums.sh`).

| Group | Rebuilds | Cells |
|---|---|---|
| battery | the seven July 2026 battery tables (`tab:*`) | 583 |
| m4_benchmark | recreio `table_m4` (M4 vs auto.arima, six frequencies x four blocks) | 192 |
| m4_objectives | ICARAI Tables 1–2 and `tabelas_paper` sections A and B | 880 |
| m4_objective_distributions | October 2026 Table 7 evidence: reproduction, 4a–4d, figures | 2,872 |
| arms | recreio objective, multistart, isolation, axes and stable tables | 299 |
| robust_cvar | contamination study (reference values transcribed from internal notes) | 19 |
| scip_ipopt | SCIP vs Ipopt summaries of the July 2026 yearly study | 23 |

## 7. Level C internals

**Configurations.** `kind = "m4"`: `frequencies`, `per_series_timeout_s`, `max_failure_rate`,
`[auto_defaults]` (every keyword passed to `Sarimax.auto`, written out; strings `"nothing"`
mean `nothing`; symbol-valued keys are converted), and `[[arms]]` with `engine = "julia"`
(optional `auto` overrides) or `engine = "r"` with `method = "auto_arima"` or `"naive2"`;
optional `[archived.<arm>]` tables point at archived per-series files for the
series-by-series comparison. `kind = "script"`: `[[steps]]` (engine, script, args,
timeout_s), `tables_group` and `tables`; the steps write records into the run directory with
the bundle's layout, and the group's generator then rebuilds the tables from the new records.

**Worker protocol.** One JSON object per line: the orchestrator sends
`{"series", "y", "s", "H"}`; the worker answers `{"series", "status", "forecast", "p", "d",
"q", "P", "D", "Q", "has_mean", "has_drift", "solver_status", "fallback", "fit_seconds"}`
after printing `{"ready": true, ...}` once. The orchestrator computes the metrics (one
implementation for every engine, `scripts/ijflib/m4data.py`).

**Outputs** (`output/runs/<experiment>/<run id>/`): `<arm>__<frequency>.csv` with columns
`experiment, arm, frequency, series, T, status, p, d, q, P, D, Q, has_mean, has_drift,
solver_status, fallback, fit_seconds, wall_seconds, smape_<block>, mase_<block>,
mase_den, forecast, run_id, config_sha256, source_fingerprint, engine`; `run.json`
(configuration and source fingerprints, image, arms, limits, per-task counts);
`events.jsonl` (task start/end, worker start, timeout, crash); `logs/`; `summary.txt`.

**Robustness.** Each finished series is appended and flushed immediately (checkpoint).
`--resume` skips series already present and refuses to continue a run whose configuration or
source fingerprint changed. A series that exceeds the deadline is recorded as TIMEOUT and its
worker killed and replaced; an exception is recorded as `ERROR:<message>`; a dead worker as
`CRASH`. A task with more than `max_failure_rate` failures is reported FAIL; failures are
never retried or dropped silently.

## 8. Integration contract for `m4_objectives`

The controlled comparison of objectives is produced by a separate workstream. To integrate
its final results:

1. **Protocol.** Replace `experiments/configs/m4_objectives.toml` by the frozen protocol in
   the format of section 7: every keyword of `auto` for every arm (including how ridge is
   expressed: the deprecated `"ridge"` objective or `mse` + `penalty = :elastic_net`,
   `alpha = 0`, `penaltyTarget = :dynamics` and its lambda), the frequencies, the time limits
   and the reference arms (`naive2`; `auto_arima` if reported). Set
   `protocol_status = "final"`.
2. **Code.** Run with the release commit of the paper (the image of this package), so that
   `source_fingerprint` and `config_sha256` in every row match the release.
3. **Outputs.** Per-series files in the schema above, one per arm and frequency
   (`<arm>__<frequency>.csv`), all series of each frequency, with their `run.json` and
   `events.jsonl`. Runs produced with `./reproduce.sh full --experiment m4_objectives` have
   this form by construction.
4. **Archive.** Add them to a new version of the artifact bundle under
   `raw/m4-objectives-controlled-<date>/` (rebuild with `build_artifact_bundle.py`, update
   `artifacts.sha256` and `artifacts.lock`), add `[archived.<arm>]` entries to the
   configuration, write the table group (`scripts/ijftables/m4_objectives_controlled.py`) and
   the expected values parsed from the table as printed in the manuscript, and update the
   manifest row (`table_status`, `expected_values`, `archived_results`).
5. **Never mix.** The historical campaigns stay under their own ids
   (`m4_objectives_icarai_*`); no table may combine them with the controlled experiment.

## 9. Release checklist (maintainers; every step needs the authors' approval)

1. Confirm the manuscript mapping: fill `manuscript_label` for every retained experiment,
   compare the manuscript's printed values with `results/expected/`, and drop or mark the
   experiments the paper does not use.
2. Integrate `m4_objectives` (section 8), or record it as pending in the paper.
3. Decide on SCIP's licence (section 3): keep SCIP.jl 0.11 and state the ZIB Academic
   License, or move the package to a SCIP release under Apache-2.0 before the release.
4. Merge the replication branch into `master` after review.
5. Build the final artifact bundle; upload it as a new Zenodo data record (test in the
   Zenodo Sandbox first with `scripts/maintainers/zenodo_deposit.py --sandbox`); record its
   version DOI, URL and SHA-256 in `artifacts.lock` (this does not change the fingerprint).
6. Build and push the image with the manually triggered workflow `ijf-image-publish.yml`;
   record the digest, fingerprint and revision in `image.lock`; archive the exported image
   (`docker save | gzip`) in the Zenodo data record if its size and the SCIP licence allow.
7. Tag the release commit (for example `ijf-2026-v1`; the package version stays 1.0.0
   because `src/` is unchanged) and publish the GitHub release; the Zenodo GitHub
   integration archives it and mints the software DOI (`.zenodo.json`).
8. Link the records: software → `isSupplementedBy` data record; both → `isSupplementTo` the
   article DOI when assigned; update `CITATION.cff` and the availability statement.

## 10. Continuous integration

`.github/workflows/ijf-replication.yml` (pull requests that touch the package or the
replication files) builds the image from the checkout and runs: host-side checks
(shellcheck, usage and missing-Docker exit codes, `checksums.sha256`, manifest
consistency); `check` (expects the absent bundle to be reported with exit code 4); `smoke`;
`tables` on the committed fixture subset of the bundle; a missing and a corrupted input
(exit 4); an image built from different code (exit 3); `fetch m4` for the weekly files; and
a two-series run of `m4_benchmark` on weekly data with every arm, then `--resume` to three
series. Reports are uploaded as workflow artifacts. Full campaigns never run in CI.
`ijf-image-publish.yml` builds and pushes the image to GHCR; it runs only when started by
hand.
