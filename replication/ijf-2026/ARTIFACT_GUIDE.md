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
  artifacts.lock                   artifact bundle parts: file, SHA-256, DOI/URL (set at release)
  artifacts.sha256                 SHA-256 of every file inside bundle part 1
  artifacts-part2.sha256           SHA-256 of every file inside bundle part 2
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
  data/fixtures/                   small committed inputs: smoke data, a 1 MB subset of the bundle
  results/expected/                reference cell values, parsed from the historical tables
  results/manuscript/              every number printed in the manuscript, with its source;
                                   errata.csv, textual_claims.csv, REVISION.toml, pending/
  results/historical/              derived tables not in the bundle (with PROVENANCE.md), the
                                   ecc4878 solver records, the recovered 2026-09 scripts/outputs
  results/reruns/                  identified re-runs kept as evidence (with RUN.md)
  reports/                         audit reports (index in reports/README.md)
  FINAL_SCIENTIFIC_AUDIT.md        summary of the final scientific audit
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
| Julia packages | Sarimax.jl from the checkout; JuMP 1.31.1, MathOptInterface 1.52.0, Ipopt.jl 1.15.0 (Ipopt 3.14.19, MUMPS 5.9.0), SCIP.jl 0.11.6 (SCIP 8.0.0), HiGHS 1.26.0, … | `environments/julia/Manifest.toml` (148 versioned packages) |
| Python | CPython 3.12.14 (python-build-standalone 20260901) | release tarball SHA-256 |
| Python packages | numpy 2.4.6, pandas 2.3.3, scipy 1.16.3, statsmodels 0.14.6, matplotlib 3.11.2 + dependencies | wheel hashes in `requirements.lock` |

The Julia environment is the one the August 2026 objective campaigns and the October
2026 controlled objective campaign ran with (their Manifests are identical package by
package), plus HiGHS, HiGHS_jll and MathOptIIS (used by one battery runner) and the
Sarimax.jl checkout; adding HiGHS_jll moved CompilerSupportLibraries_jll from 1.1.1 to 1.1.2,
the only version that differs. A re-run of those campaigns in the image therefore uses the
same JuMP, MathOptInterface, Ipopt and MUMPS builds as the archived runs (on Linux instead
of Windows). The build verifies the R library against `renv.lock`, the
Julia environment against its Manifest, and the copied source against the fingerprint
passed as a build argument; any mismatch fails the build.

**Identity.** `scripts/fingerprint.sh` hashes (carriage returns stripped) `Project.toml`,
`src/`, and `docker/`, `environments/`, `experiments/`, `scripts/`, `data/` (without
`data/external/`), `results/`, `experiment_manifest.csv`, `artifacts.sha256`,
`artifacts-part2.sha256` and `checksums.sha256` of this directory, excluding Markdown. The image stores it as the label
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

The bundle is the Zenodo data record of the paper. It has **two parts**: two tarballs of the
same record that extract into the same directory. Each part's files are listed with their
SHA-256 in its own inventory, and `artifacts.lock` records each tarball's SHA-256 (and, at
release, its DOI and URL).

| Part | Tarball | Built by | Inventory |
|---|---|---|---|
| 1 | `sarimax-ijf-2026-artifacts-v1.tar.gz` (296 MB, 535 files) | `scripts/maintainers/build_artifact_bundle.py`, from the experiment workspaces | `artifacts.sha256` |
| 2 | `sarimax-ijf-2026-artifacts-part2-v1.tar.gz` (299 MB, 368 files) | `scripts/maintainers/build_artifact_part2.py`, from the controlled campaign and the September 2026 raw archive | `artifacts-part2.sha256` |

Part 1 comes from the repository history, the three `paper/replication-staging-*` branches,
the ForecastTester experiment workspace and the `sarimax-replication` folder (the paths are
arguments). Part 2 holds what was produced or received after part 1 was frozen: the
controlled objective campaign and the raw archive of the September 2026 stable and admissibility
campaigns. Part 2 does not modify
part 1, so part 1's recorded SHA-256 stays valid. Both builds are deterministic: gzip and tar
carry no timestamps or owners.

| Directory | Part | Content |
|---|---|---|
| `raw/battery-2026-07/` | 1 | July 2026 battery: validation, forecasting, architecture, solver and energy records (jsonl), logs, simulated inputs; PJME data omitted |
| `raw/m4-objectives-icarai-2026-08/` | 1 | 37 objective campaigns (cel_*, obj_*, req_*, censo_*, smoke_*), auto.arima/Naive2 baselines for M/Q/Y/W, campaign index, validity and probe files |
| `raw/m4-objectives-rerun-a1ebc0d-2026-08/` | 1 | re-run of the huber and ridge cells at a1ebc0d, with logs |
| `raw/m4-recreio-2026-08/` | 1 | campaigns A–F (`actual` redacted, per-block metrics added) and verification files |
| `raw/m4-baselines-r-2026-07/` | 1 | auto.arima and Naive2 per-series metrics, all six M4 frequencies, four horizon blocks |
| `raw/robust-cvar-monthly-2026-08/`, `raw/admissibility-2026-08/`, `raw/initialization-order-search-2026-08/`, `raw/scip-ipopt-yearly-2026-07/`, `raw/stable-admissibility-gavea-2026-09/tables/` | 1 | the smaller studies (see the manifest) |
| `derived/m4-objective-distributions-2026-10/` | 1 | the per-series sets of the October 2026 Table 7 evidence |
| `historical-tables/` | 1 | the tables as generated at the time (source of `results/expected/`) |
| `raw/m4-objectives-controlled-2026-10/` | 2 | the controlled objective campaign: frozen protocol and environment; every per-series record of the 16 cells and of the weekly Table 3 protocol run (`test` redacted, per-block metrics added); ledgers, audits, event logs, orchestrator logs; the same results in the per-series schema of `full` |
| `derived/m4-objectives-controlled-2026-10/` | 2 | the independent scorer's tables, reconciliation and validity files, the reconciliation report and the manuscript update proposal (source of `results/expected/m4_objectives_controlled.csv`) |
| `raw/stable-admissibility-gavea-2026-09/brutos_gavea/` | 2 | the raw archive of the September 2026 campaigns as received, with the M4 test column `a` removed from the forecast files |
| `derived/stable-admissibility-gavea-2026-09/` | 2 | lag-1/seasonal MASE denominator ratios (hourly, quarterly) and the tables of 2026-10-17 (source of `results/expected/stable_admissibility_gavea.csv`) |
| `original-scripts/` | 1, 2 | verbatim runners, table scripts, environment files and notes of each campaign |
| `PROVENANCE.csv`, `part2/PROVENANCE.csv` | 1, 2 | for each file: origin (git object or workspace file), original SHA-256, transform |

Every transform that changes content is recorded in the PROVENANCE files:

* **M4 test values are removed** wherever an archived file carried them: campaign A–F `actual`, controlled-campaign `test` and the September 2026 archive's `a`. Before removal, the per-block sMAPE and MASE are computed from them:
  * campaigns A–F use the definitions of campaign A's `make_table_m4.jl`;
  * the controlled campaign uses `ijflib.m4data`, checked against the runner's own totals to 1e-9 for all 381,795 records.
* **Part 2 adds derived files:** the per-series files in the `full` schema and the MASE denominator ratios.

The `m4_benchmark` and `m4_objectives_controlled` table groups regenerate the historical tables from the redacted files to the last printed digit. That also validates the transforms.

**Determinism.**

* **Part 1 sources are pinned.** Its git sources are pinned to the commits bundle v1 was built from: the tips of the three `paper/replication-staging-*` branches on 2026-10-17, and `308461e`, the last commit that tracked the July 2026 battery. Keep these commits reachable (tag them) before deleting those branches.
* **Summation order is fixed.** The per-block metrics of the redacted campaign A–F files are summed left to right, because `sum()` became compensated in Python 3.12 and would change last bits.
* **Part 1 was rebuilt in two other environments:**
  * macOS arm64 (Python 3.14): 299 of its files matched byte for byte;
  * Windows (Python 3.12), in the environment that ran the controlled campaign: 446 matched. There, 11 original scripts of the admissibility and contamination studies differ (later local edits) and 10 sources are missing.

  The remaining files come from experiment workspaces that are not in the repository.
* **Part 2 was built twice**, with the same tarball SHA-256 both times.

**Fetching.** `./reproduce.sh fetch artifacts` downloads each part from the location in `artifacts.lock`, checks its SHA-256, extracts it safely and verifies every file against its inventory. `--from FILE` (once per part) uses local tarballs instead. Completion markers:

* a part is marked complete (`.complete`, `.complete-<part name>`) only after every one of its files has been verified;
* the whole bundle is marked complete (`.complete-all`) only when every part is.

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
| m4_benchmark | campaign A's `table_m4` (M4 vs auto.arima, six frequencies x four blocks) | 192 |
| m4_paired | paired differences and marginal distributions against auto.arima (manuscript Tables 4 and S8, Figure 2); reference values in `results/historical/m4-paired-2026-09/` | 420 |
| m4_objectives | Tables 1–2 of the August 2026 objective campaigns and `tabelas_paper` sections A and B | 880 |
| m4_objective_distributions | October 2026 Table 7 evidence: reproduction, 4a–4d, figures | 2,872 |
| arms | the campaign B–F objective, multistart, isolation, axes and stable tables | 299 |
| robust_cvar | contamination study (reference values transcribed from internal notes) | 19 |
| scip_ipopt | SCIP vs Ipopt summaries of the July 2026 yearly study | 23 |

**The manuscript.** `results/manuscript/manuscript_values.csv` lists the 672 numbers printed in
the tables, figures and experimental sections of the 2026-10-16 revision, each with its LaTeX
label and its source (a cell of a table group, a derived table in `results/historical/`, or
none with the reason; `results/manuscript/README.md`). After the groups run, `tables`
compares each one with its regenerated cell (`scripts/ijflib/manuscript.py`) and reports one
row per manuscript item; `scripts/maintainers/check_manuscript_mapping.py` makes the same
comparison against `results/expected/` without the bundle. When the manuscript changes, update
the file and run that script.

## 7. Level C internals

**Configurations.** `kind = "m4"`: `frequencies`, `per_series_timeout_s`, `max_failure_rate`,
`[auto_defaults]` (every keyword passed to `Sarimax.auto`, written out; strings `"nothing"`
mean `nothing`; symbol-valued keys are converted), and `[[arms]]` with `engine = "julia"`
(optional `auto` overrides) or `engine = "r"` with `method = "auto_arima"` or `"naive2"`;
optional `[archived.<arm>]` tables point at archived per-series files for the
series-by-series comparison. `kind = "script"`: `[[steps]]` (engine, script, args,
timeout_s), `tables_group` and `tables`; the steps write records into the run directory with
the bundle's layout, and the group's generator then rebuilds the tables from the new records.
Every configuration states `protocol_status`: `final` (a frozen protocol whose results are
archived), `archived` (the protocol of an archived historical campaign) or `provisional`;
`superseded_by = "<id>"` marks a historical campaign that another experiment replaces.
`full --all` runs the experiments that carry a manuscript label, leaving out provisional and
superseded ones (each can still be run with `--experiment`).

**Worker start-up.** All workers of a task are started at once and collected as they report
ready (each spends about a minute loading and warming up the package); a worker that fails to
start is retried on its own, up to three times.

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

## 8. Integration of `m4_objectives` (done 2026-10-20)

The controlled comparison of objectives was produced by a separate workstream (2026-10-17/19). Each point of the integration contract was handled as follows.

1. **Protocol.** `experiments/configs/m4_objectives.toml` holds the frozen protocol, with
   `protocol_status = "final"`. Every keyword the campaign passed to `auto` is written out
   (copied from the archived `protocol/experiment_protocol.toml`, frozen before the first fit).
   * Ridge is the package's dedicated `"ridge"` objective, not an elastic-net re-expression.
   * The orchestration safety limit is the protocol's 21,600 s.
   * The reference arm is `naive2`. The archived Table 5 took auto.arima from the stored
     baselines and never re-ran it.
2. **Code.** Every fit ran at `14dd581` (v1.0.0) from a clean worktree, but with the campaign's
   own runner on Windows, not with this package's image. Two consequences follow.
   * The archived rows carry the campaign's fingerprint and protocol SHA-256, not this
     package's `source_fingerprint`.
   * The image's Julia environment was rebased on the campaign's Manifest (section 3), so a
     re-run differs from the archived run only by platform and orchestration.
   * `full --experiment m4_objectives` compares such a re-run series by series with the
     archive.
3. **Outputs.** The archive keeps the campaign's own records: one JSON line per series and
   attempt, with coefficients, criterion values, solver statuses and timings. The builder
   derives from them per-series files in the schema of section 7, one per objective and
   frequency (`per_series/<objective>__<frequency>.csv.gz`). They contain every series of
   every frequency.
4. **Archive.** The results are in bundle part 2 under `raw/m4-objectives-controlled-2026-10/`
   (section 4), rather than in a rebuilt part 1, so part 1's published checksum is unaffected.
   The integration also added:
   * the `[archived.<arm>]` entries in the configuration;
   * the table group `scripts/ijftables/m4_objectives_controlled.py` (Tables 5 and 6, marginal
     quantiles, the weekly reconciliation and the ECDF figure);
   * the manifest row.

   The expected values are parsed from the tables written by the campaign's independent
   scorer, because the manuscript does not print the controlled values yet. They are its
   proposed replacements.
5. **Never mix.** The historical campaigns keep their own ids (`m4_objectives_icarai_*`) and
   table groups. `m4_objectives_icarai_innovations` is marked `superseded_by = "m4_objectives"`,
   so `full --all` runs the controlled experiment only. No table combines the two.

## 9. Release checklist (maintainers; every step needs the authors' approval)

1. **Resolve the manuscript findings** (`LIMITATIONS.md`, item 1; corrections with exact text
   in `FINAL_SCIENTIFIC_AUDIT.md`, section C):
   * transcribe the canonical 2026-10-19 revision (`reports/manuscript_traceability.md`):
     fill the pending template of the controlled items and re-check every other row;
   * the Table 7 Ipopt times and the PJME certificate (registered errata, release gate);
   * the description of the monthly protocol behind Table 3 (`reports/monthly_protocol.md`);
   * the printed values that have no archived source (`reports/provenance_gaps.md`: archive
     their records, or qualify the statements);
   * the name of the package's `AIR_PASSENGERS` series (`reports/airpassengers_dataset.md`).

   Then run `scripts/maintainers/check_manuscript_mapping.py --release`: it must pass (every
   registered erratum corrected and removed from `errata.csv`). `ijf-image-publish.yml` runs
   the same gate before it publishes anything.
2. **Decide on SCIP's licence** (section 3, `reports/scip_licence.md`): keep SCIP 8.0.0 and
   settle its redistribution, or move to SCIP 8.0.3 or later (Apache 2.0) and re-run the SCIP
   evidence as new identified runs.
3. **Tag the pinned commits.** Tag the commits of the `paper/replication-staging-*` branches
   that the bundle builder pins (section 4) before those branches are deleted.
4. **Merge** the replication branch into `master` after review.
5. **Archive the bundle.**
   * Part 1 already exists (its SHA-256 is in `artifacts.lock`); part 2 is rebuilt identically
     with `build_artifact_part2.py`.
   * If the paper keeps the certification results of Section 5.6, the August 2026
     certification batch has to be found and archived as a further part, with a table group.
   * Upload both tarballs and their inventories to one Zenodo data record. Test in the Zenodo
     Sandbox first: `scripts/maintainers/zenodo_deposit.py --bundle-dir DIR1 --part2-dir DIR2`.
   * Record the version DOI and each tarball's URL in `artifacts.lock`. This does not change
     the fingerprint.
6. **Publish the image.** Build and push it with the manually triggered workflow
   `ijf-image-publish.yml`; record the digest, fingerprint and revision in `image.lock`; make
   the GHCR package public. If its size and the SCIP licence allow, archive the exported image
   (`docker save | gzip`) in the Zenodo data record.
7. **Tag the release.** `src/` now differs from v1.0.0 (the information-criterion fix for models
   with regressors, the elastic-net fix for L1 weights of zero and the dataset alias, CHANGELOG
   "Unreleased"), so release the package as
   1.0.1 first, then tag the replication release commit (for example `ijf-2026-v1`) and publish
   the GitHub release. The Zenodo GitHub
   integration archives it and mints the software DOI (`.zenodo.json`).
8. **Link the records.** Software → `isSupplementedBy` the data record; both → `isSupplementTo`
   the article DOI once assigned. Then update `CITATION.cff` and the availability statement.

## 10. Continuous integration

`.github/workflows/ijf-replication.yml` runs on pull requests and pushes to `master` that
touch the package or the replication files, and by hand. Full campaigns never run in CI.

* **Host checks** (no Docker): shellcheck; the usage and missing-Docker exit codes;
  `checksums.sha256` against the committed files and against a fresh regeneration;
  `check_package.py` (manifest, table groups, manuscript mapping); and
  `check_recovered_2026_09.py` (the derived tables of Tables S6 and S7 and Figure S1 against
  the recovered outputs).
* **Manuscript release check** (its own job, so that it does not stop the container tests):
  `check_manuscript_mapping.py` fails on any printed value that differs from its source and is
  not a registered erratum, and on a registered erratum that no longer differs; it reports the
  five categories, the untranscribed canonical revision and the textual claims in the step
  summary. The release gate (`--release`, registered errata fail too) is shown for information
  and enforced by `ijf-image-publish.yml`.
* **Container**, on `linux/amd64` (the reference) and `linux/arm64` (experimental): the image
  is built from the checkout with the same Dockerfile and build arguments as
  `./reproduce.sh build` (buildx adds a layer cache), then: the default command without the
  unpublished bundle (smoke must pass; the run must end with exit 4, never SUCCESS); the
  battery, robust-CVaR and SCIP/Ipopt tables from the committed fixture subset; a missing and
  a corrupted archived input (exit 4); an image refused because the checkout changed after it
  was built (exit 3); `fetch m4` for the weekly files with SHA-256 verification;
  `m4_benchmark` on a random sample of 2 weekly series per arm, then `--resume` to 3 (the
  sample is nested); the controlled objective campaign (`m4_objectives`) on 3 weekly series
  per arm, compared series by series with its archived run (the weekly per-series files of
  bundle part 2 are in the fixture subset); and the battery's validation experiment (Julia,
  Python and R steps). Reports are uploaded as workflow artifacts.

`.github/workflows/ijf-image-publish.yml` builds and pushes the image to GHCR, pulls it back
by digest on a clean runner, checks its fingerprint against the checkout, runs the smoke
tests in it and prints the lines for `image.lock`. It runs only when started by hand.

The Dockerfile declares the identity build arguments (`SOURCE_FINGERPRINT`,
`SOURCE_REVISION`) just before the final layer: a build argument invalidates every later
`RUN`, so an edit to a replication script rebuilds only that layer, not R, Python and Julia.
