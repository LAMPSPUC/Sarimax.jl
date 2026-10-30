# SARIMAX.jl IJF artifact bundle

Archived experimental results behind the values printed in the article *SARIMAX.jl:
Open-Source Optimization-Based SARIMAX Modeling in Julia* (International Journal of
Forecasting, special issue "Advances in Open Source Forecasting Software").

This bundle is the input of `./reproduce.sh tables` in the
[Sarimax.jl repository](https://github.com/LAMPSPUC/Sarimax.jl) (`replication/ijf-2026/`),
which rebuilds the tables from these files and compares every cell with the value printed at
the time. It contains results, not software: the package, the container definition and the
scripts that read this bundle are in the repository release that cites it.

## Experiments

| Directory | Experiment | In the article |
|---|---|---|
| `battery-2026-07` | Validation against statsmodels and R `forecast`; MA(1) global certificates; local (Ipopt) against certified global (SCIP) estimation (July 2026) | Table 2, Section 5.6, Table 7 |
| `m4-benchmark-2026-08` | M4, six frequencies, `initialization = :innovations`, against auto.arima (campaign A); the stable objective on the weekly and yearly series (campaign F) | Tables 3, 4, S12, Figure 2, Table S4, Section 5.3; Tables S15 and S16 |
| `m4-baselines-2026-07` | auto.arima and Naive2 per-series accuracy on the six M4 frequencies, by horizon block | the reference of every M4 table |
| `m4-objectives-controlled-2026-10` | Controlled comparison of the MSE, MAE, Huber and ridge objectives on four M4 frequencies, and the weekly MSE run under the Table 3 protocol | Tables 5, 6, S13, S14, Figure 3, Table S4, Sections 5.1 and 5.4 |
| `stable-admissibility-2026-09` | The stable (CVaR) objective on the monthly, quarterly, daily and hourly series; the 2x2x2x2 stationarity x invertibility x MA-normalization grid on the monthly series (September 2026) | Tables S15 and S16, Sections 5.4 and 5.5 |

Each experiment has a directory under some of these four folders:

| Folder | Contents |
|---|---|
| `raw/` | Results as the runners wrote them (one row per series, or one record per fit), with the run logs and the recorded environment. |
| `derived/` | Files computed from public data for one table group (`stable-admissibility-2026-09`: the MASE denominator ratios described below). |
| `historical-tables/` | The tables as they were generated at the time. The expected values of `./reproduce.sh tables` were parsed from these files. |
| `original-scripts/` | Verbatim copies of the runners, table scripts and environment files that produced the results. They document the protocol; the replication package runs its own ports of them and never executes these copies. Comments and paths are those of the machines the campaigns ran on. |

Comments in the original scripts and in some raw files refer to notes of the campaigns' working
directories (for example `REPRODUCE.md`). Those notes are not part of this record; what the
article relies on from them (package commits, recorded and reconstructed environments, protocol
settings) is stated per experiment in `experiment_manifest.csv`, `LIMITATIONS.md` and
`reports/` of the replication package.

`PROVENANCE.csv` gives, for every file, its source (a commit and path of the Sarimax.jl
repository, or a path in the authors' ForecastTester workspace), the SHA-256 of the original
bytes and the transform applied. `artifacts.sha256` in the repository lists the SHA-256 of
every file of this bundle; `./reproduce.sh check` verifies them.

## Transforms

* **M4 test values are not redistributed** (the official files carry no licence). Where an
  archived file stored them, the per-block sMAPE and MASE were computed from the forecasts
  and the test values with the campaign's own definitions, stored as `smape_<block>` and
  `mase_<block>` (blocks `short`, `medium`, `long`, `total`), and the test values were
  removed: the column `actual` of the campaign A and F files, the field `test` of the
  controlled-campaign records (the computed totals agree with the runner's own sMAPE and
  MASE to a relative 1e-9), and the column `a` of the September forecast files (`*_prev.csv`).
* **Per-series files of the controlled campaign** (`per_series/`, `per_series_p1_table3_protocol/`)
  are derived from its records, one row per series (final attempt), in the schema that
  `./reproduce.sh full` writes, so that a new run can be compared series by series.
* **MASE denominators.** The stable campaigns of the hourly and quarterly series stored MASE
  with the lag-1 in-sample naive denominator. `derived/stable-admissibility-2026-09/` holds,
  per series, the ratio of the lag-1 to the seasonal in-sample naive MAE of the official
  training series, which converts those values exactly to the M4 definition.
* `historical-tables/m4-objectives-controlled-2026-10/table3_weekly_reconciliation.csv` keeps
  the rows of the three weekly MSE runs reported in the article (the six-frequency campaign,
  the controlled cell and the controlled-environment run under the Table 3 protocol).
* Large text files are gzip-compressed (`.gz`). The campaign A and F files record the removal
  of `actual` in comment lines at their top; no other content is changed.

## Data that is not redistributed

* **M4 competition data** (github.com/Mcompetitions/M4-methods, commit
  `b8f01984fde0b562aa6f1a9181a242ce076ba73e`): `./reproduce.sh fetch m4` downloads the
  official files and verifies their SHA-256.
* **PJM hourly energy consumption (PJME)**, used by one instance of Table 7: its licence is
  not established, so the data file is not included. The replication package obtains it from a
  pinned source and verifies it against `original-scripts/battery-2026-07/SHA256SUMS`.

## Known limitations

See `replication/ijf-2026/LIMITATIONS.md` and `experiment_manifest.csv` in the repository,
which give, per experiment, the package commit, the recorded or reconstructed environment and
what can and cannot be re-executed. In short: the campaigns ran on several package commits
before the release; some environments were reconstructed rather than recorded; the archived
SCIP certificates record the termination status and relative gap but not the dual bound; and
the monthly September runs and the 2x2x2x2 grid ran through the authors' harness, whose
environment file for the grid was not preserved.
