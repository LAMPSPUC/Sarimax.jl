# SARIMAX.jl IJF artifact bundle

Archived experimental material for the paper *SARIMAX.jl: Open-Source Optimization-Based
SARIMAX Modeling in Julia* (International Journal of Forecasting, special issue "Advances in
Open Source Forecasting Software").

This bundle is the input of `./reproduce.sh tables` in the
[Sarimax.jl repository](https://github.com/LAMPSPUC/Sarimax.jl) (`replication/ijf-2026/`).
It contains results, not software. The software, the container definition and the scripts that
read this bundle are in the repository release that cites it.

## Layout

| Directory | Contents |
|---|---|
| `raw/` | Original per-series experimental results, as the campaign runners wrote them (one directory per campaign, named after the host or study and the month it ran), plus their logs where they exist. |
| `derived/` | Archived intermediate datasets that earlier table scripts built from `raw/` (the October 2026 Table 7 evidence). |
| `historical-tables/` | The tables exactly as they were generated at the time. The expected values used by `./reproduce.sh tables` were extracted from these files. |
| `original-scripts/` | Verbatim copies of the runners, table scripts, environment files and reproduction notes that produced each campaign. They document the protocols and are not executed by the replication package. |
| `PROVENANCE.csv` | For every file: where it came from (a git object of the Sarimax.jl repository or a file of the authors' workspace), the SHA-256 of the original bytes, and the transform applied (`copy`, `gzip`, or the redaction described below). |

`artifacts.sha256` in the repository lists the SHA-256 of every file in this bundle;
`./reproduce.sh check` verifies them.

## Data that is not redistributed

* **M4 competition data.** The official files (github.com/Mcompetitions/M4-methods, commit
  `b8f01984fde0b562aa6f1a9181a242ce076ba73e`) carry no licence, so this bundle does not
  redistribute them. `./reproduce.sh fetch m4` downloads them and verifies their SHA-256.
  The recreio campaign files stored the M4 test values next to each forecast (column
  `actual`). In this bundle that column was removed; before removal, the per-block sMAPE and
  MASE were computed from `forecast` and `actual` with the campaign's own definitions and
  stored as `smape_<block>` / `mase_<block>`. PROVENANCE.csv gives the SHA-256 of each
  original file.
* **PJM hourly load (PJME).** The licence of the Kaggle copy used in the July 2026 battery is
  not established, so the data files of `raw/battery-2026-07/energy/data/` are omitted. The
  energy runner obtains them from the Sarimax.jl repository history at a pinned commit and
  verifies their SHA-256.

## Known limitations of the archived material

See `replication/ijf-2026/LIMITATIONS.md` and `REPRODUCIBILITY_STATUS.md` in the repository.
In short: several campaigns ran on feature-branch commits that are not part of a release;
some environments were reconstructed rather than recorded; SCIP runs recorded the termination
status and relative gap but not the dual bound; and the raw files of the September 2026
GAVEA campaigns are not part of this bundle.
