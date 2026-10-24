# SARIMAX.jl IJF artifact bundle, part 2

This archive is the second part of the input of `./reproduce.sh tables` in the
replication package of "SARIMAX.jl: Open-Source Optimization-Based SARIMAX Modeling in Julia"
(`replication/ijf-2026/` in https://github.com/LAMPSPUC/Sarimax.jl). It extracts into the same
directory as part 1 (`sarimax-ijf-2026-artifacts-v1`); `./reproduce.sh fetch artifacts` does
both and verifies every file against `artifacts.sha256` in the repository.

## Contents

`raw/m4-objectives-controlled-2026-10/`: the controlled comparison of estimation objectives
(MSE, MAE, Huber, Ridge) on M4 monthly (48,000), quarterly (24,000), yearly (23,000) and weekly
(359) series, run 2026-10-17 to 2026-10-19 with Sarimax.jl v1.0.0 (`14dd581`).

* `protocol/`: the frozen protocol (`experiment_protocol.toml`, frozen before the first fit),
  the Julia environment (`env/Project.toml`, `env/Manifest.toml`) and the recorded system and
  package versions (`environment_manifest.json`);
* `runs/m4obj-6957c71f60f1/<frequency>/<objective>/`: one JSON record per series and attempt
  (order, coefficients, criterion values, solver status and iterations, timings, forecast,
  per-block sMAPE/MASE), the ledger, the completeness audit and the event log of each cell;
* `runs/m4obj-9f302efccda9-P1-table3-protocol/weekly/mse/`: the single extra run, weekly MSE
  under the protocol of the manuscript's Table 3 (`stationary = true`);
* `per_series/<objective>__<frequency>.csv.gz` and `per_series_p1_table3_protocol/`: the same
  results in the per-series schema of `./reproduce.sh full`;
* `logs/`: the orchestrator logs of every cell and of the runner's preflight tests.

`original-scripts/m4-objectives-controlled-2026-10/`: the runner (`runner/`) and the
independent scorer (`scoring/`) exactly as executed.

`derived/m4-objectives-controlled-2026-10/`: the tables, reconciliation and validity files
the independent scorer produced on 2026-10-19 (the reference values of the table group
`m4_objectives_controlled`).

`raw/stable-admissibility-gavea-2026-09/brutos_gavea/`: the raw archive of the September 2026 stable and admissibility campaigns
(`brutos_gavea.zip`, shared by Davi Valladao on 2026-10-16): stable (CVaR) campaigns, the
over-differencing-guard campaign, the 2x2x2x2 admissibility census and diagnostics, with their
own `REPRODUCE.md`; `SHA256SUMS.original` lists the checksums of the files as received.

`derived/stable-admissibility-gavea-2026-09/`: per-series ratios of the lag-1 to the seasonal
in-sample naive MAE of the M4 hourly and quarterly training series (see below).

## Transforms

Every file's origin, the SHA-256 of the original bytes and the transform applied are listed in
`part2/PROVENANCE.csv`.

* M4 test values are not redistributed here (their licence is not established). The
  controlled-campaign records stored them in the field `test`; before removal, the builder
  computed the per-block sMAPE and MASE from `forecast` and `test` and checked that the
  runner's own total sMAPE and MASE agree with them (relative tolerance 1e-9). The
  forecast files (`*_prev.csv`) stored them in the column `a`, which was removed.
* The stable campaigns of hourly and quarterly wrote MASE with the lag-1 in-sample naive
  denominator; multiplying by `ratio_lag1_to_seasonal` gives the M4 (seasonal) MASE exactly.

The official M4 data are obtained with `./reproduce.sh fetch m4` (a pinned commit of
Mcompetitions/M4-methods, checked by SHA-256).
