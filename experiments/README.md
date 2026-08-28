# M4 objective-function experiments — replication material (host ICARAI)

This directory holds the scripts, raw outputs and table generators for the M4 benchmark
campaigns run on the host **ICARAI**. It covers the objective-function cells
(`mse`, `huber`, `mae`, `ridge`) under two initializations, across four M4 frequencies, and
the over-differencing guard experiment.

It is one of several per-host contributions. It does **not** cover the full set of
experiments behind the manuscript; campaigns run on other hosts are not represented here.
See `REPRODUCE.md` for exactly which table cells this host produced and which it did not.

## Layout

```
env/          Project.toml and Manifest.toml of the environment the campaigns ran in
scripts/      the runner, the worker, the table generator, and the reproduction probe
results/
  campaigns.csv   provenance index: one row per campaign
  validity.csv    per-campaign error, censoring and solver-status columns
  reproduction_probe_dev.csv   output of the probe behind REPRODUCE.md section 7
  raw/            campaign outputs, gzipped, one file per campaign
  baselines/      auto.arima and Naive2 reference metrics (produced on another host)
tables/       manuscript tables, regenerated from results/
reproduce.sh  single entry point
SHA256SUMS    checksums for every file above
```

## Reading the raw outputs

One row per series. Beyond the fitted orders and the metrics, every row carries the
provenance of the run that produced it: the `Sarimax.jl` commit, the Julia version, the
JuMP / MathOptInterface / Ipopt versions, the operating system and architecture, the host,
and the full cell configuration (objective, initialization, time cap, guard flag,
frequency).

Metrics are reported over four horizon blocks — `short`, `medium`, `long`, `total` — in the
M4 convention for each frequency. The `forecast` column stores the point forecast for every
step, so any metric at any horizon cut can be recomputed without refitting.

The `status` column is `OK`, `ERROR:<message>`, `REMOTE:<message>` or `TIMEOUT`. Failure
rows keep the same width as success rows and carry `-1` / `NaN` in the numeric fields.

## Regenerating the tables

`results/raw/` is sufficient: the table generator refits nothing.

```bash
python scripts/make_tables.py
```

## Re-running a campaign

Re-running requires the M4 datasets, which are not redistributed here (see `REPRODUCE.md`,
"Inputs not produced here"). With them in place:

```bash
SARIMAX_SRC=/path/to/Sarimax.jl M4_DATASETS=/path/to/datasets \
  julia --project=/path/to/Sarimax.jl scripts/run_cell.jl 0 10 out.csv mse monthly 3600 innovations 0 true
```

`REPRODUCE.md` gives the exact argument list for every campaign, in prose, so that a table
caption can be checked against the code without reading Julia.

## Checking reproduction against a newer package

`scripts/probe_reproduction.jl` re-runs every table cell on the first N monthly series
against an arbitrary `Sarimax.jl` checkout, with all arguments explicit, and
`scripts/compare_probe.py` diffs the result against the stored campaign rows. This is how
section 7 of `REPRODUCE.md` was established, and it is the way to re-establish it if the
package moves.

```bash
julia --project=<checkout> scripts/probe_reproduction.jl <checkout> 120 probe.csv
python scripts/compare_probe.py
```
