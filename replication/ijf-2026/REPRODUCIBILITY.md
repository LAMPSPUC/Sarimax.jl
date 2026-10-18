# What "reproduced" means in this package

## Four classes of results, kept apart

| Class | Where | Mutable? |
|---|---|---|
| Original raw experimental results: per-series forecasts, metrics, orders, solver statuses, logs | artifact bundle, `raw/` | never |
| Reference results: the archived inputs of the published tables (raw files and the derived per-series sets built from them) and the historical tables themselves | bundle `raw/`, `derived/`, `historical-tables/`; derived tables not in the bundle in `results/historical/`; machine-readable reference values in `results/expected/`; the manuscript's printed values in `results/manuscript/` | never |
| Regenerated outputs: tables and figures rebuilt by `./reproduce.sh tables` | `output/tables`, `output/figures` | rewritten by each run |
| New executions: results of `./reproduce.sh full` | `output/runs/<experiment>/<run id>/` | appended until the run completes |

A new execution never reads archived results in place of its own, and nothing writes into
the bundle (it is mounted read-only).

## Level A — smoke

Each smoke quantity is compared with a value from one of three sources, stated next to it in
`experiments/validation/smoke_expected.json` and in the report:

* **archived**: the July 2026 battery's own records for the same fit (e.g. `forecast::Arima`
  CSS and ML on the simulated AR(1), statsmodels SARIMAX). Tolerances follow the precision
  those records were written with (four decimals for the R records);
* **analytic**: values computed by hand or by a solver-free method (e.g. the global minimum
  of a one-dimensional conditional-sum-of-squares function on a 2,000,001-point grid, against
  which SCIP's certified optimum is checked; the binding invertibility margin);
* **reference**: values recorded from the release environment, with tolerances that absorb
  cross-platform floating point (BLAS, libm, MUMPS pivoting) and never a change of method.

A smoke test also fails if an engine cannot start, raises, times out, or does not produce a
quantity it is expected to produce.

## Level B — tables

A table cell PASSES when the regenerated value agrees with the historical value to the
precision the historical table printed it: |generated − expected| ≤ ½ unit in the last
printed decimal. Where the historical file printed full precision, six decimals are used.
Text cells (statuses, labels) must match exactly. The expected values are parsed from the
historical tables (`scripts/maintainers/extract_expected.py`), never computed by the
generators they check. A group whose inputs are missing or fail their checksum reports every
one of its cells as MISSING; a crashing generator reports FAIL.

**The manuscript.** After the table groups, `tables` takes every number printed in the
manuscript (`results/manuscript/manuscript_values.csv`: tables, figures and the numbers of the
experimental sections) and compares it with the regenerated cell it comes from, at the
precision the manuscript prints. A difference fails the manuscript item; a value whose source
group could not run is MISSING; a value that can only be compared with a derived table whose
raw inputs are not archived is reported as consistent with that table, and a value without an
archived source is listed with the reason. Neither of the last two is ever counted as
reproduced: both are WARN in the report and are counted separately in the summary
(`output/reports/manuscript_cells.csv` holds the cell-level comparison).

Conventions that matter for the M4 tables (documented in `scripts/ijftables/common.py`):
OWA is the **ratio of aggregated means**, `0.5·mean(MASE)/mean(MASE_Naive2) +
0.5·mean(sMAPE)/mean(sMAPE_Naive2)` over one common set of series, never the mean of
per-series OWA (the smoke test demonstrates that the two differ); sMAPE and MASE follow the
M4 harness definitions, with one in-sample seasonal-naive scale per series shared by every
horizon block.

## Level C — full re-execution

`./reproduce.sh full` runs an experiment's protocol (its configuration in
`experiments/configs/`) from the original inputs on the release code. `full --all` runs the
re-runnable experiments behind the results the manuscript reports; experiments the manuscript
does not report and protocols marked provisional are left out (and named), and can still be
run one by one with `--experiment`. Its report states,
per task, how many series completed and how many ended in ERROR, TIMEOUT or CRASH, and, where
archived per-series results exist, how many series reproduce them (same order; same total
sMAPE within a relative 1e-4). Because several archived campaigns ran on other commits or
platforms (see LIMITATIONS.md), a full re-run is a re-execution of the protocol; it is a
reproduction of the archived numbers only where the report says the series agree.
`--limit N` checks the execution path on the first N series of each task and is labelled as
such in every report; it is not a result.

## Identity of the code that ran

The image carries a source fingerprint (`scripts/fingerprint.sh`): a SHA-256 over the
package source, the pinned environments, the runners, the table generators, the fixtures and
the expected values. `reproduce.sh` recomputes it on the checkout and refuses an image whose
fingerprint differs (`--allow-unverified-image` overrides this and labels every report
UNVERIFIED); inside the container, `check` recomputes it over the image's own copy. Every
row written by `full` carries the run id, the configuration's SHA-256, the source
fingerprint and the engine versions, and a run refuses to resume if its configuration or
source changed.

## Determinism

No random number generator is used on the fit path of SARIMAX.jl, `auto.arima` or Naive2,
so the same inputs give the same estimates on the same platform. Parallel runs use one BLAS
thread per worker process. Across platforms, small floating-point differences are possible;
they are within the stated tolerances for the smoke tests, and the archived campaigns were
themselves produced on Windows and macOS, while the image runs Linux.
