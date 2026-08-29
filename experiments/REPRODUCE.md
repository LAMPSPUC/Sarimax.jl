# Reproduction contract — M4 objective-function campaigns, host ICARAI

This document states, for every campaign run on this host, what was computed, under which
build of `Sarimax.jl`, with which arguments, on what hardware, and what about it is and is
not reproducible. Where a fact could not be established from the evidence on the machine it
is written as **NOT KNOWN**, never as an estimate.

## 1. Two version identities, which are not the same

Two distinct versions appear in this package and must not be conflated.

**The campaign commit** is the `Sarimax.jl` commit under which a given set of results was
actually produced. It is historical, it differs between campaigns on this very host, and it
is what appears in the `sarimax_commit` column of every raw row. The campaigns here ran
under four different commits: `aa68d57`, `87f7bfb`, `6a11698` and `5b2ec6b`.

**The publication version** is the released version of the package alongside which this
replication material is distributed: **`v1.0.0`, at commit `a1ebc0d`**.

The two are not the same, and none of the four campaign commits is an ancestor of the
release. **No archived result in `results/raw/` was produced under `v1.0.0`**, and nothing
here should be read as claiming that it was. Section 7 answers, cell by cell, whether
running the scripts under `v1.0.0` reproduces the archived numbers — three cells do not,
and it says why.

## 2. Scope — what this host produced, and what it did not

This host ran the **objective-function cells** and the **over-differencing guard
experiment** on four M4 frequencies, between 2026-08-22 and 2026-08-26. Concretely, it
produced every cell of the two tables in `tables/`:

* **Table 1** (`tables/table1_horizon_total.txt`) — MASE, sMAPE and OWA at the total
  horizon, for monthly, quarterly, yearly and weekly. `mse`, `huber` and `mae` appear under
  both `initialization = :zeroed` and `initialization = :innovations`; **`ridge` appears
  under `:innovations` only**.
* **Table 2** (`tables/table2_owa_by_block.txt`) — the same cells, OWA broken out by the
  short / medium / long / total horizon blocks.

There is no `ridge` / `:zeroed` cell. No such campaign was run on this host, and
`scripts/make_tables.py` iterates `ridge` over both initializations, finds no
`cel_ridge_zeroed_<freq>` file, and renders that position as `n/a` in both tables. The
`n/a` is therefore an absent campaign, not a suppressed or failed one. The `ridge` column of
both tables consists of the four `obj_ridge_innov_<freq>` campaigns and nothing else.

The manuscript is not present on this host, so these tables are named by their generated
titles, not by a manuscript table number. **NOT KNOWN:** which numbered table of the
manuscript each corresponds to. That mapping has to be supplied by whoever holds the
manuscript.

This host did **not** produce: the daily or hourly frequencies, the `auto.arima` and Naive2
baselines, the SCIP global-optimality experiments, the runtime-attribution study, or any of
the campaigns dated before 2026-08-22. Julia was installed on this machine on 2026-08-22 at
18:06; nothing here predates that.

## 3. Campaign inventory

Wall-clock is measured from the creation timestamp of the campaign log to its last write.
CPU-hours are the sum of the recorded per-series fitting time, which excludes orchestration
and process startup — the two therefore do not divide into each other exactly.

| campaign | table cell | n | Sarimax commit | evidence | wall h | CPU h |
|---|---|---:|---|---|---:|---:|
| `cel_huber_zeroed_monthly` | T1/T2 monthly / huber / :zeroed | 48000 | `87f7bfb` | reflog | 5.02 | 24.26 |
| `cel_mae_zeroed_monthly` | T1/T2 monthly / mae / :zeroed | 48000 | `87f7bfb` | reflog | 3.88 | 17.37 |
| `cel_mse_zeroed_monthly` | T1/T2 monthly / mse / :zeroed | 48000 | `87f7bfb` | reflog | 2.97 | 11.52 |
| `censo_ridge_monthly` | ridge census (monthly) -- not a table cell | 48000 | `87f7bfb` | reflog | 2.70 | 11.78 |
| `obj_huber_innov_monthly` | T1/T2 monthly / huber / :innovations | 48000 | `5b2ec6b` | stamped | 10.49 | 37.28 |
| `obj_mae_innov_monthly` | T1/T2 monthly / mae / :innovations | 48000 | `5b2ec6b` | stamped | 9.87 | 29.96 |
| `obj_ridge_innov_monthly` | T1/T2 monthly / ridge / :innovations | 48000 | `5b2ec6b` | stamped | 2.77 | 11.63 |
| `req_false_monthly` | guard arm requireTerms=false (monthly) | 48000 | `5b2ec6b` | stamped | 3.92 | 16.38 |
| `req_true_monthly` | guard arm requireTerms=true (monthly) -- also T1/T2 mse / :innovations | 48000 | `5b2ec6b=9720; 87f7bfb=38280` | stamped | 3.39 | 16.46 |
| `smoke_innov_monthly` | apparatus smoke (monthly) -- not a table cell | 1000 | `aa68d57` | reflog | 0.17 | 0.67 |
| `smoke_ridge_monthly` | apparatus smoke (monthly) -- not a table cell | 1000 | `87f7bfb` | reflog | 0.11 | 0.45 |
| `cel_huber_zeroed_quarterly` | T1/T2 quarterly / huber / :zeroed | 24000 | `87f7bfb` | reflog | 0.97 | 3.85 |
| `cel_mae_zeroed_quarterly` | T1/T2 quarterly / mae / :zeroed | 24000 | `87f7bfb` | reflog | 0.79 | 2.45 |
| `cel_mse_zeroed_quarterly` | T1/T2 quarterly / mse / :zeroed | 24000 | `87f7bfb` | reflog | 0.70 | 1.81 |
| `censo_ridge_quarterly` | ridge census (quarterly) -- not a table cell | 24000 | `87f7bfb` | reflog | 0.63 | 1.56 |
| `obj_huber_innov_quarterly` | T1/T2 quarterly / huber / :innovations | 24000 | `5b2ec6b` | stamped | 1.21 | 5.30 |
| `obj_mae_innov_quarterly` | T1/T2 quarterly / mae / :innovations | 24000 | `5b2ec6b` | stamped | 1.03 | 3.69 |
| `obj_ridge_innov_quarterly` | T1/T2 quarterly / ridge / :innovations | 24000 | `5b2ec6b` | stamped | 0.63 | 1.54 |
| `req_false_quarterly` | guard arm requireTerms=false (quarterly) | 24000 | `87f7bfb` | stamped | 0.79 | 2.30 |
| `req_true_quarterly` | guard arm requireTerms=true (quarterly) -- also T1/T2 mse / :innovations | 24000 | `87f7bfb` | stamped | 0.79 | 2.31 |
| `cel_huber_zeroed_weekly` | T1/T2 weekly / huber / :zeroed | 359 | `87f7bfb` | reflog | 0.08 | 0.40 |
| `cel_mae_zeroed_weekly` | T1/T2 weekly / mae / :zeroed | 359 | `87f7bfb` | reflog | 0.07 | 0.41 |
| `cel_mse_zeroed_weekly` | T1/T2 weekly / mse / :zeroed | 359 | `87f7bfb` | reflog | 0.16 | 0.24 |
| `censo_ridge_weekly` | ridge census (weekly) -- not a table cell | 359 | `87f7bfb` | reflog | 0.03 | 0.14 |
| `obj_huber_innov_weekly` | T1/T2 weekly / huber / :innovations | 359 | `6a11698` | stamped | 0.09 | 0.55 |
| `obj_mae_innov_weekly` | T1/T2 weekly / mae / :innovations | 359 | `6a11698` | stamped | 0.07 | 0.38 |
| `obj_mse_innov_weekly` | T1/T2 weekly / mse / :innovations | 359 | `6a11698` | stamped | 0.04 | 0.18 |
| `obj_ridge_innov_weekly` | T1/T2 weekly / ridge / :innovations | 359 | `6a11698` | stamped | 0.04 | 0.14 |
| `cel_huber_zeroed_yearly` | T1/T2 yearly / huber / :zeroed | 23000 | `87f7bfb` | reflog | 0.51 | 0.83 |
| `cel_mae_zeroed_yearly` | T1/T2 yearly / mae / :zeroed | 23000 | `87f7bfb` | reflog | 0.51 | 0.56 |
| `cel_mse_zeroed_yearly` | T1/T2 yearly / mse / :zeroed | 23000 | `87f7bfb` | reflog | 0.51 | 0.45 |
| `censo_ridge_yearly` | ridge census (yearly) -- not a table cell | 23000 | `87f7bfb` | reflog | 0.51 | 0.37 |
| `obj_huber_innov_yearly` | T1/T2 yearly / huber / :innovations | 23000 | `5b2ec6b=17140; 6a11698=5860` | stamped | 0.67 | 1.09 |
| `obj_mae_innov_yearly` | T1/T2 yearly / mae / :innovations | 23000 | `5b2ec6b` | stamped | 0.51 | 0.67 |
| `obj_ridge_innov_yearly` | T1/T2 yearly / ridge / :innovations | 23000 | `5b2ec6b` | stamped | 0.51 | 0.35 |
| `req_false_yearly` | guard arm requireTerms=false (yearly) | 23000 | `87f7bfb` | stamped | 0.51 | 0.41 |
| `req_true_yearly` | guard arm requireTerms=true (yearly) -- also T1/T2 mse / :innovations | 23000 | `87f7bfb` | stamped | 0.51 | 0.42 |

Two campaigns span more than one commit, because they were interrupted and resumed after
the checkout moved. This is recorded per row, not averaged away:

* `req_true_monthly` — 38 280 rows under `87f7bfb`, 9 720 rows under `5b2ec6b`.
* `obj_huber_innov_yearly` — 5 860 rows under `6a11698`, 17 140 rows under `5b2ec6b`.

### How the commit was established

`stamped` means the commit is written on every row of the original output; the runner
recorded it. `reflog` means the original runner did not yet have that feature and the
commit was recovered afterwards from the `Sarimax.jl` HEAD reflog, which shows that HEAD did
not move at any point inside that campaign's wall-clock window. The reflog is evidence
about HEAD only: it does not prove the working tree was clean. Two independent cross-checks
support the reconstruction — `smoke_innov_monthly` (reflog-derived `aa68d57`) is
byte-identical to the first 1 000 rows of `req_true_monthly` (stamped `87f7bfb`), and the
`deriva` probe run at `4349364` is byte-identical to the corresponding `cel_*` cells
(reflog-derived `87f7bfb`) across 300 series and three objectives.

### Duplicate files in the historical layout

The files historically named `obj_mse_innov_monthly.csv`, `obj_mse_innov_quarterly.csv` and
`obj_mse_innov_yearly.csv` were **byte-for-byte copies** of `req_true_monthly.csv`,
`req_true_quarterly.csv` and `req_true_yearly.csv`. The `mse` / `:innovations` cell of the
tables was never a separate run: it is the `requireTerms = true` arm of the guard
experiment, whose configuration is identical to that cell. This package stores each file
once, under the `req_true_*` name, and `scripts/make_tables.py` reads it in both roles.
Weekly is the exception: `obj_mse_innov_weekly` was a genuine separate run.

`censo_ridge_monthly.csv` was seeded by copying `smoke_ridge_monthly.csv` and resuming; its
first 1 000 rows are byte-identical to that file. Both ran under `87f7bfb`.

## 4. Exact configuration, argument by argument

Every campaign called `Sarimax.auto`, then `Sarimax.predict!`, and scored the forecast with
the M4 sMAPE and MASE definitions copied from the parent harness. The arguments below are
the ones the campaigns ran with. `scripts/cell_worker.jl` passes **all** of them
explicitly, including those whose value equals the package default, so that the script
denotes a fixed computation rather than whatever the default happens to be on the day it is
run.

Constant across every campaign on this host:

| argument | value |
|---|---|
| `d`, `D` | `-1`, `-1` (chosen by the integration tests) |
| `maxp`, `maxq` | `5`, `5` |
| `maxP`, `maxQ` | `2`, `2` |
| `maxd`, `maxD` | `2`, `1` |
| `maxOrder` | `5` |
| `informationCriteria` | `"aicc"` |
| `allowMean`, `allowDrift` | `nothing`, `nothing` (decided internally) |
| `integrationTest` | `"kpssShort"` |
| `seasonalIntegrationTest` | `"seas"` |
| `searchMethod` | `"stepwise"` |
| `seasonalForm` | `:multiplicative` |
| `stationary` | `false` |
| `stationarityMargin` | `1e-6` |
| `invertible` | `false` |
| `invertibilityMargin` | `1e-6` |
| `assertStationarity` | `true` |
| `assertInvertibility` | `true` |
| `constrainedRefit` | `false` |
| `rootMargin` | `1e-2` |
| `optimizer` | `Ipopt.Optimizer`, no attributes set |
| `warmStartFromBox` | `true` |
| `multistart` | `false` |
| `cvarLevel` | `0.9` (inert: no campaign used `stable`) |
| `outlierDetection` | `false` |
| `parallel` | `false` (parallelism is between processes) |
| `requireTermsWhenOverDifferenced` | `true`, except the `req_false_*` arms |
| `requireMAWhenDoublyDifferenced` | `false` |
| `exog` | `nothing` — no campaign used exogenous regressors |
| `seasonality` | `12` monthly, `4` quarterly, `1` yearly, `1` weekly |

In prose: the search is the stepwise Hyndman–Khandakar search scored by AICc, over
`p, q <= 5` and `P, Q <= 2` with total order capped at 5, with `d` and `D` chosen by the
KPSS-short integration test and the seasonal-strength test rather than fixed. Coefficients
are estimated **free** — `stationary = false`, `invertible = false` — and candidates whose
roots fall inside a `1e-2` margin are **rejected** rather than constrained, which is R's
scheme in `forecast::auto.arima`; the `1e-6` domain margins are inert under a free
parameterisation. Fits are warm-started from the box. Ipopt is the solver throughout, with
default attributes. The over-differencing guard requiring at least one ARMA term when
`d + D >= 2` is on, except in the `req_false_*` arm that exists to measure it.

Two arguments are deliberately **not** passed:

* `lambda` and `alpha` — under `objectiveFunction = "ridge"` the package fixes the
  shrinkage at `sqrt(effective sample size)` by construction and **raises `ArgumentError`
  if `lambda` is supplied**. Declaring it explicitly is not possible for this objective.
  They belong to `elastic_net`, which no campaign here used.
* `exogDynamics` and `penaltyTarget` — these arguments did not exist in `Sarimax.auto` at
  any of the campaign commits. They were added on `dev` in `4e03136` (2026-08-28), after
  every campaign here had finished.

### What varies between campaigns

| family | `objectiveFunction` | `initialization` | `maxTimeSeconds` | `requireTerms...` |
|---|---|---|---|---|
| `cel_*_zeroed_*` | `mse`, `huber`, `mae` | `:zeroed` | `600.0`, uniform | `true` |
| `obj_*_innov_*` | `mse`, `huber`, `mae`, `ridge` | `:innovations` | `nothing` (no cap) | `true` |
| `req_true_*` / `req_false_*` | `mse` | `:innovations` | `nothing` | `true` / `false` |
| `censo_ridge_*`, `smoke_*` | `ridge`, `mse` | `:innovations` | production rule | `true` |

The **production rule** is `maxTimeSeconds = T <= 5 * (5 + 2s) ? 120.0 : nothing`: a 120 s
cap on short series only. It applies to the two smoke runs and the four ridge censuses.
**Partially NOT KNOWN:** for those six campaigns the exact command line is not recoverable
(see section 8); the rule above is the script default at the time and is consistent with the
observed timings, but it is an inference, not a record.

Separate from the model cap, the runner imposes an **orchestration deadline** per series —
900 s for the `:zeroed` family, 3 600 s for the `:innovations` family — after which it kills
the worker and records `TIMEOUT`. It is a property of the runner, not of the estimator, but
it censors results and is reported as such in `results/validity.csv`.

## 5. Hardware, environment and determinism

| | |
|---|---|
| host | ICARAI |
| CPU | Intel Core i7-12700, 12 physical / 20 logical cores |
| RAM | 64 GB |
| OS | Windows 10 Pro 10.0.19045, `x86_64-w64-mingw32` |
| Julia | 1.10.12 (juliaup, installed 2026-08-22 18:06) |
| JuMP | 1.31.1 |
| MathOptInterface | 1.52.0 |
| Ipopt | 1.15.0 |
| SCIP | 0.11.6 — present in the Manifest, **not used**; the optimizer was Ipopt throughout |
| workers | 10 processes, `BLAS.set_num_threads(1)` in each |

Total: **58.7 hours** of wall-clock, **210.2 CPU-hours**, 859 872 fitted series across 37
campaigns.

`env/Manifest.toml` is the manifest these campaigns actually resolved against, committed
here verbatim. It is **not** a manifest regenerated today: its `julia_version` header reads
`1.10.11` because it was generated on a different host on 2026-08-01 and then used unchanged
on this one, and its file timestamp is unchanged since. It was honoured exactly — every
package it pins is present in this machine's depot at that version and at exactly one
version. `env/Project.toml` is the package's own `Project.toml`, identical at all campaign
commits, declaring `version = "0.3.0"`.

Note that `Manifest.toml` is listed in the package's `.gitignore`, so it is not tracked on
`dev`; this package carries its own copy under `env/` for that reason.

### Seeds

There is no seed to fix. `Random.seed!` and `rand` occur in `Sarimax.jl` only inside the
simulation entry points, which none of these campaigns call: the fit-and-forecast path
reached by `auto` and `predict!` draws no random numbers. The `seed` column of every raw row
records `n/a-deterministic` to state this positively rather than leave it blank.

### What is not bit-deterministic

The estimates themselves are. Two things are not:

1. **The `tempo` column.** Wall-clock per series depends on machine load and on which of the
   ten workers took the job. It will not reproduce and is not meant to.
2. **Which series get censored.** Censoring is triggered by the orchestration deadline,
   which is a wall-clock threshold. On a slower machine, or under different load, a
   different set of series can cross it. On this host exactly one series was censored in
   859 872 (`obj_huber_innov_monthly`, series 36044, T = 2726, killed at 3 600 s), so the
   exposure is small — but it is not zero, and on materially slower hardware it would grow.

Order of reduction is not a source of nondeterminism here: series are independent and each
row is computed by a single worker with a single BLAS thread. Interleaving affects the order
rows are appended to the file, not their contents.

## 6. Validity columns

`results/validity.csv` carries, per campaign: error rate, orchestration-timeout censoring,
model-cap censoring, the full solver-status breakdown, and the Huber fallback count.

Across all 37 campaigns and 859 872 series: **0 errors, 0 remote failures, 1 censored
series** (rate 1.2e-6). Solver status is `LOCALLY_SOLVED` for every series except three:
two `ALMOST_LOCALLY_SOLVED` in `cel_huber_zeroed_monthly` and one in
`cel_mae_zeroed_monthly`. The Huber fallback fired 32 times in total, at most 15 in any one
campaign (0.03% of `cel_huber_zeroed_monthly`).

**NOT KNOWN — and this is a real gap:** the solver statuses of the *internal candidates*
that the stepwise search fitted and discarded. The runner recorded only the winning
candidate's `solverStatus`. A candidate that ended in `OTHER_ERROR` or `ITERATION_LIMIT` was
absorbed by `auto` and left no trace in these files. Recovering those counts requires
re-running with instrumentation; they cannot be reconstructed from the stored outputs. The
`internal_candidate_statuses` column records `not-recorded` rather than `0`.

The four `censo_ridge_*` campaigns and the two smoke runs predate the `solver` column
entirely; their breakdown reads `not-recorded` for the same reason.

## 7. Does the explicit-argument script reproduce the tables under `v1.0.0`?

This section answers, per cell, the question that matters for a reader who installs the
released package and runs these scripts.

It is answered by **measurement**, not by reading the diff. `scripts/probe_reproduction.jl`
runs the first 120 M4 monthly series through every cell of both families, with all arguments
declared explicitly, against a clean checkout of the **`v1.0.0` tag (`a1ebc0d`)**. Its
output is `results/reproduction_probe_v1.0.0.csv`; `scripts/compare_probe.py` diffs it
against the stored campaign rows for the same series. The environment is held fixed: same
host, same Julia 1.10.12, same `env/Manifest.toml`. Only the package checkout differs.

| table cell | campaign commit | series | identical | different order | reproduces under `v1.0.0`? |
|---|---|---:|---:|---:|---|
| `mse` / `:zeroed` | `87f7bfb` | 120 | 119 | 1 | **Almost** — see (a) |
| `mae` / `:zeroed` | `87f7bfb` | 120 | 120 | 0 | **Yes**, bit-identical |
| `mse` / `:innovations` | `87f7bfb` | 120 | 120 | 0 | **Yes**, bit-identical |
| `mae` / `:innovations` | `5b2ec6b` | 120 | 120 | 0 | **Yes**, bit-identical |
| `huber` / `:zeroed` | `87f7bfb` | 120 | 99 | 19 | **No** — see (b) |
| `huber` / `:innovations` | `5b2ec6b` | 120 | 99 | 18 | **No** — see (b) |
| `ridge` / `:innovations` | `5b2ec6b` | 120 | **15** | **34** | **No** — see (c) |

An earlier revision of this section reported the same probe against `dev` at `fc2c482`,
before the `v1.0.0` tag existed. That run is kept as `results/reproduction_probe_dev.csv`
and is worth comparing against: the two probes differ **only** in the two `huber` rows,
which isolates the cause in (b) to the single change that landed between those two commits.

### (a) `mse` / `:zeroed` — one series in 120

Series 50 selects a different order and its sMAPE moves by 0.0254. Over the probe the mean
sMAPE moves from 7.7610 to 7.7613. The cause was **not isolated**; the pattern is consistent
with a near-tie in AICc resolved differently, but that is a conjecture and is recorded as
such. Identical under both `dev` and `v1.0.0`, so it is not related to any change in the
release.

### (b) `huber` — the warm-start scale fix moves the selection, not the result

21 of 120 series differ in each initialization. The cause is known: the release contains a
fix to how a warm start seeds the residual vector. The seed arrived in the series' original
units while the model works in standardized units, which made the warm start a large guess
rather than an informed one; the fix divides by the scale. The `huber` branch of `auto`
fits an `mse` model first and passes it as `warmStart`, so it is the objective this path
reaches. `mse` and `mae` are untouched, which the probe confirms: `warmStartFromBox` only
engages when `stationary || invertible`, and these campaigns ran with both `false`.

The full re-run of all eight `huber` cells under `v1.0.0` measures the size of it. The
selected order changes in 1.6% (yearly, `:zeroed`) to 15.4% (monthly, `:innovations`) of
series, and the aggregate barely moves: **|ΔOWA| ≤ 0.0083 in every cell, and ≤ 0.0010 in the
three large frequencies.** The visible gain is elsewhere — the count of fits that degrade
to `mse` falls from 32 to 21 across the eight cells.

### (c) `ridge` — the objective itself changed

Only 15 of 120 series match; 34 select a different order. This is a **behavioural change,
not a default change**, and no amount of argument declaration recovers it.

At the campaign commits the penalized-initialization branch covered `mse` and `ridge`
together and closed with

```julia
@objective(jumpModel, Min, S * fator)
```

where `S = RSS + presample + λ‖β‖²` and `fator` is the determinant factor
`∏ⱼ(1-κⱼ²)^(-j/T)`. On `v1.0.0`, `ridge` is its own branch and closes with

```julia
@objective(jumpModel, Min, sum(ϵ.^2) + presampleSquares(jumpModel, penalizado) + λ*sum(coefs.^2))
```

with **no determinant factor**. The shrinkage constant is unchanged —
`λ = sqrt(max(T - lb + 1, 1))` on both sides, over the same AR/MA coefficient set — so the
divergence is the determinant term, not the penalty. The commits that produced the archived
`ridge` cells sit on `feat/ridge-innovations` and `feat/ridge-determinante-sobre-dev`,
neither of which is an ancestor of the release.

The full re-run of the four `ridge` cells measures it. In the three large frequencies the
aggregate is small and not one-signed: **monthly +0.0024, quarterly −0.0010, yearly
−0.0021** — the largest is a quarter of the 0.01 threshold used elsewhere in this work —
even though a quarter of the monthly series select a different order (12 236 of 48 000,
17 984 worse against 17 959 better, a near-perfect cancellation with no concentration).

**Weekly is the exception, and it is three series.** ΔOWA is **+0.1431**, of which one
series accounts for 39.9%, three for 86.6% and five for 97.5%; head to head the two
implementations are a coin flip (110 worse, 108 better, 141 identical). The mechanism is
specific: without the determinant factor the fit drops the MA terms under `d = 2` — series
61 and 248 go from `(1,2,4)` to `(1,2,0)` and their sMAPE goes from ≈2 to 135 and 98. With
`d + D ≥ 2` and no MA term to damp the second difference, the extrapolation diverges. These
land in exactly the bucket the over-differencing guard admits, since it requires only one
ARMA term. The effect does not generalize: in monthly that bucket **shrinks** under
`v1.0.0` (1 829 → 1 598) and the mean ARMA term count barely moves (2.793 → 2.775).

### Scope of the probe

The probe covers monthly, on 120 series. The full re-runs cover all four frequencies for
`ridge` and `huber`; `mse` and `mae` were deliberately not re-run, since the probe shows
them unaffected and the code path that the release changed is not reached with
`stationary = false, invertible = false`. Nothing here establishes anything about the
`req_false_*` guard arms, which are not table cells.

## 8. Declared unknowns

Listed so they can be repaired rather than discovered later.

1. **Which numbered manuscript table each generated table corresponds to.** The manuscript
   is not on this host. The tables are named by their generated titles.

2. **The exact source of the runner for 18 of the 37 campaigns.** The experiment scripts
   live in the parent `ForecastTester` working tree and are **not under version control**
   there. The runner was edited in place on 2026-08-24 at 16:02 — after the `cel_*`,
   `censo_ridge_*` and `smoke_*` campaigns had run — and the earlier revision is
   unrecoverable. `scripts/run_cell.jl` and `scripts/cell_worker.jl` in this package are
   descendants of the surviving revision, with all arguments made explicit. What the lost
   revision *computed* is recoverable from the outputs and the log headers, and is what
   section 4 documents; that the reconstruction is byte-equivalent to the lost source is
   **not verified** for those campaigns, though the cross-checks in section 3 are consistent
   with it.

3. **The exact command line for the six production-rule campaigns.** PowerShell truncated
   the invocation in the captured logs. The cap rule stated in section 4 is the script
   default of that revision, consistent with the observed timings, but it is an inference.

4. **The solver statuses of discarded internal candidates**, as described in section 6.

5. **The Julia patch version for campaigns other than `obj_huber_innov_monthly`.** Version
   `1.10.12` is written directly in a stack trace captured in that campaign's log. For the
   others it is inferred — but the inference is tight: `~/.julia/juliaup` contains exactly
   one installed toolchain, `julia-1.10.12+0.x64.w64.mingw32`, dated 2026-08-22 18:06,
   before the first campaign started, and no other version was ever installed on this host.

## 9. Inputs not produced here

Two inputs are required to regenerate the tables and are not outputs of this host.

**The `auto.arima` and Naive2 baselines.** `results/baselines/` holds the per-series MASE
and sMAPE of `forecast::auto.arima` and of Naive2, for four frequencies and four horizon
blocks. They were produced on a **different host** on 2026-07-18 and copied here; this
package slims them to the three columns the tables need. R is not installed on this
machine, so they could not have been produced here and were not re-derived. Their
provenance — R version, `forecast` version, hardware — is **NOT KNOWN from this host** and
must come from whoever ran them.

**The M4 datasets.** `Monthly-train.csv`, `Monthly-test.csv` and their counterparts for the
other frequencies are the public M4 competition data. They are not redistributed here. The
row index of the training file is the series id used throughout this package and in the
baselines.

## 10. File integrity

`SHA256SUMS` covers every file in this directory. Verify with:

```bash
./reproduce.sh verify
```
