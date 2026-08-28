# Reproduction contract

This document is the contract. It states, for every result in `results/raw/`: which table
it produces, the exact configuration it was produced under, the package commit that
produced it, the hardware and wall time it cost, whether re-running it today reproduces it,
and what is known not to be reproducible and why.

Nothing here is uniformised. Where campaigns ran under different commits, different solver
stacks, or different semantics for the same keyword, they are reported that way.

---

## 0. Two version numbers that must not be conflated

There are two distinct facts about every result, and they are different values.

**The commit a result was PRODUCED UNDER.** Historical, per campaign, and recorded in the
`sarimax_commit` column of every row. It differs between campaigns in this directory: the
frequencies of the headline benchmark alone span two commits. This is the value that
answers "what code computed this number".

**The package version this material is PUBLISHED ALONGSIDE.** A single value, the same for
the whole directory. This is the value that answers "what release does this material
document".

At the time of writing, the second value **does not yet exist**. This material is staged
against `dev`; no release tag has been cut. Every statement below therefore names a commit,
never a release. When the release is cut, the publication version can be added to this
section — it must not be back-written into the per-campaign rows, and no result may be
described as "produced under" it.

Every verification reported in section 3 was run against **`3d7f883`**, the `dev` head at
consolidation time. A release cut later than that will contain changes not covered by these
checks; see section 6.

---

## 1. Environment

| | |
|---|---|
| Machine | Intel i7-5820K, 6 cores / 12 threads, 3.30 GHz, 64 GB RAM |
| OS | Windows 10 Pro (build 19045), x86_64 |
| Julia | 1.9.4 |
| Parallelism | 10 worker processes, `BLAS.set_num_threads(1)` in each |

`env/Project.toml` and `env/Manifest.toml` pin the environment. **Both manifests are
committed and neither was regenerated**: they are the files that were on disk, not fresh
resolutions made to look like originals.

Two manifests are shipped because the dependency stack changed mid-way through the
campaigns:

| file | MathOptInterface | Ipopt | JuMP | SCIP | HiGHS |
|---|---|---|---|---|---|
| `env/Manifest.toml` | 1.48.0 | 1.6.3 | 1.22.2 | 0.11.14 | 1.9.1 |
| `env/Manifest-moi131.toml` | 1.31.0 | 1.6.3 | 1.22.2 | 0.11.14 | 1.9.1 |

MathOptInterface 1.31.0 does not recognise that the residual block of the objective is
diagonal and declares the whole Lagrangian Hessian as possibly non-zero. On one
representative cell the non-zero count is 133,920 under 1.31.0 against 2,592 under 1.48.0.
The effect is on **cost, not on the solution** — which is verified empirically in section 3,
not assumed. Use `env/Manifest.toml` unless reproducing timings.

### What the environment attribution can and cannot claim

**No campaign in this directory recorded its solver stack at run time.** The original
provenance stamp captured the package commit, the working-tree state and the Julia version,
but not the versions of MathOptInterface, Ipopt or JuMP. The `moi`, `ipopt` and `jump`
columns in `results/raw/` therefore read `not_recorded` for every row. They are not filled
in with a reconstruction, because a reconstruction in a data column is indistinguishable
from a measurement.

The reconstruction, with its evidence, is here instead:

| campaign | reconstructed MathOptInterface | evidence | confidence |
|---|---|---|---|
| A — yearly, quarterly | 1.31.0 | ran 22 Aug; the manifest was updated to 1.48.0 on 23 Aug and the pre-update file was preserved as `Manifest-moi131.toml`. These two frequencies were never re-run afterwards, while weekly/hourly/daily were re-run specifically to move onto 1.48.0 | high, circumstantial |
| A — weekly, hourly, daily, monthly | 1.48.0 | ran after the manifest update; the three re-runs were named for it | high, circumstantial |
| D, E, F | 1.48.0 | ran 24–26 Aug, after the update | high, circumstantial |
| B, C | unknown | ran 18–19 Aug, before the preserved pre-update manifest was written. Whether the stack was unchanged between those dates and 23 Aug cannot be established from anything on disk | **unknown — do not assert** |

---

## 2. Campaigns

Each campaign's configuration is stated twice on purpose: once in the runner's
configuration function, and once here in prose. A table legend and the code can then be
checked against each other without reading Julia.

Common to every campaign below, unless stated otherwise:

- objective `mse`; multiplicative seasonal form
- `stationary = true` with domain margin `1e-6`; `invertible = false` with domain margin
  `1e-6`; `assertStationarity = true`; `assertInvertibility = true`; root-rejection margin
  `1e-2`; `constrainedRefit = false`
- search `stepwise`, criterion `aicc`, differencing tests `kpssShort` / `seas`
- order bounds `maxp = 5`, `maxq = 5`, `maxP = 2`, `maxQ = 2`, `maxOrder = 5`,
  `maxd = 2`, `maxD = 1`; `d` and `D` selected by test (`-1`)
- `multistart = false`, `warmStartFromBox = false`, `outlierDetection = false`,
  `requireTermsWhenOverDifferenced = false`, `requireMAWhenDoublyDifferenced = false`
- `cvarLevel = 0.9` (inert unless the objective is `stable`)
- optimizer: Ipopt, at the version pinned by the manifest
- `exogDynamics`: **not applicable.** Every series here is univariate and no `exog` is
  passed, so the exogenous-equation semantics never engage. It is deliberately absent
  rather than pinned to a value that would not describe anything.

### A — M4 benchmark, `initialization = :innovations`

**Table produced:** the headline accuracy table — this package against
`forecast::auto.arima`, per frequency, split into short / medium / long horizon blocks and
a total. Generator: `tables/make_table_m4.jl`. The manuscript's name for this table is not
recorded on this machine; see section 5.

**Differs from the common configuration in:** `initialization = :innovations`;
`maxTimeSeconds = nothing` (**no cap**); whole series used for fitting (no history
truncation).

The absence of a cap matters and is not a detail. With a 120 s cap, 35% of weekly series
truncate and the truncated ones carry 86% of the gap against the reference — the number
becomes a property of the clock rather than of the method.

**This campaign spans two commits. It is not uniform and is not reported as if it were.**

| frequency | rows | commit | reconstructed MOI | CPU | wall @10w | errors | non-`LOCALLY_SOLVED` | censored |
|---|---|---|---|---|---|---|---|---|
| yearly | 23,000 | `aa68d57` | 1.31.0 | 1.4 h | 0.15 h | 0 | 0 | n/a — no cap |
| quarterly | 24,000 | `aa68d57` | 1.31.0 | 14.5 h | 1.45 h | 0 | 0 | n/a — no cap |
| weekly | 359 | `aa68d57` | 1.48.0 | 0.5 h | 0.05 h | 0 | 0 | n/a — no cap |
| hourly | 414 | `aa68d57` | 1.48.0 | 4.7 h | 0.47 h | 0 | 0 | n/a — no cap |
| daily | 4,227 | `aa68d57` | 1.48.0 | 9.1 h | 0.91 h | 0 | 0 | n/a — no cap |
| monthly | 48,000 | `3d7f883` | 1.48.0 | 51.3 h | 5.13 h | 0 | 0 | n/a — no cap |

Every fit in all six frequencies returned `LOCALLY_SOLVED`; the error rate is 0 across
99,997 series. Because there is no time cap, the censoring rate is zero by construction —
the long tail is a cost property, not a truncation. For reference, fits exceeding 120 s:
hourly 22 (5.3%), quarterly 17, monthly 16, daily 3, yearly 1, weekly 0.

**What differs between `aa68d57` and `3d7f883`:** 189 non-comment lines in
`src/models/sarima.jl`, including the change of the `initialization` default from `:zeroed`
to `:innovations`. That default change cannot affect this campaign, which passes
`initialization` explicitly. The objective block actually used — `mse` under a penalised
pre-sample — is **byte-for-byte identical** between the two commits.

### B — Objectives, M4 monthly

**Tables produced:** the win rate against the reference on a random sample of the
population, and the comparison between objective functions at fixed initialization.
Files: `results/raw/objectives_monthly.csv` and `objectives_monthly_penalized.csv`.

**Differs from the common configuration in:** `maxTimeSeconds = 120.0`; per-cell objective
and initialization as below. Sample: `Random.seed!(20260819)`, 4,000 series for the `mse`
cell, a nested 1,500-series subsample for the others.

| cell | objective | initialization | rows |
|---|---|---|---|
| `mse` | mse | `:penalized` | 4,000 |
| `mse_zeroed` | mse | `:zeroed` | 1,500 |
| `ridge` | ridge | `:zeroed` | 1,499 |
| `mae` | mae | `:zeroed` | 1,500 |
| `huber` | huber | `:zeroed` | 1,500 |
| `ridge_pen` | ridge | `:penalized` | 528 |
| `mae_pen` | mae | `:penalized` | 542 |
| `huber_pen` | huber | `:penalized` | 529 |

Commit: **not recorded.** These two files predate the provenance stamp. Cross-referencing
their write times (19 Aug, 03:31 and 14:24) against the commit log places them under
`4e5faaf`, whose window runs from 19 Aug 00:05 to 20 Aug 11:13. That is an inference from
file timestamps, not an attestation, and the `sarimax_commit` column reads `not_recorded`
accordingly.

Cost: `objectives_monthly.csv` 32.0 CPU-h (3.2 h wall); `objectives_monthly_penalized.csv`
55.4 CPU-h (5.5 h wall). Errors: 0 in both.

**Censoring is severe in the `_pen` file and must be reported with any number drawn from
it:** 389 of 1,600 fits (**24.3%**) reached the 120 s cap. In `objectives_monthly.csv` the
rate is 99 of 10,000 (1.0%).

**The three `_pen` cells did not measure what their name says.** See section 4, finding 2.

### C — Multistart, M4 monthly

**Table produced:** the paired comparison of deterministic multistart against the single
zero start. File: `results/raw/multistart_random.csv`.

**Differs from the common configuration in:** `initialization = :penalized`;
`maxTimeSeconds = 120.0`; `multistart` is the treatment axis, `false` in the `base` arm and
`true` in the `multi` arm. Sample: `Random.seed!(20260818)`, 400 series, both arms on each
series.

Commit: **not recorded**; timestamps place it under `07662a7` (18 Aug 15:35 to 19 Aug
00:05). Same caveat as campaign B.

Cost: 6.9 CPU-h, 0.7 h wall. Errors 0. Censoring: 34 of 800 fits (4.3%) at the 120 s cap.

### D — Same-commit isolation

**Table produced:** the distance between the production configuration and `:innovations`,
both arms built from one package tree. File: `results/raw/isolation.csv`.

This campaign exists because that distance had only ever been measured across campaigns
three weeks apart, and three weeks of package code were measured to move weekly OWA by
0.0320 — larger than the distance being estimated. Any such cross-campaign difference
confounds the treatment with version drift.

Arms: `production_v0_6`, which calls the frozen wrapper in `scripts/wrapper_v0_6.jl`; and
`innovations`, which is campaign A's configuration exactly. Frequencies weekly (359) and
yearly (23,000), both arms on every series: 46,718 rows.

Commit `3d7f883`. Cost 5.6 CPU-h, 0.6 h wall. Errors 0. The production arm caps short
series at 120 s and 8 of its 23,359 fits (0.03%) reached that cap; the `innovations` arm is
uncapped and has none.

**This campaign carries an unrecoverable provenance defect.** See section 4, finding 3.

### E — Axes 2×2, weekly

**Table produced:** the isolated and combined effect of the two production-wrapper axes
that were otherwise unexplained. Files: `results/raw/axes_weekly.csv` (commit `aa68d57`)
and `axes_weekly_earlier_commit.csv` (commit `6e8e0bd`).

Cells: `base`, `req` (`requireTermsWhenOverDifferenced = true`), `tests`
(`integrationTest = "kpssShort"`, `seasonalIntegrationTest = "seas"`), and `req+tests`.
359 weekly series × 4 cells = 1,436 rows per file. No time cap.

**The `base` and `req` cells deliberately leave the two differencing-test keywords to the
package default** — that is the treatment, not an oversight. At both commits the defaults
were `integrationTest = "kpssShort"` and `seasonalIntegrationTest = "seas"`, which is why
those two axes were measured to be inert on this frequency.

Cost: 1.7 and 2.8 CPU-h. Errors 0 in both. No cap, so no censoring; 8 fits in the
earlier-commit file exceeded 120 s.

### F — `stable` objective against `mse`

**Table produced:** the three-arm paired comparison of the CVaR-of-squared-residuals
objective against `mse`. File: `results/raw/stable_weekly_yearly.csv`.

**Differs from the common configuration in:** `objectiveFunction` and `cvarLevel` per arm.

| arm | objective | cvarLevel |
|---|---|---|
| `mse` | mse | 0.9 (inert) |
| `stable_090` | stable | 0.9 |
| `stable_050` | stable | 0.5 |

Frequencies weekly (359) and yearly (23,000), whole populations, three arms each: 70,077
rows. Commit `3d7f883`. Cost 17.2 CPU-h, 1.7 h wall. Errors 0. No time cap; 14 fits
exceeded 120 s.

**The runner on disk could not regenerate this file, and the third arm has been
reconstructed.** See section 4, finding 1.

**Coverage is weekly and yearly only.** Runs extending this campaign to daily, quarterly
and the remaining frequencies were attempted and failed — see section 5.

---

## 3. Does it reproduce?

The question: running the script **with every argument explicit**, under the current code,
does it reproduce the archived numbers?

Method: `scripts/verify_reproduction.jl` draws a seeded sample of series already present in
each archived file, re-fits them under the explicit configuration and the current tree, and
compares sMAPE and selected order against the stored values. This re-fits a sample; it does
not re-run a campaign.

All checks below ran against **`3d7f883`** under `env/Manifest.toml` (MathOptInterface
1.48.0). No release tag existed at consolidation time, so no check can be reported against
one.

| campaign | reproduces? | evidence |
|---|---|---|
| **A — M4 benchmark** | **Yes** | Two independent checks, below. |
| **B — objectives, five main cells** | *pending* | |
| **B — objectives, three `_pen` cells** | **No, and the reason is known** | Behaviour change, not a default change. See section 4, finding 2. |
| **C — multistart** | *pending* | |
| **D — isolation** | *see the note below the table* | |
| **E — axes** | **Yes** | 100 of 100 sampled series identical across all four cells (25 each), sMAPE and selected order exact, delta 0.0 throughout. Archived at `aa68d57`, re-fitted at `3d7f883`. |
| **F — stable** | *pending* | |

### Campaign A, check 1 — seeded sample

150 of 150 sampled series identical, 25 per frequency, sMAPE to four decimals and selected
order both exact, delta 0.0 throughout. This covers the two-commit split and both solver
stacks: the yearly and quarterly rows were produced at `aa68d57` under MathOptInterface
1.31.0 and were re-fitted at `3d7f883` under 1.48.0 with no difference.

### Campaign A, check 2 — full population, and it was free

Campaign F's `mse` arm is, by design, campaign A's configuration re-measured. The two were
run by different runners, on different days, at different commits: campaign A's weekly and
yearly rows at `aa68d57`, campaign F's control arm at `3d7f883`. Comparing them series by
series is therefore a complete cross-commit reproduction check that costs no computation at
all:

| frequency | series compared | differing |
|---|---|---|
| weekly | 359 | **0** |
| yearly | 23,000 | **0** |

Largest absolute difference in either sMAPE or MASE across all 23,359 pairs: **0**. The
aggregate agrees too — weekly sMAPE 8.3979 and MASE 2.4403, yearly 15.2441 and 3.4067, in
both campaigns.

This is stronger than the sampled check and it settles finding 4 empirically: the commit
split inside campaign A, and the MathOptInterface change under it, do not move the numbers.
The correct attribution still has to be stated per frequency, because the claim being
verified here is "the results agree", not "the results came from the same place".

It also verifies the `mse` control arm the campaign F design depends on: an arm intended to
reproduce the headline table, which does.

---

## 4. Divergences between what a script does and what its results claim

These are the defects found during consolidation. They are listed before the convenience
material because they are what a reviewer will test first.

### Finding 1 — the `stable` runner could not regenerate its own results

`results/raw/stable_weekly_yearly.csv` contains three arms: `mse`, `stable_090`,
`stable_050`, 23,359 series each. The runner as found in the harness working tree declared
only two:

```julia
const ARMS = [("mse","mse",0.9), ("stable_050","stable",0.5)]
```

The `stable_090` arm had been removed from the runner after the run completed. The file on
disk therefore produced two arms where the data has three, while the companion analyser
still required all three by name. Running the pair as found produces no table.

**Resolution:** the arm is restored in `scripts/run_stable.jl`, and the restoration is
announced in a banner at the top of that file rather than applied silently. Three
independent sources attest it: the arm label in the data, the analyser's required arm list,
and the runner's own header text and log line ("3 arms").

**What remains unverified:** `cvarLevel` was never written to the archived rows. The value
0.9 for `stable_090` is inferred from the arm label and from the package default at that
commit. It is **not** established by the artefact. The new runner writes `cvar_level` as a
column so this cannot recur.

### Finding 2 — three cells measured a different estimator than their name says

The `ridge_pen`, `mae_pen` and `huber_pen` cells of campaign B pair a non-`mse` objective
with `initialization = :penalized`. What that combination means changed after the campaign
ran, with no change to the script:

- **At the campaign commit**, the penalised pre-sample treatment was implemented for `mse`
  only. The package emitted a `@warn` and the fit fell through to the ordinary branch — it
  became `:free`. Under a ten-worker run a `@warn` is invisible in the log. **All three
  cells measured `:free`.**
- **Under the current code**, the pre-sample block covers nine objectives including these
  three. The same cells now genuinely receive the penalised block and the guard does not
  fire.

So re-running produces different numbers for the same cell names, and the difference is a
change of estimator, not of default.

**Consequence for the manuscript:** any legend describing these cells as `:penalized`
describes something that did not run. Either the legend says `:free`, or the rows are
regenerated under the current code and relabelled. This cannot be settled inside the
replication package — it is a claim in the paper.

Compounding it: 24.3% of the fits in that file reached the 120 s cap.

### Finding 3 — the provenance stamp checked the wrong scope, and a production wrapper ran uncommitted

The original stamp verified that the **package** working tree was clean. It did not check
the **harness** repository, where the production wrapper lives.

That wrapper carried uncommitted modifications from shortly after the last harness commit
through the whole campaign window. The modification changes the short-series trigger:

```
committed in the harness repo   short = length(y) <= 150            (fixed)
what actually ran               short = length(y) <= 5 * (5 + 2s)   (relative)
```

At `s = 1` the thresholds are 35 and 150 — not a cosmetic difference on yearly or weekly,
where the branch selects between `:free` and `:zeroed` and changes the time cap.

Every affected output nonetheless reads `tree=clean`, because the stamp never looked. This
affects campaign D's `production_v0_6` arm and any earlier production run.

**Resolution:** `scripts/provenance.jl` now checks both trees and records both in every row
(`sarimax_tree`, `harness_tree`). `scripts/wrapper_v0_6.jl` freezes the version that
actually ran, with a banner saying so.

**What cannot be resolved:** the harness repository state for the archived runs is
unrecoverable after the fact. Those columns read `not_recorded` and will stay that way.

### Finding 4 — the headline table spans two commits and two solver stacks

Reported in full in section 2A. It is listed here because a summary elsewhere recorded the
entire table as a single commit and a single MathOptInterface version, which is wrong for
five of six frequencies on the commit and for two on the solver stack. The empirical check
in section 3 shows the discrepancy does not move the numbers — but it had to be checked,
not assumed, and the correct attribution is the one in section 2A.

### Finding 5 — internal search failures are not instrumented anywhere

The per-fit telemetry counters are recorded **per model**, and `auto` returns a single
candidate — the winner. Any candidate that failed inside the search is absorbed and appears
nowhere. Measured on a representative cell: wall 13.81 s against 0.07 s of accounted solve
time and `fitCount` 1.

**Consequence:** the count of internal `OTHER_ERROR` and `ITERATION_LIMIT` outcomes cannot
be produced from the archived artefacts. It is not a column that was dropped; it was never
captured. Obtaining it requires instrumenting the search loop and re-running. The
`solver_status` column that does exist describes **the returned candidate only**, and the
`LOCALLY_SOLVED` rates in section 2 must be read with that scope.

---

## 5. Declared unknowns

Listed rather than estimated.

1. **Which manuscript table each script feeds.** The manuscript is not on this machine.
   Section 2 names the generator and the output file for each campaign, which is as far as
   the attribution can be carried from here.
2. **The commit for campaigns B and C.** Inferred from file timestamps against the commit
   log; recorded as `not_recorded` in the data.
3. **The solver stack for campaigns B and C.** Both predate the earliest preserved
   manifest. Unknown.
4. **`cvarLevel` for the `stable_090` arm.** Inferred from the arm label; never recorded.
5. **Whether changes between `aa68d57` and `3d7f883` outside the objective block affect the
   `mse` + `:innovations` path.** The objective block is verified byte-identical and the
   sampled re-fits are exact, which is strong evidence of no effect; a line-by-line audit of
   the remaining 189 changed lines was not performed.
6. **The harness repository state for every archived run.** Unrecoverable; see finding 3.
7. **`stable` beyond weekly and yearly.** Extensions to daily and quarterly were attempted
   three times and failed — twice with solver-level crashes in the linear solver, once with
   a worker stack-size failure. No output was produced. The empty files those attempts left
   behind are not shipped.

---

## 6. Known not to be bit-for-bit reproducible

- **Timings.** Wall and CPU times are machine and load dependent, and the runs shared the
  machine. They are reported for order of magnitude and for paired ratios, not as
  benchmarks. Campaigns A, D, E and F additionally span two MathOptInterface versions whose
  cost characteristics differ by orders of magnitude on long series; the accuracy figures
  do not depend on this, the timings do.
- **Anything from campaign B's `_pen` cells** under current code — by construction, see
  finding 2.
- **Anything depending on the production wrapper** where the harness checkout differs from
  the frozen copy in `scripts/` — see finding 3.
- **A release cut after `3d7f883`.** All verification in section 3 was run against that
  commit. Changes merged afterwards are not covered, and the two pull requests open at
  consolidation time are not covered. Re-running `scripts/verify_reproduction.jl` against
  the release is the check that closes this gap, and it costs minutes.

Determinism within a fixed commit and manifest: every sampled re-fit reproduced exactly, to
four decimals of sMAPE and to the selected order. Sampling in campaigns B, C and F is
seeded and the seed is written to every row. No stage is known to be non-deterministic; the
parallel scheduler affects the order rows are written, not their contents.
