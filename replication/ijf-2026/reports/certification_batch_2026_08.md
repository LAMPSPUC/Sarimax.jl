# The August 2026 certification batch, Table S2 and Section 5.5: the records received on 2026-10-21

Reconciliation of the material received on 2026-10-21 (archived in
`results/historical/certification-2026-08/`, see its `PROVENANCE.md`) with the values the
2026-10-16 manuscript prints. It answers the requests listed in `provenance_gaps.md` (items 1, 3,
4, 5) and `missing_global_artifacts.md` (G1). Nothing was re-executed: the comparisons are
against the received record, computed by `scripts/maintainers/check_certification_2026_08.py`.

**What kind of evidence this is.** The per-instance raw outputs of the certification batch, and
the per-replication or per-fit outputs of the three diagnostics, are lost: they ran in a
temporary directory that no longer exists. What was received is a reconstruction from the record
of the session that ran them (console tables, and the harness rebuilt from the session's patches
and checked by execution), with the source of every row declared. The manuscript mapping
therefore classifies these values as **derived-only**: they agree, or not, with a recovered
summary; none is reproduced from raw records.

## Effect on the manuscript inventory (2026-10-16 transcription)

| category | before | after |
|---|---|---|
| raw-reproduced | 603 | 603 |
| derived-only | 19 | 56 (+37: Table S5 12, Section 5.6 8, Table S2 9, Section 5.5 8) |
| differing | 2 | 6 (+4: the Section 5.6 seasonal-grid figures, registered errata) |
| no evidence | 46 | 5 (Section 5.5 ranking: 12.1%, 16.5%, OWA +0.0097; Table S3: 2) |
| checked at run time only | 2 | 2 |

## 1. The certification batch (Table S5, Section 5.6)

**What exists.** The 46 instances, complete (`casos()` of the harness: batch 1, ids 1-32, cells
A-D, run 2026-08-21; batch 2, ids 33-46, cells E-G, run 2026-08-22), with T, order, mode,
data-generating process and seed; real M4 series 2772, 7389, 7510 (ids 28-30). Status, time and
certified relative gap for 44 instances. Objectives: Ipopt for 9 instances (26, 33-38, 44, 45),
SCIP and dual bound for 7 (26, 33, 36, 37, 38, 44, 45); batch 1 records Ipopt - SCIP for all 32.
The harness: lifted products `uϕΦ = ϕΦ`, `uθΘ = θΘ` in boxes `±B²`, `B = 1 - 1e-6`; innovation
bounds `|ε_t| <= sqrt(f_Ipopt)·(1 + 1e-9)` with `f_Ipopt` the Ipopt objective of the same model;
full warm start. Environment: Julia 1.12.3, JuMP 1.31.1, SCIP.jl 0.11.6 on SCIP 8.0.0, Ipopt.jl
1.15.0; SCIP `limits/time = 600` and `display/verblevel = 0`, nothing else; batch 1 in 8
processes of 4 instances, batch 2 in 5 processes; serial BLAS.

**What is lost or never existed.** Node counts (never recorded); SCIP logs (never written);
SCIP's raw status beyond JuMP's `termination_status`; the Ipopt coefficients and status; primal
and dual bounds of the 37 instances without objectives. Ids 34 and 35 (T = 480) never finished:
34 ran 7 h 40 min under a 600 s `limits/time` before an external 900 s clock was added, and both
were killed by it. The dual bound of id 37 is SCIP's `-1e+20` sentinel: no valid bound.

**Table S5: the counts agree.** The record does not say which instances each row counts. The
assignment by the harness's own cells reproduces every count:

| Table S5 row | certified / time limit (printed) | instances | record |
|---|---|---|---|
| non-seasonal grid, T in {30, ..., 240} | 12 / 0 | cell A, ids 1-12 | 12 / 0 |
| extended, T = 480 | 1 / 0 | id 33 (34, 35 unfinished) | 1 / 0 |
| extended, T = 960 | 1 / 2 | ids 36-38 | 1 / 2 (37 without a valid dual bound) |
| seasonal grid, T in {36, 60} | 0 / 12 | cell B, ids 13-24 | 0 / 12 |
| seasonal/adversarial diagnostics | 2 / 4 | cell C, ids 25-30 | 2 / 4 |
| free pre-sample mode, T in {60, 120} | 3 / 0 | ids 31, 32 (cell D), 46 (cell G) | 3 / 0 |

Two caveats the text should reflect (textual claims T16, T17):

* **The T = 480 row omits two unfinished instances.** Three were run; one certified (136 s).
* **The assignment is not unique, and either one leaves out instances the text relies on.** Under
  the harness's cells, the diagnostics row counts the three M4 series (ids 28-30), which failed
  the harness's own noise-floor assertion (SCIP objective below `0.25 (T - 1)`, which the harness
  marks as a wrong box). Ids 25-27 with the seasonal cancellation cases 39, 41, 43 of cell F give
  the same 2 / 4. Either way the table counts 37 of the 46 instances and leaves out most of cell F,
  including **id 44, the first certificate of Section 5.6**; id 45 (the second) is counted only
  if it replaces 31 or 32 in the last row.

**Section 5.6: the instance values agree, the seasonal-grid figures do not.**

| printed | record | status |
|---|---|---|
| near-cancellation instance: Ipopt 46.660913, SCIP feasible 45.314976 | id 26 (cell C, (1,0,1)(1,0,1)_12, Φ = 0.700, Θ = -0.699, T = 60): 46.660913, 45.314976, dual 30.738742, TIME_LIMIT | agrees |
| certified instance 1: local 57.868044, optimum 57.536983 (0.575%) | id 44 (cell F, regular cancellation, slack 5e-2, `:zeroed`): 57.868044, 57.536983, gap 1.09e-9, OPTIMAL in 521 s | agrees |
| certified instance 2: local 54.706978, optimum 53.681295 (1.911%) | id 45 (cell G, regular cancellation, `:innovations`): 54.706978, 53.681295, gap 1.07e-8, OPTIMAL in 36 s | agrees |
| seasonal grid: incumbent 2.5% below the local objective | cell B: largest Ipopt - SCIP difference 2.52e-9 (absolute): the incumbent is the Ipopt point | **differs** (erratum) |
| seasonal grid: smallest remaining gap 1.9% | 1.94e-4 relative, 0.019% | **differs** (erratum) |
| seasonal grid: largest remaining gap 4.9% | 4.90e-1 relative, 49% | **differs** (erratum) |
| seasonal grid: median remaining gap 1.2% | 1.51e-2 relative, 1.5% (median of 12; the 6th value is 1.23e-2) | **differs** (erratum) |

The printed seasonal figures carry the mantissas of the recorded values with the exponents lost:
`2.5e-9` became 2.5%, `1.9e-4` 1.9%, `4.9e-1` 4.9%. The corrections proposed are in
`results/manuscript/errata.csv`; the wording of the paper is decided by its authors.

**Re-running: the batch's instances are recovered exactly.** The harness draws its synthetic
series from `MersenneTwister` (the coefficients through `Base.hash((cell, draw, P, Q))`, the
series from `MersenneTwister(1000 + id)`). `Base.hash` gives the same values under Julia 1.10
and 1.11, but MersenneTwister's seeding changed in Julia 1.11: under the replication environment
(Julia 1.10.12) the harness draws other series, and its Ipopt objectives differ from the record
for every synthetic instance. Run under Julia 1.11.9, the harness's own data code
(`experiments/runners/global/certification_2026_08_instances.jl`) gives series that, fitted
with the harness's model in the replication environment, reproduce **every Ipopt objective the
record holds**: ids 5, 26, 27, 33-38, 44 and 45 to the printed precision (to 5e-7 where six
decimals were printed), and the three M4 series (ids 28-30, read from the M4 file, no random
draw) to within 4e-5 of the SCIP objectives the harness printed. The 46 instances are therefore
written as data (`experiments/configs/certification_2026_08_instances.csv`), and
`experiments/runners/global/run_certification_2026_08.jl` re-runs the batch on them with the
archived harness loaded verbatim, recording what the batch did not: SCIP's raw status, node
count, coefficients and one log per case, and Ipopt's status and coefficients
(`./reproduce.sh full --experiment global_optimization_grid --jobs N`; then
`compare_certification_2026_08.py` compares the run with the record and recomputes Table S5 and
the Section 5.6 figures). A run is a new execution: SCIP's times and the gaps of time-limited
instances depend on the machine and the number of parallel jobs. **Not run yet:** SCIP needs the
Linux container (the Windows build crashes), which was not available where this was prepared.

## 2. Table S2: the exogenous-convention simulation (LAMPSPUC/Sarimax.jl#18)

All nine printed values agree with the per-DGP aggregate printed by the simulation's reader
(`resultados_agregados/h_discriminador_3dgp_20-08_1350.txt`). Archived: the generator
`diag_gera.jl` (seeds 9000 + s, s = 1..20; T = 180, H = 12, three regressors, b = [2.0, -1.5,
0.8]), the Julia and R fitters (`diag_jl.jl`; `diag_R.R` with `forecast::Arima(xreg =, method =
"ML")`) and the reader. The per-replication outputs are lost; the generator is deterministic, so
a port to a runner would re-create them as a new execution.

## 3. Section 5.5: the initialization diagnostic (LAMPSPUC/Sarimax.jl#20)

All five printed values agree with the records:

* **0.2076 -> 0.0863** are arm A (`:penalized`, free pre-sample, no cross term) and arm C
  (`:innovations`, no determinant term) of the 10-seed simulation (`g4.jl`, `g4p.jl`; T = 200,
  `MersenneTwister(500 + r)`, target R `arima(method = "ML")` on the same data). The per-seed
  table is archived and its medians reproduce the printed ones. The same runs report arms B and D
  (with the cross term or the determinant: 0.4294 and 0.7984); the text compares A with C.
* **47.4% -> 0.2%** of fits at the 200-iteration limit and **-28%** wall time (0.848 s -> 0.609 s
  per series; 1,273 s -> 914 s in total) are the paired 1,500-series gate at commit `f90e554`,
  whose runner was reconstructed (`m4_runner.jl` of 2026-08-14 with the `M4_INIT`
  parametrization). Only the aggregate survives; the per-fit outputs are lost.

## 4. Section 5.5: the likelihood-ranking diagnostic (LAMPSPUC/Sarimax.jl#17)

The paragraph merges two measurements, and only part of it is recovered:

* **1,099 paired fits** is the AICc-parity comparison with R on the seasonal P x Q grid
  (`aicq.jl`, `aicq.R`; the first 100 series of a 400-series sample whose ids are in
  `rand400.csv`): the number of parameters agrees in 1,099 of 1,099 fits.
* **21.6 -> 3.3** is a different measurement (`mixdist.jl`): the median CSS -> ML log-likelihood
  distance with the `seasonalPresamplePrior` flag off (n = 14) and on (n = 12), **in the same
  revision (2026-08-16), not paired**, and on 14 and 12 fits, not 1,099. "Before" and "after"
  are the flag, not two revisions.
* **12.1% -> 16.5%** mixed seasonal orders and **OWA +0.0097** are not recovered. The
  collaborator points to `g_rand400.csv` and `fix_rand400.csv` (archived); no evident definition
  of a mixed seasonal order (P > 0 and Q > 0, over all series, over seasonal series or over
  seasonally differenced series) gives 12.1% or 16.5% from them (they give 3.8%-12.5%), and the
  OWA needs the Naive2 baseline, which they do not hold. The summary they cite
  (`FRONTEIRA_FINAL.md`, sections 15-16) was not supplied. These three values stay without
  evidence.

## What remains open

* Table S5 and Section 5.6: the per-instance raw records (lost); which instances Table S5 counts;
  the four seasonal-grid figures (errata); ids 34-35 and 44 in the text (T16, T17).
* Section 5.5: the source of 12.1% -> 16.5% and OWA +0.0097, and the wording that merges two
  measurements (T13).
* Porting the batch's instances and the diagnostics' scripts to runners of this package
  (new executions, on approval).
