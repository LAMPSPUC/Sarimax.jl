# The August 2026 certification batch, Table S2 and Section 5.5: the records received on 2026-10-21

What the material received on 2026-10-21 (archived in
`results/historical/certification-2026-08/`, see its `PROVENANCE.md`) contains and which results
it supports (`provenance_gaps.md`, `missing_global_artifacts.md`, G1). Nothing was re-executed
here: the values are computed from the received record by
`scripts/maintainers/check_certification_2026_08.py`, which writes them to
`results/historical/certification-2026-08/manuscript_sources.csv`; the manuscript mapping
compares the submitted manuscript's printed values with them.
The batch's author re-ran the certification batch on 2026-10-21; that re-run is compared with the
record in section 1.

**What kind of evidence this is.** The per-instance raw outputs of the certification batch, and
the per-replication or per-fit outputs of the three diagnostics, are lost: they ran in a
temporary directory that no longer exists. What was received is a reconstruction from the record
of the session that ran them (console tables, and the harness rebuilt from the session's patches
and checked by execution), with the source of every row declared. The manuscript mapping
therefore classifies these values as **derived-only**: they agree with a recovered summary;
none is reproduced from raw records.

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

**Table S5.** The submitted table lists the 46 instances by the harness's cells (A-G), with the
outcome counts of the August record and of the 2026-10-21 re-run. The record gives, by cell,
OPTIMAL / TIME_LIMIT / not returned: A 12/0/0, B 0/12/0, C 2/4/0, D 2/0/0, E 2/2/2, F 3/3/0,
G 2/0/0, in total 23/21/2 (quantities `s5_cell_*`), as printed. Id 44, the first certificate of
Section 5.6, is in cell F; ids 34 and 35 (T = 480) are the two instances that did not return.

**Section 5.6: the instance values and the seasonal-grid figures agree.**

| printed | record | status |
|---|---|---|
| near-cancellation instance: Ipopt 46.660913, SCIP feasible 45.314976 | id 26 (cell C, (1,0,1)(1,0,1)_12, Φ = 0.700, Θ = -0.699, T = 60): 46.660913, 45.314976, dual 30.738742, TIME_LIMIT | agrees |
| certified instance 1: local 57.868044, optimum 57.536983 (0.575%) | id 44 (cell F, regular cancellation, slack 5e-2, `:zeroed`): 57.868044, 57.536983, gap 1.09e-9, OPTIMAL in 521 s | agrees |
| certified instance 2: local 54.706978, optimum 53.681295 (1.911%) | id 45 (cell G, regular cancellation, `:innovations`): 54.706978, 53.681295, gap 1.07e-8, OPTIMAL in 36 s | agrees |
| seasonal grid: largest absolute difference between SCIP incumbent and Ipopt objective 2.52e-9 | cell B: largest Ipopt - SCIP difference 2.52e-9: the incumbent is the Ipopt point | agrees |
| seasonal grid: remaining normalized gaps 1.94e-4 to 4.90e-1, median 1.51e-2 | 1.94e-4, 4.90e-1, median of 12 1.51e-2 (the 6th value is 1.23e-2) | agrees |

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
instances depend on the machine and the number of parallel jobs. This runner has not been run
in the replication container (SCIP needs Linux; the Windows build crashes).

**The re-run of 2026-10-21.** The batch's author re-ran the 46 instances with the batch's harness
(recording extended, model unchanged) in the batch's environment (Julia 1.12.3, the batch's
Manifest, SCIP 8.0.0), under an expectation written beforehand; it is archived as
`results/reruns/certification-2026-08-rerun-2026-10-21/` (`RUN.md`) and checked in CI by
`scripts/maintainers/check_certification_rerun_2026_10.py`. The 43 series it wrote are the
package's instances exactly. `import_certification_rerun_2026_10.py` writes it in the runner's
format and `compare_certification_2026_08.py` compares it with the record:

| | record (August) | re-run (2026-10-21) |
|---|---|---|
| certified (OPTIMAL) | 23 | the same 23, certified gap equal to three digits in all; certified objective equal where recorded (33, 36, 44, 45) |
| time limit | 21 | 20 of them again; id 38 (T = 960) did not return in the queue, and stopped at the time limit when run alone (2026-10-22) |
| never returned | 34, 35 (T = 480) | 34, 35 and 38 in the queue (3-4 SCIP processes at once), under a 900 s and then a 3600 s clock; run alone, 34 stops at the time limit and 35 stays in SCIP's presolve (`completesol` heuristic), measured |
| Ipopt objective / Ipopt - SCIP | 9 / 32 recorded | 9 / 32 agree |
| Table S5, by cell (OPTIMAL / TIME_LIMIT / not returned) | 23/21/2 | 23/20/3 (cell E 2/1/3: ids 34, 35, 38 did not return in the queue); run alone on 2026-10-22, 34 and 38 stop at the time limit and 35 does not return (Table S7) |
| Section 5.6 ids 26, 44, 45 | 46.660913 / 45.314976; 57.868044 / 57.536983; 54.706978 / 53.681295 | the same values |
| seasonal grid: largest Ipopt - SCIP; gaps smallest / largest / median | 2.52e-9; 1.94e-4 / 4.90e-1 / 1.51e-2 | 2.52e-9; 1.73e-4 / 4.63e-1 / 5.86e-2 |

The re-run therefore reproduces every certificate and every instance value the text relies on,
with per-instance records and SCIP logs; what it does not reproduce is what depends on the
machine's load: times, the gaps of instances stopped at the time limit, and whether id 38 returns.

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
  **Re-run 2026-10-22** (`results/reruns/gate-1500-f90e554-2026-10-22/RUN.md`): the code of `f90e554` (supplied as a git bundle)
  with the August Manifest, on another platform: 48.3% -> 0.2% of fits at the iteration limit;
  under `:innovations` 1,494 of 1,500 series equal the August per-series record (six select
  another order, not because of a time limit); sMAPE / MASE 13.8812 / 0.9512 and 13.9497 /
  0.9476 against 13.9000 / 0.9528 and 13.9395 / 0.9471. The wall-time reduction is -43% there
  and is not a property of the method: times depend on the machine, and the August -28% is not
  paired.

## 4. Section 5.5: the likelihood-ranking diagnostic (LAMPSPUC/Sarimax.jl#17)

Two separate measurements, which the record keeps apart:

* **1,099 paired fits** is the AICc-parity comparison with R on the seasonal P x Q grid
  (`aicq.jl`, `aicq.R`; the first 100 series of a 400-series sample whose ids are in
  `rand400.csv`): the number of parameters agrees in 1,099 of 1,099 fits.
* **21.6 -> 3.3** is a different measurement (`mixdist.jl`): the median CSS -> ML log-likelihood
  distance with the `seasonalPresamplePrior` flag off (n = 14) and on (n = 12), **in the same
  revision (2026-08-16), not paired**, and on 14 and 12 fits, not 1,099. "Before" and "after"
  are the flag, not two revisions.
* The session record also reports **12.1% -> 16.5%** mixed seasonal orders and **OWA
  +0.0097**; they are not recovered. The collaborator points to `g_rand400.csv` and
  `fix_rand400.csv` (archived); no evident definition of a mixed seasonal order (P > 0 and Q > 0,
  over all series, over seasonal series or over seasonally differenced series) gives 12.1% or
  16.5% from them (they give 3.8%-12.5%), and the OWA needs the Naive2 baseline, which they do
  not hold. The manuscript does not print these values.

## Can the diagnostics be re-run?

Checked against the repository's history (every branch):

| diagnostic | script's package dependency | re-run |
|---|---|---|
| Table S2 (`diag_*.jl`, `diag_R.R`, 2026-08-20) | committed APIs; the revision of pull request 18 (`de56f50`) is in the repository | **possible**, as a new execution: the generator draws from `MersenneTwister`, so its data must be written under Julia >= 1.11 (as for the certification batch), and the R arm needs the container. Not done |
| Section 5.5 initialization, 10 seeds (`g4.jl`, `g4p.jl`, 2026-08-21) | keywords `exactCrossTerm` and `crossTermMinObs`, which **no commit of the package ever had** | **not possible**: the code the four arms ran against was never committed |
| Section 5.5 initialization, 1,500-series gate | the package at commit `f90e554`, on no branch; supplied as a git bundle on 2026-10-22 | **done**, as a new execution (`results/reruns/gate-1500-f90e554-2026-10-22/`) |
| Section 5.5 ranking, CSS -> ML distance (`mixdist.jl`, 2026-08-16 04:33) | keyword `seasonalPresamplePrior`, never committed (the prior was merged unconditionally at 16:17 the same day, `3e6cf20`, and reverted on 2026-08-18, `9e406a8`); the R reference `rmix.csv`, which defines the 14 and 12 fits, lost | **not possible** as measured |
| Section 5.5 ranking, AICc parity (`aicq.jl`, `aicq.R`) | committed APIs (2026-08-13 on) | possible in principle; the order of the 400-series list it took the first 100 from is not certain (`rand400.csv` holds the ids) |

## What remains open

* Table S5 and Section 5.6: the August batch's per-instance raw records are lost, and the
  2026-10-21 re-run stands in for them (with logs) for everything but load-dependent quantities
  (times, the gaps of time-limited instances, whether ids 34, 35 and 38 return).
* Section 5.5 and Table S9: the ranking diagnostics survive as aggregates only; the regenerated
  AICc pairing the manuscript mentions is not deposited (`results/manuscript/textual_claims.csv`,
  C20).
* Porting the batch's instances and the diagnostics' scripts to runners of this package
  (new executions, on approval).
