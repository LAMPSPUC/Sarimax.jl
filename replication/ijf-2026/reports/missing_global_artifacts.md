# Missing global-optimization evidence

Status of every global-optimization result whose evidence is incomplete, with what would close
it. Status values: **AWAITING EVIDENCE** (the original records exist elsewhere and must be
supplied), **RE-RUN AVAILABLE** (the package can produce a new, identified run). Each section
ends with what the submitted manuscript states, which the manuscript mapping checks against the
records (`results/manuscript/manuscript_values.csv`). Nothing here was re-executed for
this report beyond the re-runs recorded in `results/reruns/full-all-limit1-2026-10-20/` and,
by the batch's author, `results/reruns/certification-2026-08-rerun-2026-10-21/`.

## G1. The August 2026 certification batch (Tables S5-S7; Section 5.6) — RECONSTRUCTED RECORD AND LOGGED RE-RUN

**Received 2026-10-21** (`results/historical/certification-2026-08/`; reconciliation in
`certification_batch_2026_08.md` §1). The per-instance raw outputs are lost; what exists is a
reconstruction from the record of the session that ran the batch: the 46 instances (`casos()`
of the harness; batch 1 ids 1-32, cells A-D; batch 2 ids 33-46, cells E-G), the three harness
snapshots that ran, the Julia Manifest (SCIP.jl 0.11.6 on SCIP 8.0.0, Ipopt.jl 1.15.0, JuMP
1.31.1, Julia 1.12.3), the SCIP parameters (`limits/time = 600`, `display/verblevel = 0`, nothing
else), and per instance the status, time and certified gap (44 of 46), the Ipopt objective (9)
and SCIP's objective and dual bound (7).

**Never recorded or lost:** node counts, SCIP logs, SCIP's raw status, the Ipopt coefficients and
status, the primal and dual bounds of 37 instances. Ids 34 and 35 (T = 480) never finished; the
dual bound of id 37 is SCIP's `-1e+20` sentinel. The re-run of 2026-10-21 records all of these,
with a SCIP log per case, for the 43 instances that finished in it.

**What the record gives:** the outcome counts by the harness's cells (23 OPTIMAL, 21
TIME_LIMIT, 2 not returned), the near-cancellation instance (id 26), the two certificates
(ids 44, 45), and the seasonal-grid figures (largest Ipopt - SCIP difference 2.52e-9; remaining
relative gaps from 1.94e-4 to 4.90e-1, median 1.51e-2).

**Runner capability:** RE-RUN AVAILABLE. `run_certification_2026_08.jl` re-runs the batch with
its own harness on its own 46 instances, written as data (under the replication environment's
Julia 1.10 MersenneTwister draws other series; the file was generated under Julia 1.11, as the
batch's 1.12.3 drew them, and reproduces every Ipopt objective the record holds), and records
nodes, bounds, statuses, coefficients and SCIP logs. About 4.6 h of solver time, divided by
`--jobs`, in the Linux container. Not run in the container.

**Re-run by the batch's author, 2026-10-21** (`results/reruns/certification-2026-08-rerun-2026-10-21/`,
`certification_batch_2026_08.md` §1): the batch's harness, recording extended, in the batch's
environment, on series identical to the package's instances. The 23 certificates are certified
again with the same gaps, the instance values of Section 5.6 and the Ipopt objectives agree, and
every Table S5 count is reproduced except T = 960 (1 / 1: id 38 did not return within 900 s, nor
within 3600 s, with other SCIP processes running); ids 34 and 35 did not return either. Run alone
on 2026-10-22, ids 34 and 38 stop at the time limit (T = 960 is then 1 / 2) and id 35
stays in SCIP's presolve, in the `completesol` heuristic that SCIP.jl's partial warm start
triggers. The gaps of the time-limited seasonal
instances differ (1.7e-4 to 4.6e-1, median 5.9e-2), as gaps at a time limit do from one machine
load to another.

**In the manuscript.** Table S5 lists the 46 instances by cell with the August and the re-run
counts (including the instances that did not return), Table S6 selected records of the re-run
(ids 26, 37, 44, 45) and Table S7 the isolated runs of 2026-10-22; Section 5.6 states the seasonal-grid figures as
recorded. All agree with the records.

## G2. Table 7, PJME ARMA(1,0,1): the 900 s certificate — RE-RUN AVAILABLE

**Records:** `ecc4878` `TIME_LIMIT` at 601.8 s (gap 1.06e-7, unlogged; the commit's log shows the
batch solve killed); `8ffc512` `OPTIMAL` at 899.958 s of 900 s (unlogged; the printed record);
re-run 2026-10-20 `TIME_LIMIT` at 900.1 s, gap 3.40e-5, dual bound 246.3712130 (logged).

**Missing:** a log of the `8ffc512` solve, and the tolerances in force.

**Command:** `./reproduce.sh fetch pjme` then `./reproduce.sh full --experiment battery_global_value`
(about 36 min in the container: the three SCIP solves at up to 900 s each). Every record now
carries the dual bound and SCIP's raw status. A run with a longer limit (e.g. 3600 s) to see
whether the instance certifies at all would cost up to 1 h per instance and is not done here.

**In the manuscript.** Table 7 prints "no", with a footnote: an unlogged archive reported
optimality at 899.958 s, and a logged re-execution stopped at 900 s with relative gap 3.4e-5.
SCIP's dual bound of that re-execution places the local solution within 4e-5 (relative) of the
global optimum.

## G3. Table 7, simulated ARMA(1,0,1): the certificate depends on the platform

Archived `OPTIMAL` (logged, 393.1 s; the unlogged `8ffc512` record 385.4 s); re-run
`TIME_LIMIT` at 900.1 s with a relative gap of 5.3e-6. The certificate is valid for the archived
run. **In the manuscript.** Table 7 prints "yes" with a footnote (certified in an archived logged
run; another platform did not close the 900-second gap), and its caption states that the SCIP
values and times are those of the archived runs.

## G4. Table 7, objective column

The SCIP objectives printed below the Ipopt ones (73.6531 vs 73.6533, 60.5256 vs 60.5257,
246.3796 vs 246.3799) are SCIP's sums of squares of residual variables that satisfy the model
equations only within its feasibility tolerance; at SCIP's coefficients the exact objective is
not below Ipopt's in any instance (`global_certification_audit.md` §5.1); from SCIP's dual
bounds, the local solutions are globally optimal to within 2.3e-6, 7.1e-6 and 3.5e-5 (relative).
**In the manuscript.** The caption of Table 7 states that the differences in reported objectives
lie within SCIP's feasibility tolerance.

## G5. SCIP settings of the archived runs — not recoverable

No archived run recorded SCIP's tolerances or its log at a verbosity that shows them. The ported
runners keep `display/verblevel = 0`; their records hold the dual bound, gaps, statuses, node
count and version, which is what the classification needs. Recording `numerics/feastol`,
`limits/gap` and `limits/absgap` in each record would be a small runner change, not made here.
