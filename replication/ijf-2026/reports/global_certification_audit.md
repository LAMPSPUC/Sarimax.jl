# Global-optimization certificates: audit of the evidence

Every SCIP result the manuscript relies on, checked against the records that exist. The
per-record table is `global_certificate_inventory.csv` (255 rows), built from the records by
`scripts/maintainers/build_global_certificates.py`; what is missing, and how to obtain it, is
in `missing_global_artifacts.md`. Numbers below are copied from the inventory.

## 1. Sources

| source | content | where |
|---|---|---|
| July 2026 battery, commit `8ffc512` (2026-07-12) | MA(1) certificates T = 8-40 (300 s), scaling study T = 40-200 (600 s), the three Table 7 instances (900 s) | bundle part 1, `raw/battery-2026-07/solver/*.jsonl` |
| same battery, commit `ecc4878` (2026-07-11) | the records the solver logs show: MA(1) certificates and Table 7 (the scaling records are identical at both commits) | `results/historical/battery-2026-07-ecc4878/`; logs in the bundle (`refresh_scip.log`, `scip_scaling_run*.log`, `global_value_run.log`) |
| re-run `full-all-limit1-2026-10-20` | a NEW execution of the Table 7 and MA(1)-certificate runners in the replication container, recording SCIP's dual bound, raw status and node count, with logs | `results/reruns/full-all-limit1-2026-10-20/` (`RUN.md`) |
| July 2026 yearly study | SCIP against Ipopt through the package API on 100 short M4 yearly series (free and constrained runs) and two probes (size, gap limit) | bundle part 1, `raw/scip-ipopt-yearly-2026-07/` |
| August 2026 certification batch | Table S5 and the seasonal-grid, near-cancellation and certified-instance numbers of Section 5.6 | **reconstructed** from the session that ran it (raw outputs lost): `results/historical/certification-2026-08/` (section 6) |

All archived battery SCIP solves used SCIP 8.0.0 through SCIP.jl 0.11.6 (SCIP_jll 0.2.1+0), on
macOS arm64 with Julia 1.11.9; the re-run used the same SCIP build for Linux x86_64 with Julia
1.10.12 (`RUN.md`).

## 2. Formulations, and what the package generates

* **Table 7 and the MA(1) certificates** are models written directly in JuMP by the runners
  (`experiments/runners/battery/run_solver_global_value.jl`, `run_solver_scip_cert.jl`,
  `run_solver_scip_scaling.jl`, ports of the archived scripts), not built by `Sarimax.fit!`:
  `eps_1` fixed at 0, `y_t = c [+ phi y_(t-1)] + theta eps_(t-1) + eps_t` as equality
  constraints, objective `sum eps_t^2`, boxes `phi, theta in [-1, 1]`, `c` and `eps` free. No
  innovation bounds, no warm start from Ipopt, no lifting (the models are non-seasonal: there
  are no products of coefficients to lift). SCIP options: `display/verblevel = 0` and
  `limits/time`; every other setting, including `limits/gap = 0` and `numerics/feastol = 1e-6`,
  is SCIP's default and was not recorded.
* **The July 2026 yearly study** went through the package API (`Sarimax.fit!` with a SCIP
  optimizer, fixed non-seasonal orders) with `limits/gap = 0.01`.
* **What the package generates for SCIP** (v1.0.0): the same nonlinear model it gives Ipopt.
  The multiplicative cross terms `phi_i Phi_k y_(t-i-sk)` and `theta_j Theta_w eps_(t-j-sw)` enter
  the fitted value as product expressions (`src/models/sarima.jl`, the `ŷ` expression after the
  comment "Box-Jenkins multiplicative SARIMA"); no lifting variables `a_ik = phi_i Phi_k` or
  `m_jw = theta_j Theta_w`, no innovation bounds and no warm start are created, and
  `includeSolverParameters!` sets no SCIP parameter. The bilinear reformulation of Section 3.2,
  the innovation bounds `|eps_t| <= sqrt(f_Ipopt)` and the Ipopt warm start of Section 5.6 were
  therefore added outside the package. The coordination e-mail of 2026-08-21 describing the
  August batch says so ("bilinear reformulation in an external harness; the package is not
  touched"); that harness is not archived.

## 3. Classification

Per record, applied by the builder script:

* **certified**: SCIP `OPTIMAL` under a zero-gap limit, relative gap 0 recorded, and either a log
  of the same run or the dual bound in the record; an unlogged re-run whose primal objective is
  identical to a logged certificate inherits it;
* **feasible with open gap**: a recorded finite gap or bound that is not closed (`TIME_LIMIT`, or
  `OPTIMAL` under `limits/gap > 0`);
* **improved feasible without certification**: a feasible point whose objective is below the
  local solution's by more than the tolerance effect of section 5.1, without a zero-gap
  certificate;
* **insufficient evidence**: no record, no gap, or records that conflict without a log.

Elapsed time is not used to classify.

## 4. Results by instance

| instance (manuscript) | records | verdict |
|---|---|---|
| MA(1), T = 8, 10, 20, 40 (Section 5.6: "closes the gap in 1.7-14.8 s") | `ecc4878` logged (1.21-14.35 s), `8ffc512` unlogged with identical objectives (1.27-14.83 s, the printed values), re-run with dual bounds (6.2-61.2 s) | **certified** in all three runs |
| MA(1) scaling, T = 40-120, 600 s (not cited by the manuscript) | logged | **certified** |
| MA(1) scaling, T = 150, 200, 600 s (not cited) | logged; `TIME_LIMIT`, relative gaps 1.02e-5 and 1.17e-5 | **feasible with open gap** |
| Table 7, simulated MA(1), T = 80, 900 s | `ecc4878` logged (71.6 s), `8ffc512` (69.2 s, printed), re-run (338.3 s, dual bound 73.65312392) | **certified** in all three runs |
| Table 7, simulated ARMA(1,0,1), T = 80, 900 s | `ecc4878` logged (393.1 s), `8ffc512` (385.4 s, printed): `OPTIMAL`; re-run: `TIME_LIMIT` at 900.1 s, relative gap 5.34e-6 | **certified** in the archived runs; **not reproduced** within 900 s in the re-run (open gap) |
| **Table 7, PJME ARMA(1,0,1), T = 80, 900 s** ("Certified: yes", 900.0 s) | section 4.1 | **insufficient evidence** for a certificate; **feasible with open gap** is established |
| Table S5; Section 5.6 seasonal grid, near-cancellation instance, certified instances 1 and 2 | reconstructed per-instance status, time and certified gap (44 of 46), objectives and dual bounds (7); no logs, no node counts | ids 44 and 45 **certified** (gaps 1.09e-9, 1.07e-8) and id 26 **improved feasible without certification** on the reconstructed record; the seasonal grid **feasible with open gap** (relative 1.9e-4 to 4.9e-1); not re-verifiable without logs (`certification_batch_2026_08.md`) |
| July 2026 yearly study (not cited by the manuscript) | 195 per-series rows, `limits/gap = 0.01` | 183 **feasible with open gap** (≤ 1%), 8 **improved feasible without certification** (SCIP's objective 0.9% to 25.8% below Ipopt's), 4 **insufficient evidence** (`TIME_LIMIT`, no gap recorded) |
| July 2026 size and gap-limit probes (not cited) | status and time only | 2 feasible with gap ≤ 1%, 6 insufficient evidence |

### 4.1 The PJME instance

Three records, which do not agree:

1. `ecc4878`: `TIME_LIMIT` at 601.8 s, relative gap 1.06e-7. The script committed with it sets
   900 s, so the record came from an earlier revision with about 600 s; the log of that
   commit shows the batch solve of this instance terminated by a signal and a segmentation
   fault inside SoPlex, with no result line. Unlogged.
2. `8ffc512`: `OPTIMAL`, relative gap 0, 899.958 s of the 900 s limit, primal objective
   bit-identical to record 1. Unlogged. This is the record Table 7 prints.
3. Re-run 2026-10-20 (logged, dual bound recorded): `TIME_LIMIT` at 900.1 s, primal
   246.3795964, dual bound 246.3712130, relative gap 3.40e-5.

`OPTIMAL` in record 2 is what `termination_status` returned, but no log shows that solve, the
only archived log of the instance shows it killed, the tolerances in force were not recorded,
and the logged re-run with the same SCIP build did not close the gap in 900 s. What is
established: SCIP's dual bound shows the local (Ipopt) solution within 3.51e-5 (relative) of the
global optimum of the tolerance-relaxed problem (re-run), and within 1.06e-7 according to the
unlogged record 1.

## 5. Two properties of the certificates that the manuscript should reflect

### 5.1 The objective differences in Table 7 are within SCIP's feasibility tolerance

SCIP reports the sum of squares of its own residual variables, which satisfy the equality
constraints only to `numerics/feastol` (1e-6). Evaluating the objective EXACTLY at the
coefficients SCIP reported (residuals generated by the recursion, so the constraints hold
exactly; column `primal_exact`):

| instance | Ipopt (exact) | SCIP reported | SCIP exact at its coefficients |
|---|---|---|---|
| simulated MA(1) | 73.6532959 | 73.6531218 | 73.6532963 |
| simulated ARMA(1,0,1) | 60.5256594 | 60.5256117 | 60.5257012 |
| PJME ARMA(1,0,1) | 246.3798557 | 246.3795705 | 246.3798565 |

(archived `8ffc512` records; the re-run gives the same picture.) In every instance the exact
objective at SCIP's coefficients is equal to or above Ipopt's (by 3.5e-9 to 6.9e-7 relative).
Likewise, every MA(1) certificate reports an objective 1.1e-6 to 2.4e-6 (relative) below the
brute-force profile minimum of the same series. The "Obj." column of Table 7 (73.6533 against
73.6531, etc.) therefore does not show SCIP finding a better solution than Ipopt: on these
instances the local solution is the global one, to within the tolerance. What the certificate
establishes is a bound: the local objective exceeds SCIP's lower bound by at most 2.3e-6
(MA(1)), 7.1e-6 (ARMA(1,0,1)) and 3.5e-5 (PJME) relative, in the logged re-run.

### 5.2 Whether a 900 s certificate closes depends on the platform

The same runners and SCIP build, on Linux x86_64 in the container, took 338 s for the MA(1)
instance (69 s archived) and stopped at 900 s with open gaps for the two ARMA(1,0,1) instances
(archived: 385 s `OPTIMAL`, and 900 s `OPTIMAL` unlogged). The MA(1) certificates of T = 8-40
were reproduced (6.2-61.2 s against 1.2-14.8 s). Times and the closing of the gap are
properties of the machine, not of the instance alone.

## 6. Missing evidence

The August 2026 certification batch (Table S5 and the Section 5.6 numbers) has no raw record:
its per-instance outputs were lost. Its author supplied on 2026-10-21 a reconstruction from the
record of the session that ran it: the 46 instances, the harness, the environment, and per
instance the status, time and certified gap, with objectives and dual bounds for a few
(`results/historical/certification-2026-08/`). SCIP ran with `display/verblevel = 0`, so no log
exists, and node counts were never recorded: its certificates cannot be re-verified beyond the
status and gap the session printed. `certification_batch_2026_08.md` reconciles it with the
manuscript; `missing_global_artifacts.md` G1 gives the remaining gaps and the wording.

## 7. Reproduction

```bash
./reproduce.sh fetch pjme
./reproduce.sh full --experiment battery_solver_comparison   # MA(1) certificates T = 8-40 (300 s each)
./reproduce.sh full --experiment battery_global_value        # Table 7 (three SCIP solves, 900 s each)
./reproduce.sh full --experiment battery_solver_scaling      # T = 40-200 (600 s each)
```

Costs on the container run of 2026-10-20: about 2 min for the certificates, 36 min for Table 7;
the scaling study runs sizes 40-300 at 600 s each and stops after two consecutive
non-certified sizes (about 1.5 h of SCIP time on the archived machine). SCIP must run in the Linux container (the
Windows SCIP_jll build crashes). Every record carries the dual bound, relative gap, raw status,
node count and SCIP version. Rebuild the inventory with

```bash
python3 replication/ijf-2026/scripts/maintainers/build_global_certificates.py \
  --bundle <unpacked bundle part 1> \
  --pjme-daily output/runs/battery_global_value/<run>/raw/battery-2026-07/energy/data/pjme_daily.csv
```
