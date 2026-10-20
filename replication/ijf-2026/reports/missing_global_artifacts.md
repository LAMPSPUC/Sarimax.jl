# Missing global-optimization evidence

Status of every global-optimization claim whose evidence is incomplete, with what would close
it. Status values: **AWAITING EVIDENCE** (the original records exist elsewhere and must be
supplied), **RE-RUN AVAILABLE** (the package can produce a new, identified run), **WORDING**
(the claim should be restated to what the archive supports). Nothing here was re-executed for
this report beyond the re-run already recorded in `results/reruns/full-all-limit1-2026-10-20/`.

## G1. The August 2026 certification batch (Table S5; Section 5.6) — AWAITING EVIDENCE

**Printed values without a record (24):** Table S5 (`tab:global_experiment`), all twelve cells:
non-seasonal grid (1,0,1), T in {30, ..., 240}: 12 certified, 0 at the time limit; non-seasonal
extended T = 480: 1 / 0; T = 960: 1 / 2; seasonal grid (1,0,1)(P,0,Q)_12, T in {36, 60}: 0 / 12;
seasonal/adversarial diagnostics: 2 / 4; free pre-sample mode, T in {60, 120}: 3 / 0. Section
5.6 (`sec:global_diagnostics`): seasonal grid incumbent 2.5% below the local objective, remaining
gaps 1.9% / 4.9% / 1.2% (smallest / largest / median), near-cancellation instance Ipopt
46.660913 and SCIP feasible 45.314976, certified instance 1 local 57.868044 and certified
57.536983 (0.575%), instance 2 local 54.706978 and certified 53.681295 (1.911%). The rows are in
`results/manuscript/manuscript_values.csv` and `global_certificate_inventory.csv`.

**What is known about the batch** (coordination e-mail of 2026-08-21, not an archived record):
32 cases, 600 s limit, a bilinear reformulation in an external harness without changes to the
package; the non-seasonal cases all `OPTIMAL` with gaps around 1e-11 (including T = 240 in about
2 min); seasonal cases at the time limit with the dual bound collected, one closed to 0.68%; the
adversarial non-seasonal cases certified, the seasonal ones not. Table S5 counts 37 instances
(12 + 1 + 3 + 12 + 6 + 3), the e-mail 32: the extended and free pre-sample rows may come from a
later run. The batch's author wrote on 2026-10-16 that its files were sent on 2026-08-22 and
offered to repackage them. They are not in this repository, its branches, the artifact bundle or
the workspace that built it.

**Records needed, per instance:** the series (or generator and seed) and T; the harness at the
commit used (JuMP model, lifting variables `a_ik = phi_i Phi_k` and `m_jw = theta_j Theta_w` and
their bounds, innovation bounds `|eps_t| <= sqrt(f_Ipopt)` and how `f_Ipopt` was obtained, warm
start); the Julia Manifest (SCIP, SCIP_jll, Ipopt versions); the SCIP parameters in force
(`limits/time`, `limits/gap`, `limits/absgap`, `numerics/feastol`, and any other); the SCIP
output (termination and raw status, primal bound, dual bound, absolute and relative gap, solving
time, nodes) or its log at `display/verblevel >= 4`; the Ipopt incumbent (objective,
coefficients, status); the number of concurrent jobs.

**Runner capability:** none. The package does not generate lifting variables, innovation bounds
or SCIP warm starts (`global_certification_audit.md` §2), and no runner of this replication
package builds that formulation, so no command regenerates the batch. Re-creating the harness
from the manuscript's description would be a NEW experiment, not a reproduction, and needs
approval.

**Cost, if the harness is supplied and re-run:** 37 instances at up to 600 s each, single-threaded
SCIP: at most 6.2 h of SCIP time, about 1.6 h of wall time with 4 parallel jobs, in the Linux
container (SCIP crashes on native Windows). The archived 2026-10-20 re-run suggests that times
on the container platform can be several times the archived ones.

**Wording if the records stay missing.** Table S5 and the numbers above cannot be printed as
archived results. Supported by the archive instead (inventory rows): "On non-seasonal MA(1)
least-squares problems, SCIP 8.0.0 closed the gap within 600 s up to T = 120 and stopped at the
time limit with relative gaps of about 1e-5 at T = 150 and T = 200; on the ARMA(1,1) instances of
Table 7 it certified the simulated case and left a relative gap of 3.4e-5 on the PJME case in a
logged 900 s run. Multiplicative seasonal instances were not certified in the archived runs." If
the seasonal and adversarial findings are kept, they should be labelled as exploratory runs whose
records are not part of the replication package.

## G2. Table 7, PJME ARMA(1,0,1): "Certified: yes", 900.0 s — WORDING (RE-RUN AVAILABLE)

**Records:** `ecc4878` `TIME_LIMIT` at 601.8 s (gap 1.06e-7, unlogged; the commit's log shows the
batch solve killed); `8ffc512` `OPTIMAL` at 899.958 s of 900 s (unlogged; the printed record);
re-run 2026-10-20 `TIME_LIMIT` at 900.1 s, gap 3.40e-5, dual bound 246.3712130 (logged).

**Missing:** a log of the `8ffc512` solve, and the tolerances in force.

**Command:** `./reproduce.sh fetch pjme` then `./reproduce.sh full --experiment battery_global_value`
(about 36 min in the container: the three SCIP solves at up to 900 s each). Every record now
carries the dual bound and SCIP's raw status. A run with a longer limit (e.g. 3600 s) to see
whether the instance certifies at all would cost up to 1 h per instance and is not done here.

**Wording:** "Certified: no (relative gap 3.4e-5 at the 900 s limit)", with the note that an
earlier archived run reported `OPTIMAL` at 899.96 s without a log; or keep "yes" only with a
footnote that the certificate was not reproduced in a logged re-run. In the text: SCIP's dual
bound shows the local solution to be within 4e-5 (relative) of the global optimum.

## G3. Table 7, simulated ARMA(1,0,1): certificate depends on the platform — WORDING

Archived `OPTIMAL` (logged, 393.1 s; printed 385.4 s); re-run `TIME_LIMIT` at 900.1 s with a
relative gap of 5.3e-6. The certificate is valid for the archived run; the table should say that
the times are those of the archived run (macOS arm64) and that whether the gap closes within
900 s depends on the hardware.

## G4. Table 7, objective column — WORDING

The SCIP objectives printed below the Ipopt ones (73.6531 vs 73.6533, 60.5256 vs 60.5257,
246.3796 vs 246.3799) are SCIP's sums of squares of residual variables that satisfy the model
equations only within its feasibility tolerance; at SCIP's coefficients the exact objective is
not below Ipopt's in any instance (`global_certification_audit.md` §5.1). The column should
print the exact objective at each solver's coefficients, or a note should say that the
differences are within the solver's tolerance and that the local solutions are globally optimal
to within 2.3e-6, 7.1e-6 and 3.5e-5 (relative, from SCIP's dual bounds).

## G5. SCIP settings of the archived runs — not recoverable

No archived run recorded SCIP's tolerances or its log at a verbosity that shows them. The ported
runners keep `display/verblevel = 0`; their records hold the dual bound, gaps, statuses, node
count and version, which is what the classification needs. Recording `numerics/feastol`,
`limits/gap` and `limits/absgap` in each record would be a small runner change, not made here.
