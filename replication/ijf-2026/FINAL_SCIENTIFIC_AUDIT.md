# Final scientific audit of the replication package (October 2026)

Scope: the scientific and reproducibility corrections requested for pull request
LAMPSPUC/Sarimax.jl#34, branch `feature/ijf-replication`. Canonical manuscript: the revision of
2026-10-19 (controlled M4 objective comparison), which was **not available** (section B). Every
finding below cites its evidence; the detailed reports are in `reports/` (index in
`reports/README.md`). Nothing was published, no campaign was re-run in full, no historical
output was altered, and SCIP stays at 8.0.0.

## A. Resolved issues

| # | issue | resolution | evidence |
|---|---|---|---|
| A1 | **AIC/AICc/BIC of models with regressors** (`exogDynamics = :armax` scored as regression with errors; regressors differenced although `fit!` does not difference them; future regressor rows skipped; a single one-dimensional regressor sent to the CSS fallback) | confirmed against an independent dense likelihood; fixed in `src/exact_likelihood.jl` (regressors as `fit!` used them, filtered ARMAX path with a steady-state start); univariate models untouched | `reports/exog_likelihood.md`; `test/exog_likelihood.jl` 21/21 (6/21 on v1.0.0); commit `1e431ef` |
| A2 | **`AIR_PASSENGERS` is not the Box-Jenkins series** | identified as `fpp2::a10` value for value; documentation corrected, `ANTIDIABETIC_DRUG_SALES` alias, identity test; data unchanged | `reports/airpassengers_dataset.md`; commit `1400ee0` |
| A3 | **Global-optimization certificates** | every SCIP record inventoried and classified (certified / feasible with open gap / improved feasible without certification / insufficient evidence), with an identified re-run that records dual bounds and logs; lifting and innovation bounds shown to be added outside the package | `reports/global_certification_audit.md`, `reports/global_certificate_inventory.csv`, `results/reruns/full-all-limit1-2026-10-20/`; commit `8243edd` |
| A4 | **Table 7 Ipopt times** (0.04 s, 0.02 s printed; 0.0018 s, 0.0020 s archived) | what is timed (`@elapsed optimize!`, the first solve including compilation) and where the printed values come from (the July 2026 experiment report, no record) established; correction proposed and registered as erratum | `reports/table7_timing_reconciliation.md`, `results/manuscript/errata.csv` |
| A5 | **Table 3 / Section 5.1 protocol** | archived settings determined from the runner (no per-fit cap, `stationary = true`, post-fit root checks at 1e-2); the 120 s limit belongs to another campaign; minimal text correction proposed | `reports/monthly_protocol.md` |
| A6 | **Manuscript reconciliation as a release check** | the check fails on any unregistered numerical mismatch and on a stale erratum, reports five categories and the untranscribed canonical revision; the release gate (registered errata fail too) runs before the image can be published | `scripts/maintainers/check_manuscript_mapping.py`, CI job "Manuscript release check", `ijf-image-publish.yml`; commits `bd8839e`, `2134161` |
| A7 | **Replacement tables mapped to `m4_objectives`** | template for Tables 5, 6, S9, S10, Figure 3, Section 5.4 pointing at `m4_objectives_controlled` cells (printed values left to be transcribed); `controlled.paired` added for S9/S10 (760 of 760 cells agree with the campaign's independent scorer) | `results/manuscript/pending/`, `reports/manuscript_traceability.md` |
| A8 | **Supplementary provenance (Tables S6, S7, Figure S1, AR(3) figure)** | the simulation script and its outputs recovered and archived; CI recomputes the derived tables from them | `results/historical/revision-2026-09/recovered/`, `check_recovered_2026_09.py`; commits `ade8916`, `eafbddd` |
| A9 | **Table S2 and the Section 5.5 initialization numbers** | the records were supplied on 2026-10-21 as a reconstruction from the session that produced them: all 9 values of Table S2 and all 5 initialization values agree with the recovered aggregates (derived-only); per-replication and per-fit outputs are lost | `reports/certification_batch_2026_08.md` §2-3; `results/historical/certification-2026-08/` |
| A10 | **CI** | documentation build (unresolved `@ref` to the new helpers) fixed in `5c4d76d`; a stale checksum list fixed in `8dc4290`; the manuscript check no longer fails on documented errata | section F |
| A11 | **Intermittent Windows CI failure** of `test/objective_functions.jl:278` (elastic net at `lambda = 0`, Ipopt `OTHER_ERROR`; seen on 2026-09-10, 2026-10-18 and twice on 2026-10-20, always 0.0349) | root cause: the auxiliary variables of the L1 term have no upper bound and, with an L1 weight of zero, nothing in the objective holding them down, so the barrier problem has no minimizer in them and the outcome depended on the CPU's BLAS kernel (reproduced locally: `lambda = 10, alpha = 0` ends `OTHER_ERROR` on every kernel tried; `lambda = 0` changes status with the kernel). Fixed in `src/penalty.jl` by creating the auxiliaries only for coefficients with a positive L1 weight; uniform `lambda > 0, alpha > 0` builds the same model as before; the `"ridge"` objective, used by the M4 campaigns, has its own penalty and is unaffected | `test/objective_functions.jl` (new testset: passes with the fix on five BLAS kernels, fails on 1.0.0); CHANGELOG "Unreleased" |
| A12 | **The August 2026 certification batch (Table S5, Section 5.6) and the Section 5.5 ranking diagnostic** | supplied on 2026-10-21 as a reconstruction (raw outputs lost), archived with its provenance and checked in CI: the 12 Table S5 counts and the 8 instance values of Section 5.6 agree; the 4 seasonal-grid figures differ (registered errata); of the ranking diagnostic, 3 values agree and 3 are not recovered, and the text merges two measurements | `reports/certification_batch_2026_08.md`, `scripts/maintainers/check_certification_2026_08.py`, `results/manuscript/errata.csv`, textual claims T7, T13, T16, T17 |

## B. Pending external evidence

| item | what is missing | who holds it | status |
|---|---|---|---|
| B1 | the 2026-10-19 manuscript: `SARIMAX_jl_IJF_revision_20261019_controlled.pdf`, `..._supplement.pdf`, `..._source.zip` | whoever maintains the manuscript (it is the paper itself, not experimental data; even the 2026-10-16 PDF and the 2026-10-11 source used for the transcription are not on this machine) | AWAITING EVIDENCE: the inventory stays the 2026-10-16 transcription (not relabelled); the template for the controlled items waits for its printed values |
| B2 | the per-instance raw records of the August 2026 certification batch | lost (the batch ran in a temporary directory); a reconstruction was supplied on 2026-10-21 | RECONSTRUCTED: the printed values are checked against it as derived-only (`certification_batch_2026_08.md` §1); the seasonal-grid figures and the composition of Table S5 need WORDING (`missing_global_artifacts.md` G1) |
| B3 | the source of 3 values of the Section 5.5 likelihood-ranking diagnostic (mixed seasonal orders 12.1% -> 16.5%, OWA +0.0097) | the collaborator who ran it; the files supplied on 2026-10-21 do not give them, and the summary they cite was not supplied | AWAITING EVIDENCE, or qualify; the other 3 values are recovered (`certification_batch_2026_08.md` §4) |
| B4 | per-replication and per-fit outputs behind Table S2 and the Section 5.5 initialization numbers | lost; scripts, seeds and aggregates were supplied on 2026-10-21 | RECOVERED as aggregates; a port of the scripts to runners would re-create the outputs as a new execution, on approval |
| B5 | a log of the 900 s PJME solve that reported `OPTIMAL` (commit `8ffc512`) | nobody: the July 2026 battery never wrote one, and the logged re-run of 2026-10-20 did not reproduce the certificate | WORDING (alternative in `missing_global_artifacts.md` G2) |
| B6 | decision on SCIP 8.0.0's ZIB Academic License for a public image | maintainers | DECISION (`reports/scip_licence.md`) |

Not counted as defects: the Zenodo records, the image digest, the DOIs and the GitHub release,
which are not published yet by design.

Of the 46 printed values that had no evidence, 41 are now checked against the records supplied
on 2026-10-21 (37 agree, 4 differ). The 5 still without evidence are 3 values of B3 and the 2
configuration rows of Table S3, which are a wording correction, not missing data. B1 is the
manuscript document, B5 is not recoverable from anyone and B6 is a decision.

## C. Manuscript corrections

The 2026-10-19 text was not available; "current" quotes the 2026-10-16 revision as recorded in
`results/manuscript/` and the earlier verification reports. The authors should apply each
correction to the wording of the 2026-10-19 revision.

| location | current | proposed | evidence |
|---|---|---|---|
| Table 7, Sim. ARMA(1,0,1), Ipopt, Time (s) | 0.04 | 0.0018 | `table7_timing_reconciliation.md` |
| Table 7, PJME ARMA(1,0,1), Ipopt, Time (s) | 0.02 | 0.0020 | same |
| Table 7, Sim. MA(1), Ipopt, Time (s) | 0.6 | 0.59, with the note "first solve of the process, including compilation" | same |
| Table 7 caption | (times not qualified) | add: "Times are wall-clock times of a single solve on the archived machine (macOS arm64); the first Ipopt time includes compilation. Whether SCIP closes the gap within 900 s depends on the hardware." | same; `global_certification_audit.md` §5.2 |
| Section 5.6 | "is obtained locally in hundredths of a second" | "is obtained locally in about two milliseconds" | same |
| Table 7, PJME ARMA(1,0,1), SCIP, Certified | yes | "no (relative gap 3.4×10⁻⁵ at the 900 s limit)"; footnote: "an earlier run reported optimality at 899.96 s, without a log" | `global_certification_audit.md` §4.1 |
| Table 7, Obj. column (or caption) | SCIP objectives printed below Ipopt's | add: "The SCIP objectives are below Ipopt's only within SCIP's feasibility tolerance; evaluated at SCIP's coefficients the objective is not lower. SCIP's dual bounds place the local solutions within 2.3×10⁻⁶, 7.1×10⁻⁶ and 3.5×10⁻⁵ (relative) of the global optimum." | §5.1 of the same report |
| Table S5 and the Section 5.6 seasonal, near-cancellation and certified-instance numbers | printed as results | keep them as results of the batch, with: the seasonal-grid figures as recorded (incumbent equal to the Ipopt point to 2.5×10⁻⁹; remaining relative gaps 1.9×10⁻⁴ to 4.9×10⁻¹, median 1.5×10⁻²); the T = 480 row with its two unfinished instances; and the instances Table S5 counts, so that the certificates of Section 5.6 (one of them in a group the table does not count) can be located | `missing_global_artifacts.md` G1; `certification_batch_2026_08.md` §1; `errata.csv` |
| Table S3, "Per-series time limit" | 120 | none | `monthly_protocol.md` |
| Table S3, "Admissibility" | post-fit root checks, margin 1e-2 | "AR stationarity imposed by the reflection parameterization (margin 10⁻⁶), MA coefficients in their box; post-fit root checks of both polynomials, margin 10⁻²" | same |
| Section 5.1, stationarity sentence | "In this experiment, stationarity and invertibility are checked after estimation using a root margin of 10⁻² rather than imposed through the reflection-coordinate parameterization." | the two-protocol sentence quoted in `monthly_protocol.md` (Tables 3-4: imposed by reflection; Tables 5-6, Figure 3: checked after estimation; root check at 10⁻² in both), and no 120 s limit for the Table 3 runs | same |
| every mention of the package's "AirPassengers" series; Table S7 case "Airline, log AirPassengers" | AirPassengers / Box-Jenkins airline data | "the monthly series of Australian anti-diabetic drug sales (`a10` in the R package fpp2, July 1991 to June 2008, 204 observations), shipped with the package as `AIR_PASSENGERS`"; Table S7: "Airline model, log a10" | `airpassengers_dataset.md` |
| Sections 3.6, 5.4, Huber threshold | "expressed on the standardized residual scale" | "on the scale of the series divided by the standard deviation of its differenced values" | `textual_claims.csv` T15 |
| Section 5.5, initialization diagnostic | printed as an archived result | keep the numbers with the source: "measured during the development of the `:innovations` mode (10 simulated seeds; a paired 1,500-series M4 sample); only aggregate records survive" | `certification_batch_2026_08.md` §3 |
| Section 5.5, likelihood-ranking diagnostic (1,099 fits ...) | one measurement on 1,099 paired fits, before and after | two measurements: the AICc parity with R on 1,099 fits; and the median CSS -> ML distance with the seasonal pre-sample prior off and on in one revision (21.6 -> 3.3, 14 and 12 fits, not paired). Remove 12.1% -> 16.5% and OWA +0.0097 unless their source is supplied | `certification_batch_2026_08.md` §4 |
| Table S2 | printed as an archived result | caption: "reported with the change that introduced `exogDynamics` (pull request 18); only the per-process aggregate is archived" | `certification_batch_2026_08.md` §2 |
| Section 4.2 (information criteria) | criteria ranked by the exact likelihood | add: "for estimators other than Gaussian ML the AICc computed this way is a ranking heuristic"; and, if the paper describes AIC/BIC with regressors, "from Sarimax.jl 1.0.1, which scores models with regressors by the likelihood of the fitted ARMAX or regression-with-errors model" | `exog_likelihood.md` |
| Tables 5, 6, S9, S10, Figure 3, Section 5.4 | historical objective campaigns (2026-10-16) | the controlled campaign, as in the 2026-10-19 revision; then fill `results/manuscript/pending/` | `manuscript_traceability.md` |

## D. Software changes

| change | files | API | effect |
|---|---|---|---|
| exact likelihood of models with regressors | `src/exact_likelihood.jl` (`fittedExogMatrix`, `armaxRegressionPath`), one metadata key in `src/models/sarima.jl` | unchanged | AIC/AICc/BIC, `auto`'s orders and its forecasts change for models with regressors; univariate models and fixed-order fits unchanged (except under `multistart = true`) |
| dataset alias | `src/datasets.jl` | adds `ANTIDIABETIC_DRUG_SALES` | none (same data) |
| elastic net with an L1 weight of zero | `src/penalty.jl` (`elasticNetPenalty!`) | unchanged | fits with `alpha = 0`, `lambda = 0` or zero entries of a heterogeneous `lambda` become well posed (they could end on a solver error); every other fit builds the same model. Affects no M4 table. Two archived experiments used such weights, so their re-runs may differ from their archived outputs: the battery architecture table's "ridge" row (`elastic_net` with `alpha = 0`; not reported in the manuscript) and part F of the recovered Figure S1 experiment (zero entries of `lambda`) |
| tests | `test/exog_likelihood.jl` (new, 21 checks), `test/datasets.jl` (+11), `test/objective_functions.jl` (+1 testset, 4 checks) | | |
| documentation | README, `docs/src/tutorial.md`, `docs/src/reference.md`, CHANGELOG "Unreleased" | | |
| replication package | release check with errata and release gate; `controlled.paired`; the certificate inventory builder; the recovered-output check; evidence and reports | | no historical output changed |

`src/` now differs from v1.0.0: the package should be released as 1.0.1 before the replication
release is tagged (`ARTIFACT_GUIDE.md` §9).

## E. Readiness

| area | state |
|---|---|
| package fixes (A1, A2) | ready for review |
| reproducibility of archived tables | ready: every table group regenerates from archived records on every cell its inputs allow |
| manuscript traceability | **blocked** on B1 (canonical revision); the release gate fails on the two registered errata until the manuscript is corrected |
| global-optimization claims | Table 7 and the MA(1) certificates are documented record by record; Table S5 and Section 5.6 checked against the reconstructed August 2026 record, with 4 registered errata and the wording of G1 |
| Section 5.5 diagnostics, Table S2 | checked against the recovered aggregates; 3 ranking values still without evidence (B3) |
| publication (Zenodo, GHCR, release) | not started by design; needs the authors' approval and B6 |

The package is not ready for release: B1 and the manuscript corrections of section C come first.

## F. Test evidence

Commands, platforms and full results: `REPRODUCIBILITY_STATUS.md`, section "Final scientific
audit (2026-10-20)". In short:

* package suite with both `src/` fixes: 1,443 pass, 0 fail, 0 error, 5 pre-existing broken;
* `test/exog_likelihood.jl` 21/21 (6/21 on v1.0.0); the new elastic-net testset passes on five
  OpenBLAS kernels (its first check fails on v1.0.0); `test/datasets.jl` 40/40;
* image rebuilt for the changed `src/` (commit `8dc4290`): smoke SUCCESS; tables 4,204 of 4,204
  cells in 9 groups (exit 4 only because the local bundle lacks inputs of the tenth group);
* **univariate M4 under the fixed code:** the controlled campaign re-run on every weekly series
  reproduces the archive on 359/359 series for each objective (same orders, sMAPE within
  1.0e-11, forecasts within 6.4e-13 relative, the same bit-identical counts as with v1.0.0),
  and 20/20 sampled quarterly and yearly series for each objective;
* release check: pull-request mode passes with the 2 registered errata; release mode fails on
  them, as designed;
* CI at `8dc4290`: IJF replication (host, manuscript check, container amd64 and arm64) and
  documentation PASS; package CI PASS except the Windows job, whose elastic-net failure is A11.
