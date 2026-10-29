# Reports

**For the submitted manuscript:** `../verification/VERIFICATION_REPORT.md` (what is verified,
how, and what remains) and `experiment_provenance.md` (source revision, environment, runner and
records of every experiment, and the operational details of the solver and initialization
studies).

**Audit reports.** The reports below examine the evidence behind particular results: each
states its sources, and its numbers are taken from records. The printed values of the submitted
manuscript are checked by `../results/manuscript/` and summarized in the verification report;
where a report quotes a printed value, the mapping is authoritative.

| report | subject |
|---|---|
| `experiment_provenance.md` | per-experiment source revisions, environments and records; global-grid and initialization operational notes |
| `global_certification_audit.md` | every SCIP result against its records; classification per record |
| `global_certificate_inventory.csv` | the per-record table (21 rows), built by `scripts/maintainers/build_global_certificates.py` |
| `missing_global_artifacts.md` | missing global-optimization evidence: what is needed, commands, costs, and what the manuscript states |
| `table7_timing_reconciliation.md` | what the times of Table 7 measure and which records they come from |
| `provenance_gaps.md` | the printed values that no table group regenerates, by kind of record, with what is missing |
| `certification_batch_2026_08.md` | the records of the August 2026 certification batch (Tables S5-S7, Section 5.6), Table S2 and the Section 5.5 diagnostics, reconstructed on 2026-10-21, and the re-run |
| `exog_likelihood.md` | the information criteria of models with regressors: defect, fix, harness and effects (`exog_likelihood/`) |
| `airpassengers_dataset.md` | identity of the package's `AIR_PASSENGERS` series (`fpp2::a10`) |
| `monthly_protocol.md` | the protocols of the six-frequency comparison (Table 3) and of the 120-second monthly experiment (Table S3) against their runners |
| `scip_licence.md` | SCIP 8.0.0 and the ZIB Academic License; a possible migration |
