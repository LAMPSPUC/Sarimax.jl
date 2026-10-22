# Provenance gaps: what the manuscript prints without an archived source

Every printed value or result whose raw source is not part of the replication package, in
order of priority, with its disposition: **recovered** (the source was found and is now
archived), **request** (the original records exist elsewhere and must be supplied),
**reconstruct** (a new, identified execution could produce comparable evidence; needs
approval), **qualify** (restate the claim to what the archive supports). The values are those of
the 2026-10-16 transcription (`results/manuscript/manuscript_values.csv`); the 2026-10-19 text
was not available.

| priority | item | values | what exists | disposition |
|---|---|---|---|---|
| 1 | Table S5; Section 5.6 seasonal grid, near-cancellation and certified instances (August 2026 certification batch) | 24 | received 2026-10-21 and archived in `results/historical/certification-2026-08/`: the 46 instances, the harness, the environment, and status, time and gap for 44 instances, RECONSTRUCTED from the record of the session that ran the batch (per-instance raw outputs lost; no node counts, no SCIP logs) | **recovered as a reconstruction**: 20 values agree (derived-only), the 4 seasonal-grid figures of Section 5.6 differ (registered errata); which instances Table S5 counts is not recorded (`certification_batch_2026_08.md` §1) |
| 2 | Table 7, PJME "Certified: yes" and the Ipopt times 0.04 s / 0.02 s | 3 | records, logs and a logged re-run that disagree with the printed cells | **qualify** (`global_certification_audit.md` §4.1, `table7_timing_reconciliation.md`); a longer re-run is available on approval |
| 3 | Section 5.5, likelihood-ranking diagnostic: 1,099 paired fits, median log-likelihood distance to R's ML 21.6 -> 3.3, mixed seasonal orders 12.1% -> 16.5%, OWA +0.0097 | 6 | received 2026-10-21 (`results/historical/certification-2026-08/`): two separate measurements that the text merges. 1,099 = the AICc-parity comparison on the P x Q grid; 21.6 -> 3.3 = the `seasonalPresamplePrior` flag off/on in one revision, n = 14 and 12, not paired. 12.1% -> 16.5% and OWA +0.0097 are not recovered from the files supplied | **partly recovered**: 3 values derived-only; **qualify** the wording (two measurements, not paired); 3 values still without evidence (`certification_batch_2026_08.md` §4) |
| 4 | Section 5.5, initialization diagnostic: median coefficient distance to `stats::arima` ML 0.2076 (free pre-sample, `:penalized`) -> 0.0863 (`:innovations`) on a simulated (1,0,1)(1,0,1)_12 with T = 200 and 10 seeds; fits at the 200-iteration limit 47.4% -> 0.2% and wall time -28% on a paired 1,500-series M4 sample | 5 | received 2026-10-21 (`results/historical/certification-2026-08/`): the 10-seed scripts with their per-seed table; the reconstructed runner and the aggregate of the 1,500-series gate (commit `f90e554`); per-fit outputs lost | **recovered**: all 5 values agree (derived-only) (`certification_batch_2026_08.md` §3). **Reconstruct** (port to a runner) is a new execution |
| 5 | Table S2, exogenous-convention simulation (3 data-generating processes x 20 replications, b = [2.0, -1.5, 0.8], orders fixed at the truth) | 9 | received 2026-10-21 (`results/historical/certification-2026-08/`): generator (seeds 9000+s), fitters, reader and the per-DGP aggregate; per-replication outputs lost | **recovered**: all 9 values agree (derived-only) (`certification_batch_2026_08.md` §2). The generator is deterministic: a port re-creates the replications as a new execution |
| 6 | Table S6 (quantile loss), Table S7 (multistart), Figure S1 (coefficient-specific penalties), and the AR(2)/AR(3) stationarity figure | 19 | the script (`efg_julia_experiments.jl`, parts E, F, G; `a_ar_stationarity.py`, part A) and its outputs, written 2026-09-11, found in the revision workspace | **recovered**: archived in `results/historical/revision-2026-09/recovered/` with SHA-256; `check_recovered_2026_09.py` (CI) shows the derived tables follow from them (Table S7 recomputed from the 360 per-start records). Remaining limit: Table S6's 200 replications were summarized inside the script (no per-replication records); the run's environment was not recorded |
| 7 | Table S3 configuration rows (120 s per fit; post-fit root checks) | 2 | the archived configuration of the Table 3 runs, which differs | **qualify** (`monthly_protocol.md`) |
| 8 | Figure 1 | 0 printed numbers | an analytic illustration; no plotting script archived | **request** the plotting script, or leave as an illustration |

## What the collaborator supplied (2026-10-21)

Items 1, 3, 4 and 5 came from work of one collaborator, who supplied on 2026-10-21 what survives
of it (`results/historical/certification-2026-08/`, `PROVENANCE.md` there): not the raw records,
which are lost, but a reconstruction from the record of the session that produced them, with the
source of each row declared. The values are therefore checked as **derived-only** against that
record (`scripts/maintainers/check_certification_2026_08.py`, in CI), never as reproductions.
The reconciliation is in `certification_batch_2026_08.md`.

## Effect on the manuscript inventory

| category (2026-10-16 transcription) | values |
|---|---|
| raw-reproduced | 603 |
| derived-only | 56 (19 of item 6; 37 of items 1, 3, 4, 5) |
| differing | 6 (Table 7 Ipopt times: item 2; Section 5.6 seasonal grid: item 1) |
| no evidence | 5 (item 3: 3; item 7: 2) |
| checked at run time only | 2 |

The 5 values without evidence: Section 5.5 3 (12.1%, 16.5%, OWA +0.0097), Table S3 2.
