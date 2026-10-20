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
| 1 | Table S5; Section 5.6 seasonal grid, near-cancellation and certified instances (August 2026 certification batch) | 24 | a coordination e-mail describing the batch; nothing else | **request** the batch files (offered by their author on 2026-10-16); otherwise **qualify** (wording in `missing_global_artifacts.md` G1). Not reconstructible by this package: the harness is not archived |
| 2 | Table 7, PJME "Certified: yes" and the Ipopt times 0.04 s / 0.02 s | 3 | records, logs and a logged re-run that disagree with the printed cells | **qualify** (`global_certification_audit.md` §4.1, `table7_timing_reconciliation.md`); a longer re-run is available on approval |
| 3 | Section 5.5, likelihood-ranking diagnostic: 1,099 paired fits, median log-likelihood distance to R's ML 21.6 -> 3.3, mixed seasonal orders 12.1% -> 16.5%, OWA +0.0097 | 6 | nothing found: no record, script, report, pull-request description or issue in the repository or the workspaces | **request** the measurement (which code revisions are "before" and "after", which 1,099 fits, the R reference); otherwise **qualify**: state the direction of the effect without the numbers, or drop the paragraph. **Reconstruct** is possible only once "before" and "after" are identified |
| 4 | Section 5.5, initialization diagnostic: median coefficient distance to `stats::arima` ML 0.2076 (free pre-sample, `:penalized`) -> 0.0863 (`:innovations`) on a simulated (1,0,1)(1,0,1)_12 with T = 200 and 10 seeds; fits at the 200-iteration limit 47.4% -> 0.2% and wall time -28% on "a paired 1,500-series sample" (the frequency is not named; the census of the same description is M4 monthly) | 5 | all five values in the description of LAMPSPUC/Sarimax.jl#20 (2026-08-21; census at commit `f90e554`); the wall times there are 0.848 s and 0.609 s per series, a 28.2% reduction; no records, seeds or script archived | **request** the records and script; until then **qualify** the source in the text ("measured during the development of the `:innovations` mode; records not archived"). **Reconstruct**: a simulation at the stated design against v1.0.0 is cheap (minutes) but is a new experiment |
| 5 | Table S2, exogenous-convention simulation (3 data-generating processes x 20 replications, b = [2.0, -1.5, 0.8], orders fixed at the truth) | 9 | all nine values (and the estimated coefficients) in the description of LAMPSPUC/Sarimax.jl#18 (2026-08-20, head `de56f50`); no runner, seeds or per-replication output | **request** the runner and output; otherwise **qualify** as "reported with the change that introduced `exogDynamics`". Unaffected by the information-criterion fix (`exog_likelihood.md`): the orders are fixed, so no ranking is involved. **Reconstruct** possible on approval (minutes) |
| 6 | Table S6 (quantile loss), Table S7 (multistart), Figure S1 (coefficient-specific penalties), and the AR(2)/AR(3) stationarity figure | 19 | the script (`efg_julia_experiments.jl`, parts E, F, G; `a_ar_stationarity.py`, part A) and its outputs, written 2026-09-11, found in the revision workspace | **recovered**: archived in `results/historical/revision-2026-09/recovered/` with SHA-256; `check_recovered_2026_09.py` (CI) shows the derived tables follow from them (Table S7 recomputed from the 360 per-start records). Remaining limit: Table S6's 200 replications were summarized inside the script (no per-replication records); the run's environment was not recorded |
| 7 | Table S3 configuration rows (120 s per fit; post-fit root checks) | 2 | the archived configuration of the Table 3 runs, which differs | **qualify** (`monthly_protocol.md`) |
| 8 | Figure 1 | 0 printed numbers | an analytic illustration; no plotting script archived | **request** the plotting script, or leave as an illustration |

## What "request" means here

The holders of the missing material are the authors of the August 2026 certification batch
(item 1) and of the development measurements behind items 3 to 5. For each, the files to ask
for are: the script at the commit used, the inputs (or generator and seed), the per-instance or
per-replication output, and the environment (Julia Manifest, solver versions, platform). Once
received they go into a new bundle part (as part 2 was built for the controlled campaign),
their table group or check is added, and their manuscript rows move from "no evidence" to
"raw-reproduced" or "derived-only".

## Effect on the manuscript inventory

| category (2026-10-16 transcription) | values |
|---|---|
| raw-reproduced | 603 |
| derived-only | 19 (now with recovered outputs: item 6) |
| differing | 2 (Table 7 Ipopt times: item 2) |
| no evidence | 46 (items 1, 3, 4, 5, 7) |
| checked at run time only | 2 |

The 46 values without evidence: Table S5 12, Section 5.6 12, Section 5.5 11, Table S2 9,
Table S3 2.
