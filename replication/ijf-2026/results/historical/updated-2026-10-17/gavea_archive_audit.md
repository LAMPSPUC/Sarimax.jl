# GAVEA archive (`brutos_gavea.zip`) — verification and tables

Source: `brutos_gavea.zip` (107,137,987 bytes), Google Drive folder `replicacao_gavea`, shared by
Davi Valladão on 2026-10-16; downloaded 2026-10-17 and extracted to
`_replication_gavea_brutos/conteudo/`. Its `SHA256SUMS`: **177/177 OK**. Its `REPRODUCE.md` is the
one on branch `paper/replication-staging-gavea`. Scripts: `scripts/w_gavea_tables.py` (recomputes
every arm from the raw rows), `scripts/x_manuscript_tables.py` (writes the manuscript tables).

## 1. A MASE defect in two campaigns, corrected exactly

The `stable` campaigns of **hourly and quarterly** (commit `dev@fc2c482+PR26`) wrote MASE with the
**non-seasonal** in-sample naive (lag 1) as denominator; M4 uses the seasonal naive (lag `s`).
Established by comparing with the `requireTerms` campaign, which re-fits the same series with the
seasonal denominator: for every series with identical sMAPE, `MASE_stable / MASE_requireTerms`
equals `den(lag s) / den(lag 1)` computed from the training data, 400/400 in each frequency. The
monthly `stable` campaign and the 2×2×2×2 census use the seasonal denominator (300/300, checked from
the census's stored forecasts); daily and weekly/yearly have `s = 1`.

As stored, those two campaigns give quarterly OWA 1.33 instead of ~0.906. MAE is unaffected, so the
M4 MASE is recovered exactly as `MASE x den(lag 1) / den(lag s)`; all tables below use it. After the
correction the hourly MSE arm of the `stable` campaign and the MSE arm of the `requireTerms`
campaign agree 411/411 series, quarterly 99.6%.

## 2. Section 5.5 — now verified

`tables/stationarity_invertibility_grid.{csv,tex}`. Monthly, 48,000 series per cell, commit
`4e8cf11`. The manuscript's "0.9094 to 0.9115" (normalization off) and "0.9095 → 0.9038" (stationary,
normalization off → on) are reproduced exactly. The eight OWA values equal those in Davi's e-mail of
2026-08-28 to four decimals (he pairs 47,989 series; the local Naive2 reference has no non-finite
value for monthly, so all 48,000 are kept).

Scope to state: the census configuration is `requireTermsWhenOverDifferenced = true`,
`maxTimeSeconds = 120`, `stationary`/`invertible` as in each cell (e-mail of 2026-08-28; the archive
does not carry the command line). The guard makes these searches subject to the stepwise defect C1
(`model_scoring_audit.md` §11b). The four normalization-off cells were produced with a patch that no
longer exists (archive `REPRODUCE.md` §5): comparable, not re-runnable.

## 3. `stable` objective — `tables/stable_objective.{csv,tex}`

Six frequencies, each paired within its own campaign (the campaigns differ in commit and host):
yearly and weekly from campaign F (RECREIO, `3d7f883`, the Table 3 cells exactly); monthly a
5,000-series subsample (`aa68d57`; the alpha = 0.5 arm stopped at 923); quarterly and hourly
`dev@fc2c482+PR26`; daily `v1.0.0@a1ebc0d` with 229 censored stable fits (5.4%: 198 MUMPS aborts,
19 deadlocks, 12 time caps) and 19 errors.

Agreement with the numbers in Davi's e-mails (2026-08-27/28, 2026-09-03): weekly, yearly, monthly
alpha = 0.5, quarterly ΔOWA and the census match. Small differences remain that the files do not
explain: monthly alpha = 0.9 (0.8913 / 0.9511 here, 0.8917 / 0.9514 in the e-mail, same N = 4,997);
daily (N 3,979 here, 3,971 in the e-mail; ΔOWA +0.0007 in both); hourly ΔOWA +0.0047 here against
+0.0058 in the e-mail, which matches the uncorrected lag-1 MASE (+0.0059).

## 4. `requireTermsWhenOverDifferenced` (archive `requireterms/`)

`tables/gavea_owa_summary.csv`, campaign `requireTerms`. Guard off → on: hourly 0.5910 → 0.5910
(411 series, no order changes), quarterly 0.9054 → 0.9053, daily 1.0324 → 1.0552. With the weekly
result (`weekly_mse_reconciliation.md`), the guard's effect is the stepwise defect C1, not the
requirement itself.

## 5. SCIP, short yearly series (July study) — `tables/scip_restricted_yearly.csv`

The files Davi attached on 2026-10-16 (`scip_julho_31-07.zip`) are already in
`ForecastTester/docker_scip/` with the same names; their hashes were not compared with the
attachment, which was not downloaded. 100 M4 yearly series with T ≤ 100, fixed order, Ipopt vs
SCIP (gap 1%, 600 s), box-only vs reflection-constrained (95 series covered; 5 with p or q ≥ 3 not
supported). Not part of Table 7 or of Section 5.6 as written; it does not supply the missing
seasonal-grid and certified-instance records (`global_optimization_audit.md` §6), which Davi
identifies as the 21-22/08 certification batch and offers to repackage.
