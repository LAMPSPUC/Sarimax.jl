# Updated results — 2026-10-17

Same file schemas as the earlier tables (objective_stable_innovations, objective_distribution_*,
objective_tail_severity, objective_union_tail). Generator: `scripts/y_updated_results.py`.

## Files (`tables/`)

| stem | content |
|---|---|
| `stable_objective_*` | `stable` (CVaR alpha 0.5 / 0.9) vs MSE, six M4 frequencies (+ the monthly alpha = 0.5 subset) |
| `admissibility_grid_*` | Section 5.5 grid: stationary × invertible × MA normalization, monthly, 48,000 series |
| `require_terms_guard_*` | `requireTermsWhenOverDifferenced` off vs on, hourly / quarterly / daily |
| `scip_restricted_yearly.csv` | July SCIP vs Ipopt study, 100 short yearly series, box vs reflection-constrained |
| `table5_reproduction.csv`, `table6_reproduction.csv` | Tables 5-6 recomputed from the ICARAI archive (16/16, 12/12 match) |
| `updated_20261017_captions.tex` | captions |

Each stem has `_summary.csv/.tex`, `_distribution_quantiles.csv`, `_paired_differences.csv/.tex`,
`_tail_severity.csv`, `_union_tail.csv/.tex`.

## Sources and provenance

| table | source | commit | host |
|---|---|---|---|
| stable yearly, weekly | campaign F, `_replication/experiments/results/raw/stable_weekly_yearly.csv` | `3d7f883` | RECREIO |
| stable quarterly, hourly | `brutos_gavea.zip`: `stable_quarterly/`, `stable_hourly/` | `dev@fc2c482+PR26` | GAVEA (Apple M4) |
| stable monthly | `brutos_gavea.zip`: `stable_monthly/` | `aa68d57` | GAVEA |
| stable daily | `brutos_gavea.zip`: `stable_daily/` | `v1.0.0@a1ebc0d` | GAVEA |
| admissibility grid | `brutos_gavea.zip`: `censo_2x2x2x2/` | `4e8cf11` | GAVEA |
| guard | `brutos_gavea.zip`: `requireterms/` (on) and the stable campaigns' MSE arms (off) | `v1.0.0@a1ebc0d` / as above | GAVEA |

`brutos_gavea.zip`: shared by Davi Valladão 2026-10-16, SHA256SUMS 177/177 OK.

## Corrections applied, and what to state

1. **MASE of the GAVEA `stable` campaigns, hourly and quarterly**: written with the lag-1 naive
   denominator; converted exactly to the M4 seasonal naive (`reports/gavea_archive_audit.md` §1).
   Without it, quarterly OWA reads 1.33.
2. Series with a non-finite reference (3 hourly) are excluded from every arm.
3. The arms of different frequencies come from different revisions and hosts: compare within a
   row, not across rows. Only yearly and weekly use the Table 3 MSE cell.
4. Section 5.5 numbers are reproduced exactly (0.9094-0.9115; 0.9095 → 0.9038).
5. The guard's effect is the stepwise defect C1 (`reports/model_scoring_audit.md` §11b), not the
   requirement itself.

## Small differences from the numbers in Davi's e-mails

Monthly alpha = 0.9 (0.8913 / 0.9511 here vs 0.8917 / 0.9514); daily N 3,979 vs 3,971 (ΔOWA
+0.0007 in both); hourly ΔOWA +0.0047 vs +0.0058 (the e-mail value matches the uncorrected MASE);
the census pairs 48,000 series here vs 47,989 (OWA equal to four decimals).
