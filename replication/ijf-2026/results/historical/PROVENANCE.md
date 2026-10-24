# Derived tables archived in the repository

Reference tables that are not in the artifact bundle v1, kept here so that the values the
manuscript prints can be traced to a file. They are DERIVED results: tables computed by the
authors' analysis scripts at the time, copied unchanged except for line endings (normalized
from CRLF to LF, as for every text file of this package; the SHA-256 of the original bytes is
listed below). Nothing here is regenerated from raw data unless a table group says so.

| Folder | Origin | Used for |
|---|---|---|
| `m4-paired-2026-09/` | `Sarimax_revisao_resultados` (Google Drive folder of the September 2026 revision analysis; export of 2026-10-10, zip SHA-256 `bdc072b6a71dbac3…`), written 2026-09-28 by `b_m4_distributional.py` | reference values of the `m4_paired` table group (manuscript Tables 4 and S8, Figure 2), which REGENERATES them from the bundle's per-series files |
| `revision-2026-09/` | same export, written by `efg_figures.py` from `efg_julia_experiments.jl` (cases E, F, G) | consistency check of manuscript Table S10, Figure S1 and Table S11. The simulation script and its outputs were recovered on 2026-10-20 and are in `revision-2026-09/recovered/` (its `PROVENANCE.md`); `check_recovered_2026_09.py` recomputes the three tables from them |
| `battery-2026-07-ecc4878/` | the July 2026 battery's solver records at commit `ecc4878`, copied byte for byte from the repository history (its `PROVENANCE.md`) | the records the archived solver logs show; compared with the bundle's `8ffc512` records in `reports/global_certification_audit.md` |
| `certification-2026-08/` | `scip_agosto_reconstrucao_21-10.zip` (5,186,807 bytes, SHA-256 `3aefb26af94ccb66…`), received 2026-10-21: the August 2026 certification batch and the records of Table S2 and Section 5.5, RECONSTRUCTED from the session that produced them (its `PROVENANCE.md`) | consistency check of Table S5, Section 5.6, Table S2 and Section 5.5 (`manuscript_sources.csv`, recomputed by `check_certification_2026_08.py`) |
| `updated-2026-10-17/` | `resultados_atualizados_20261017.zip` (65,560 bytes, SHA-256 `44075fadd0a8965f…`): the tables, table scripts and reports of the update of 2026-10-17 | the stable-objective, admissibility-grid and guard tables, which the `stable_admissibility_gavea` table group REGENERATES from the raw archive in bundle part 2 (its reference values are parsed from the copies of these tables archived there) |

The 2026-10-17 tables were computed from `brutos_gavea.zip` (107,137,987 bytes, 177 files
verified against its own SHA256SUMS, shared on 2026-10-16), the raw archive of the September 2026
stable and admissibility campaigns, together with campaign F and the stored auto.arima/Naive2 baselines. That
raw archive is in bundle part 2 (`raw/stable-admissibility-gavea-2026-09/brutos_gavea/`), and
the `stable_admissibility_gavea` table group rebuilds these tables from it (1,028 cells; see
`experiment_manifest.csv`). Two facts recorded in
`updated-2026-10-17/gavea_archive_audit.md` matter when reading them: the stable campaigns of
the hourly and quarterly frequencies stored MASE with the lag-1 naive denominator, which the
tables convert exactly to the M4 seasonal denominator; and the arms of different frequencies
come from different commits and hosts, so they are compared within a row, not across rows.

| File | SHA-256 of the original | SHA-256 after LF normalization |
|---|---|---|
| `m4-paired-2026-09/m4_distribution_quantiles_full.csv` | `c443e297e1e429fa…` | `a1ae1ff62a4dba1d…` |
| `m4-paired-2026-09/m4_paired_differences_full.csv` | `eb8d56130c4960d3…` | `3bda00ea33d33146…` |
| `revision-2026-09/multistart_kstart.csv` | `0c662f76ab2bf7c0…` | `e0aa5439d1098590…` |
| `revision-2026-09/quantile_controlled_example.csv` | `17adeaf180134cf1…` | `522f82214b3a09d8…` |
| `revision-2026-09/weighted_regularization_demo.csv` | `5dcab279783ddad6…` | `8f80a654c35e4555…` |
| `updated-2026-10-17/UPDATED_RESULTS_20261017.md` | `1c648f201c4688fd…` | same |
| `updated-2026-10-17/gavea_archive_audit.md` | `94316881220d5316…` | same |
| `updated-2026-10-17/scripts/common.py` | `62f882f1d7667f3e…` | same |
| `updated-2026-10-17/scripts/s_table5_check.py` | `3f27e73fd62b6710…` | same |
| `updated-2026-10-17/scripts/w_gavea_tables.py` | `82b84243b284ab2f…` | `86d7a7092a3004dc…` |
| `updated-2026-10-17/scripts/x_manuscript_tables.py` | `15dd5689268412cb…` | same |
| `updated-2026-10-17/scripts/y_updated_results.py` | `ee6ccef0a7aae3cb…` | `87a338af65a6426e…` |
| `updated-2026-10-17/tables/admissibility_grid_distribution_quantiles.csv` | `e087fa119b69849b…` | `0906c6d28ea150de…` |
| `updated-2026-10-17/tables/admissibility_grid_paired_differences.csv` | `3c534c81daa42a57…` | `fd5ee14952b95e86…` |
| `updated-2026-10-17/tables/admissibility_grid_paired_differences.tex` | `04f4d32768a19ed8…` | `bc80e1835cab4cf2…` |
| `updated-2026-10-17/tables/admissibility_grid_summary.csv` | `526d21a6ea3d0337…` | `e067f348dfe6cb64…` |
| `updated-2026-10-17/tables/admissibility_grid_summary.tex` | `8e6de67e8bf25b39…` | `91db2b33c39b87ca…` |
| `updated-2026-10-17/tables/admissibility_grid_tail_severity.csv` | `3ee8834c0b81bc9a…` | `937c85a8f2cbf73d…` |
| `updated-2026-10-17/tables/admissibility_grid_union_tail.csv` | `0ecf8f6ffd1743b2…` | `5ce00d867c0e7d0b…` |
| `updated-2026-10-17/tables/admissibility_grid_union_tail.tex` | `fdc2a4842df0982d…` | `a9adc106c31468bb…` |
| `updated-2026-10-17/tables/require_terms_guard_distribution_quantiles.csv` | `b393d334cf603596…` | `4e1b25e5921a86eb…` |
| `updated-2026-10-17/tables/require_terms_guard_paired_differences.csv` | `5a2394f171e85e5d…` | `2614265e55bcef94…` |
| `updated-2026-10-17/tables/require_terms_guard_paired_differences.tex` | `aef5879cc5d3d3cb…` | `1f595947439bad2c…` |
| `updated-2026-10-17/tables/require_terms_guard_summary.csv` | `645e00df18d4170a…` | `113a16589cbc34e5…` |
| `updated-2026-10-17/tables/require_terms_guard_summary.tex` | `3a8c175d3c0313ad…` | `0aad65bf8b6714c6…` |
| `updated-2026-10-17/tables/require_terms_guard_tail_severity.csv` | `d0ca20f0f4650829…` | `687bf293db35a548…` |
| `updated-2026-10-17/tables/require_terms_guard_union_tail.csv` | `3d200a7ff6736976…` | `4e391cbc44491b2e…` |
| `updated-2026-10-17/tables/require_terms_guard_union_tail.tex` | `2d3de86fe7e41e58…` | `256fa5ceeede9333…` |
| `updated-2026-10-17/tables/scip_restricted_yearly.csv` | `bf1d5164188c04be…` | `fc7f35c28b50bfe4…` |
| `updated-2026-10-17/tables/stable_objective_distribution_quantiles.csv` | `7fca42800b521fa0…` | `5ab28df3c556319a…` |
| `updated-2026-10-17/tables/stable_objective_paired_differences.csv` | `ac1989d957880e4b…` | `8c4730401b6db83f…` |
| `updated-2026-10-17/tables/stable_objective_paired_differences.tex` | `a52940f8d5c6ad1c…` | `8a7c8d8a04de0b90…` |
| `updated-2026-10-17/tables/stable_objective_summary.csv` | `7876ef71f1d257d5…` | `92a7c8f0a49c531b…` |
| `updated-2026-10-17/tables/stable_objective_summary.tex` | `d72920f92fcddb51…` | `ccdd130353a7e411…` |
| `updated-2026-10-17/tables/stable_objective_tail_severity.csv` | `7c632dcf2699cc2f…` | `91b16e8bb16d2405…` |
| `updated-2026-10-17/tables/stable_objective_union_tail.csv` | `085ca021baf40d93…` | `cfdb46b20a8eba55…` |
| `updated-2026-10-17/tables/stable_objective_union_tail.tex` | `c03baf2c7f6e8d10…` | `1f039f72ba55f89c…` |
| `updated-2026-10-17/tables/table5_reproduction.csv` | `cb0f43c55d5858e0…` | `1705ebc730fdf998…` |
| `updated-2026-10-17/tables/table6_reproduction.csv` | `6bf062e691c2bbb0…` | `06cf40f9e82fd451…` |
| `updated-2026-10-17/tables/updated_20261017_captions.tex` | `569dc785e171a05e…` | same |
