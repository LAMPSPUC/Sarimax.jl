# The August 2026 certification batch and the Section 5.5 / Table S2 records, as received

`received/` holds the material that the collaborator who ran the August 2026 SCIP certification
batch (Table S5, Section 5.6) and the development measurements of Table S2 and Section 5.5
supplied on 2026-10-21, in reply to the request for the batch's per-instance records (items a-f)
and for the records of those three diagnostics (items g-i). It arrived as
`scip_agosto_reconstrucao_21-10.zip` (5,186,807 bytes, SHA-256
`3aefb26af94ccb6639594c14e911783bc345ae2b3053b072e13129b1a43059d7`, 47 files verified against
the `SHA256SUMS` inside it).

**This is a reconstruction, not the raw records.** The batch ran in a temporary directory on the
collaborator's machine that no longer exists; the per-instance CSV files of both batches were
never sent and are lost. What survives was recovered from the record of the session that ran
the batch: tables printed to the console, and the harness rebuilt by re-applying the patches
written in that session, checked by execution (each snapshot loads and counts 32 or 46 cases
under SCIP 8.0.0). `received/00_INVENTARIO.md` (Portuguese, the collaborator's own inventory)
states item by item what exists and what was lost; `received/tabela_scip_lotes1-2_RECONSTRUIDA.csv`
gives the source of every row in its column `fonte`. Times in file names are UTC.

| folder / file | what it is |
|---|---|
| `00_INVENTARIO.md` | the collaborator's inventory of items a-i: what exists, where, what was lost |
| `RESULTADO_SCIP_LOTE1_21-08.md`, `RESULTADO_SCIP_LOTE2_22-08.md`, `GATE_SCIP_21-08.md`, `GATE_TERMO_CRUZADO_20-08.md` | the batch reports and pre-registered gates written at the time (summaries; batch 2's report had never been sent) |
| `tabela_scip_lotes1-2_RECONSTRUIDA.csv` | one row per instance (ids 1-46): cell, label, mode, T, SCIP status, time, certified gap, Ipopt - SCIP (batch 1), objectives and dual bound where recorded, noise-floor assertion, source |
| `harness/` | `scip_prog_lote1_21-08_1843.jl` (batch 1), `scip_prog_lote2_22-08_1014.jl` and `_1435.jl` (batch 2; the second adds `max_wall_time = 600` to Ipopt), and `scip_wd.sh`, the external 900 s clock of ids 34 and 35. `casos()` defines the 46 instances |
| `ambiente/` | the batch's `Project.toml` and `Manifest.toml` (Julia 1.12.3, JuMP 1.31.1, SCIP.jl 0.11.6 on SCIP 8.0.0, Ipopt.jl 1.15.0), and the Manifest's modification time |
| `resultados_agregados/` | the console outputs recovered from the session record: per-batch tables, case 26, the refuted and certified cases, the external-clock log of 34/35, and the outputs of items g, h, i |
| `scripts_g/`, `scripts_h/`, `scripts_i/` | the scripts of the Section 5.5 initialization diagnostic and its 1,500-series gate (g), of the Table S2 simulation (h), and of the Section 5.5 ranking diagnostic (i) |
| `dados_em_disco/` | `rand400.csv`, `g_rand400.csv`, `fix_rand400.csv`: per-series results of the 400-series sample of item i (no M4 test values) |

## Transforms

`received/SHA256SUMS` lists the bytes archived here. They equal the received bytes except:

* The 13 files of `resultados_agregados/` ended with a line the session's terminal appended
  (`Shell cwd was reset to ...`, a path that contains an e-mail address); the line is removed and
  the file ends with a newline. Nothing the program printed is changed.
* `00_INVENTARIO.md`: an e-mail address is replaced by `[the author's address]`.
* `tabela_scip_lotes1-2_RECONSTRUIDA.csv` had CRLF line endings and is kept with LF, as every
  text file of this package.
* Not copied (5.5 MB each; no printed number of the manuscript depends on them): the per-series
  accuracy of the 48,000-fit `:innovations` census of item g, by total and by horizon block.
  They remain in the received zip.

| file | SHA-256 received | SHA-256 archived |
|---|---|---|
| `00_INVENTARIO.md` | `709fd50b9909adfa…` | `9ff10fa1e1747f14…` |
| `tabela_scip_lotes1-2_RECONSTRUIDA.csv` | `4f63d0a8c2b8772e…` | `3e32266017da409a…` |
| `resultados_agregados/g_10sementes_bracos_AB_21-08_0005.txt` | `287e0b1ea88edf46…` | `bb1a5004a723ee1c…` |
| `resultados_agregados/g_10sementes_bracos_CD_21-08_0109.txt` | `bfc1854b2cba67d4…` | `b2b0d3815e7b0028…` |
| `resultados_agregados/g_censo48k_telemetria_21-08_1646.txt` | `c9c6ba3cb367cea3…` | `6aa2fa65df51f0bc…` |
| `resultados_agregados/g_gate_1500_21-08_1104.txt` | `8ca5ab2dd9b20ffc…` | `e74ef10e78a14d4a…` |
| `resultados_agregados/h_discriminador_3dgp_20-08_1350.txt` | `2d41c14a4185e181…` | `fe6e627327986e6d…` |
| `resultados_agregados/i_distancia_css_ml_16-08_0434.txt` | `d128b2fda1a312f1…` | `b7a132a2648cbd2f…` |
| `resultados_agregados/i_paridade_aicc_1099_16-08_0406.txt` | `07d26f3445fd5b64…` | `35097ee9ebdc0582…` |
| `resultados_agregados/lote1_caso26_21-08_1945.txt` | `400c679b3aae3583…` | `763b1ad15a132f0a…` |
| `resultados_agregados/lote1_tabela_32_casos_21-08_1945.txt` | `3634c8698eae1731…` | `2bc2e5036969129e…` |
| `resultados_agregados/lote2_casos34_35_relogio_externo_23-08_1634.txt` | `79bdfa5054bededb…` | `ec178503cd9d9f30…` |
| `resultados_agregados/lote2_celulas_EFG_22-08_1042.txt` | `72f05122dcee85bf…` | `8a4abd7d732ffba5…` |
| `resultados_agregados/lote2_parcial_12de14_22-08_1041.txt` | `69a616b7fbd5a293…` | `314f2f1c039d70dd…` |
| `resultados_agregados/lote2_refutacoes_certificadas_22-08_1042.txt` | `d107d2072c51de4e…` | `ec63fd3d7b0cda85…` |
| `dados_em_disco/censo_blocos.csv` (not copied) | `2d8a5af3c77e7ba2…` | – |
| `dados_em_disco/owa_pareado.csv` (not copied) | `19728aa22594078e…` | – |

## What is checked

`scripts/maintainers/check_certification_2026_08.py` (CI, without the artifact bundle) verifies
`received/SHA256SUMS`; that the consolidated table agrees, row by row, with the per-batch tables
recovered separately (44 rows: the 32 of batch 1 and the 12 of batch 2 that finished); that the
printed medians and wall-time reduction follow from their own per-seed rows and totals; and that
`manuscript_sources.csv` follows from these files. `manuscript_sources.csv` holds the values the
manuscript mapping compares the printed numbers with (`source_kind = derived`): a consistency
check against a reconstructed record, never a reproduction. The reconciliation, the instance
assignment of Table S5 and what disagrees are in `reports/certification_batch_2026_08.md`.
