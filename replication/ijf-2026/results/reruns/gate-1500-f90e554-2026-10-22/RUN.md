# Re-run `gate-1500-f90e554-2026-10-22`

A NEW execution of the paired 1,500-series gate of 2026-08-21, the source of the Section 5.5
initialization numbers "fits at the 200-iteration limit 47.4% -> 0.2%" and "wall time -28%"
(`reports/certification_batch_2026_08.md` §3). The per-series outputs of August are lost; its
aggregate survives (`received/referencia/agosto_gate_1500_agregado.txt`, the same file as
`results/historical/certification-2026-08/received/resultados_agregados/g_gate_1500_21-08_1104.txt`).
The author of the August run packaged the code and runner for this re-run on 2026-10-22
(`gate_g_1500_para_luiz_22-10.zip`, SHA-256 `f62ecf2f3d39a4a3…482fc284`).

| | |
|---|---|
| code | the package at commit `f90e554`, which is on no branch of the repository: `received/Sarimax_f90e554.bundle` holds it (tag `regen/g-f90e554-22-10`, commit `f90e55445089dc1f82b87a96c8b88ff4165aacbb`). The copy that ran was checked against the tag's tree file by file; `src/models/sarima.jl` SHA-256 `833072403a766069…`, as each log prints |
| environment | Julia 1.12.3 and the August Manifest (SHA-256 `05f0db2d…4dcf`, the file archived as `results/historical/certification-2026-08/received/ambiente/Manifest.toml`), instantiated afresh; `JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `BLAS.set_num_threads(1)` |
| runner | `received/scripts/m4_gate1500.jl`, the runner of 2026-08-21 with two changes the author lists in its header (data folder from `M4_DATA`; the stamp is the SHA-256 of `sarima.jl`). Configuration as in August: `auto` stepwise, `kpssShort` / `seas`, `objectiveFunction = "mse"`, `warmStartFromBox = true`, `maxTimeSeconds = 120`, `requireTermsWhenOverDifferenced = true`, `stationary = invertible = false` with roots checked at `rootMargin = 1e-2`; the package limits each candidate to 10 s of wall time. Arm by `M4_INIT` (`penalized`, `innovations`) |
| data | M4 monthly training and test files, SHA-256 equal to `received/referencia/M4_sha256.txt`; the 1,500 series of `Random.seed!(20260814); sort(randperm(48000)[1:1500])` (`received/referencia/gate1500_sids.txt`) |
| execution | 2026-10-22, Windows x86_64, one arm at a time with no other workload: `:penalized` 17:53-19:08, `:innovations` 19:08-19:51 (UTC-3) |
| pilot | before the run, the 30 longest of the 1,500 series under `:innovations` (`pilot/`): 29 equal to the August per-series record; the 30th (sid 31336) differs again when re-run alone in a warmed-up process, without a time limit |

## Result

`output/agregado.txt` (the August reader `received/scripts/le_gate.jl`) against the August
aggregate:

| arm | sMAPE | MASE | fits at the 200-iteration limit | status OK |
|---|---|---|---|---|
| `:penalized`, August | 13.9000 | 0.9528 | 47.4% | 1500 |
| `:penalized`, re-run | 13.8812 | 0.9512 | 48.3% | 1500 |
| `:innovations`, August | 13.9395 | 0.9471 | 0.2% | 1500 |
| `:innovations`, re-run | 13.9497 | 0.9476 | 0.2% | 1500 |

* **`:innovations`, series by series** (`output/valida_inov.csv`, the author's validator against
  the August per-series accuracy of the 48,000-series census, which per the author equals the
  gate's; the means over the 1,500 series equal the August aggregate, 13.9395 / 0.9471): 1,494 of
  1,500 series have the same order and the same sMAPE and MASE to six decimals. The six that
  differ (sids 9350, 25520, 26926, 27908, 31336, 34822) select another order; none of them is
  among the series where a solve stopped at a time limit, and sid 31336 differs identically when
  re-run alone. They are differences between this platform's and the August platform's
  numerics, not of load.
* **Time limits:** the Ipopt warnings of a solve stopped at its time limit (8 in the `:penalized`
  log, 4 in the `:innovations` log) all fall in one block of five series (positions 1141-1145),
  in both arms; no divergent series is in it.
* **`:penalized`:** there is no August per-series record, only the aggregate. The share of fits at
  the iteration limit is 48.3% against 47.4%.
* **Wall time:** 4,444 s and 2,520 s in total here (-43%), against 1,273 s and 914 s in August
  (-28%). Times depend on the machine, and the August `:innovations` arm ran alongside the census,
  so the August -28% is not a paired measurement (`received/00_LEIA.md`); neither figure is a
  property of the method.

What the re-run supports: the near-elimination of fits at the iteration limit under
`:innovations` (0.2% in both runs, against 47-48% under `:penalized`) and the paired accuracy of
the two arms (sMAPE and MASE within 0.02 of August). What it does not support: the size of the
wall-time reduction.

## Files

* `received/`: the files of the author's package used here: `00_LEIA.md`, the code bundle, the
  three scripts and `referencia/` without `owa_pareado.csv` (5.5 MB, the per-series accuracy of
  the 48,000-series census; its values for the 1,500 series are the reference columns of
  `output/valida_inov.csv`). The code copy and its Manifest are not duplicated: the bundle and
  `results/historical/certification-2026-08/received/ambiente/Manifest.toml` hold them. One file
  changed: the August aggregate ended with a line the author's terminal appended, which carried
  an e-mail address; it is removed (`received_transforms.csv`, both SHA-256). Every other file
  equals the author's list.
* `output/`: the two arms' per-series records (`m1500_pen.csv`, `m1500_inov.csv`), their logs, the
  aggregate and the per-series validation.
* `pilot/`: the pilot's lists, records, logs and validations.
* The validations and the pilot list were written with CRLF line endings on Windows and are kept
  with LF, as every text file of this package; nothing else in `output/` and `pilot/` is changed.
* `SHA256SUMS`: every file of the run.
* `manuscript_sources.csv`: the values the manuscript mapping compares Table S9 and Section 5.5
  with, recomputed from the per-series records by `scripts/maintainers/check_gate_rerun_2026_10.py`
  (which also checks `SHA256SUMS` and the printed aggregate).
