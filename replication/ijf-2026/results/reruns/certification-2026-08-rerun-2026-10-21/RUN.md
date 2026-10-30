# Re-run `certification-2026-08-rerun-2026-10-21`

A NEW execution of the August 2026 certification batch (Table S5, Section 5.6), kept separate
from the batch's reconstructed record (`results/historical/certification-2026-08/`) it is
compared with. The batch's author ran it on 2026-10-21 and supplied it on 2026-10-22
(`scip_rerun_replicacao_21-10.zip`, SHA-256 `cb802a22582e5d5e…1d008b`, 154 files and the author's
`SHA256SUMS`). It records, per instance, what the August batch did not: the series, Ipopt's
status and coefficients, SCIP's raw status, node count, coefficients and a full log.

| | |
|---|---|
| expectation | written before the run (`received/PREREG_RERUN_21-10.md`): certified objectives equal to the record to about 1e-6; times and the gaps of time-limited instances not comparable |
| harness | `received/scip_rerun.jl`: the archived harness `scip_prog_lote2_22-08_1435.jl` with a patch that changes only what is recorded (`received/diff_vs_snapshot_22-08_1435.patch`): model, bounds, warm start and SCIP `limits/time = 600` unchanged; SCIP's display level back to its default (one log per case); Ipopt's `max_wall_time = 600` for all 46 cases (in August from 2026-08-22 14:35 UTC on; Ipopt took under 2 s in every case) |
| environment | the batch's own: Julia 1.12.3 and the batch's Manifest (SHA-256 `05f0db2d…4dcf`, the file archived as `results/historical/certification-2026-08/received/ambiente/Manifest.toml`; `received/Manifest_21-10.toml` is the same bytes): JuMP 1.31.1, SCIP.jl 0.11.6 on SCIP 8.0.0, Ipopt.jl 1.15.0. The harness does not load the package |
| instances | the 46 of the batch. The 43 the run wrote are those of `experiments/configs/certification_2026_08_instances.csv`: the 40 synthetic series value for value, the three M4 series (ids 28-30) the standardized M4 monthly training series 2772, 7389 and 7510 |
| execution | 2026-10-21, 19:20-20:35 (UTC-3): 4 processes at `nice 10`, one case per process (`received/roda_caso.sh`), each killed after 900 s by an external clock, serial BLAS; the machine was shared (1-minute load average 6.4-12.0 at the cases' start). Ids 34, 35 and 38 were tried again under a 3600 s clock (20:40-21:40) |
| not done | the replication container: this run is the batch's environment, not the package's. `./reproduce.sh full --experiment global_optimization_grid` would be another execution |

## Result

`raw/certification-2026-08/summary.md` and `comparison.csv` (one row per instance), written by
`compare_certification_2026_08.py`:

* **43 of 46 cases finished.** The 23 instances the batch certified are certified again, with the
  same certified gap to three digits; where the record holds the certified objective (ids 33, 36,
  44, 45) it agrees at its printed precision. 20 of the 21 time-limited instances stop at the
  time limit again (the author's `RESULTADO_RERUN_21-10.md` writes 19 of 20; its own
  `COMPARACAO_RERUN_vs_AGOSTO.csv` and the record count 21). The Ipopt objective agrees in the 9 instances whose record holds it, and
  Ipopt - SCIP in the 32 of batch 1.
* **Ids 34, 35 (T = 480) and 38 (T = 960) did not return** in this run: each process was
  killed at 900 s, and again at 3600 s, with 3 or 4 SCIP processes running at once. Their logs
  end where SCIP's output buffer was last flushed, not where SCIP was: SCIP block-buffers its
  output when it goes to a file, and a killed process loses the buffer (the author's correction
  of 2026-10-22; the SCIP part of the 900 s and 3600 s logs has the same size, 8,531 bytes for
  34 and 8,960 for 38). 34 and 35 did not return in August either; 38 stopped at the time limit
  in 600.9 s in August, without the external clock.
* **Isolated runs, 2026-10-22** (`received-2026-10-22/`): the same harness and parameters, one
  SCIP process at a time. Id 34: time limit at 600.01 s, 182 nodes, incumbent 455.242254 (the
  Ipopt point), dual bound 432.000151, certified gap 5.11e-2. Id 38: time limit at 600.01 s, 111
  nodes, incumbent 969.442960 (the Ipopt point), dual bound 950.769961, certified gap 1.93e-2.
  Id 35: killed at 900 s twice; stack samples at 2, 12 and 14 minutes are inside SCIP's presolve,
  in the `completesol` heuristic (a sub-SCIP solving an LP), which runs because SCIP.jl 0.11.6
  passes every warm start as a partial solution. Why 34 and 38 return alone and not with other
  SCIP processes running was not measured. A diagnostic with `completesol` switched off
  (`diag/`, a changed parameter, not a replication) is kept but not used.
* **Table S5**, by the harness's cells (OPTIMAL / TIME_LIMIT / not returned): every cell as in the
  record except cell E, 2/1/3 against 2/2/2 (id 38, T = 960, did not return in the 2026-10-21
  queue); in all, 23/20/3 against 23/21/2. In the isolated runs of 2026-10-22, ids 34 and 38 stop
  at the time limit and id 35 does not return.
* **Section 5.6**: id 26 Ipopt 46.660913 and SCIP 45.314976 (time limit), id 44 57.868044 ->
  57.536983 (certified, gap 1.09e-9) and id 45 54.706978 -> 53.681295 (certified, gap 1.07e-8),
  as in the record. Seasonal grid (cell B): largest Ipopt - SCIP 2.52e-9 as in the record;
  remaining relative gaps 1.73e-4 / 4.63e-1 / 5.86e-2 (smallest / largest / median), against
  1.94e-4 / 4.90e-1 / 1.51e-2 in August: the gaps at a time limit depend on the machine's load.
* Id 37 stops at the time limit without closing its root node (0 nodes, dual bound `-1e+20`), as in
  August.

## Files

* `received/`: the files as received, except that the host name in the stamp line of 49 logs is
  replaced by `[redacted]`, and the two CSV files written with CRLF line endings
  (`COMPARACAO_RERUN_vs_AGOSTO.csv`, `tabela_rerun_21-10.csv`) are kept with LF, as every text
  file of this package (`received_transforms.csv`, both SHA-256 of each). `SHA256SUMS.received`
  is the author's list (it also hashes itself); `SHA256SUMS` lists the archived bytes.
* `received-2026-10-22/`: the author's correction (`scip_correcao_22-10.zip`, 16 files with its
  list): the updated `RESULTADO_RERUN_21-10.md`, the isolated runs of ids 34, 35 and 38
  (`logs/`, `saida/`) and the diagnosis of id 35 (`diag/`). The host name in 5 logs is replaced
  by `[redacted]` (`received_transforms.csv`); the folder has its own `SHA256SUMS` and
  `SHA256SUMS.received`. Its `RESULTADO_RERUN_21-10.md` supersedes the one in `received/`, which
  is kept as received.
* `raw/certification-2026-08/records.csv`: the run in the format of `run_certification_2026_08.jl`,
  written by `experiments/runners/global/import_certification_rerun_2026_10.py` from `received/saida/`
  (and, for the three cases killed, from their logs).
* `raw/certification-2026-08/comparison.csv`, `summary.md`: `compare_certification_2026_08.py`.

Regenerate and check (no solver runs): `python
scripts/maintainers/check_certification_rerun_2026_10.py` verifies the files against both lists,
the instances against the package's, and that the three outputs follow from them.
