# July 2026 battery: solver records of commit `ecc4878`

The artifact bundle (`raw/battery-2026-07/solver/`) holds the records of commit `8ffc512`
(2026-07-12, "Battery re-run on v0.3.0"). Two of its solver files were different at the
previous commit, `ecc4878` (2026-07-11), the one whose solver logs the bundle also holds
(`refresh_scip.log` for the MA(1) certificates, `global_value_run.log` for Table 7). They are
copied here byte for byte from the repository history, so that every SCIP record behind the
manuscript can be compared:

| file | git blob at `ecc4878` | at `8ffc512` (bundle) |
|---|---|---|
| `scip_cert_results.jsonl` | `11a92d268a4e480566d17cd2a038ceea226e3abb` | `0298cb1aef950562537f312a102c6803f92b57d0` |
| `global_value_results.jsonl` | `a23540c9592f94f386ca56cc85f55925e2e9ad9e` | `a00d9e6e82e823a0bcb86cfb9c93f28a82297064` |

`scip_scaling_results.jsonl` is identical at both commits. Source path in both:
`experiments/results/raw/solver/` (branch `v0.2`; the directory was untracked from the
default branch on 2026-07-15 by `74eb882`).

The `ecc4878` records are the ones the solver logs show, with one exception: the PJME SCIP
record of `global_value_results.jsonl` (TIME_LIMIT at 601.8 s, relative gap 1.06e-7) has no
log; the log of that commit shows the batch run of the instance terminated by a signal. See
`reports/global_certification_audit.md`.
