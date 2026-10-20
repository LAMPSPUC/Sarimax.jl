# Table 7: the Ipopt times

Table 7 (`tab:local_global_forecasts`) prints Ipopt times of 0.6 s (simulated MA(1)), 0.04 s
(simulated ARMA(1,0,1)) and 0.02 s (PJME ARMA(1,0,1)); Section 5.6 says the PJME fit "is
obtained locally in hundredths of a second". The manuscript check reports the two last cells as
DIFFER. This note identifies what was timed and where the printed values come from.

## What is timed

`run_solver_global_value.jl` (archived script and its port, identical in this respect) builds the
JuMP model, then records `rt = @elapsed optimize!(m)`: the wall time of JuMP's `optimize!` call,
which copies the model into the solver and solves it. Model construction, the forecast and the
writing of the record are outside it. Each instance is solved once; the three Ipopt solves run in
one Julia process in the order MA(1), ARMA(1,0,1) sim, ARMA(1,0,1) PJME, so the first one also
pays the compilation of the Ipopt/MOI code path.

## The numbers

| instance | `ecc4878` record | `8ffc512` record (bundle) | archived log (`ecc4878`) | July 2026 experiment report | Table 7 | re-run 2026-10-20 (container) |
|---|---|---|---|---|---|---|
| sim MA(1) | 0.5752 s | 0.5925 s | `rt=0.6s` | 0.6 s | 0.6 | 1.866 s |
| sim ARMA(1,0,1) | 0.001807 s | 0.001815 s | `rt=0.0s` | **0.04 s** | **0.04** | 0.006036 s |
| PJME ARMA(1,0,1) | 0.002354 s | 0.002048 s | `rt=0.0s` | **0.02 s** | **0.02** | 0.012339 s |

Sources: `results/historical/battery-2026-07-ecc4878/global_value_results.jsonl`;
`raw/battery-2026-07/solver/global_value_results.jsonl` and `global_value_run.log` (bundle
part 1); `experiments/experiment_report.md` §"Global-value experiment" at commit `ecc4878`
(branch `v0.2`); `results/reruns/full-all-limit1-2026-10-20/battery_global_value/`.

* No record gives 0.04 s or 0.02 s. The printed values match the July 2026 experiment report,
  whose table also lists the SCIP times of the `ecc4878` records (71.6, 393.1, 601.8 s) but
  Ipopt times that are not those records' (0.0018 and 0.0024 s). The report's Ipopt times
  cannot be traced to any archived run; the log prints them rounded to 0.0.
* The printed SCIP times (69.2, 385.4, 900.0 s) are those of the `8ffc512` records, so Table 7
  combines Ipopt times from the report with SCIP times from a later run.
* The first Ipopt time (0.6 s) includes compilation; the next two show the solve cost, about 2
  ms in the archived run.

## Proposed correction (for the authors)

Either print the archived measurement:

> Time (s), Ipopt: 0.59 (first solve of the process, including compilation), 0.0018, 0.0020

and in Section 5.6 replace "in hundredths of a second" by "in about two milliseconds"; or print
no Ipopt times and state qualitatively that each local solve took milliseconds, against 69-900
s for SCIP. In both cases the caption should say that times are wall-clock times of a single
solve on the archived machine (macOS arm64), that the first Ipopt time includes compilation, and
that re-runs on other hardware give other times (the container re-run: 1.87, 0.0060 and 0.0123
s). Until the manuscript changes, the manuscript check keeps reporting the two cells as DIFFER.
