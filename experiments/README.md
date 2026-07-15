# SARIMAX.jl Experiments (Replication Package)

Reproducible benchmark pipeline that populates the four experiment tables of the manuscript.
Bounded software-architecture claims; no fabricated results.

## Layout

```
experiments/
  README.md                  this file
  benchmark_status.md        living status / blockers / file->table mapping
  experiment_protocol.md     design of the four blocks
  experiment_report.md       final report (commands, results, interpretation bounds)
  .venv-benchmarks/          Python env (statsmodels baseline)
  scripts/                   run_*.{jl,py,R}, combine_*, setup/
  results/
    raw/{environment,validation,forecasting,architecture,solver}/   JSONL + raw outputs
    processed/               intermediate merged data
  tables/                    table_*.csv + table_*.tex (manuscript-ready)
```

## Prerequisites

- Julia **1.11** (package compat excludes 1.12): `julia +1.11 --project=.`
- Python venv: `source experiments/.venv-benchmarks/bin/activate`
- R 4.6.1: `/opt/homebrew/bin/Rscript`

## Manuscript mapping

| Table | Source |
|-------|--------|
| Validation (`tab:validation_implementations`) | `tables/table_validation.tex` |
| Forecasting (`tab:forecast_oos`) | `tables/table_forecasting.tex` |
| Architecture (`tab:architecture_extensibility`) | `tables/table_architecture_checks.tex` |
| Solver (`tab:solver_comparison`) | `tables/table_solver_comparison.tex` |

Tables are NOT auto-inserted into `../chapters/experiments.tex`; see `experiment_report.md` for
recommended replacements.
