# Recovered scripts and outputs of the September 2026 revision analysis

The derived tables one directory up (`quantile_controlled_example.csv`, `weighted_regularization_demo.csv`,
`multistart_kstart.csv`) were listed until 2026-10-20 as "derived table only: the simulation
script and its outputs are not archived". Both were found in the analysis workspace of the
September 2026 revision (`paper_results_revision/scripts/` and `paper_results_revision/data/`,
written on 2026-09-11 between 18:17 and 18:38 local time) and are copied here byte for byte
(they were already LF text; `SHA256SUMS` lists every file).

| file | what it is |
|---|---|
| `scripts/efg_julia_experiments.jl` | parts E (quantile loss on a skewed AR(1), 200 replications, Table S6), F (coefficient-specific penalty weights on a six-regressor model, Figure S1) and G (K = 30 starts per case and initialization, Table S7). `SEED = 20260911` (E), +1 (F), +2 (G). Its header states that it ran against the package at master `f96feac`; the run itself did not record a commit |
| `scripts/efg_figures.py`, `scripts/common.py` | the tables and figures of E, F, G written from the outputs below (`common.py` holds the workspace paths) |
| `scripts/a_ar_stationarity.py` | part A: the AR(2)/AR(3) stationarity regions drawn through the package's reflection recursion and validated by polynomial roots (the AR(3) cross-section figure) |
| `outputs/quantile_example_summary.csv`, `quantile_example_paths.csv`, `quantile_example_ordering.txt` | part E: the 200-replication summary (the source of Table S6), one representative path, and the ordering check (held in 200 of 200 replications) |
| `outputs/weighted_penalty_table.csv`, `weighted_penalty_path.csv`, `weighted_penalty_meta.txt` | part F: the coefficient table and the full regularization path |
| `outputs/multistart_kstart.csv` | part G: one record per start (360 = 6 cases x 2 initializations x 30 starts) with objective, coefficients, status, iterations and seconds |
| `outputs/ar_stationarity_validation.json` | part A: the root-based validation counts |
| `outputs/_log_part_eg.txt`, `outputs/_log_part_a.txt` | the console logs of the runs |

`scripts/maintainers/check_recovered_2026_09.py` verifies, in CI and without the artifact
bundle, that the derived tables follow from these outputs: Tables S6 and the Figure S1 table
equal the summaries the Julia script wrote, and Table S7 is recomputed from the 360 per-start
records with the rule of `efg_figures.py` (every column of every row agrees).

Limits. Part E's summary aggregates 200 replications inside the Julia script; the
per-replication values were not written, so Table S6 can be recomputed from its summary but
not from per-replication records. The runs' environment (Julia version, Manifest) was not
recorded. "Airline, log AirPassengers" in part G is the package's `AIR_PASSENGERS` file,
which is `fpp2::a10` (see `src/datasets.jl`), not the Box-Jenkins series.

Re-running them is a NEW execution against the current package (`julia --project=<package>
efg_julia_experiments.jl E F G`, then `python efg_figures.py`); part G's 360 solves took 18 s in
total in the archived run. It was not done for this package.
