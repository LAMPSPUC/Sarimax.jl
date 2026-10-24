# Experiment provenance

Which source revision, environment, runner and records stand behind each experiment of the
paper, and the operational details of the solver and initialization studies that the paper
summarizes. The machine-readable form is `../experiment_manifest.csv`; the paper's printed
values are mapped to these records in `../results/manuscript/manuscript_values.csv`.

A historical source state is named only where the records identify it. Where they do not, the
public release is **not** substituted for it.

## Source revisions and environments

| experiment | paper | source state | environment | runner and records |
|---|---|---|---|---|
| Controlled objective comparison (MSE, Huber, MAE, Ridge) | Tables 5, 6, S13, S14; Figure 3; Section 5.4 | Sarimax.jl v1.0.0, commit `14dd58164867ef025d87468f266f36f1930ca61f` | Julia 1.10.12, JuMP 1.31.1, MathOptInterface 1.52.0, Ipopt.jl 1.15.0 with Ipopt 3.14.19 (campaign Manifest, bundle part 2) | `original-scripts/m4-objectives-controlled-2026-10/runner/run_campaign.jl`; protocol `experiment_protocol.toml` (P2); per-series records, bundle part 2 |
| Weekly re-execution under the six-frequency protocol | Table S4; Section 5.1 | the same commit `14dd581` | the same environment | the controlled campaign's single extra run (stationarity by reflection) |
| Six-frequency comparison with `auto.arima` | Tables 3, 4, S12; Figure 2; Section 5.3 | campaign A: commit `aa68d57` (monthly `3d7f883`) | Julia 1.9.4 | bundle part 1, `original-scripts/m4-recreio-2026-08/scripts/run_m4.jl`; per-series records `raw/m4-recreio-2026-08/` |
| `auto.arima` reference scores | Tables 3, 5 | archived outputs of `forecast::auto.arima` | **not recorded**: the R and forecast versions of this campaign are not established (the R 4.6.1 of the matched-implementation checks is not that campaign's) | bundle part 1, `raw/m4-baselines-r-2026-07/` |
| Matched-implementation checks | Table 2 | the July 2026 battery | Python 3.9.6 with statsmodels 0.14.6; R 4.6.1 with forecast (`raw/battery-2026-07/environment/`) | `experiments/runners/battery/`; records `raw/battery-2026-07/` |
| Exogenous-convention simulation | Table S2 | reconstruction at `4e5faaf`; alternative exogenous API check at `925ebfb` | as recorded with the received scripts | `results/historical/certification-2026-08/received/` (generator, fitters, per-DGP aggregate) |
| AICc pairing (1,099 fits) | Section 5.5; Supplementary text | `c73c1bb` | | `aicq.jl`, `aicq.R`; aggregate `i_paridade_aicc_1099` (received) |
| CSS-to-ML likelihood distance | Section 5.5 | `c73c1bb` with a reconstructed, uncommitted patch (`seasonalPresamplePrior`, merged in `3e6cf20` and reverted in `9e406a8`) | | `mixdist.jl`; aggregate `i_distancia_css_ml` (received) |
| Ten-seed initialization diagnostic | Table S8; Section 5.5 | snapshots based on `3578849`, checked against `f90e554`; the keywords of arms B and D were never committed | | `g4.jl`, `g4p.jl`; per-seed rows (received) |
| Paired 1,500-series initialization comparison | Table S9; Section 5.5 | `f90e554` (on no branch; `results/reruns/gate-1500-f90e554-2026-10-22/received/Sarimax_f90e554.bundle`) | Julia 1.12.3 and the August Manifest (`results/historical/certification-2026-08/received/ambiente/Manifest.toml`) | August aggregate; per-series records of the rerun, `results/reruns/gate-1500-f90e554-2026-10-22/` |
| Global-optimization grid (46 cases) | Tables S5, S6, S7; Section 5.6 | a standalone JuMP harness, not the package API (`results/historical/certification-2026-08/received/harness/`) | Julia 1.12.3, JuMP 1.31.1, SCIP.jl 0.11.6 on SCIP 8.0.0, Ipopt.jl 1.15.0 (the same Manifest) | August reconstruction `results/historical/certification-2026-08/`; rerun with logs `results/reruns/certification-2026-08-rerun-2026-10-21/` |
| Held-out local and global fits | Table 7; Section 5.6 | the July 2026 battery (records of `ecc4878` and `8ffc512`) | as recorded; SCIP 8.0.0 | bundle part 1, `raw/battery-2026-07/solver/`; logged rerun `results/reruns/full-all-limit1-2026-10-20/` |
| Stable/CVaR campaigns | Tables S15, S16; Section 5.4 | per campaign: weekly and yearly `3d7f883`; monthly `aa68d57`; quarterly and hourly `fc2c482` with PR 26; daily `a1ebc0d` | per campaign (`env/Manifest.toml` of each archive) | `stable_admissibility_gavea` table group; bundle part 2 |
| Stationarity, invertibility and MA normalization census | Section 5.5 | `4e8cf11`; the four normalization-off cells used a patch that no longer exists | lost Manifest; the eight cells ran in one environment | `stable_admissibility_gavea` (admissibility grid) |
| Quantile, weighted-regularization and multistart examples | Tables S10, S11; Figure S1 | the September 2026 simulation scripts | | `results/historical/revision-2026-09/recovered/` |

Two behaviours of the search at `14dd581` qualify every automatic selection made with that
revision: the stepwise search can keep its neighbourhood when the incumbent becomes the null
model, and the Gaussian likelihood that ranks candidates mishandles the drift level. Both can
change the selected order, and using one search for all objectives does not make their effects
cancel. The controlled comparison describes that implementation.

## Global-optimization grid

* **Records.** The August results are a reconstruction from the session record of the run, not
  per-instance solver logs. The October rerun records objectives, bounds, raw solver status,
  coefficients, node counts and times, with a SCIP log per case: 43 cases returned and 3 were
  interrupted by the external clock, covering all 46 identifiers. 20 of the 21 August time-limit
  outcomes keep that status; case 38 did not return in the main rerun. The isolated runs of 22
  October are kept apart and are not substituted into the main campaign.
* **Gaps.** For a feasible objective f and a lower bound L the paper reports
  g = (f - L) / max(|f|, 1e-12); the relative excess of the local objective is
  (f_Ipopt - f) / max(|f|, 1e-12). The two differ, and both can differ from the gap SCIP
  displays: the isolated normalized gaps of cases 34 and 38 are 5.1054% and 1.9262%, while their
  logs display about 5.38% and 1.96%. The dual value -1e+20 of case 37 is SCIP's sentinel, not a
  bound.
* **Times.** SCIP's own solving time and the elapsed time around the solver call are both kept;
  for case 44 they are 175.132 s and 177.412 s.
* **Case 35.** It did not return under the external limits in the shared and the isolated runs.
  Stack samples of the isolated runs, at 2, 12 and 14 minutes, place it in presolve, inside the
  `completesol` heuristic (a sub-SCIP solving an LP), which runs because SCIP.jl 0.11.6 passes
  every warm start as a partial solution. SCIP's log is block-buffered when written to a file,
  so the last printed line of a killed process does not show where it stopped. A diagnostic
  with that heuristic switched off changes a solver parameter and is not counted as a
  replication.
* **Scope.** The solver records concern the standalone harness; the package commit recorded in
  the rerun's stamp is environment metadata and does not make the harness the package API.

## Paired 1,500-series initialization comparison

**What the iteration percentage counts.** Each record is one automatic model-selection search,
that is one series. Its `iterMax` is the largest Ipopt iteration count among the candidate
models that search fitted (`iterMax = maximum(iters)` in `m4_gate1500.jl`), and the reader
`le_gate.jl` reports the share of searches with `iterMax >= 200`. The percentage is therefore
the share of searches in which **at least one candidate** reached the 200-iteration ceiling; it
is neither a solver failure rate nor the share of individual fits at the ceiling.
`scripts/maintainers/check_gate_rerun_2026_10.py` recomputes it, and the mean sMAPE and MASE,
from the per-series records.

**The rerun.** `results/reruns/gate-1500-f90e554-2026-10-22/` (`RUN.md`): the source tree of
`f90e554` (identical to the tag in the bundle), the M4 input hashes, Julia 1.12.3 and the August
Manifest, one configuration at a time on an otherwise idle machine. Both configurations returned
forecasts for all 1,500 series:

| configuration | mean sMAPE | mean MASE | searches at the ceiling | August aggregate |
|---|---|---|---|---|
| `:penalized` | 13.8812 | 0.9512 | 48.3% | 13.9000 / 0.9528 / 47.4% |
| `:innovations` | 13.9497 | 0.9476 | 0.2% | 13.9395 / 0.9471 / 0.2% |

Under `:innovations`, 1,494 of 1,500 series reproduce the August per-series record (order, and
sMAPE and MASE to six decimals); the other six select another order. The time-limit warnings
(eight under `:penalized`, four under `:innovations`) fall in one block of five other series, so
the order disagreements are not among the fits that hit a time limit.

**Timing.** The rerun's totals are 4,444 s and 2,520 s; the August averages were 0.848 s and
0.609 s per series. The two come from different execution contexts, and the August timing was
not a paired benchmark: neither establishes a transferable runtime reduction. The comparison
also gives no uniform forecasting ranking: the lower sMAPE and the lower MASE belong to
different configurations.
