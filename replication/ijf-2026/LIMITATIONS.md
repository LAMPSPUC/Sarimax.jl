# Known limitations

Limitations of the archived material and of this replication package, as established while
assembling it. Each is also recorded per experiment in `experiment_manifest.csv`
(`known_limitations`) and `REPRODUCIBILITY_STATUS.md`.

## The manuscript itself

1. **Not every printed number has an archived source.** The 672 numbers printed in the
   tables, figures and experimental sections of the 2026-10-16 revision are listed in
   `results/manuscript/manuscript_values.csv` and checked by `tables`. Compared with the
   historical tables, 599 agree, 2 differ, 2 are checked only against regenerated values, 23
   agree with derived tables whose raw inputs are not archived, and 46 have no archived source:
   * **Table 7 (`tab:local_global_forecasts`), Ipopt times.** The manuscript prints 0.04 s
     (simulated ARMA(1,0,1)) and 0.02 s (PJME); the archived records
     (`raw/battery-2026-07/solver/global_value_results.jsonl`) give 0.0018 s and 0.0020 s, and
     the historical table printed 0.0. `tables` reports Table 7 as FAIL until the manuscript or
     the archive is corrected; the "hundredths of a second" of Section 5.6 rests on these cells.
   * **Table 3, monthly cell.** Section 5.1 and Table S3 describe the monthly comparison as
     time-limited (120 s per fit) with stationarity checked after estimation; the archived
     monthly cell (campaign A, 3d7f883) had no time cap and imposed stationarity through the
     reflection parameterization. Either the text or the cell's source must change.
   * **Table S4.** Its two weekly MSE values are the two archived experiments, 0.8977
     (campaign A, aa68d57) and 0.9429 (ICARAI, 6a11698). The manifest records the settings in
     which they differ: stationarity by reflection (A) or not (ICARAI), `warmStartFromBox`
     (false, true), `requireTermsWhenOverDifferenced` (false, true), the commit and the Julia
     version (1.9.4, 1.10.12).
   * **No archived source** (46 values): Table S2 (exogenous-convention simulation), Table S5
     and the certification numbers of Section 5.6 (seasonal-grid gaps, the near-cancellation
     and the two certified instances), and the initialization (0.2076 → 0.0863; 47.4% → 0.2%;
     −28%) and likelihood-ranking (1,099 fits; 21.6 → 3.3; 12.1% → 16.5%; +0.0097) numbers of
     Section 5.5. Their runners, records and logs must be archived, or the statements
     removed.
   * **Derived tables only** (23 values): Tables S6 and S7 and Figure S1 agree with the
     tables of the September 2026 revision analysis, whose simulation scripts are not
     archived; the admissibility numbers of Section 5.5 (0.9094–0.9115; 0.9095 → 0.9038) agree
     with the derived tables of 2026-10-17, whose raw archive is not in the bundle (item 7).
   * Figure 1 is an analytic illustration; its plotting script is not archived.

## Provenance of the archived results

2. **Several M4 campaigns ran on commits that are not part of a release.** The objective
   campaigns of August 2026 (`m4_objectives_icarai_*`) ran on feature-branch commits
   87f7bfb, 6a11698 and 5b2ec6b, with mixed commits inside some cells (monthly mse: 38,280
   series at 87f7bfb, 9,720 at 5b2ec6b). Their tables regenerate exactly from the archived
   per-series files, but a re-run executes the protocol on the release code, and the
   estimators changed before the release (huber's warm-start scale, ridge's determinant
   factor: the a1ebc0d re-run moved ridge's weekly OWA by +0.1431). These campaigns are
   superseded by the controlled `m4_objectives` experiment.
3. **The headline M4 campaign** (`m4_benchmark`, recreio campaign A) ran at two commits
   (aa68d57 for five frequencies, 3d7f883 for monthly), both ancestors of the release, on
   Julia 1.9.4 with two MathOptInterface versions that were reconstructed, not recorded.
4. **The auto.arima and Naive2 baselines** were computed in July 2026 on another host via
   RCall. The R version (4.4.1) and forecast version (8.23.0) are inferred from logs and
   scripts of the same workspace, not recorded with the results. The replication image uses
   exactly R 4.4.1 and forecast 8.23.0.
5. **The July 2026 battery** (tables `tab:*`) was produced on macOS arm64 with Julia 1.11.9,
   R 4.6.1 + forecast 9.0.2 and Python 3.9.6 from a v0.3.0 working tree whose Manifest
   recorded a path, so uncommitted changes cannot be excluded; its solver-scaling table comes
   from an earlier run (v0.1.3). The replication image uses Julia 1.10.12, R 4.4.1 and
   Python 3.12, and the release package: re-runs are expected to agree closely but not
   digit for digit where the estimator changed between v0.3.0 and v1.0.0.
6. **Smaller studies** (contamination/CVaR, admissibility margins, initialization, order
   search, multistart, isolation, axes) ran from un-versioned scripts in the ForecastTester
   workspace; their package commits were not recorded. Their raw results and original
   scripts are archived, but their runners were not ported (`full` is unavailable for them).
7. **The September 2026 GAVEA campaigns** (stable objective on hourly, quarterly, daily and
   monthly; the 2x2x2x2 stationarity x invertibility x log-det census) are BLOCKED for
   regeneration: their raw files are in a separate archive, `brutos_gavea.zip` (107,137,987
   bytes, 177 files with their own SHA256SUMS, shared on 2026-10-16), that is not yet part of
   the artifact bundle; their environment ran Julia 1.12.3 (outside the package's compat), and
   the census Manifest and one patch are lost. The derived tables computed from that archive
   on 2026-10-17 are kept in `results/historical/updated-2026-10-17/`; they found that the
   stable campaigns of the hourly and quarterly frequencies stored MASE with the lag-1 naive
   denominator and converted it exactly to the M4 seasonal denominator.

## Global optimization evidence

8. **No archived SCIP result carries a dual bound.** The July 2026 battery recorded SCIP's
   termination status and `relative_gap()`, but not the dual bound, tolerances or a SCIP log;
   two status strings in its solver table are hard-coded in the scripts rather than read
   from the solver; and the PJME instance's record changed from TIME_LIMIT to OPTIMAL
   (899.958 s of a 900 s limit) between two commits without a log of the second run.
   These results can be stated as "SCIP reported OPTIMAL with relative gap 0" but cannot be
   re-verified from the archive. The July 2026 M4 yearly study ran SCIP with
   `limits/gap = 0.01`: its "OPTIMAL" rows are certified only to a 1% relative gap. The
   global-optimization grid of Table S5 and the certified instances of Section 5.6 (the
   August 2026 certification batch) are not archived at all (item 1).
9. The ported runners and the smoke test record the dual bound, relative gap, raw status,
   primal status and SCIP version, and the smoke test checks the certificate against an
   independent dense-grid optimum. A brute-force grid is an upper bound, not a proof; SCIP's
   certificate is numerical (its feasibility and optimality tolerances), not exact
   arithmetic.

## Data

10. **M4** has no stated licence, so it is not redistributed. `fetch m4` downloads the
    official files at commit b8f01984 of Mcompetitions/M4-methods and verifies them; the
    archived recreio files had the M4 test values (`actual`) removed for the same reason,
    after their per-block metrics were computed from them.
11. **PJME** (PJM hourly load, Kaggle copy): licence not established; fetched from this
    repository's history at commit 308461e and verified, not redistributed.
12. **`datasets/airpassengers.csv` is not the Box-Jenkins airline series.** The package
    documentation (`src/datasets.jl`) calls it the classic Box-Jenkins airline data, but the
    file holds 204 monthly observations from 1991-07 to 2008-06 (values 2.81 to 29.67), the
    shape of fpp2's `a10` series (Australian antidiabetic drug sales); the airline series has
    144 observations from 1949 to 1960. The validation and forecasting tables label this
    file "airpassengers"; the label should be corrected in the manuscript and the package
    documentation.

## This package

13. **The container image and the artifact bundle are not published yet.** Publishing (GHCR,
    Zenodo) is a maintainer action at release time. Until then the image is built locally
    and the bundle must be supplied with `--from`/`--artifacts`. The bundle tarball exists on
    the maintainer host that built it; 299 of its 535 files (everything sourced from git and
    the Table 7 evidence) were rebuilt byte for byte on a second machine, the other 236 come
    from workspaces that only that host holds.
14. **`linux/arm64` is experimental.** The Dockerfile handles both architectures and CI builds
    and exercises arm64 on an arm64 runner, but only `linux/amd64` is the reference platform
    (see REPRODUCIBILITY_STATUS.md for the latest result). On Apple Silicon, Docker runs the
    amd64 image under emulation, which is slow; `--platform linux/arm64 --build` builds a
    native image.
15. **Full re-execution of the M4 campaigns was not performed** while preparing the
    package (it costs hundreds of CPU-hours). The runner was exercised on subsets, which
    checks the execution path, checkpointing and resumption, not the scientific result.
16. **Timing columns** of a re-run measure your machine and its load; they are not
    expected to match the archived ones.
17. **The controlled objective experiment (`m4_objectives`) is pending.** Its configuration
    is a placeholder (release defaults with the objective varied); its table generator and
    expected values will be added with its final outputs (see ARTIFACT_GUIDE.md).

## Found while re-running the battery on the release (to be assessed by the authors)

18. **AIC/BIC of models with regressors.** The release scores information criteria with the
    exact Gaussian likelihood, which removes `X*beta` from `y` as for regression with ARMA
    errors. That is not the ARMAX model `fit!` estimates by default (`exogDynamics = :armax`):
    for the battery's ARX row it gives AIC 1025.87, where CSS, statsmodels and R all give
    855.12. If confirmed, this affects `auto`'s order selection whenever regressors are used.
19. **The archived PJME SARIMAX.jl numbers came from worse local optima.** On 3 of the 6
    rolling origins the v0.3.0 solve stopped at a higher sum of squares; re-solving the
    unscaled v0.3.0 problem reproduces the archived MAE/RMSE exactly, while the release's
    internally rescaled solve finds lower-SSE fits (MAE 2396.26 against 2407.89; RMSE 2883.84
    against 2869.13).
20. **Fragile model selection in the architecture table.** The release's `auto` chooses
    (2,1,2)(0,1,1)12 over the archived (1,1,1)(0,1,1)12; the two are 0.055 AICc apart.
21. **Definition changes since v0.3.0** move some battery cells by construction: `aic`/`bic`
    use the exact likelihood (the CSS values are still recorded by the ported runners), and
    `elastic_net` is a single-stage estimator. Every such change is listed in
    `experiments/runners/battery/PORTING.md`.
