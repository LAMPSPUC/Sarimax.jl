# Known limitations

Limitations of the archived material and of this replication package, as established while
assembling it. Each is also recorded per experiment in `experiment_manifest.csv`
(`known_limitations`) and `REPRODUCIBILITY_STATUS.md`.

## The manuscript itself

1. **The manuscript source was not available** when the package was assembled. The
   reference values that `tables` checks are the values printed in the historical tables
   that the experiment scripts generated (`historical-tables/` in the bundle), not values
   read from the manuscript. LaTeX labels are given where the historical tables carried them
   (the July 2026 draft); for the August–October 2026 experiments the corresponding
   manuscript tables are unknown. Before publication, each manuscript table must be mapped
   to a manifest row and its printed values compared with `results/expected/`.

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
   monthly; the 2x2x2x2 stationarity x invertibility x log-det census) are BLOCKED: their raw
   files travelled in a separate archive that was not available, their environment ran
   Julia 1.12.3 (outside the package's compat), and the census Manifest and one patch are
   lost. Only their scripts and probe outputs are archived.

## Global optimization evidence

8. **No archived SCIP result carries a dual bound.** The July 2026 battery recorded SCIP's
   termination status and `relative_gap()`, but not the dual bound, tolerances or a SCIP log;
   two status strings in its solver table are hard-coded in the scripts rather than read
   from the solver; and the PJME instance's record changed from TIME_LIMIT to OPTIMAL
   (899.958 s of a 900 s limit) between two commits without a log of the second run.
   These results can be stated as "SCIP reported OPTIMAL with relative gap 0" but cannot be
   re-verified from the archive. The July 2026 M4 yearly study ran SCIP with
   `limits/gap = 0.01`: its "OPTIMAL" rows are certified only to a 1% relative gap.
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
    and the bundle must be supplied with `--from`/`--artifacts`.
14. **`linux/arm64` has not been tested.** Every component exists for arm64 and the
    Dockerfile handles both architectures, but only `linux/amd64` was built and run.
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
