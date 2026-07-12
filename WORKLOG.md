# WORKLOG — Sarimax.jl modernization

Handoff log for AI/human collaborators. **Read this before continuing any work.**
Update the "Current state" and "Next steps" sections after every completed batch.

## How to work on this repo

- Branch: `v0.2` (contains v0.2.0 + v0.3.0 work; Project.toml at 0.3.0).
- Test gate: `julia +1.11 --project=. -e 'using Pkg; Pkg.test()'` (~10 min, ~620 assertions).
  Full suite green is REQUIRED before every commit. One commit per coherent batch.
- Coverage measurement: `Pkg.test(coverage=true, julia_args=["--pkgimages=no"])` —
  the `--pkgimages=no` is required on Julia 1.11 (precompiled pkgimages record zero counts).
  Last measured: **91.5%** total (2026-07-12).
- Push: remote may be HTTPS; the `gh` credential helper is configured. Changes under
  `.github/workflows/` need the `workflow` token scope (or SSH remote).

## Non-negotiable design decisions (user-approved; do NOT relitigate)

1. **No Kalman filter, ever.** Package identity = CSS estimation as an explicit JuMP
   optimization problem. "Exact ML" is out of scope; `initialization = :warmup` is the
   R-compatible CSS convention (matches `arima(method="CSS")` to ~1e-5; pinned in
   `test/reference_values.jl`).
2. **Multiplicative seasonal form is the default** (`seasonalForm = :multiplicative`);
   `:additive` is the legacy fallback; `:free` reserved (throws).
3. **OCSB lag-selection IC call looks argument-scrambled — it is deliberate.**
   It reproduces pmdarima exactly; "fixing" it breaks 13/31 fixtures. See the comment
   in `src/statistical_tests.jl`. Do not "correct" it.
4. Sparse K counting (|coef|>1e-5) only for elastic-net fits; everything else counts
   declared parameters.
5. SCIP/Alpine stay hard dependencies (identity: certified global optima out of the box).
6. snake_case API with camelCase `@deprecate` shims until v1.0; keyword args stay
   camelCase until v1.0.
7. `allowMean`/`allowDrift` mutually exclusive; drift = differentiated-t regressor.

## Editing gotchas (learned the hard way)

- `src/models/sarima.jl` contains Unicode identifiers (ϕ θ Φ Θ ϵ ŷ σ²) with
  inconsistent normalization: **string-matching edits frequently miss**. Prefer
  line-number surgery or regex with ASCII-only anchors; always `assert` matches in
  edit scripts and verify with a compile check (`julia --project=. -e 'using Sarimax'`).
- `using Sarimax, StateSpaceModels` in the same scope makes `SARIMA` ambiguous
  (both export it) — use `import StateSpaceModels` in scripts/tests.
- MLJModelInterface "light mode": do NOT call `MLJModelInterface.nrows` inside the
  package — count rows via Tables (see `src/mlj.jl`).
- Logging macros with parentheses can't take `key = val` groups: use space form
  (`@warn "msg" exception = e`).
- Test files are `include`d inside one outer `@testset` in `runtests.jl`; `using`
  statements at file top level are fine (include evaluates at module scope).

## Completed batches (all pushed to origin/v0.2)

| Commit | Batch | Content |
|---|---|---|
| 6644db2 | A | Bug fixes: exog forecast indexing, seasonal predict bounds, stepwise constant toggle, solver status check, OCSB cleanup |
| c654361 | B | Declared-CSS loglik/ICs, honest K, same-sample auto search, integrated-scale forecast variances, concentrated "ml", runtests wrapper (suite never ran past sarima_fit.jl before!) |
| b134415 | C | Deps cleanup, compat julia=1.10, stepwise default, formulation disclosure docs, CHANGELOG, v0.2.0 |
| a213e0f | D | Multiplicative Box-Jenkins default (fit + predict cross terms, form-aware polynomials) |
| 2375b42 | E | snake_case + @deprecate shims, StatsAPI (coef/residuals/nobs/fitted/vcov/stderror), cssResiduals + numerical-Hessian standard errors, v0.3.0 |
| de139c5 | F | initialization=:zeroed|:warmup (R-CSS parity ~1e-5), R coefficient fixtures in CI, solver gate in auto, Aqua (fixed 6 unbound type params) |
| 8767296 | — | CI matrix 1.6 → 1.10 |
| da1d38d | G | Real drift term, stationary=true (reflectionToAR/Levinson), readable show/print with coefficient table + std errors |
| b4843d3 | H | ljung_box_test, jarque_bera_test, boxcox_transform/inverse/lambda (Guerrero), cross_validation (rolling origin) |
| 1ff3c94 | I | stepwiseSearch refactor 716→85 lines (bit-for-bit equivalent, pinned), gridSearch restructure, auto(parallel=true) opt-in |
| 7a4726f | J | Tables.jl input, RecipesBase plot recipe, MLJ SARIMAForecaster, CI macOS, CONTRIBUTING.md, @inferred tests |

## Current state (2026-07-12, batch K IN PROGRESS)

**Batch K — test improvements** (implemented, gate suite running):
- [x] K1: removed dead `set_optimal_start_values` from src/fit.jl
- [x] K2: prediction-interval nominal-coverage tests (test/statistical_properties.jl):
      random walk 100 replicas ×4 horizons, estimated AR(1) 50 replicas ×3
- [x] K3: Monte Carlo bias test AR(1) (20 replicas, T=300)
- [x] K4: SCIP path test — validated standalone: OPTIMAL certificate in ~3s at T=40,
      θ agrees with Ipopt to 4e-4

**Batch L — documentation** (next):
- [x] L1: README rewritten end-to-end: runnable quickstart with REAL output (auto on
      AIR_PASSENGERS → SARIMA(4,1,1)(0,1,1)[12], AICc 465.457), fit-options tour,
      verified-against-R coefficient table, formulation/comparability section,
      diagnostics/Box-Cox/CV examples, ecosystem section
- [x] L2: docs/make.jl now `using Sarimax` (was include)
- [x] L3: docs/src/reference.md rewritten name-based (no stale signatures), all new
      exports included; StatsAPI methods documented via `f(::SARIMAModel)` entries
- [x] L4: docs/src/tutorial.md written (was EMPTY): FPP-style identify→estimate→
      diagnose→forecast→validate + SARIMAX section with ARX warning
- [x] L5: doctests remain OFF (legacy fake jldoctest blocks in docstrings); index.md
      feature list refreshed. Local docs build validation launched.

## Next steps after L (prioritized backlog)

1. auto improvements #1-#4 from the comparison report (2026-07-12 conversation):
   final refit of the winner without minConditioningObs; fix maxP/maxQ cap formula
   (`n/3*m` should be `n/(3m)`, src/models/sarima.jl ~line 1995); unit-root proximity
   margin in checkModelStationarityInvertibility (R uses 0.001); initial-phase cleanup
   of stepwiseSearch (IC=Inf pattern instead of considerModel juggling).
2. auto #5: `lambda = :auto` (Box-Cox integrated into auto + back-transform in predict!).
3. Convert legacy fake jldoctests to real ones; enable doctest=true.
4. Re-run the experiments/ manuscript battery on the new code (validation becomes
   numerical equality vs R-CSS with :warmup; exog forecasting tables were contaminated
   by the pre-A exog indexing bug). Update final_claims (§4 softened, §5 rewritten:
   boundary pile-up was an additive-form artifact; test counts now ~620).
5. v1.0: remove camelCase shims, rename keyword args, Spec/Fit immutable refactor, JET.
6. Phase 3: missing-data-as-decision-variable spike (optimization-native imputation).
