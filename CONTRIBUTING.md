# Contributing to Sarimax.jl

Thanks for your interest! PRs for bug fixes, new models and documentation are
very welcome. For non-trivial changes, please open an issue first to discuss
the design.

## Development setup

```julia
] dev https://github.com/LAMPSPUC/Sarimax.jl
] test Sarimax
```

The test suite includes cross-implementation reference fixtures (KPSS/OCSB vs
statsmodels/pmdarima, coefficients vs R `arima(method = "CSS")`) and Aqua.jl
quality checks. All of it must stay green.

## What to know before touching the estimation code

Sarimax.jl is deliberately **not** a Kalman-filter package. Estimation is
conditional least squares (CSS) formulated as an explicit JuMP optimization
problem — that is the package's identity and its extension point (custom
objectives, constraints, regularization, solver swaps, certified global optima
via SCIP). Read the "Model formulation and comparability" section of the README
before proposing changes to the likelihood, the information criteria or the
seasonal polynomial: every convention there is chosen to be *declared and
verifiable* against R's CSS mode.

Conventions:

- Public API is snake_case; keyword arguments are currently camelCase (a v1.0
  rename is planned). Deprecated camelCase function names live in
  `src/deprecated.jl`.
- New statistical behavior needs a reference test: either a fixture generated
  from R/statsmodels/pmdarima (document the exact call and version in a
  comment — see `test/reference_values.jl`) or an exact analytical case (see
  the ψ-weight and drift tests).
- Every `@testset` runs inside one outer testset in `test/runtests.jl`; never
  rely on file execution order.
- Solver interactions must check `termination_status`; candidates with failed
  status are not selectable in `auto`.

## Release checklist

- Update `CHANGELOG.md` (Keep-a-Changelog format).
- Bump `version` in `Project.toml` (SemVer; coefficient-changing behavior is
  breaking).
- Full test suite + Aqua green on Julia 1.10 and the latest stable release.
