```@raw html
<div style="width:100%; height:150px;border-width:4px;border-style:solid;padding-top:25px;
        border-color:#000;border-radius:10px;text-align:center;background-color:#99DDFF;
        color:#000">
    <h3 style="color: black;">Star us on GitHub!</h3>
    <a class="github-button" href="https://github.com/LAMPSPUC/Sarimax.jl" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LAMPSPUC/Sarimax.jl on GitHub" style="margin:auto">Star</a>
    <script async defer src="https://buttons.github.io/buttons.js"></script>
</div>
```

# Sarimax.jl Documentation

## Introduction

Sarimax.jl is a groundbreaking Julia package that revolutionizes SARIMA (Seasonal Autoregressive Integrated Moving Average) modeling by seamlessly integrating with the JuMP framework — a powerful optimization modeling language. Unlike traditional SARIMA implementations, Sarimax.jl leverages JuMP's optimization capabilities to provide precise and highly customizable SARIMA models.

### Key Features

* Multiplicative Box-Jenkins SARIMA (additive form available)
* Swappable objective functions: MSE, MAE (L1), Huber, quantile / pinball loss
  ("quantile"), concentrated Gaussian CSS ("ml"), exact treatment of the initial
  observations ("ml_exact"), ridge, penalized elastic net, and a tail-oriented CVaR
  criterion ("stable")
* Coefficient-specific regularization weights (`lambda` per coefficient or per block),
  which is what an adaptive Lasso needs
* Certified globally optimal estimates via SCIP; any JuMP solver via `fit!(optimizer=…)`
* Automatic order selection (Hyndman-Khandakar stepwise, grid, opt-in parallel)
* Stationarity/invertibility **by construction** (reflection-coefficient parameterizations)
* Exogenous variables (ARX by default, regression-with-ARIMA-errors available)
  and outlier dummies inside `auto`
* StatsAPI: `coef`, `stderror`, `vcov`, `residuals`, … with CSS standard errors
* Residual diagnostics (Ljung-Box, Jarque-Bera), Box-Cox (Guerrero λ), temporal
  cross-validation, scenario simulation
* Several CSS conditioning conventions; the default `:innovations` penalizes a
  free pre-sample block, and `initialization = :warmup` matches R's
  `arima(method = "CSS")` to ~1e-5 (pinned in CI)
* Tables.jl input, Plots.jl recipe, MLJ wrapper

## Model formulation and comparability

Before comparing Sarimax.jl outputs with `forecast` (R) or `statsmodels` (Python), be aware of four deliberate design differences:

1. **Seasonal form.** Since v0.3 the default is the **multiplicative** Box-Jenkins SARIMA ``\phi(B)\Phi(B^s)y'_t = \theta(B)\Theta(B^s)\epsilon_t`` — coefficients are directly comparable with R/statsmodels given the same estimation method (item 3). The pre-v0.3 additive form (no cross terms) remains available via `seasonalForm = :additive` in `fit!` and `auto`.
2. **Exogenous variables (ARX by default).** Regressors enter a dynamic-regression/ARX model: the AR terms act on the observed series, so the coefficient is an impact multiplier conditional on past ``y``. R's `Arima(xreg=)` and statsmodels' `SARIMAX(exog=)` fit regression-with-ARIMA-errors instead, where the coefficient is the usual marginal effect. These are different model families and coincide only when the autoregressive polynomial is unitary and there is no differencing. Both forms ship on `fit!` and on `auto`: `exogDynamics = :armax` (default) and `exogDynamics = :regression_errors`.
3. **Estimation and information criteria (CSS).** Estimation is conditional least squares / concentrated conditional Gaussian ML formulated as a JuMP optimization problem; there is no Kalman filter. `loglike`, `aic`, `aicc` and `bic` follow the CSS convention with full Gaussian constants — comparable to R's `arima(..., method = "CSS")`, not to exact-ML defaults.
4. **What the optimization formulation buys.** Swappable objectives (MSE, MAE, Huber, CVaR, ridge, elastic net), custom constraints, an invertible-MA parameterization (`fit!(model; invertible = true)`), and certified global optima via SCIP.

## Estimation criteria

Every objective is a different criterion over the SAME SARIMAX equations and the same
initialization constraints. Changing the criterion never changes the dynamics — that is
the point of writing the model as a JuMP program:

```
coefficients + innovations = optimization variables
SARIMAX dynamics           = algebraic constraints
estimation criterion       = objective function
```

Throughout, ``\varepsilon_t`` is the model **innovation**, defined by
``y_t = \hat y_t + \varepsilon_t``; after a fit, the realized values are the **fitted
residuals** (`model.ϵ`).

### Quantile (pinball) loss

`objectiveFunction = "quantile"` minimizes the check loss of Koenker and Bassett over the
innovations,

```math
\min_{\vartheta,\varepsilon}\;\sum_t \rho_\tau(\varepsilon_t),
\qquad
\rho_\tau(\varepsilon) = \tau\max(\varepsilon, 0) + (1-\tau)\max(-\varepsilon, 0)
                       = \varepsilon\,\bigl(\tau - \mathbb{1}\{\varepsilon < 0\}\bigr),
```

with ``\tau \in (0,1)`` given by `quantileLevel` (default `0.5`). In the linearized form
the package actually builds, ``\varepsilon_t = \varepsilon_t^{+} - \varepsilon_t^{-}`` with
``\varepsilon_t^{+}, \varepsilon_t^{-} \ge 0`` — the same decomposition `"mae"` uses — and
the objective is ``\sum_t [\tau\varepsilon_t^{+} + (1-\tau)\varepsilon_t^{-}]``.

**Sign convention.** In this package ``\varepsilon_t = y_t - \hat y_t``, so a *positive*
innovation is an *under-prediction*, and ``\tau`` is the weight it carries. A high ``\tau``
therefore makes under-prediction expensive and pushes the fitted location **up**: at the
optimum a fraction ``\tau`` of the fitted residuals lies at or below zero. ``\tau = 0.9``
estimates an upper conditional quantile, ``\tau = 0.1`` a lower one.

**Relation to MAE.** At ``\tau = 0.5`` the loss is symmetric and equals
``\lvert\varepsilon\rvert/2``, so `"quantile"` is the *same estimator* as `"mae"` — same
coefficients, fitted values, fitted residuals and forecasts — while the reported objective
*value* is half of `"mae"`'s. The factor is not absorbed, because ``\rho_\tau`` above is
the standard definition.

The level is recorded in `model.metadata["quantileLevel"]`: the same series and orders at
two levels are two different estimates.

`quantileLevel` is **refused** by every other objective rather than ignored — as is
`cvarLevel`, the level of `"stable"`, by every objective but that one. Under an objective
that does not read it the level never reaches the optimization, so accepting it would let a
caller believe they had selected an estimator they did not, and a warning would be invisible
in a parallel sweep. Both keywords default to `nothing`; omitting one applies its documented
default (`0.5` and `0.9`).

!!! warning "This is an estimation criterion, not a forecasting mode"
    Fitting with the pinball loss does **not** make `predict!` return calibrated
    probabilistic quantile forecasts, and the package makes no such claim. Explicit
    quantile forecasts are a separate API and statistical question.

Under `auto`, a candidate is *fitted* with the pinball loss but *ranked* by the package's
declared criterion machinery (the Gaussian likelihood behind `aic`/`aicc`/`bic`), never by
comparing raw pinball values across specifications.

### Composing a loss with a penalty

The loss and the coefficient penalty are two independent axes of the objective:

```math
\min_{\vartheta,\varepsilon}\; \underbrace{L(\varepsilon)}_{\texttt{objectiveFunction}} \;+\;
\underbrace{\sum_j \lambda_j\left[\alpha\lvert\psi_j\rvert + \frac{1-\alpha}{2}\psi_j^2\right]}_{\texttt{penalty}}
```

subject to the same SARIMAX equations. `penalty = :elastic_net` adds the term to whichever
loss `objectiveFunction` selected, so a quantile fit with a lasso penalty is one call:

```julia
fit!(model; objectiveFunction = "quantile", quantileLevel = 0.9,
     penalty = :elastic_net, alpha = 1.0, lambda = 20.0)
```

`objectiveFunction = "elastic_net"` is `"mse"` with `penalty = :elastic_net` **when the
pre-sample block is not priced** (`initialization` in `:zeroed`, `:free`) — there the two
spellings emit the same expression and give bit-identical fits.

!!! warning "They differ under `:penalized` and `:innovations`"
    Under the modes that price the free pre-sample block — including `:innovations`, the
    **default** — the two fit terms are not the same. `"mse"` prices that block as a
    concentrated Gaussian likelihood and carries the determinant factor
    ``\prod_j (1-\kappa_j^2)^{-j/T}``; the `"elastic_net"` objective's fit term is plain
    ``\sum_t \varepsilon_t^2`` plus the pre-sample squares, with no determinant.

    This asymmetry predates the `penalty` keyword — it is how the two branches have always
    been written — and neither is changed. But it means that on the default path
    `objectiveFunction = "elastic_net"` and `objectiveFunction = "mse", penalty = :elastic_net`
    are **different estimators**, and a study should name which one it used. Pinned by a
    test in `test/loss_penalty_composition.jl`.

| | |
|---|---|
| **admitted losses** | `"mse"`, `"mae"`, `"huber"`, `"quantile"`, `"ml"` |
| **refused: scale** | `"ml_exact"` (log scale), `"stable"` (mean scale) |
| **refused: reachability** | `"bilevel"` — the moving-average coefficients are not decision variables there |
| **refused: double specification** | `"elastic_net"`, `"ridge"` — they already carry a penalty |

The admitted set is decided on **scale**: `lambda` defaults to the square root of the
effective sample because the fit term is a *sum over observations*, so the two sides of the
objective are commensurable. The refusals are errors rather than warnings — the combination
is fixed at the call site, and a mis-scaled penalty is invisible in the fitted coefficients.

The penalty is recorded in `model.metadata["penalty"]` and drives the sparse parameter
count: what shrinks a coefficient to zero is the penalty, not the loss.

## Regularization

The `"elastic_net"` objective is the conventional penalized estimator

```math
\min_{\vartheta,\varepsilon}\; L(\varepsilon) \;+\; \sum_j \lambda_j\left[\alpha\lvert\psi_j\rvert + \frac{1-\alpha}{2}\psi_j^2\right],
\qquad 0 \le \alpha \le 1,
```

where ``L(\varepsilon)`` is the innovation loss and ``\psi_j`` runs over the penalized
coefficients. The intercept and the drift are always excluded, since penalizing the level
has no shrinkage interpretation here. ``\alpha = 0`` recovers a ridge-type penalty and
``\alpha = 1`` a lasso-type one.

A **scalar** `lambda` is the uniform case ``\lambda_j = \lambda`` and reproduces the
classical form ``\lambda[\alpha\lVert\psi\rVert_1 + \frac{1-\alpha}{2}\lVert\psi\rVert_2^2]``
exactly. It defaults to the square root of the effective sample size, matching the scale of
the sum-form objective.

!!! warning "The default `lambda` is a scale convention, not a tuning rule"
    ``\lambda = \sqrt{n_{\text{eff}}}`` exists so that the penalty is commensurable with a
    fit term written as a **sum over observations** rather than a mean — it is the
    package's sum-vs-mean bookkeeping, nothing more.

    It is **not** an optimal or universally calibrated regularization parameter, and it
    carries no such claim for any loss. It was never calibrated against MAE, Huber or the
    quantile loss in particular: those fit terms live on a different numerical scale from
    the squared one, so the same ``\lambda`` shrinks by a different amount under each.

    For any substantive regularized analysis — and for anything reported in a paper —
    **select or supply `lambda` explicitly**, by cross-validation or by whatever criterion
    the study defends, and record the value. Relying on the default makes the shrinkage an
    artefact of the package's internal scaling rather than a choice you can justify.

### Coefficient-specific weights

``\lambda_j`` is a per-coefficient *strength*; ``\alpha`` remains the L1/L2 mixing
parameter, and the two never trade places. Three shapes are accepted:

| `lambda` | meaning |
|---|---|
| `2.0` | one strength for every penalized coefficient |
| `[2.0, 0.5, 0.0]` | one weight per penalized coefficient, in the order of `penaltyCoefficientNames(model)` |
| `(ar = 2.0, ma = [0.5, 0.0])` | keyed by block; each value a scalar or a per-coefficient vector |

Block keys are `:ar`, `:ma`, `:sar`, `:sma`, `:exog` (the Greek coefficient names `:ϕ`,
`:θ`, `:Φ`, `:Θ`, `:β` are accepted as aliases). The flat-vector ordering is
`[ar; ma; sar; sma; exog]`, restricted to the blocks the model has and `penaltyTarget`
admits — call `penaltyCoefficientNames` rather than reconstructing it:

```julia
model = SARIMA(y, X, 2, 0, 1)
penaltyCoefficientNames(model)                            # ["ar1","ar2","ma1","exog:x1","exog:x2"]
penaltyCoefficientNames(model; penaltyTarget = :exogenous) # ["exog:x1","exog:x2"]
```

`0.0` is a legal weight and means "leave this coefficient unpenalized"; negative, `NaN`
and `Inf` are rejected, as are wrong lengths and unknown block keys. A structured `lambda`
must name *every* penalized block of the model: filling an unnamed block with the default
would make `lambda = (ar = 0.0,)` read as "penalize nothing" while the moving-average block
stayed at the default.

### Which blocks are penalized

Selected with `penaltyTarget`, on `fit!` and on `auto`:

| `penaltyTarget` | Penalized blocks |
|---|---|
| `:all` (default) | autoregressive, moving-average and exogenous coefficients |
| `:dynamics` | autoregressive and moving-average coefficients only |
| `:exogenous` | exogenous coefficients only |

Targeting the regressors alone is the usual choice when the point of the fit is
selecting among them while leaving the dynamics unshrunk.

### Exogenous regressors are not standardized

The endogenous series is divided by its own standard deviation before the model is built,
and the autoregressive and moving-average coefficients are dimensionless, so a penalty over
them is scale-free. **Exogenous coefficients are not.** Each ``\beta_j`` carries the units
of its own regressor, and the package does not standardize the regressor matrix.

The consequence is direct: with two regressors on very different scales,

```math
\lambda_{\beta_1} = \lambda_{\beta_2}
\quad\text{does NOT imply comparable effective shrinkage.}
```

A regressor measured in thousands and one measured in units receive penalties that differ
by that factor, whichever penalty is in play — lasso, ridge-type, elastic net, or
adaptive-Lasso weights, all of which act on the coefficient rather than on the standardized
effect.

Two ways to get comparable shrinkage, both the caller's choice:

1. **Standardize the regressors before fitting** — divide each column by its standard
   deviation (or use whatever normalization the analysis defends) and remember that the
   fitted ``\beta`` is then on the standardized scale and must be mapped back for
   interpretation.
2. **Supply coefficient-specific weights that absorb the scale**, e.g.
   ``\lambda_j = \lambda \cdot s_j`` with ``s_j`` the standard deviation of regressor
   ``j``, through the vector or block form of `lambda`.

The package deliberately does **not** standardize regressors for you: doing so would change
what the coefficients mean and would break backward compatibility for every existing
SARIMAX fit.

!!! warning "`objectiveFunction = \"ridge\"` is deprecated"
    It is the fixed-``\lambda`` case of this same penalty: ``\alpha = 0`` over the dynamics
    blocks, with ``\lambda = \sqrt{n_{\text{eff}}}`` chosen internally. It ignores `lambda`,
    `alpha` and `penaltyTarget` alike, which is why every guard has to special-case it, and
    it will be removed in v2.0.

    Carry a fit over exactly with

    ```julia
    fit!(model; objectiveFunction = "mse", penalty = :elastic_net,
         alpha = 0.0, penaltyTarget = :dynamics,
         lambda = 2 * model.metadata["ridgeLambda"])
    ```

    **Twice** the recorded value, because the elastic-net L2 term is
    ``\frac{1-\alpha}{2}\psi_j^2`` and `"ridge"` carries no ``\frac{1}{2}``; and
    `metadata["ridgeLambda"]` because ``n_{\text{eff}}`` discounts the CSS conditioning and
    cannot be rebuilt from the series length. The equivalence is pinned by a test.

### Adaptive Lasso

Heterogeneous weights are exactly what an adaptive Lasso needs. The package does **not**
run the two-stage procedure — it accepts the weights the procedure produces:

```julia
first = SARIMA(y, X, 0, 0, 0)
fit!(first; objectiveFunction = "mse")

β̃ = [first.exogCoefficients...]
γ = 1.0
w = 1.0 ./ abs.(β̃) .^ γ              # w_j = 1 / |β̃_j|^γ

model = SARIMA(y, X, 0, 0, 0)
fit!(model; objectiveFunction = "elastic_net", alpha = 1.0,
     penaltyTarget = :exogenous, lambda = 10.0 .* w)
```

!!! warning "Cap the weight when a first-stage coefficient is near zero"
    ``1/|\tilde\beta_j|^\gamma`` diverges as ``\tilde\beta_j \to 0``, and an infinite weight
    is **rejected** — the package does not silently transform what you pass, so `Inf`
    raises an `ArgumentError` rather than being quietly turned into a large number. That
    refusal is deliberate: a weight you did not choose is a shrinkage you cannot report.

    Cap or floor it yourself, and say which you did:

    ```julia
    ε = 1e-6
    w = 1.0 ./ max.(abs.(β̃), ε) .^ γ        # floor the coefficient
    w = min.(1.0 ./ abs.(β̃) .^ γ, 1e6)      # or cap the weight
    ```

    A coefficient that is numerically zero at the first stage is also the case where the
    adaptive weight is doing the most work, so the floor is a modelling choice worth
    stating rather than a numerical detail.

## Known limitations

- `exactLoglike` refuses a share of seasonal-AR candidates when the ``\psi`` tail does
  not decay within the truncation window, and the criterion falls back to the CSS
  plug-in for those. The fallback is recorded in
  `model.metadata["criterionFallback"]` and penalized during search, so the behaviour
  is degraded rather than silently wrong
  ([#15](https://github.com/LAMPSPUC/Sarimax.jl/issues/15)).
- In the `mse` + `:penalized` objective the determinant exponent divides by the
  effective sample rather than by the number of observations
  ([#14](https://github.com/LAMPSPUC/Sarimax.jl/issues/14)).
- `cssResiduals`, and therefore `vcov`/`stderror`, implement the zeroed recursion and
  do not reproduce the free pre-sample block modes.

## Installation

Sarimax.jl can be installed using Julia's built-in package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add Sarimax
```

Or, you can install it by using `Pkg` directly:

```julia
using Pkg
Pkg.add("Sarimax")
```

To use the development version, you can install directly from the GitHub repository:

```julia
Pkg.add(url = "https://github.com/LAMPSPUC/Sarimax.jl.git")
```

## Quick Start

To start using Sarimax.jl, simply import the package:

```julia
using Sarimax
```

Check out our [Tutorial](#tutorial) section for detailed examples of how to use the package.

## License

Sarimax.jl is licensed under the [MIT License](https://opensource.org/licenses/MIT). This means you are free to use, modify, and distribute the code, subject to the terms and conditions of the MIT license.

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/LAMPSPUC/Sarimax.jl). Pull requests for bug fixes and new features are also appreciated.

For more detailed information about the package functionality, please refer to the following sections:

```@contents
Pages = [
    "tutorial.md",
    "api.md",
    "examples.md"
]
Depth = 2
```

