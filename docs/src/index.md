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
* Swappable objective functions: MSE, MAE (L1), Huber, concentrated Gaussian CSS
  ("ml"), exact treatment of the initial observations ("ml_exact"), ridge,
  penalized elastic net, and a tail-oriented CVaR criterion ("stable")
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
2. **Exogenous variables (ARX by default).** Regressors enter a dynamic-regression/ARX model: the AR terms act on the observed series, so the coefficient is an impact multiplier conditional on past ``y``. R's `Arima(xreg=)` and statsmodels' `SARIMAX(exog=)` fit regression-with-ARIMA-errors instead, where the coefficient is the usual marginal effect. These are different model families and coincide only when the autoregressive polynomial is unitary and there is no differencing. Both forms ship on `fit!`: `exogDynamics = :armax` (default) and `exogDynamics = :regression_errors`. Order selection through `auto` always uses the ARX form.
3. **Estimation and information criteria (CSS).** Estimation is conditional least squares / concentrated conditional Gaussian ML formulated as a JuMP optimization problem; there is no Kalman filter. `loglike`, `aic`, `aicc` and `bic` follow the CSS convention with full Gaussian constants — comparable to R's `arima(..., method = "CSS")`, not to exact-ML defaults.
4. **What the optimization formulation buys.** Swappable objectives (MSE, MAE, Huber, CVaR, ridge, elastic net), custom constraints, an invertible-MA parameterization (`fit!(model; invertible = true)`), and certified global optima via SCIP.

## Regularization

The `"elastic_net"` objective is the conventional penalized estimator

```math
\min_{\vartheta,\varepsilon}\; L(\varepsilon) \;+\; \lambda\left[\alpha\lVert\Theta\rVert_1 + \frac{1-\alpha}{2}\lVert\Theta\rVert_2^2\right],
\qquad 0 \le \alpha \le 1,
```

where ``L(\varepsilon)`` is the residual loss and ``\Theta`` collects the penalized
coefficient blocks: the autoregressive and moving-average coefficients and, when
present, the exogenous ones. The intercept and the drift are excluded, since
penalizing the level has no shrinkage interpretation here. ``\alpha = 0`` recovers a
ridge-type penalty and ``\alpha = 1`` a lasso-type one. `lambda` defaults to the square
root of the effective sample size, matching the scale of the sum-form objective.

Exogenous coefficients carry the units of their own regressor, which the package does
not standardize, so scale-comparable regressors are the caller's responsibility.

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

