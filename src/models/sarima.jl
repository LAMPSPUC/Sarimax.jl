"""
Iteration ceiling applied to a fit whose caller asked for a bounded solve (see the
`maxTimeSeconds` handling in [`fit!`](@ref)). A wall-clock limit is not enough on long
series: Ipopt checks it only between iterations, and one iteration of the
`initialization = :free` problem (≈2T variables) can outlast the whole budget. A
well-scaled fit converges in far fewer iterations than this, so hitting the ceiling is
itself the signal that the candidate is not worth more time.
"""
const MAX_ITER_CAPPED_FIT = 200

"""
Default confidence level of the `"stable"` objective, which minimises the Conditional
Value-at-Risk of the squared residuals. At `α` the fit optimises the mean of the worst
`(1-α)` fraction of squared residuals, so higher `α` concentrates on a thinner tail and
`α → 1` degenerates to min-max. Note this is a *conservative* (worst-case) objective,
not a robust one: it fits the tail rather than discounting it — `"mae"` is the robust
choice in this package.
"""
const DEFAULT_CVAR_LEVEL = 0.9

"""
    DEFAULT_HUBER_DELTA

Threshold where the `"huber"` objective switches from quadratic to linear, in units of the
residual standard deviation. 1.345 is the classical choice: it buys 95% of the efficiency
of least squares under Gaussian errors while bounding the influence of an outlier.

It is a bare constant rather than a keyword because the endogenous data is standardized
internally before the fit (`yValues ./ yScale`), so residuals already live on a unit scale
and the threshold needs no calibration per series. A package that did not standardize would
have to estimate sigma first and scale delta by it.
"""
const DEFAULT_HUBER_DELTA = 1.345

"""
Margem de REJEICAO de candidatos: quao longe do circulo unitario a menor raiz tem que estar
para o candidato ser considerado admissivel na SELECAO de ordem.

Vale `1e-2` para casar com o `forecast::auto.arima`: o `myarima` dele poe `ic = Inf` em
qualquer candidato cuja raiz minima caia a menos de 1% do circulo unitario.

NAO usar este valor nas parametrizacoes por construcao — ver [`DEFAULT_DOMAIN_MARGIN`].
`rootMargin` limita o modulo das raizes a `> 1 + rho`, enquanto `stationarityMargin` e
`invertibilityMargin` limitam os coeficientes de reflexao a `|kappa| <= 1 - rho`. A
correspondencia e exata na ordem 1: para um AR(1), `kappa = phi` e a raiz e `1/phi`, entao
`|raiz| > 1.01` equivale a `|phi| < 0.9901`. Mas as duas margens respondem a perguntas
diferentes — uma e regra de selecao, a outra e o dominio do estimador.
"""
const DEFAULT_ROOT_MARGIN = 1e-2

"""
Margin that keeps the DOMAIN of the by-construction parameterization open, used by
`stationarityMargin` and `invertibilityMargin`.

The stationary likelihood exists on `|kappa| < 1` and diverges at the boundary; the only role
of this margin is to stop the solver from evaluating exactly `|kappa| = 1`, for which `1e-6`
suffices. It is NOT an admissibility rule: inadmissible candidates are rejected by
`rootMargin`/`assertStationarity` at the selection stage.

The two margins must stay distinct. Setting this one to the selection margin (`1e-2`) imposes
a SELECTION rule as an ESTIMATION constraint: measured over 88 AR(p) fits on M4 monthly
levels, where 57 of the 88 series have a smallest root below 1.02, 45.5% of the fits then
terminate against the bound and the median distance to R's exact-ML `phi` rises from 0.00005
to 0.00234, truncating legitimate near-unit-root estimates. R has no such bound: with
`transform.pars = TRUE` it parameterizes the AR coefficients through `tanh`, whose range is
the open interval `(-1, 1)`.
"""
const DEFAULT_DOMAIN_MARGIN = 1e-6
"""
Teto de modelos que uma busca stepwise pode visitar, equivalente ao `nmodels = 94` do
`forecast::auto.arima`.

This is the counterpart to the scope of `maxOrder`: R does not bound `p+q+P+Q` in the local
search, but it does bound how many models that search fits. `stepwiseSearch`, the default
method, already implemented this as `maxModels = 94`.

Note when measuring cost: since the default path was already bounded, changing this ceiling
does NOT explain time differences observed with `searchMethod = "stepwise"`.
"""
const DEFAULT_NMODELS = 94

"""
Default CSS conditioning convention used when fitting.

`:innovations` leaves the pre-sample block FREE (the innovations before `t = 1` are decision
variables rather than imposed zeros) and PENALIZES that block in the objective, so the fit
error is summed from `t = 1` instead of starting after the conditioning.

These are two independent axes, not rungs of a ladder: `:free` opens the block without
penalizing it, `:penalized` opens and penalizes it, `:zeroed` holds it fixed at zero.

The default is not `:zeroed` because, under a 2x2 test of solver starting point against mode,
`:zeroed` was the only mode whose optimum depends on where the solver starts; `:free`,
`:penalized` and `:innovations` reached the same optimum from different starts. Invariance is
not globality — there is a case with Ipopt certifiably 1.911% above the global optimum within
an invariant mode — but the start dependence of `:zeroed` means the same data and the same
call could return different estimates depending on the initial point.

BREAKING: callers relying on the previous behaviour must pass `initialization = :zeroed`
explicitly.
"""
const DEFAULT_INITIALIZATION = :innovations

"""
    admissibleCoefficientBound(order::Int, i::Int) -> Float64

Bound `|coef_i| <= C(order, i)` for the FREE parameterization (no stationarity or
invertibility by construction). It is chosen for one property: **it excludes no admissible
model**.

A stationary AR(p) polynomial factors as `phi(z) = prod_i (1 - alpha_i z)` with
`|alpha_i| < 1`, so each coefficient is, up to sign, an elementary symmetric function of the
`alpha`, and `|phi_i| <= e_i(1,...,1) = C(p, i)`. The same holds for the MA part. Any point
outside this box is therefore necessarily inadmissible, and the box is equivalent to having
no bound at all as far as selectable models are concerned.

    p = 1 -> (1)        p = 2 -> (2, 1)        p = 3 -> (3, 3, 1)

At order 1 this coincides with `[-1, 1]`; from order 2 on it does not. A flat `[-1, 1]` on
every index would simultaneously admit inadmissible points (an MA(2) at `(0.5, -0.9)` is in
the box and not invertible) and exclude admissible ones (R's ML at `(1.51, 0.79)` is
invertible and outside it).

Keeping a finite bound, rather than leaving the coefficient free as `stats::arima` does, is
deliberate: R evaluates the likelihood with a Kalman filter, which does not propagate the
recursion `eps_t = y_t - sum_j theta_j eps_(t-j)`. This package does propagate it, and
outside the invertible region it blows up numerically within about 100 steps. Since the bound
excludes nothing selectable, it gives R's freedom without the overflow.

Admissibility itself is still enforced where it always was: at REJECTION, via
[`DEFAULT_ROOT_MARGIN`] / `assertStationarity` / `ensureAdmissible!`.
"""
admissibleCoefficientBound(order::Int, i::Int) = Float64(binomial(order, i))

"""
The `SARIMAModel` struct represents a SARIMA model. It contains the following fields:

- `y`: The time series data.
- `p`: The autoregressive order for the non-seasonal part.
- `d`: The degree of differencing.
- `q`: The moving average order for the non-seasonal part.
- `seasonality`: The seasonality period.
- `P`: The autoregressive order for the seasonal part.
- `D`: The degree of seasonal differencing.
- `Q`: The moving average order for the seasonal part.
- `metadata`: A dictionary containing model metadata.
- `exog`: Optional exogenous variables.
- `c`: The constant term.
- `trend`: The trend term.
- `ϕ`: The autoregressive coefficients for the non-seasonal part.
- `θ`: The moving average coefficients for the non-seasonal part.
- `Φ`: The autoregressive coefficients for the seasonal part.
- `Θ`: The moving average coefficients for the seasonal part.
- `ϵ`: The residuals.
- `exogCoefficients`: The coefficients of the exogenous variables.
- `σ²`: The variance of the residuals.
- `fitInSample`: The in-sample fit.
- `forecast`: The forecast.
- `silent`: Whether to suppress output.
- `allowMean`: Whether to include a mean term in the model.
- `allowDrift`: Whether to include a drift term in the model.
- `keepProvidedCoefficients`: Whether to keep the provided coefficients.
- `lambda`: Strength of the elastic-net penalty (`lambda >= 0`). Defaults to the square
  root of the effective sample size. Only the `"elastic_net"` objective honours it; the
  `"ridge"` objective fixes its own value and rejects the argument.
- `alpha`: Mixing parameter of the elastic-net penalty (0 <= alpha <= 1); `alpha = 1` is
  lasso, `alpha = 0` is ridge.
"""
mutable struct SARIMAModel{Fl<:AbstractFloat} <: SarimaxModel
    y::TimeArray
    p::Int
    d::Int
    q::Int
    seasonality::Int
    P::Int
    D::Int
    Q::Int
    metadata::Dict{String,Any}
    exog::Union{TimeArray,Nothing}
    c::Union{Fl,Nothing}
    trend::Union{Fl,Nothing}
    ϕ::Union{Vector{Fl},Nothing}
    θ::Union{Vector{Fl},Nothing}
    Φ::Union{Vector{Fl},Nothing}
    Θ::Union{Vector{Fl},Nothing}
    ϵ::Union{Vector{Fl},Nothing}
    exogCoefficients::Union{Vector{Fl},Nothing}
    σ²::Fl
    fitInSample::Union{TimeArray,Nothing}
    forecast::Union{TimeArray,Nothing}
    silent::Bool
    allowMean::Bool
    allowDrift::Bool
    keepProvidedCoefficients::Bool
    lambda::Union{Fl,Nothing}
    alpha::Union{Fl,Nothing}
    icOffset::Union{Fl,Nothing}
    function SARIMAModel{Fl}(
        y::TimeArray,
        p::Int,
        d::Int,
        q::Int;
        seasonality::Int = 1,
        P::Int = 0,
        D::Int = 0,
        Q::Int = 0,
        exog::Union{TimeArray,Nothing} = nothing,
        c::Union{Fl,Nothing} = nothing,
        trend::Union{Fl,Nothing} = nothing,
        ϕ::Union{Vector{Fl},Nothing} = nothing,
        θ::Union{Vector{Fl},Nothing} = nothing,
        Φ::Union{Vector{Fl},Nothing} = nothing,
        Θ::Union{Vector{Fl},Nothing} = nothing,
        ϵ::Union{Vector{Fl},Nothing} = nothing,
        exogCoefficients::Union{Vector{Fl},Nothing} = nothing,
        σ²::Fl = 0.0,
        fitInSample::Union{TimeArray,Nothing} = nothing,
        forecast::Union{TimeArray,Nothing} = nothing,
        silent::Bool = true,
        allowMean::Bool = true,
        allowDrift::Bool = false,
        keepProvidedCoefficients::Bool = false,
        lambda::Union{Fl,Nothing} = nothing,
        alpha::Union{Fl,Nothing} = nothing,
        icOffset::Union{Fl,Nothing} = nothing,
    ) where {Fl<:AbstractFloat}
        @assert p >= 0
        @assert d >= 0
        @assert q >= 0
        @assert P >= 0
        @assert D >= 0
        @assert Q >= 0
        @assert seasonality >= 1
        yMetadata = Dict()
        granularityInfo = identify_granularity(timestamp(y))
        yMetadata["granularity"] = granularityInfo.granularity
        yMetadata["frequency"] = granularityInfo.frequency
        yMetadata["weekDaysOnly"] = granularityInfo.weekdays
        yMetadata["startDatetime"] = timestamp(y)[1]
        yMetadata["endDatetime"] = timestamp(y)[end]
        if !isnothing(exog)
            @assert yMetadata["startDatetime"] == timestamp(exog)[1] "The endogenous and exogenous variables must start at the same timestamp"
            @assert yMetadata["endDatetime"] <= timestamp(exog)[end] "The exogenous variables must end after the endogenous variables"
            @assert granularityInfo == identify_granularity(timestamp(exog)) "The endogenous and exogenous variables must have the same granularity, frequency and pattern"
        end
        return new{Fl}(
            y,
            p,
            d,
            q,
            seasonality,
            P,
            D,
            Q,
            yMetadata,
            exog,
            c,
            trend,
            ϕ,
            θ,
            Φ,
            Θ,
            ϵ,
            exogCoefficients,
            σ²,
            fitInSample,
            forecast,
            silent,
            allowMean,
            allowDrift,
            keepProvidedCoefficients,
            lambda,
            alpha,
            icOffset,
        )
    end
end

typeofModelElements(model::SARIMAModel) = eltype(values(model.y))

"""
    conditioningLags(p, q, P, Q, s, seasonalForm)

Number of pre-sample observations the CSS recursion must condition on. Under the
multiplicative form the AR/MA polynomials reach the cross lags `p + s*P` / `q + s*Q`.
"""
conditioningLags(p::Int, q::Int, P::Int, Q::Int, s::Int, seasonalForm::Symbol) =
    seasonalForm === :multiplicative ? max(p + s * P, q + s * Q) : max(p, s * P, q, s * Q)

"""
    modelSeasonalForm(model::SARIMAModel)

The seasonal form the model was (or will be) fitted with: `:multiplicative`
(Box-Jenkins, default) or `:additive` (pre-v0.3 behavior).
"""
modelSeasonalForm(model::SARIMAModel) = Symbol(get(model.metadata, "seasonalForm", "multiplicative"))

"""
    print(model::SARIMAModel)

Prints the full fitted-model summary (alias for `show(stdout, MIME"text/plain"(), model)`).
"""
function print(model::SARIMAModel)
    show(stdout, MIME("text/plain"), model)
    println()
end

modelSpecString(model::SARIMAModel) =
    "SARIMA($(model.p),$(model.d),$(model.q))($(model.P),$(model.D),$(model.Q))[$(model.seasonality)]"

"""
    Base.show(io::IO, model::SARIMAModel)

Compact one-line representation: specification and fit status.
"""
function Base.show(io::IO, model::SARIMAModel)
    Base.print(io, modelSpecString(model), isFitted(model) ? " | fitted" : " | not fitted")
    return nothing
end

"""
    Base.show(io::IO, ::MIME"text/plain", model::SARIMAModel)

Full model summary: specification, seasonal form, estimation convention,
coefficient table with CSS standard errors, and fit statistics.
"""
function Base.show(io::IO, ::MIME"text/plain", model::SARIMAModel)
    println(io, modelSpecString(model))
    form = get(model.metadata, "seasonalForm", "multiplicative")
    init = get(model.metadata, "initialization", "zeroed")
    deterministic = String[]
    model.allowMean && push!(deterministic, "mean")
    model.allowDrift && push!(deterministic, "drift")
    detStr = isempty(deterministic) ? "none" : join(deterministic, " + ")
    println(io, "Seasonal form: ", form, " | Estimation: CSS (", init, ") | Deterministic: ", detStr)
    if !isFitted(model)
        Base.print(io, "Status: not fitted — run fit!(model)")
        return nothing
    end
    names = StatsAPI.coefnames(model)
    estimates = StatsAPI.coef(model)
    ses = try
        StatsAPI.stderror(model)
    catch
        fill(NaN, length(estimates))
    end
    if !isempty(names)
        wname = max(maximum(length.(names)), length("coefficient"))
        rule = "─"^(wname + 28)
        println(io, rule)
        println(io, rpad("coefficient", wname), "  ", lpad("estimate", 12), "  ", lpad("std. error", 12))
        println(io, rule)
        for (nm, est, se) in zip(names, estimates, ses)
            seStr = isnan(se) ? "—" : @sprintf("%.4f", se)
            println(io, rpad(nm, wname), "  ", lpad(@sprintf("%.4f", est), 12), "  ", lpad(seStr, 12))
        end
        println(io, rule)
    end
    Base.print(
        io,
        @sprintf(
            "σ² = %.6g | n = %d | loglik = %.3f | AIC = %.3f | AICc = %.3f | BIC = %.3f",
            model.σ²,
            length(model.ϵ),
            loglike(model),
            aic(model),
            aicc(model),
            bic(model)
        )
    )
    solverStatus = get(model.metadata, "solverStatus", "")
    solverStatus in ("", "OPTIMAL", "LOCALLY_SOLVED", "ALMOST_OPTIMAL", "ALMOST_LOCALLY_SOLVED") ||
        Base.print(io, "\n⚠ solver status: ", solverStatus)
    model.keepProvidedCoefficients && Base.print(
        io,
        "\n(provided coefficients kept fixed; set keepProvidedCoefficients = false to re-estimate)",
    )
    return nothing
end

"""
    SARIMA constructor.

    Parameters:
    - y: TimeArray with the time series.
    - p: Int with the autoregressive order for the non-seasonal part.
    - d: Int with the degree of differencing.
    - q: Int with the moving average order for the non-seasonal part.
    - seasonality: Int with the seasonality period.
    - P: Int with the autoregressive order for the seasonal part.
    - D: Int with the degree of seasonal differencing.
    - Q: Int with the moving average order for the seasonal part.
    - silent: Bool to supress output.
    - allowMean: Bool to include a mean term in the model.
    - allowDrift: Bool to include a drift term in the model.
"""
function SARIMA(
    y::TimeArray,
    p::Int,
    d::Int,
    q::Int;
    seasonality::Int = 1,
    P::Int = 0,
    D::Int = 0,
    Q::Int = 0,
    silent::Bool = true,
    allowMean::Bool = true,
    allowDrift::Bool = false,
    lambda::Union{Nothing,<:AbstractFloat} = nothing,
    alpha::Union{Nothing,<:AbstractFloat} = nothing,
    icOffset::Union{Nothing,<:AbstractFloat} = nothing,
)
    modelFl = eltype(values(y))
    return SARIMAModel{modelFl}(
        y,
        p,
        d,
        q;
        seasonality = seasonality,
        P = P,
        D = D,
        Q = Q,
        silent = silent,
        allowMean = allowMean,
        allowDrift = allowDrift,
        lambda = lambda,
        alpha = alpha,
        icOffset = icOffset,
    )
end

"""
    SARIMA constructor to initialize model with provided coefficients.

    Parameters:
    - y: TimeArray with the time series.
    - exog: TimeArray with the exogenous variables.
    - arCoefficients: Vector with the autoregressive coefficients.
    - maCoefficients: Vector with the moving average coefficients.
    - seasonalARCoefficients: Vector with the autoregressive coefficients for the seasonal component.
    - seasonalMACoefficients: Vector with the moving average coefficients for the seasonal component.
    - mean: Float with the mean term.
    - trend: Float with the trend term.
    - exogCoefficients: Vector with the exogenous coefficients.
    - d: Int with the degree of differencing.
    - D: Int with the degree of seasonal differencing.
    - seasonality: Int with the seasonality period.
    - silent: Bool to supress output.
    - allowMean: Bool to include a mean term in the model.
    - allowDrift: Bool to include a drift term in the model.
 """
function SARIMA(
    y::TimeArray;
    exog::Union{TimeArray,Nothing} = nothing,
    arCoefficients::Union{Vector{<:AbstractFloat},Nothing} = nothing,
    maCoefficients::Union{Vector{<:AbstractFloat},Nothing} = nothing,
    seasonalARCoefficients::Union{Vector{<:AbstractFloat},Nothing} = nothing,
    seasonalMACoefficients::Union{Vector{<:AbstractFloat},Nothing} = nothing,
    mean::Union{<:AbstractFloat,Nothing} = nothing,
    trend::Union{<:AbstractFloat,Nothing} = nothing,
    exogCoefficients::Union{Vector{<:AbstractFloat},Nothing} = nothing,
    d::Int = 0,
    D::Int = 0,
    seasonality::Int = 1,
    silent::Bool = true,
    allowMean::Bool = true,
    allowDrift::Bool = false,
    lambda::Union{<:AbstractFloat,Nothing} = nothing,
    alpha::Union{<:AbstractFloat,Nothing} = nothing,
    icOffset::Union{<:AbstractFloat,Nothing} = nothing,
)

    if isnothing(arCoefficients) &&
       isnothing(maCoefficients) &&
       isnothing(seasonalARCoefficients) &&
       isnothing(seasonalMACoefficients)
        throw(
            InvalidParametersCombination(
                "At least one of the AR, MA, seasonal AR or seasonal MA coefficients must be provided",
            ),
        )
    end

    if (!isnothing(seasonalARCoefficients) || !isnothing(seasonalMACoefficients)) &&
       seasonality == 1
        throw(
            InvalidParametersCombination(
                "The seasonality must be provided if seasonal AR and/or MA coefficients are provided",
            ),
        )
    end

    if isnothing(exog) && !isnothing(exogCoefficients)
        throw(
            InvalidParametersCombination(
                "Exogenous coefficients were provided but no exogenous variable was passed",
            ),
        )
    end

    if !isnothing(exog) && length(colnames(exog)) != length(exogCoefficients)
        throw(
            InvalidParametersCombination(
                "The number of exogenous coefficients must match the number of exogenous variables",
            ),
        )
    end

    if !isnothing(lambda) && lambda < 0
        throw(
            InvalidParametersCombination(
                "The lambda value must be non-negative",
            ),
        )
    end

    if !isnothing(alpha) && (alpha < 0 || alpha > 1)
        throw(
            InvalidParametersCombination(
                "The alpha value must be between 0 and 1",
            ),
        )
    end

    p = isnothing(arCoefficients) ? 0 : length(arCoefficients)
    q = isnothing(maCoefficients) ? 0 : length(maCoefficients)
    P = isnothing(seasonalARCoefficients) ? 0 : length(seasonalARCoefficients)
    Q = isnothing(seasonalMACoefficients) ? 0 : length(seasonalMACoefficients)
    c = isnothing(mean) ? nothing : mean
    trend = isnothing(trend) ? nothing : trend
    allowMean = !isnothing(mean) || allowMean
    allowDrift = !isnothing(trend) || allowDrift

    modelFl = eltype(values(y))
    return SARIMAModel{modelFl}(
        y,
        p,
        d,
        q;
        seasonality = seasonality,
        P = P,
        D = D,
        Q = Q,
        exog = exog,
        c = c,
        trend = trend,
        ϕ = arCoefficients,
        θ = maCoefficients,
        Φ = seasonalARCoefficients,
        Θ = seasonalMACoefficients,
        exogCoefficients = exogCoefficients,
        silent = silent,
        allowMean = allowMean,
        allowDrift = allowDrift,
        keepProvidedCoefficients = true,
        lambda = lambda,
        alpha = alpha,
        icOffset = icOffset,
    )
end

"""
    SARIMA constructor.

    Parameters:
    - y: TimeArray with the time series.
    - exog: TimeArray with the exogenous variables.
    - p: Int with the order of the AR component.
    - d: Int with the degree of differencing.
    - q: Int with the order of the MA component.
    - seasonality: Int with the seasonality period.
    - P: Int with the order of the seasonal AR component.
    - D: Int with the degree of seasonal differencing.
    - Q: Int with the order of the seasonal MA component.
    - silent: Bool to supress output.
    - allowMean: Bool to include a mean term in the model.
    - allowDrift: Bool to include a drift term in the model.
"""
function SARIMA(
    y::TimeArray,
    exog::Union{TimeArray,Nothing},
    p::Int,
    d::Int,
    q::Int;
    seasonality::Int = 1,
    P::Int = 0,
    D::Int = 0,
    Q::Int = 0,
    silent::Bool = true,
    allowMean::Bool = true,
    allowDrift::Bool = false,
    lambda::Union{Nothing,<:AbstractFloat} = nothing,
    alpha::Union{Nothing,<:AbstractFloat} = nothing,
    icOffset::Union{Nothing,<:AbstractFloat} = nothing,
)
    modelFl = eltype(values(y))
    return SARIMAModel{modelFl}(
        y,
        p,
        d,
        q;
        seasonality = seasonality,
        P = P,
        D = D,
        Q = Q,
        exog = exog,
        silent = silent,
        allowMean = allowMean,
        allowDrift = allowDrift,
        lambda = lambda,
        alpha = alpha,
        icOffset = icOffset,
    )
end

"""
    fillFitValues!(
        model::SARIMAModel,
        c::Fl,
        trend::Fl,
        ϕ::Vector{Fl},
        θ::Vector{Fl},
        ϵ::Vector{Fl},
        σ²::Fl,
        fitInSample::TimeArray;
        Φ::Union{Vector{Fl}, Nothing}=nothing,
        Θ::Union{Vector{Fl}, Nothing}=nothing,
        exogCoefficients::Union{Vector{Fl}, Nothing}=nothing
    ) where Fl<:AbstractFloat

Fills the SARIMA model with fitted values.

# Arguments
- `model::SARIMAModel`: The SARIMA model to be filled.
- `c::Fl`: The intercept value.
- `trend::Fl`: The trend value.
- `ϕ::Vector{Fl}`: The autoregressive coefficients.
- `θ::Vector{Fl}`: The moving average coefficients.
- `ϵ::Vector{Fl}`: The residuals.
- `σ²::Fl`: The model's σ².
- `fitInSample::TimeArray`: The fitted values.
- `Φ::Union{Vector{Fl}, Nothing}`: The seasonal autoregressive coefficients. Default is `nothing`.
- `Θ::Union{Vector{Fl}, Nothing}`: The seasonal moving average coefficients. Default is `nothing`.
- `exogCoefficients::Union{Vector{Fl}, Nothing}`: The exogenous variable coefficients. Default is `nothing`.

"""
function fillFitValues!(
    model::SARIMAModel,
    c::Fl,
    trend::Fl,
    ϕ::Vector{Fl},
    θ::Vector{Fl},
    ϵ::Vector{Fl},
    σ²::Fl,
    fitInSample::TimeArray;
    Φ::Union{Vector{Fl},Nothing} = nothing,
    Θ::Union{Vector{Fl},Nothing} = nothing,
    exogCoefficients::Union{Vector{Fl},Nothing} = nothing,
) where {Fl<:AbstractFloat}
    model.c = c
    model.trend = trend
    model.ϕ = ϕ
    model.θ = θ
    model.ϵ = ϵ
    model.σ² = σ²
    model.Φ = Φ
    model.Θ = Θ
    model.fitInSample = fitInSample
    model.exogCoefficients = exogCoefficients
end

"""
    isFitted(model::SARIMAModel)

Returns `true` if the SARIMA model has been fitted.

# Arguments
- `model::SARIMAModel`: The SARIMA model.

# Returns
- `Bool`: `true` if the model has been fitted; otherwise, `false`.

"""
function isFitted(model::SARIMAModel)
    hasResiduals = !isnothing(model.ϵ)
    hasFitInSample = !isnothing(model.fitInSample)
    estimatedAR = (model.p == 0) || !isnothing(model.ϕ)
    estimatedMA = (model.q == 0) || !isnothing(model.θ)
    estimatedSeasonalAR = (model.P == 0) || !isnothing(model.Φ)
    estimatedSeasonalMA = (model.Q == 0) || !isnothing(model.Θ)
    estimatedIntercept = !model.allowMean || !isnothing(model.c)
    estimatedExog = isnothing(model.exog) || !isnothing(model.exogCoefficients)
    return hasResiduals &&
           hasFitInSample &&
           estimatedAR &&
           estimatedMA &&
           estimatedSeasonalAR &&
           estimatedSeasonalMA &&
           estimatedIntercept &&
           estimatedExog
end

"""
    get_hyperparameters_number(model::SARIMAModel)

Returns the number of estimated parameters `K` of a SARIMA model (including σ²),
as used by the information criteria.

For regular objectives every declared parameter counts, regardless of the
magnitude of its estimate — a coefficient estimated near zero was still
estimated. For `elastic_net` fits the count is instead the
number of ACTIVE coefficients (|coef| > 1e-5), the standard effective-degrees-
of-freedom estimate for L1-type regularization (Zou, Hastie & Tibshirani, 2007).

# Arguments
- `model::SARIMAModel`: The SARIMA model.

# Returns
- `Int`: The number of parameters `K`.

"""
function get_hyperparameters_number(model::SARIMAModel)
    # Keyed on the OBJECTIVE that fitted the model, not on `lambda`/`alpha` being set:
    # those fields do not by themselves change the estimate, so letting them switch the
    # count would move the criterion without moving the fit.
    #
    # The non-zero count applies to `elastic_net` at any `alpha`. Restricting it to the
    # lasso case (`alpha = 1`), the only one with theoretical backing (Zou, Hastie &
    # Tibshirani 2007 — under ridge the count degenerates to the nominal one, since ridge
    # shrinks without zeroing), would be a policy change rather than a fix.
    usesSparseCount = get(model.metadata, "objectiveFunction", "") == "elastic_net"
    if isFitted(model) && usesSparseCount
        hyperparametersNumber = 1
        fields = [:c, :trend, :ϕ, :θ, :Φ, :Θ, :exogCoefficients]
        for field in fields
            fieldValue = getfield(model, field)
            isnothing(fieldValue) && continue
            fieldValues = [fieldValue...]
            length(fieldValues) > 0 && (hyperparametersNumber += length(filter(x-> abs(x) > 1e-5, fieldValues)))
        end
        return hyperparametersNumber
    end
    k = (model.allowMean) ? 1 : 0
    k = (model.allowDrift) ? k + 1 : k
    β = isnothing(model.exog) ? 0 : length(colnames(model.exog))
    # `K = ncoef + 1` (the +1 is sigma^2), the `forecast::Arima` convention. The `n` of the
    # criteria comes from `criterionLoglikeAndN`: `T = length(diffY)` (R's
    # `n* = n - d - D*m`) when the exact likelihood is used, and
    # `length(observedResiduals) = T - lb + 1` on the CSS fallback, where the conditioned
    # residuals discount `lb` and are therefore NOT R's `n*`.
    #
    # Free pre-sample values are not counted as parameters. Charging them would make the
    # criterion incomparable with R's; under `:penalized` the objective already prices the
    # pre-sample block, and under `:free` the rejection rule is the defense, as in R.
    return model.p + model.q + model.P + model.Q + k + β + 1
end

function get_hyperparameters_number(model::JuMP.Model)
    # Read through the object dictionary (`model[:c]`) rather than `variable_by_name`: the
    # build disables the variables' string names, so `variable_by_name` would return
    # `nothing` for `c` and `trend` and silently drop them from the count. The object
    # dictionary is populated by `@variable` regardless of string names.
    c = haskey(model, :c) ? model[:c] : nothing
    trend = haskey(model, :trend) ? model[:trend] : nothing
    hyperparametersNumber = (c !== nothing && abs(value(c)) > 1e-5) ? 1 : 0
    hyperparametersNumber += (trend !== nothing && abs(value(trend)) > 1e-5) ? 1 : 0

    hyperparameters = [:ϕ, :θ, :Φ, :Θ, :exogCoefficients]
    # Access if the value is near zero (absolute value less than 1e-6)
    for hyperparameter in hyperparameters
        hyperparameterValue = []
        try
            hyperparameterValue = value.(model[hyperparameter])
        catch
            continue
        end
        hyperparametersNumber += length(filter(x-> abs(x) > 1e-5, hyperparameterValue))
    end
    return hyperparametersNumber + 1
end

"""
    fit!(
        model::SARIMAModel;
        silent::Bool=true,
        optimizer::DataType=Ipopt.Optimizer,
        objectiveFunction::String="mse"
        automaticExogDifferentiation::Bool=false
        invertible::Bool=false
        invertibilityMargin::AbstractFloat=DEFAULT_DOMAIN_MARGIN
    )

Estimate the SARIMA model parameters via conditional least squares (CSS) formulated
as a JuMP optimization problem: the residuals are decision variables tied to the data
by the model dynamics, and the first `lb-1` pre-sample residuals are fixed at zero.
No Kalman filter / exact likelihood is used; the `"ml"` objective is the concentrated
conditional Gaussian likelihood (equivalent to least squares in the coefficients).
The resulting optimal parameters as well as the residuals and the model's σ² are
stored within the model.
The default objective function used to estimate the parameters is the mean squared error (MSE),
but it can be changed to the maximum likelihood (ML) by setting the `objectiveFunction` parameter to "ml".

# Arguments
- `model::SARIMAModel`: The SARIMA model to be fitted.
- `silent::Bool`: Whether to suppress solver output. Default is `true`.
- `optimizer::DataType`: The optimizer to be used for optimization. Default is `Ipopt.Optimizer`.
- `mipSolver::DataType`: The MIP sub-solver used by the Alpine global optimizer for its
  lower-bounding step. Default is `SCIP.Optimizer`, which can solve the mixed-integer quadratic
  (MIQP) relaxations arising from quadratic objectives such as `"mse"`. If `HiGHS.Optimizer` is
  supplied it only works with the linear `"mae"` objective; a warning is issued otherwise. Only
  relevant when `optimizer = Alpine.Optimizer`.
- `objectiveFunction::String`: The objective function used for estimation. Default is "mse".
- `automaticExogDifferentiation::Bool`: Whether to automatically differentiate the exogenous variables. Default is `false`.
- `invertible::Bool`: When `true`, the (seasonal) moving-average coefficients are generated from
  bounded reflection coefficients `κ` via [`reflectionToMA`](@ref), guaranteeing an invertible MA
  polynomial by construction instead of imposing only box bounds on `θ`/`Θ`. Not compatible with
  the `"bilevel"` objective. Default is `false`, mirroring `stats::arima`, which does NOT constrain
  the MA during optimization and only converts to an invertible representation afterwards.

  Note the two regions do not nest: the free path's box (see [`admissibleCoefficientBound`])
  admits non-invertible points, while the invertibility region contains points outside any
  per-coefficient box. They coincide only for `q = Q = 1`.
- `invertibilityMargin::AbstractFloat`: Margin `ρ ∈ [0, 1)` that bounds the reflection coefficients to
  `[-(1-ρ), 1-ρ]`, keeping the solution `ρ` away from the unit circle. Only used when `invertible=true`.
  Default is `DEFAULT_DOMAIN_MARGIN` (`1e-6`): enough to keep the domain open, small enough
  not to truncate near-unit-root estimates. Do NOT set it to the rejection margin
  [`DEFAULT_ROOT_MARGIN`] — that imposes a selection rule as an estimation constraint.
- `seasonalForm::Symbol`: `:multiplicative` (Box-Jenkins, default) or `:additive`.
- `exogDynamics::Symbol`: Semantics of the exogenous block. `:armax` (default) is the
  dynamic-regression/ARX form, in which the AR terms act on the observed series and the
  coefficient is an impact multiplier conditional on past `y`. `:regression_errors` is
  regression with ARIMA errors, the form `forecast::Arima(xreg=)` estimates, in which the
  coefficient is the usual marginal effect. The two coincide only when the autoregressive
  polynomial is unitary and there is no differencing, so the choice is observable only
  when regressors are present.
- `penaltyTarget::Symbol`: Which coefficient blocks the `"elastic_net"` penalty applies
  to: `:all` (default), `:dynamics` for the autoregressive and moving-average blocks only,
  or `:exogenous` for the regressors only. The intercept and the drift are never
  penalized. Ignored by every other objective.
- `stationary::Bool`: When `true`, the (seasonal) AR coefficients are generated from
  bounded reflection coefficients (partial autocorrelations) via [`reflectionToAR`](@ref),
  guaranteeing a stationary AR polynomial by construction (exact under `:multiplicative`;
  per-block only under `:additive`). Default is `true`, mirroring `stats::arima` with
  `transform.pars = TRUE`, which parameterizes the AR through `tanh`. When `false`, the
  coefficients are box-bounded by [`admissibleCoefficientBound`], which excludes no
  admissible model.
- `stationarityMargin::AbstractFloat`: Margin in `[0, 1)` bounding the AR reflection
  coefficients to `[-(1-margin), 1-margin]`. Only used when `stationary = true`.
  Default is `DEFAULT_DOMAIN_MARGIN` (`1e-6`), the AR analogue of `invertibilityMargin`.
- Missing observations: `NaN` entries in the endogenous series are supported for
  stationary models (`d = D = 0`, `mse`/`ml` objectives, no exogenous regressors). Each
  gap becomes a free decision variable whose residual is retained in the objective,
  yielding the two-sided conditional smoother; σ², the log-likelihood, the effective
  sample size and the residual diagnostics all exclude the imputed indices. The imputed
  values are written back into `model.y` and recorded in `model.metadata["nMissing"]`.
- `initialization::Symbol`: CSS conditioning convention. The five modes are points in a
  SPACE of independent choices, not rungs on a ladder. Three are known: whether the
  pre-sample block is free or held at zero; whether that block is penalised in the
  objective; and how the block is parameterised. Read the axes below, not the names —
  two modes sharing the first two axes still estimate different models.
  `:innovations` (**default**) is free-and-penalised: the pre-sample innovations are
  decision variables, the objective carries the determinant factor
  `∏ⱼ(1-κⱼ²)^(-j/T)`, and the fit-error sum starts at `t = 1` for EVERY objective
  function rather than after the conditioning lag.
  `:penalized` is free-and-penalised too, but parameterises the pre-sample block
  differently: it leaves the pre-sample differenced values AND the pre-sample residuals
  free at the same time, whereas `:innovations` generates the pre-sample past from the
  innovations alone. The two are NOT interchangeable — measured on 6 M4 monthly series
  at fixed order (2,1,2), they disagree in all 6, with `φ₁` flipping from `0.886` to
  `-0.618` on one of them. Which of the two admits the other's optimum is an open
  question, not a documented property.
  `:free` estimates the pre-sample residuals and the pre-sample differenced endogenous
  values as free variables (Box-Jenkins backcasting / unconditional least squares) and
  keeps every differenced observation in the objective, but does NOT penalise the block.
  `:zeroed` fixes the pre-sample residuals at zero and drops the first
  `max(p+sP, q+sQ)` differenced observations — it is the only mode whose optimum was
  measured to depend on the solver's starting point.
  `:warmup` conditions only on the AR-side lags and warm-starts the MA recursion from the
  beginning of the differenced sample, matching R's `arima(..., method = "CSS")`.
  The free-and-penalised concentrated likelihood tracks the exact (Kalman) likelihood up
  to a near-constant log-determinant term, making information criteria comparable across
  candidate orders. Exact-likelihood (Kalman) initialization is out of scope by design.
  See [`DEFAULT_INITIALIZATION`].
- `warmStart::Union{Nothing,SARIMAModel}`: A previously fitted model of the same
  specification whose solution seeds this solve (residual vector, coefficients and —
  under the reflection parameterisations — their reflection-space preimages via
  [`arToReflection`](@ref)/[`maToReflection`](@ref)). A starting point only: the
  problem being solved is unchanged. Default is `nothing`.
- `warmStartFromBox::Bool`: When `true` and a constrained fit is requested
  (`stationary` and/or `invertible`), first solves the cheap unconstrained (box)
  problem and warm-starts the constrained one from it, falling back through three
  tiers if it does not converge within the budget: full constraints → invertibility
  only → the unconstrained seed. The tier reached is stored in
  `model.metadata["warmStartTier"]` (1, 2 or 3). Default is `false`.
- `maxTimeSeconds::Union{Nothing,Real}`: Budget for each solve. Sets both the solver
  wall-clock limit and (for Ipopt) an iteration ceiling — a wall-clock limit alone is
  only checked between iterations and cannot bound a single expensive iteration. On a
  budget cut the fit returns with the corresponding solver status
  (`TIME_LIMIT`/`ITERATION_LIMIT`) instead of running unbounded. Default is `nothing`
  (no limit).

Internally the differenced series is scaled by its standard deviation before the
model is built and the scale-dependent estimates are mapped back afterwards, keeping
the objective well conditioned for large-magnitude data; AR/MA coefficients are
invariant to this and results where the solver already converged are unchanged.

# Example
```jldoctest
julia> airPassengers = load_dataset(AIR_PASSENGERS)

julia> model = SARIMA(airPassengers,0,1,1;seasonality=12,P=0,D=1,Q=1)

julia> fit!(model)
```
"""
function fit!(
    model::SARIMAModel;
    silent::Bool = true,
    optimizer::Union{DataType,MOI.OptimizerWithAttributes} = Ipopt.Optimizer,
    mipSolver::DataType = SCIP.Optimizer,
    objectiveFunction::String = "mse",
    automaticExogDifferentiation::Bool = false,
    alpha::Union{Nothing,<:AbstractFloat} = nothing,
    lambda::Union{Nothing,<:AbstractFloat} = nothing,
    invertible::Bool = false,
    invertibilityMargin::AbstractFloat = DEFAULT_DOMAIN_MARGIN,
    minConditioningObs::Int = 0,
    seasonalForm::Symbol = :multiplicative,
    initialization::Symbol = DEFAULT_INITIALIZATION,
    # Semantics of the exogenous block. See `arAcc` in the model construction for the two
    # equations.
    #
    # The default is `:armax`: the dynamic-regression/ARX form, in which `beta` is an
    # impact multiplier conditional on past `y`. Under `:regression_errors` -- regression
    # with ARIMA errors, the form `forecast::Arima(xreg=)` estimates and the one Hyndman
    # recommends in "The ARIMAX model muddle" (2010) -- `beta` is the marginal effect
    # instead. The two coincide only when the autoregressive polynomial is unitary and
    # there is no differencing.
    #
    # The choice is only observable with regressors present: at `nExog == 0` `arAcc` takes
    # the same branch under both modes.
    exogDynamics::Symbol = :armax,
    # Which coefficient blocks the `elastic_net` penalty applies to: `:all` (default),
    # `:dynamics` for the AR/MA blocks only, or `:exogenous` for the regressors only.
    # The intercept and the drift are never penalized. Ignored by every other objective.
    penaltyTarget::Symbol = :all,
    # Extra window of pre-sample innovations under `initialization = :innovations`; it has
    # no effect in any other mode. The default covers one seasonal cycle beyond what the
    # order requires, enough for the null initial condition to decay.
    presampleBurnIn::Int = 12,
    # R's default: `stats::arima` with `transform.pars = TRUE` parameterizes the AR part
    # through `tanh`, i.e. stationary BY CONSTRUCTION on an open domain. The MA part stays
    # free (see `invertible`), which is the other half of R's behaviour.
    stationary::Bool = true,
    stationarityMargin::AbstractFloat = DEFAULT_DOMAIN_MARGIN,
    warmStart::Union{Nothing,SARIMAModel} = nothing,
    maxTimeSeconds::Union{Nothing,Real} = nothing,
    warmStartFromBox::Bool = false,
    cvarLevel::AbstractFloat = DEFAULT_CVAR_LEVEL,
    multistart::Bool = false,
)
    @assert 0.0 < cvarLevel < 1.0 "cvarLevel must lie strictly between 0 and 1."
    penaltyTarget in (:all, :dynamics, :exogenous) || throw(
        ArgumentError(
            "penaltyTarget must be :all, :dynamics or :exogenous, got $(penaltyTarget)",
        ),
    )
    exogDynamics in (:armax, :regression_errors) || throw(
        ArgumentError(
            "exogDynamics must be :armax or :regression_errors, got $(exogDynamics)",
        ),
    )
    # Two-phase warm-start orchestration: when a stationarity/invertibility-by-
    # construction fit is requested with `warmStartFromBox`, first solve the cheap
    # unconstrained (box) problem, then solve the constrained one warm-started from it
    # under the `maxTimeSeconds` cap; if the constrained solve does not converge in
    # time, fall back to the (valid) unconstrained fit. This keeps the guarantee when
    # affordable and a usable model otherwise, addressing the O(T) reflection blow-up on
    # long series. Purely an optimization technique (a starting point).
    #
    # The Huber objective is linear in the tail, so far from the optimum a large residual
    # costs little and the surface is nearly flat — Ipopt wanders and, on the M4 monthly,
    # returned LOCALLY_INFEASIBLE with forecasts of +-1e5 on orders that "mse" solved
    # cleanly. Six series ended above MASE 100 (one at 5e5) where no other objective, not
    # even Naive2, exceeded 38.55: the ceiling elsewhere is a property of those series,
    # the excess was numerical.
    #
    # The classical remedy for an M-estimator is to start from least squares (IRLS begins
    # at OLS). Solve `mse` first, warm start Huber from it, and keep the `mse` fit when the
    # Huber solve does not reach an acceptable status — so Huber is never worse than the
    # starting point it refines.
    #
    # Bounding `u` to [-delta, delta] was tried instead and made things worse: it is exact
    # at the optimum but adds 2(T-lb+1) active box constraints, and LOCALLY_INFEASIBLE went
    # from 2 to 3 of the six diagnosed series.

    # MULTISTART {zero, CSS}: the starts are the zero vector and the CSS fit, which is the
    # path `stats::arima` itself takes (CSS then ML) — deterministic, with no new constant
    # to calibrate.
    #
    # The tie-break is by CRITERION, not by SSE. At fixed order `K` and `n` coincide across
    # starts, so comparing AICc compares the criterion's likelihood; by SSE the lower
    # in-sample error would win, which is not the better forecast.
    #
    # At fixed order the effect is small and never negative — the zero start remains a
    # candidate. The step pays off inside the SEARCH, where a marginally better criterion
    # can change the selected order, so measure it through `auto` rather than through a
    # fixed-order fit.
    if multistart && isnothing(warmStart)
        passaM = (;
            silent, optimizer, mipSolver, automaticExogDifferentiation, alpha, lambda,
            invertible, invertibilityMargin, minConditioningObs, seasonalForm,
            stationary, stationarityMargin, maxTimeSeconds, warmStartFromBox, cvarLevel,
        )
        criterio(m) = try
            v = aicc(m)
            isfinite(v) ? v : Inf
        catch
            Inf
        end
        okStatus(m) = get(m.metadata, "solverStatus", "") in
                      ("LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED", "TIME_LIMIT")

        # semente CSS: o mesmo objetivo, condicionado (`:zeroed`)
        semente = nothing
        try
            cand = deepcopy(model)
            fit!(cand; passaM..., objectiveFunction = objectiveFunction,
                 initialization = :zeroed, multistart = false)
            okStatus(cand) && (semente = cand)
        catch
        end

        # partida do zero. Sob `:zeroed` ela e IDENTICA a semente (mesma inicializacao,
        # mesmos argumentos): reaproveita em vez de pagar o mesmo ajuste duas vezes.
        aZero = nothing
        if initialization === :zeroed
            aZero = semente
        else
            try
                cand = deepcopy(model)
                fit!(cand; passaM..., objectiveFunction = objectiveFunction,
                     initialization = initialization, multistart = false)
                okStatus(cand) && (aZero = cand)
            catch
            end
        end
        # partida da semente CSS
        aCSS = nothing
        if !isnothing(semente)
            try
                cand = deepcopy(model)
                fit!(cand; passaM..., objectiveFunction = objectiveFunction,
                     initialization = initialization, warmStart = semente, multistart = false)
                okStatus(cand) && (aCSS = cand)
            catch
            end
        end

        cands = filter(!isnothing, [aZero, aCSS])
        if isempty(cands)
            # nenhuma partida convergiu: cai no caminho normal e deixa o erro aparecer la
            return fit!(model; passaM..., objectiveFunction = objectiveFunction,
                        initialization = initialization, multistart = false)
        end
        vencedor = cands[argmin([criterio(c) for c in cands])]
        # `:y` is on the list because the fit imputes missing values INSIDE the model
        # itself (`model.y = ...`); without copying it, the winner's `y` and its residuals
        # would come from different samples on series with gaps.
        for f in (:ϕ, :θ, :Φ, :Θ, :c, :trend, :ϵ, :σ², :fitInSample, :exogCoefficients,
                  :icOffset, :y)
            hasproperty(model, f) && setfield!(model, f, getfield(vencedor, f))
        end
        merge!(model.metadata, vencedor.metadata)
        model.metadata["multistartPartidas"] = length(cands)
        model.metadata["multistartVenceuCSS"] = (vencedor === aCSS)
        return model
    end

    if objectiveFunction == "huber" && isnothing(warmStart)
        passa = (;
            silent, optimizer, mipSolver, automaticExogDifferentiation, alpha, lambda,
            invertible, invertibilityMargin, minConditioningObs, seasonalForm,
            initialization, stationary, stationarityMargin, maxTimeSeconds,
            warmStartFromBox, cvarLevel,
        )
        base = deepcopy(model)
        fit!(base; passa..., objectiveFunction = "mse")
        aceitavel(m) = get(m.metadata, "solverStatus", "") in
                       ("LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED")
        ok = false
        try
            fit!(model; passa..., objectiveFunction = "huber", warmStart = base)
            ok = aceitavel(model)
        catch
            ok = false
        end
        if !ok
            model.c = base.c
            model.trend = base.trend
            model.ϕ = base.ϕ
            model.θ = base.θ
            model.Φ = base.Φ
            model.Θ = base.Θ
            model.ϵ = base.ϵ
            model.σ² = base.σ²
            model.fitInSample = base.fitInSample
            model.exogCoefficients = base.exogCoefficients
            merge!(model.metadata, base.metadata)
            model.metadata["huberFallback"] = true
        else
            model.metadata["huberFallback"] = false
        end
        return model
    end

    if warmStartFromBox && (stationary || invertible) && isnothing(warmStart)
        common = (;
            silent, optimizer, mipSolver, objectiveFunction, automaticExogDifferentiation,
            alpha, lambda, minConditioningObs, seasonalForm, initialization, cvarLevel,
        )
        seed = deepcopy(model)
        fit!(seed; common..., stationary = false, invertible = false,
             stationarityMargin = 0.0, invertibilityMargin = 0.0,
             maxTimeSeconds = maxTimeSeconds)
        # Accumulates the cost of the attempts made ON `model` (tiers 1 and 2). Kept out of
        # the metadata because the tier-3 `merge!` would overwrite it; reconciled before
        # returning.
        modelTimings = Dict{String,Float64}("build" => 0.0, "solve" => 0.0, "count" => 0.0)
        accumulate!() = begin
            modelTimings["build"] += get(model.metadata, "buildTimeSec", 0.0)
            modelTimings["solve"] += get(model.metadata, "solveTimeSec", 0.0)
            modelTimings["count"] += 1.0
        end
        solved() = get(model.metadata, "solverStatus", "") in
                   ("LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED")
        tryFit(st, inv, sMargin, iMargin) = begin
            ok = false
            try
                fit!(model; common..., stationary = st, invertible = inv,
                     stationarityMargin = sMargin, invertibilityMargin = iMargin,
                     warmStart = seed, maxTimeSeconds = maxTimeSeconds)
                ok = solved()
            catch
                ok = false
            end
            accumulate!()
            ok
        end
        # tier 1: full stationarity + invertibility by construction
        converged = tryFit(stationary, invertible, stationarityMargin, invertibilityMargin)
        tier = 1
        # tier 2: relax stationarity, keep invertibility — only when tier 1 failed for a
        # *numerical/feasibility* reason (fast). If tier 1 was cut off by a budget limit
        # (time or iterations), tier 2 is just as constrained and would burn the same
        # budget again, so skip straight to the box result.
        hitLimit = get(model.metadata, "solverStatus", "") in
                   ("TIME_LIMIT", "ITERATION_LIMIT", "OTHER_LIMIT", "MEMORY_LIMIT")
        if !converged && stationary && invertible && !hitLimit
            converged = tryFit(false, true, 0.0, invertibilityMargin)
            tier = 2
        end
        # tier 3: fall back to the unconstrained fit — which is exactly `seed`, already
        # solved above. Re-solving it here doubled the cost of every failing candidate
        # (the dominant term in the stepwise search on hard series), so copy it instead.
        if !converged
            model.c = seed.c
            model.trend = seed.trend
            model.ϕ = seed.ϕ
            model.θ = seed.θ
            model.Φ = seed.Φ
            model.Θ = seed.Θ
            model.ϵ = seed.ϵ
            model.σ² = seed.σ²
            model.fitInSample = seed.fitInSample
            model.exogCoefficients = seed.exogCoefficients
            merge!(model.metadata, seed.metadata)
            tier = 3
        end
        model.metadata["warmStartTier"] = tier
        # The box solve happens in `seed`, a separate object, so its cost does not reach the
        # accumulators of `model` on its own, and at tier 3 the `merge!` above overwrites
        # `model`'s with `seed`'s. Without this reconciliation the telemetry of a
        # warm-started candidate reports a subset of what it cost. Rebuild the sum
        # explicitly.
        model.metadata["buildTimeSecTotal"] =
            get(modelTimings, "build", 0.0) + get(seed.metadata, "buildTimeSecTotal", 0.0)
        model.metadata["solveTimeSecTotal"] =
            get(modelTimings, "solve", 0.0) + get(seed.metadata, "solveTimeSecTotal", 0.0)
        model.metadata["fitCount"] =
            Int(get(modelTimings, "count", 0.0)) + get(seed.metadata, "fitCount", 0)
        return model
    end

    Fl = typeofModelElements(model)
    isFitted(model) &&
        @info("The model has already been fitted. Overwriting the previous results")
    @assert objectiveFunction ∈ ["mae", "mse", "ml", "bilevel", "elastic_net", "stable", "ridge", "huber", "ml_exact"] "The objective function $objectiveFunction is not supported. Please use 'mae', 'mse', 'ml', 'bilevel', 'elastic_net', 'stable', 'ridge', 'huber' or 'ml_exact'"
    @assert !(invertible && MACoefficientsAreModelParameters(objectiveFunction)) "The invertible MA parameterization is not compatible with the '$objectiveFunction' objective (MA coefficients are treated as outer parameters there)."
    @assert 0.0 <= invertibilityMargin < 1.0 "invertibilityMargin (ρ) must lie in [0, 1)."
    @assert 0.0 <= stationarityMargin < 1.0 "stationarityMargin must lie in [0, 1)."
    if objectiveFunction == "elastic_net"
        @assert (!isnothing(alpha) || !isnothing(model.alpha)) "In elastic net objective function, alpha must be specified"
    end

    # Deprecated in v1.0, scheduled for removal in v2.0. The objective optimizes the
    # moving-average coefficients in an outer loop, which costs orders of magnitude more
    # solver time than the objectives that keep them as decision variables, and it has no
    # remaining use that those do not cover.
    if objectiveFunction == "bilevel"
        @warn(
            "objectiveFunction = \"bilevel\" is deprecated and will be removed in " *
            "Sarimax v2.0. Use \"mse\" (or \"ml\") instead; they estimate the same " *
            "moving-average coefficients as decision variables at a fraction of the cost.",
            maxlog = 1
        )
    end

    model.allowMean && model.allowDrift && throw(
        InvalidParametersCombination(
            "allowMean and allowDrift are mutually exclusive: in the differenced " *
            "equation they were perfectly collinear. Use allowMean for d+D == 0 " *
            "and allowDrift for d+D == 1.",
        ),
    )
    seasonalForm === :free && throw(ArgumentError("seasonalForm :free is planned for a later release"))
    seasonalForm in (:multiplicative, :additive) ||
        throw(ArgumentError("seasonalForm must be :multiplicative or :additive"))
    # Invalid ARGUMENT COMBINATIONS raise rather than warn. The combination is fixed at
    # the call site and has no valid interpretation, so the caller can check it in advance;
    # a warning is invisible in a parallel sweep, and a sweep in which some cells silently
    # mean something else is a broken sweep. Run-time DEGRADATION is the separate case
    # (`ml_exact` falling back to CSS depends on the candidate, whose order varies inside
    # `auto`) and warns instead, since raising there would abort the search.
    #
    # The list below must mirror EXACTLY the penalized-objective gate (search for
    # `&& penalizado` in this file). `:innovations` is named next to `:penalized` because
    # the gate is `penalizado = initialization in (:penalized, :innovations)`; an objective
    # this guard admits but the gate does not cover would leave the pre-sample values
    # unpenalized and silently degrade the fit to `:free`.
    #
    # The penalized pre-sample block is implemented for the losses whose block term can be
    # written in the SAME loss as the data, which is what prevents mixing scales (an L2
    # block penalty against an L1 data loss). Under `mse` the block enters as a sum of
    # squares and the determinant factor applies. Under `mae` and `huber` it enters as the
    # data loss itself and the determinant factor does NOT apply: that factor comes from
    # concentrating sigma^2, a step that holds only under a quadratic loss. The remaining
    # objectives add the block through `presampleSquares`.
    if initialization in (:penalized, :innovations) &&
       !(objectiveFunction in
         ("mse", "mae", "huber", "ml", "ml_exact", "ridge", "stable", "bilevel", "elastic_net"))
        throw(
            ArgumentError(
                "initialization = :$(initialization) is implemented for objectiveFunction " *
                "in (\"mse\", \"mae\", \"huber\", \"ml\", \"ml_exact\", \"ridge\", " *
                "\"stable\", \"bilevel\", \"elastic_net\"); got \"$(objectiveFunction)\". " *
                "The pre-sample values would stay unpenalized, i.e. the fit would " *
                "silently degrade to :free.",
            ),
        )
    end
    initialization in (:zeroed, :warmup, :free, :penalized, :innovations) || throw(
        ArgumentError(
            "initialization must be :zeroed, :warmup, :free, :penalized or :innovations",
        ),
    )

    isnothing(lambda) || (model.lambda = lambda)
    isnothing(alpha) || (model.alpha = alpha)
    model.metadata["seasonalForm"] = String(seasonalForm)
    model.metadata["exogDynamics"] = String(exogDynamics)
    model.metadata["initialization"] = String(initialization)
    # Recorded because the PARAMETER COUNT depends on the objective that actually fitted
    # the model, not on `lambda`/`alpha` being set — see `get_hyperparameters_number`.
    model.metadata["objectiveFunction"] = objectiveFunction

    # The `ridge` objective fixes `lambda = sqrt(nEff)` internally and ignores the
    # argument. Accepting it silently would let the caller believe they control the
    # shrinkage while the fit does not change; `elastic_net` is the objective that honours
    # a caller-supplied lambda.
    if objectiveFunction == "ridge" && !isnothing(lambda)
        throw(
            ArgumentError(
                "objectiveFunction = \"ridge\" ignores `lambda`: the shrinkage is fixed at " *
                "sqrt(effective sample size) by construction. Passing it would have no " *
                "effect on the fit; drop the argument or use \"elastic_net\".",
            ),
        )
    end

    # Cost telemetry (performance attribution). A search costs (number of fits) x (cost per
    # fit), and the cost per fit splits into BUILDING the JuMP problem (grows with T and
    # with the order — about 2T variables under `:free`) and SOLVING it (grows with
    # numerical difficulty). Without separating the two, "more candidates", "larger models"
    # and "harder solves" are indistinguishable, and they call for different remedies.
    # Purely observational: nothing here changes the estimate.
    fitStartTime = time()

    diffY = differentiate(model.y, model.d, model.D, model.seasonality)

    if !isnothing(model.exog)
        if automaticExogDifferentiation
            diffExog, exogMetadata =
                automatic_differentiation(model.exog; seasonalPeriod = model.seasonality)
            model.metadata["exog"] = exogMetadata
            diffY = TimeSeries.merge(diffY, diffExog)
        else
            diffY = TimeSeries.merge(diffY, model.exog)
        end
    end

    T = length(diffY)

    # Drift enters as the differentiated deterministic-time regressor: for
    # d + D == 0 it is the linear trend t itself; for d = 1 it reduces to a
    # constant (classic drift); for d + D > 1 it vanishes (not identifiable).
    driftValues::Vector{Fl} = if model.allowDrift
        model.d + model.D > 1 && @warn(
            "Drift with d + D > 1 is not identifiable (the differenced trend is zero)."
        )
        diffT = differentiate(collect(Fl, 1:length(values(model.y))), model.d, model.D, model.seasonality)
        diffT[end-T+1:end]
    else
        ones(Fl, T)
    end

    yValues = values(diffY)[:, 1]
    nExog = isnothing(model.exog) ? 0 : size(values(diffY), 2) - 1
    exogValues = isnothing(model.exog) ? [] : values(diffY)[:, 2:end]

    # Numerical conditioning: the CSS objective is quadratic in the data, so a series
    # living around 1e4 produces an objective near 1e8. Ipopt then drops into its
    # restoration phase and a *single* iteration can run for minutes — no time or
    # iteration cap helps, because both are only checked between iterations. Measured
    # on M4 daily series 3441: 208s without converging on the raw data versus 1.1s
    # (converged) on the same series scaled. So solve in units of the differenced
    # series' standard deviation and map the scale-dependent estimates back below.
    # AR/MA coefficients are invariant under this rescaling; only c, trend, the
    # exogenous coefficients, the residuals and their variance carry the factor.
    yScale = let finiteY = filter(isfinite, yValues)
        s = isempty(finiteY) ? one(Fl) : Fl(Statistics.std(finiteY))
        (isfinite(s) && s > zero(Fl)) ? s : one(Fl)
    end
    yValues = yValues ./ yScale

    # Missing-data support: NaN entries in the (differenced) endogenous series
    # are treated as free decision variables (Section: missing observations).
    missingMask::Vector{Bool} = isnan.(yValues)
    hasMissing = any(missingMask)
    if hasMissing
        (model.d + model.D > 0) && throw(
            ArgumentError(
                "Missing-data estimation currently supports only stationary " *
                "models (d = D = 0); differencing propagates gaps and requires " *
                "re-integration handling that is not implemented yet.",
            ),
        )
        isnothing(model.exog) || throw(
            ArgumentError("Missing-data estimation with exogenous regressors is not yet supported."),
        )
        objectiveFunction in ("mse", "ml") || throw(
            ArgumentError("Missing-data estimation supports only the 'mse' and 'ml' objectives."),
        )
        model.keepProvidedCoefficients && throw(
            ArgumentError("Missing-data estimation is not compatible with provided fixed coefficients."),
        )
    end

    residualLags =
        initialization in (:free, :penalized, :innovations) ? 0 :
        initialization === :warmup ?
        (
            seasonalForm === :multiplicative ?
            model.p + model.seasonality * model.P :
            max(model.p, model.seasonality * model.P)
        ) :
        conditioningLags(model.p, model.q, model.P, model.Q, model.seasonality, seasonalForm)
    lb = max(residualLags, minConditioningObs) + 1

    mod = Model(optimizer)
    # The variables' string names serve only printing and `variable_by_name`, and each is
    # one allocated String per variable — with about 2T variables under `:free` that is pure
    # build work. Build is about 22% of the cost of a search, stable across the regimes
    # tested. The only consumer was `get_hyperparameters_number(::JuMP.Model)`, which now
    # reads the object dictionary.
    set_string_names_on_creation(mod, false)

    if (model.allowMean)
        @variable(mod, c)
    else
        @variable(mod, c in Parameter(1.0))
        set_parameter_value(mod[:c], 0.0)
    end

    if (model.allowDrift)
        @variable(mod, trend)
    else
        @variable(mod, trend in Parameter(1.0))
        set_parameter_value(mod[:trend], 0.0)
    end

    @variable(mod, β[1:nExog])
    if stationary
        # Stationary-by-construction AR parameterization: the AR coefficients are
        # generated from bounded reflection coefficients (partial autocorrelations)
        # via reflectionToAR. Under :multiplicative this guarantees stationarity of
        # the full polynomial (each factor is stationary); under :additive it
        # constrains each block's own polynomial only (necessary, not sufficient).
        ρs = stationarityMargin
        @variable(mod, ϕ[1:model.p])
        @variable(mod, Φ[1:model.P])
        if model.p > 0
            @variable(mod, -(1 - ρs) <= κAR[1:model.p] <= (1 - ρs))
            ϕκ = reflectionToAR(κAR)
            @constraint(mod, [i = 1:model.p], ϕ[i] == ϕκ[i])
            for i = 1:model.p
                set_start_value(κAR[i], 0.0)
                set_start_value(ϕ[i], 0.0)
            end
        end
        if model.P > 0
            @variable(mod, -(1 - ρs) <= κSAR[1:model.P] <= (1 - ρs))
            Φκ = reflectionToAR(κSAR)
            @constraint(mod, [k = 1:model.P], Φ[k] == Φκ[k])
            for k = 1:model.P
                set_start_value(κSAR[k], 0.0)
                set_start_value(Φ[k], 0.0)
            end
        end
    else
        # A bound that EXCLUDES no admissible model — see `admissibleCoefficientBound`. The
        # plain `-1 <= phi_i <= 1` excludes legitimate estimates from order 2 on: in a
        # stationary AR(2), |phi_1| reaches 2. Measured against `stats::arima` over 125
        # fits, 11.2% of R's ML estimates fall outside that box and 16.5% of our fits
        # terminated against it.
        @variable(
            mod,
            -admissibleCoefficientBound(model.p, i) <=
            ϕ[i = 1:model.p] <=
            admissibleCoefficientBound(model.p, i)
        )
        @variable(
            mod,
            -admissibleCoefficientBound(model.P, k) <=
            Φ[k = 1:model.P] <=
            admissibleCoefficientBound(model.P, k)
        )
    end
    @variable(mod, ϵ[1:T])

    fix.(ϵ[1:lb-1], 0.0)

    # initialization = :free — pre-sample values as free decision variables (Box-Jenkins
    # backcasting / unconditional least squares): the pre-sample residuals and the
    # pre-sample (differenced) endogenous values needed by the AR terms are estimated
    # instead of conditioned at zero, and every differenced observation enters the
    # objective. Pre-sample variables enter only through the recursion, not the objective,
    # and are not counted as hyperparameters.
    #
    # initialization = :innovations — the pre-sample block parameterized BY INNOVATIONS
    # ONLY.
    #
    # `:free` and `:penalized` leave `yback` and `ϵpre` free at the same time, which
    # overparameterizes the initial block: there are more degrees of freedom than the exact
    # quadratic form admits, so the minimum falls below it. On a (1,0,1)(1,0,1)[12] with
    # T = 200 the ratio S / y'Omega^-1 y is 0.925 under `:free` against 1.014 under
    # `:innovations`, with the reference at 1.
    #
    # The correct profiling is by MINIMUM NORM OVER INNOVATIONS: choose e_{-M..0}
    # minimizing ||e||^2 subject to reproducing y, which gives y'(Psi Psi')^-1 y and
    # converges to y'Omega^-1 y — exact to 1e-16 in the reference harness, and within 0.6%
    # already at the `q + s*Q` window the package uses.
    #
    # This matters because the determinant term of the exact likelihood was derived FOR
    # y'Omega^-1 y. Adding it to a different quadratic form breaks the balance between the
    # two, and the imbalance is largest where the two forms diverge most. `:free` is a
    # legitimate CSS objective on its own; it fails only when paired with the exact
    # determinant.
    #
    # Here `yback` stops being a free variable and is GENERATED by the recursion from the
    # pre-sample innovations, with everything before the window set to zero.
    inovacoes = initialization === :innovations
    freeInit = initialization in (:free, :penalized, :innovations)
    penalizado = initialization in (:penalized, :innovations)
    yLo = freeInit ? 1 - (model.p + model.seasonality * model.P) : 1
    epsLo = freeInit ? 1 - (model.q + model.seasonality * model.Q) : 1
    if inovacoes
        # The innovations window covers the AR block too, since `yback` is generated from
        # it. With no AR block (p = P = 0) there is no pre-sample `z` to generate, and
        # creating variables and constraints for it would be pure cost: `:innovations`
        # coincides with `:penalized` by construction in that case.
        if model.p == 0 && model.P == 0
            yLo = 1
        else
            epsLo = min(epsLo, yLo) - presampleBurnIn
            yLo = min(yLo, epsLo + 1)
        end
    end
    if freeInit && epsLo <= 0
        @variable(mod, ϵpre[epsLo:0])
        for t0 = epsLo:0
            set_start_value(ϵpre[t0], 0.0)
        end
        epsAcc = OffsetArrays.OffsetVector(Any[[ϵpre[t0] for t0 = epsLo:0]; ϵ], epsLo - 1)
    else
        epsAcc = OffsetArrays.OffsetVector(Any[ϵ...], 0)
    end

    # Represent missing endogenous values as free variables so the model
    # relates each gap to its neighbours; keeping their residuals in the
    # objective yields the (two-sided) conditional smoother.
    local yData
    if hasMissing
        missIdx = findall(missingMask)
        @variable(mod, ymiss[missIdx])
        startVal = any(.!missingMask) ? Statistics.mean(yValues[.!missingMask]) : 0.0
        for m in missIdx
            set_start_value(ymiss[m], startVal)
        end
        yData = Vector{JuMP.AffExpr}(undef, T)
        for t = 1:T
            yData[t] = missingMask[t] ? one(Fl) * ymiss[t] : convert(JuMP.AffExpr, yValues[t])
        end
    else
        yData = yValues
    end

    # The real scope of missing data (d = D = 0, `mse`/`ml` objectives, no exogenous
    # regressors) is enforced above and is independent of the initialization mode: with
    # `d >= 1` it rejects `:zeroed` exactly as it rejects the others.
    if freeInit && yLo <= 0
        if inovacoes
            yAcc = nothing   # adiado: depende de theta/Theta, criados adiante
        else
            @variable(mod, yback[yLo:0])
            for t0 = yLo:0
                set_start_value(yback[t0], 0.0)
            end
            yAcc = OffsetArrays.OffsetVector(Any[[yback[t0] for t0 = yLo:0]; yData], yLo - 1)
        end
    else
        yAcc = OffsetArrays.OffsetVector(Any[yData...], 0)
    end

    # SEMANTICS OF THE EXOGENOUS BLOCK. Two distinct equations, both standard, and the
    # choice between them is not an implementation detail (Hyndman, "The ARIMAX model
    # muddle", 2010):
    #
    #   :armax               phi(B) Phi(B^s) y_t = X_t'b + theta(B) Theta(B^s) e_t
    #   :regression_errors   y_t = X_t'b + eta_t ,
    #                        phi(B) Phi(B^s) eta_t = theta(B) Theta(B^s) e_t
    #
    # They coincide iff the autoregressive polynomial (regular and seasonal) is unitary and
    # there is no differencing. Outside that class `b` means different things: an impact
    # multiplier conditional on past `y` under :armax, the usual marginal regression effect
    # under :regression_errors. `forecast::Arima(xreg=)` implements the second, so a
    # comparison between the two packages outside that class compares different models —
    # measured at about +-0.40 in delta log(RMSE) on a DGP with T = 180.
    #
    # The implementation is a single substitution: the AR terms operate on
    # eta_t = y_t - X_t'b instead of y_t. The free pre-sample values already represent eta
    # directly (there is no X before the sample starts), so indices <= 0 pass through
    # unchanged. The product b*phi makes the expression bilinear; the CSS objective is
    # already non-convex through the MA terms, so this does not change the problem class.
    arAcc = if inovacoes
        nothing   # adiado junto com yAcc
    elseif nExog == 0 || exogDynamics === :armax
        yAcc
    else
        OffsetArrays.OffsetVector(
            Any[
                t <= 0 ? yAcc[t] :
                yAcc[t] - sum(β[j] * exogValues[t, j] for j = 1:nExog) for t = yLo:T
            ],
            yLo - 1,
        )
    end
    # Free pre-sample values are penalized as parameters in the information criteria: the
    # ULS approximation leaves them unpenalized, which lets high-seasonal-order candidates
    # absorb s*P backcast degrees of freedom for free.
    #
    # Under `:penalized` that charge no longer applies to whoever pays the prior in the
    # OBJECTIVE. The pre-sample innovations are a priori iid N(0, sigma^2), so adding them
    # to the sum of squares IS their prior: shrunk rather than free, and not counted. On
    # the AR side the Levinson form covers the first `p` positions of `yback`; the
    # remaining `s*P` stay free and are still charged. Under `:innovations` there is no
    # free `yback` at all — the past is generated — so there is no pre-sample AR block to
    # penalize and no hyper-parameter to count on that side.
    nYback, nEpsPre = inovacoes ? (0, max(0, 1 - epsLo)) :
                                  (max(0, 1 - yLo), max(0, 1 - epsLo))
    # The credit depends on the AR term having been BUILT, not merely on `:penalized` being
    # requested. The Levinson form needs the reflection coefficients, which exist only under
    # `stationary = true`; with the free parameterization there is no `κAR` and the `p`
    # positions of `yback` remain without a prior. Crediting regardless understates AICc by
    # about 2*min(p, nYback) points and biases selection towards high p.
    penalARAtivo = penalizado && stationary && model.p > 0 && nYback > 0
    nYbackPago = penalARAtivo ? min(model.p, nYback) : 0
    model.metadata["nPresampleFree"] =
        !freeInit ? 0 :
        penalizado ? (nYback - nYbackPago) : (nYback + nEpsPre)

    if MACoefficientsAreModelParameters(objectiveFunction)
        @variable(mod, θ[i = 1:model.q] in Parameter(i))
        @variable(mod, Θ[i = 1:model.Q] in Parameter(i))
    elseif invertible
        # Invertible MA parameterization: the MA coefficients θ (Θ) are generated
        # from bounded reflection coefficients κ (κseasonal) through reflectionToMA.
        # θ/Θ stay registered as variables (so the rest of the code is unchanged)
        # and are linked to the reflection recursion by equality constraints.
        ρ = invertibilityMargin
        @variable(mod, θ[1:model.q])
        @variable(mod, Θ[1:model.Q])
        if model.q > 0
            @variable(mod, -(1 - ρ) <= κ[1:model.q] <= (1 - ρ))
            θκ = reflectionToMA(κ)
            @constraint(mod, [j = 1:model.q], θ[j] == θκ[j])
            for i = 1:model.q
                set_start_value(κ[i], 0.0)
                set_start_value(θ[i], 0.0)
            end
        end
        if model.Q > 0
            @variable(mod, -(1 - ρ) <= κseasonal[1:model.Q] <= (1 - ρ))
            Θκ = reflectionToMA(κseasonal)
            @constraint(mod, [j = 1:model.Q], Θ[j] == Θκ[j])
            for i = 1:model.Q
                set_start_value(κseasonal[i], 0.0)
                set_start_value(Θ[i], 0.0)
            end
        end
    else
        # MA free during optimization, as in R, which converts to the invertible
        # representation only afterwards. The bound is the non-excluding one (see
        # `admissibleCoefficientBound`), not `[-1, 1]`, which from q = 2 on leaves out
        # legitimate estimates: on series 36291 R's ML gives theta = (1.511, 0.794) for an
        # invertible MA(2), unreachable under that box.
        @variable(
            mod,
            -admissibleCoefficientBound(model.q, j) <=
            θ[j = 1:model.q] <=
            admissibleCoefficientBound(model.q, j)
        )
        @variable(
            mod,
            -admissibleCoefficientBound(model.Q, w) <=
            Θ[w = 1:model.Q] <=
            admissibleCoefficientBound(model.Q, w)
        )
        for i = 1:model.q
            set_start_value(mod[:θ][i], 0.0)
        end

        for i = 1:model.Q
            set_start_value(mod[:Θ][i], 0.0)
        end
    end

if inovacoes
        # Pre-sample y GENERATED by the recursion from the free innovations. Indices before
        # the window are zero; the window is long enough for the null initial condition to
        # decay (the effect decays as the largest inverse-root modulus raised to the number
        # of periods).
        #
        # The past is a DECISION VARIABLE tied by an equality constraint, not a nested
        # expression. Chaining `@expression` instead makes JuMP inline the recursion, so
        # z_0 expands into a degree-|W| polynomial in the coefficients: measured at 0.183s
        # against 57.0s on the same M4 series (312x) without adding a single variable. The
        # blow-up is in the EXPRESSION, not the dimension. Variable plus equality keeps the
        # Jacobian sparse and costs +|W| variables and +|W| constraints.
        #
        # The cross terms `phi*Phi*zpre` and `theta*Theta*epsilon` are degree 3, not
        # bilinear. That is acceptable because the in-sample yhat already carries
        # `theta*Theta*epsilon` in every multiplicative fit.
        @variable(mod, zpre[yLo:0])
        for t0 = yLo:0
            set_start_value(zpre[t0], 0.0)
        end
        # The pre-sample generator must satisfy THE SAME equation as the in-sample block,
        # which includes `c` and `trend`. Omitting them leaves `zpre` hovering near zero
        # while `yData` carries the level, so the first AR contributions come out
        # systematically wrong. This is invisible on zero-mean synthetic data, where `c` is
        # fixed at 0.
        # `driftValues` only exists for t = 1..T; for the past it is extrapolated by the
        # same rule: constant when there is differencing, linear when there is not.
        driftPre(t0) =
            length(driftValues) >= 2 ?
            driftValues[1] + (t0 - 1) * (driftValues[2] - driftValues[1]) :
            (isempty(driftValues) ? zero(Fl) : driftValues[1])
        lerY(t) = t < yLo ? zero(AffExpr) : (t <= 0 ? one(Fl) * zpre[t] : yData[t])
        lerE(t) = t < epsLo ? zero(AffExpr) : (t <= 0 ? one(Fl) * mod[:ϵpre][t] : one(Fl) * mod[:ϵ][t])
        for t0 = yLo:0
            @constraint(
                mod,
                zpre[t0] ==
                mod[:c] +
                mod[:trend] * driftPre(t0) +
                lerE(t0) +
                sum(mod[:ϕ][i] * lerY(t0 - i) for i = 1:model.p; init = zero(AffExpr)) +
                sum(mod[:θ][j] * lerE(t0 - j) for j = 1:model.q; init = zero(AffExpr)) +
                sum(mod[:Φ][k] * lerY(t0 - model.seasonality * k) for k = 1:model.P; init = zero(AffExpr)) +
                sum(mod[:Θ][w] * lerE(t0 - model.seasonality * w) for w = 1:model.Q; init = zero(AffExpr)) -
                (model.seasonality > 1 && seasonalForm === :multiplicative ?
                    sum(mod[:ϕ][i] * mod[:Φ][k] * lerY(t0 - i - model.seasonality * k)
                        for i = 1:model.p, k = 1:model.P; init = zero(AffExpr)) : zero(AffExpr)) +
                (model.seasonality > 1 && seasonalForm === :multiplicative ?
                    sum(mod[:θ][j] * mod[:Θ][w] * lerE(t0 - j - model.seasonality * w)
                        for j = 1:model.q, w = 1:model.Q; init = zero(AffExpr)) : zero(AffExpr))
            )
        end
        yAcc = OffsetArrays.OffsetVector(
            Any[[zpre[t0] for t0 = yLo:0]; yData], yLo - 1)
        arAcc = if nExog == 0 || exogDynamics === :armax
            yAcc
        else
            OffsetArrays.OffsetVector(
                Any[
                    t <= 0 ? yAcc[t] :
                    yAcc[t] - sum(β[j] * exogValues[t, j] for j = 1:nExog) for t = yLo:T
                ],
                yLo - 1,
            )
        end
    end

    model.keepProvidedCoefficients && setProvidedCoefficients!(mod, model, yScale)
    includeSolverParameters!(mod, silent; mipSolver = mipSolver, objectiveFunction = objectiveFunction)

    if model.seasonality > 1 && seasonalForm === :multiplicative
        # Box-Jenkins multiplicative SARIMA: the polynomial products yield the
        # cross terms -phi_i*Phi_k*y_(t-i-sk) and +theta_j*Theta_w*eps_(t-j-sw).
        @expression(
            mod,
            ŷ[t = lb:T],
            c +
            trend * driftValues[t] +
            sum(β[i] * exogValues[t, i] for i = 1:nExog) +
            sum(ϕ[i] * arAcc[t-i] for i = 1:model.p if (t - i >= yLo)) +
            sum(θ[j] * epsAcc[t-j] for j = 1:model.q if (t - j >= epsLo)) +
            sum(
                Φ[k] * arAcc[t-(model.seasonality*k)] for
                k = 1:model.P if (t - (model.seasonality * k) >= yLo)
            ) +
            sum(Θ[w] * epsAcc[t-(model.seasonality*w)] for w = 1:model.Q if (t - (model.seasonality * w) >= epsLo)) -
            sum(
                ϕ[i] * Φ[k] * arAcc[t-i-(model.seasonality*k)] for
                i = 1:model.p, k = 1:model.P if (t - i - (model.seasonality * k) >= yLo)
            ) +
            sum(
                θ[j] * Θ[w] * epsAcc[t-j-(model.seasonality*w)] for
                j = 1:model.q, w = 1:model.Q if (t - j - (model.seasonality * w) >= epsLo)
            )
        )
    elseif model.seasonality > 1
        @expression(
            mod,
            ŷ[t = lb:T],
            c +
            trend * driftValues[t] +
            sum(β[i] * exogValues[t, i] for i = 1:nExog) +
            sum(ϕ[i] * arAcc[t-i] for i = 1:model.p if (t - i >= yLo)) +
            sum(θ[j] * epsAcc[t-j] for j = 1:model.q if (t - j >= epsLo)) +
            sum(
                Φ[k] * arAcc[t-(model.seasonality*k)] for
                k = 1:model.P if (t - (model.seasonality * k) >= yLo)
            ) +
            sum(Θ[w] * epsAcc[t-(model.seasonality*w)] for w = 1:model.Q if (t - (model.seasonality * w) >= epsLo))
        )
    else
        @expression(
            mod,
            ŷ[t = lb:T],
            c +
            trend * driftValues[t] +
            sum(β[i] * exogValues[t, i] for i = 1:nExog) +
            sum(ϕ[i] * arAcc[t-i] for i = 1:model.p if (t - i >= yLo)) +
            sum(θ[j] * epsAcc[t-j] for j = 1:model.q if (t - j >= epsLo))
        )
    end

    includeModelConstraints!(mod, yData, T, objectiveFunction, lb)

    objectiveFunctionDefinition!(mod, model, objectiveFunction, T, lb, cvarLevel, yValues,
                                 penalizado, yLo, epsLo; penaltyTarget = penaltyTarget)

    isnothing(warmStart) || applyWarmStart!(
        mod,
        warmStart,
        lb,
        T;
        epsScale = yScale,
        stationary = stationary,
        invertible = invertible,
        stationarityMargin = stationarityMargin,
        invertibilityMargin = invertibilityMargin,
    )

    # Safeguard: cap the solve so a pathological (e.g. numerically hard) instance fails
    # fast instead of blowing the per-series budget. The caller decides the fallback
    # (e.g. use the box warm-start result).
    #
    # The wall-clock limit alone does NOT bound the work: Ipopt only checks it between
    # iterations, and on long series with `initialization = :free` (≈2T variables) a
    # single iteration can already take longer than the cap — measured overshoots of
    # 2x-6x on M4 daily. Bounding the iteration count is the hard limit that actually
    # holds, so cap both.
    if !isnothing(maxTimeSeconds)
        set_time_limit_sec(mod, Float64(maxTimeSeconds))
        if solver_name(mod) == "Ipopt"
            set_optimizer_attribute(mod, "max_iter", MAX_ITER_CAPPED_FIT)
        end
        # The bilevel objective already set a deliberate 1s cap on the INNER solve, and
        # the line above would silently undo it. That matters more than it looks: the
        # outer loop calls the inner solve once per function evaluation, hundreds of
        # times, so handing each of them the whole budget multiplies the cost instead of
        # bounding it. Keep whichever cap is tighter.
        if objectiveFunction == "bilevel"
            set_time_limit_sec(mod, min(1.0, Float64(maxTimeSeconds)))
        end
    end

    # See `fitStartTime`: everything up to here builds the JuMP problem, what follows is the
    # solve. `solveTimeSec` measures the wall clock on the Julia side (including the
    # post-processing in `optimizeModel!`, such as the bilevel outer loop); `solverTimeSec`
    # is the time the solver itself reports.
    buildElapsed = time() - fitStartTime
    solveElapsed = @elapsed optimizeModel!(mod, model, objectiveFunction)
    model.metadata["buildTimeSec"] = buildElapsed
    model.metadata["solveTimeSec"] = solveElapsed
    # ...Total ACCUMULATES per model object, while the fields above hold only the last fit.
    # The distinction matters because the same model is fitted more than once on two paths:
    # `warmStartFromBox` (box solve plus up to two constrained tiers) and `ensureAdmissible!`
    # (refit on top of the candidate). Overwriting would lose the real cost, and at warm-start
    # tier 3 would report precisely the cheapest of the three solves.
    model.metadata["buildTimeSecTotal"] =
        get(model.metadata, "buildTimeSecTotal", 0.0) + buildElapsed
    model.metadata["solveTimeSecTotal"] =
        get(model.metadata, "solveTimeSecTotal", 0.0) + solveElapsed
    model.metadata["fitCount"] = get(model.metadata, "fitCount", 0) + 1
    model.metadata["solverStatus"] = string(termination_status(mod))
    # The objective value is recorded so that a check comparing TWO formulations at the
    # same coefficients has something to compare. Diagnostic only, not a test hook.
    model.metadata["objectiveValue"] = try
        objective_value(mod)
    catch
        nothing
    end
    model.metadata["solverTimeSec"] = try
        MOI.get(mod, MOI.SolveTimeSec())
    catch
        missing
    end
    model.metadata["solverIterations"] = try
        MOI.get(mod, MOI.BarrierIterations())
    catch
        missing
    end
    silent || @info(
        "The model has been fitted with the objective function $objectiveFunction: $(objective_value(mod))"
    )

    fittedValues::Vector{Fl} = Vector(OffsetArrays.no_offset_view(value.(ŷ)))
    # Back to the original units (see `yScale`); must precede the re-integration
    # below, which combines these with the untouched observed series.
    fittedValues .*= yScale
    fittedOriginalLengthDifference = length(values(model.y)) - length(fittedValues)
    initialValuesLength = model.d + model.D * model.seasonality
    initialValuesOffset =
        fittedOriginalLengthDifference > initialValuesLength ?
        fittedOriginalLengthDifference - initialValuesLength + 1 : 1
    originalValues = values(model.y)

    integratedFit = [
        integrate(
            originalValues[initialValuesOffset+i-1:fittedOriginalLengthDifference+i-1],
            [fittedValues[i]],
            model.d,
            model.D,
            model.seasonality,
        )[end] for i = 1:length(fittedValues)
    ]
    lengthIntegratedFit = length(integratedFit)
    fitInSample::TimeArray =
        TimeArray(timestamp(model.y)[end-lengthIntegratedFit+1:end], integratedFit)

    residualMissingMask = hasMissing ? missingMask[lb:end] : nothing
    residualsVariance = computeSARIMAModelVariance(
        mod,
        objectiveFunction,
        get_hyperparameters_number(model),
        lb;
        missingMask = residualMissingMask,
    )

    # Scale-dependent estimates return to the original units; ϕ/θ/Φ/Θ do not, being
    # invariant under a rescaling of the data.
    residualsVariance = residualsVariance * yScale^2
    c = (is_valid(mod, c) ? value(c) : 0.0) * yScale
    trend = (is_valid(mod, trend) ? value(trend) : 0.0) * yScale
    exogCoefficients = isnothing(model.exog) ? nothing : value.(β) .* yScale
    # The complete residual vector (including the smoothed values at missing
    # indices) is stored so the forecast recursion can seed its MA terms; the
    # mask records which of them are not real innovations.
    residuals::Vector{Fl} = value.(ϵ)[lb:end] .* yScale

    if hasMissing
        # `yValues` and the imputed variables both live on the internal scale here,
        # so undo the scaling once, after filling the gaps.
        imputed = copy(yValues)
        for m in missIdx
            imputed[m] = value(ymiss[m])
        end
        imputed .*= yScale
        model.y = TimeArray(timestamp(model.y), imputed, colnames(model.y))
        model.metadata["missingResidualMask"] = residualMissingMask
        model.metadata["nMissing"] = length(missIdx)
        model.metadata["imputedIndices"] = missIdx
    end

    fillFitValues!(
        model,
        c,
        trend,
        value.(ϕ),
        value.(θ),
        residuals,
        residualsVariance,
        fitInSample;
        Φ = value.(Φ),
        Θ = value.(Θ),
        exogCoefficients = exogCoefficients,
    )
end

"""
    MACoefficientsAreModelParameters(objectiveFunction::String)

Determines if the moving average coefficients are treated as model parameters based on the objective function.

# Arguments
- `objectiveFunction::String`: The objective function used.

# Returns
- `Bool`: `true` if the moving average coefficients are treated as model parameters; otherwise, `false`.
"""
function MACoefficientsAreModelParameters(objectiveFunction::String)
    return objectiveFunction == "bilevel"
end

"""
    reflectionToMA(κ)

Maps a vector of reflection coefficients `κ = (κ₁,…,κ_q)` to moving-average
coefficients `θ = (θ₁,…,θ_q)` through the recursion

    θ₁⁽¹⁾ = κ₁
    θ_m⁽ᵐ⁾ = κ_m,                                  m = 2,…,q
    θ_i⁽ᵐ⁾ = θ_i⁽ᵐ⁻¹⁾ + κ_m · θ_{m−i}⁽ᵐ⁻¹⁾,        i = 1,…,m−1
    θ_j    = θ_j⁽q⁾,                                j = 1,…,q

When the reflection coefficients satisfy |κ_m| < 1 the resulting MA polynomial is
invertible. The entries of `κ` may be numbers or JuMP variables/expressions; the
returned vector then contains the corresponding (possibly nonlinear) expressions.
"""
function reflectionToMA(κ)
    q = length(κ)
    q == 0 && return Any[]
    prev = Any[κ[1]]
    for m = 2:q
        cur = Vector{Any}(undef, m)
        cur[m] = κ[m]
        for i = 1:(m-1)
            cur[i] = prev[i] + κ[m] * prev[m-i]
        end
        prev = cur
    end
    return prev
end

"""
    reflectionToARStages(κ)

Every stage of the Levinson-Durbin recursion, not just the last: `stages[m]` holds the
AR(m) coefficients implied by `κ[1:m]`. The exact treatment of the initial observations
needs them, because the one-step prediction of `y_t` for `t <= p` is the AR(t-1) fit —
the full AR(p) is not available yet at that point in the sample.
"""
function reflectionToARStages(κ)
    p = length(κ)
    p == 0 && return Vector{Vector{Any}}()
    est = Vector{Vector{Any}}(undef, p)
    prev = Any[κ[1]]
    est[1] = copy(prev)
    for m = 2:p
        cur = Vector{Any}(undef, m)
        cur[m] = κ[m]
        for i = 1:(m-1)
            cur[i] = prev[i] - κ[m] * prev[m-i]
        end
        prev = cur
        est[m] = copy(cur)
    end
    return est
end



"""
    reflectionToAR(κ)

Maps reflection coefficients (partial autocorrelations) `κ` to AR coefficients
via the Levinson-Durbin recursion (the AR analogue of [`reflectionToMA`](@ref),
with the opposite sign):

    φ_m⁽ᵐ⁾ = κ_m
    φ_i⁽ᵐ⁾ = φ_i⁽ᵐ⁻¹⁾ − κ_m · φ_{m−i}⁽ᵐ⁻¹⁾

When |κ_m| < 1 the resulting AR polynomial is stationary. Entries may be
numbers or JuMP variables/expressions.
"""
function reflectionToAR(κ)
    p = length(κ)
    p == 0 && return Any[]
    prev = Any[κ[1]]
    for m = 2:p
        cur = Vector{Any}(undef, m)
        cur[m] = κ[m]
        for i = 1:(m-1)
            cur[i] = prev[i] - κ[m] * prev[m-i]
        end
        prev = cur
    end
    return prev
end

"""
    arToReflection(ϕ)

Inverse of [`reflectionToAR`](@ref): maps AR coefficients back to reflection
coefficients (partial autocorrelations) via the step-down Levinson-Durbin
recursion. Used to warm-start the stationarity-by-construction fit from an
unconstrained (box) solution. Numeric only.
"""
function arToReflection(ϕ::AbstractVector)
    p = length(ϕ)
    p == 0 && return Float64[]
    a = collect(float.(ϕ))
    κ = zeros(Float64, p)
    for m = p:-1:1
        κ[m] = a[m]
        if m > 1
            d = 1 - κ[m]^2
            abs(d) < 1e-8 && (d = d >= 0 ? 1e-8 : -1e-8)
            aprev = Vector{Float64}(undef, m - 1)
            for i = 1:(m-1)
                aprev[i] = (a[i] + κ[m] * a[m-i]) / d   # AR: opposite sign of MA
            end
            @inbounds a[1:m-1] .= aprev
        end
    end
    return κ
end

"""
    maToReflection(θ)

Inverse of [`reflectionToMA`](@ref): maps MA coefficients back to reflection
coefficients via the step-down recursion (MA sign convention). Numeric only.
"""
function maToReflection(θ::AbstractVector)
    q = length(θ)
    q == 0 && return Float64[]
    a = collect(float.(θ))
    κ = zeros(Float64, q)
    for m = q:-1:1
        κ[m] = a[m]
        if m > 1
            d = 1 - κ[m]^2
            abs(d) < 1e-8 && (d = d >= 0 ? 1e-8 : -1e-8)
            aprev = Vector{Float64}(undef, m - 1)
            for i = 1:(m-1)
                aprev[i] = (a[i] - κ[m] * a[m-i]) / d   # MA sign
            end
            @inbounds a[1:m-1] .= aprev
        end
    end
    return κ
end

"""
    applyWarmStart!(mod, ws, lb, T; stationary, invertible, stationarityMargin, invertibilityMargin)

Seeds the JuMP variables of a SARIMA fit with the solution of a previous
(cheaper) fit `ws`, so Ipopt starts near-feasible. The dominant win is seeding
the O(T) residual vector `ϵ` (making the T identity constraints y = ŷ + ϵ almost
satisfied at iteration 0); coefficients and — under the reflection
parameterization — their reflection preimages (via [`arToReflection`](@ref) /
[`maToReflection`](@ref), clamped to the bound box) are seeded too. Scalar
mean/trend are left at their defaults. No-op for variables absent in this
formulation.
"""
function applyWarmStart!(
    mod::Model,
    ws::SARIMAModel,
    lb::Int,
    T::Int;
    epsScale::Real = 1,
    stationary::Bool,
    invertible::Bool,
    stationarityMargin::AbstractFloat,
    invertibilityMargin::AbstractFloat,
)
    od = JuMP.object_dictionary(mod)
    clampκ(v, margin) = clamp(v, -(1 - margin - 1e-4), (1 - margin - 1e-4))

    if !isnothing(ws.ϵ) && haskey(od, :ϵ)
        # `ws.ϵ` vem em unidades ORIGINAIS — o construtor devolve `value.(ϵ) .* yScale` —
        # enquanto o `ϵ` deste modelo vive na escala padronizada (`yValues ./ yScale`).
        # Semear o vetor cru punha a partida MAIS infactivel que a partida fria: medido em
        # 28/08 pelo `inf_pr` da iteracao 0 do Ipopt, de ~13 para 337-7.720 em series daily.
        # Sem esta divisao o `warmStart` e um chute grande, nao uma semente informada.
        escala = (isfinite(epsScale) && epsScale > 0) ? epsScale : one(epsScale)
        ϵv = mod[:ϵ]
        n = min(length(ws.ϵ), T - lb + 1)
        for k = 1:n
            set_start_value(ϵv[lb-1+k], ws.ϵ[k] / escala)
        end
    end

    seedBlock!(sym, coefs, useRefl, refl, margin, ksym) = begin
        (isnothing(coefs) || isempty(coefs) || !haskey(od, sym)) && return
        for i in eachindex(coefs)
            set_start_value(mod[sym][i], coefs[i])
        end
        if useRefl && haskey(od, ksym)
            κ = refl(coefs)
            for i in eachindex(κ)
                set_start_value(mod[ksym][i], clampκ(κ[i], margin))
            end
        end
    end

    seedBlock!(:ϕ, ws.ϕ, stationary, arToReflection, stationarityMargin, :κAR)
    seedBlock!(:Φ, ws.Φ, stationary, arToReflection, stationarityMargin, :κSAR)
    seedBlock!(:θ, ws.θ, invertible, maToReflection, invertibilityMargin, :κ)
    seedBlock!(:Θ, ws.Θ, invertible, maToReflection, invertibilityMargin, :κseasonal)
    return nothing
end

function getParametersVector(model::SARIMAModel)
    parametersVector::Vector{Symbol} = Vector{Symbol}()
    model.allowMean && push!(parametersVector, :c)
    model.allowDrift && push!(parametersVector, :trend)
    model.p > 0 && push!(parametersVector, :ϕ)
    model.q > 0 && push!(parametersVector, :θ)
    model.P > 0 && push!(parametersVector, :Φ)
    model.Q > 0 && push!(parametersVector, :Θ)
    !isnothing(model.exog) && push!(parametersVector, :β)
    return parametersVector
end

"""
    setProvidedCoefficients!(jumpModel::Model, model::SARIMAModel)

Sets the provided coefficient values from a `SARIMAModel` to the corresponding parameters in a `jumpModel`.

# Arguments
- `jumpModel::Model`: The target model where the coefficients will be set.
- `model::SARIMAModel`: The source model containing the coefficients.

# Description
This function assigns the provided coefficients from the `model` to the corresponding parameters in the `jumpModel` if they are not `nothing`.

# Details
- If `model.c` is not `nothing`, it sets `jumpModel[:c]` to `model.c`.
- If `model.trend` is not `nothing`, it sets `jumpModel[:trend]` to `model.trend`.
- If `model.ϕ` is not `nothing`, it sets `jumpModel[:ϕ]` to `model.ϕ`.
- If `model.θ` is not `nothing`, it sets `jumpModel[:θ]` to `model.θ`.
- If `model.Φ` is not `nothing`, it sets `jumpModel[:Φ]` to `model.Φ`.
- If `model.Θ` is not `nothing`, it sets `jumpModel[:Θ]` to `model.Θ`.
- If `model.exogCoefficients` is not `nothing`, it sets `jumpModel[:β]` to `model.exogCoefficients`.

`yScale` is the factor the endogenous series was divided by before the model was
built (see [`fit!`](@ref)). The provided values are expressed in the user's units, so
the scale-dependent ones (`c`, `trend`, `β`) are converted to the internal units here;
otherwise fixing them would state a different model than the caller asked for, and the
un-scaling applied to the results afterwards would not return the values given.
`ϕ`, `θ`, `Φ` and `Θ` are dimensionless and pass through untouched.
"""
function setProvidedCoefficients!(
    jumpModel::Model,
    model::SARIMAModel,
    yScale::Real = 1.0,
)
    !isnothing(model.c) && fix(jumpModel[:c], model.c / yScale)
    !isnothing(model.trend) && fix(jumpModel[:trend], model.trend / yScale)
    !isnothing(model.ϕ) && fix.(jumpModel[:ϕ], model.ϕ; force = true)
    !isnothing(model.θ) && fix.(jumpModel[:θ], model.θ; force = true)
    !isnothing(model.Φ) && fix.(jumpModel[:Φ], model.Φ; force = true)
    !isnothing(model.Θ) && fix.(jumpModel[:Θ], model.Θ; force = true)
    !isnothing(model.exogCoefficients) &&
        fix.(jumpModel[:β], model.exogCoefficients ./ yScale; force = true)
end

"""
    includeSolverParameters!(model::Model)

Includes solver-specific parameters in the JuMP model.

# Arguments
- `model::Model`: The JuMP model to which solver parameters will be included.

"""
function includeSolverParameters!(
    model::Model,
    isSilent::Bool = true;
    mipSolver::DataType = SCIP.Optimizer,
    objectiveFunction::String = "mse",
)
    isSilent && solver_name(model) != "Alpine" && set_silent(model)
    if solver_name(model) == "Gurobi"
        set_optimizer_attribute(model, "NonConvex", 2)
    elseif solver_name(model) == "Alpine"
        # Alpine relaxes the (bilinear/quadratic) SARIMAX model into a sub-problem that is
        # passed to a MIP sub-solver. SCIP is the default because it can solve the
        # mixed-integer quadratic (MIQP) relaxations produced by quadratic objectives
        # such as "mse". HiGHS only handles the linear (MILP) relaxation of the "mae"
        # objective; warn if it is requested for a quadratic objective.
        if occursin("HiGHS", string(mipSolver)) && objectiveFunction != "mae"
            @warn(
                "The HiGHS MIP sub-solver cannot solve the mixed-integer quadratic (MIQP) " *
                "relaxation that Alpine builds for the '$objectiveFunction' objective. " *
                "Use objectiveFunction=\"mae\" (linear) with HiGHS, or keep the default " *
                "SCIP mip_solver for quadratic objectives such as 'mse'."
            )
        end
        ipopt =
            isSilent ? optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0) :
            optimizer_with_attributes(Ipopt.Optimizer)
        mip =
            (isSilent && occursin("SCIP", string(mipSolver))) ?
            optimizer_with_attributes(mipSolver, "display/verblevel" => 0) :
            optimizer_with_attributes(mipSolver)
        set_optimizer_attribute(model, "nlp_solver", ipopt)
        set_optimizer_attribute(model, "mip_solver", mip)
    elseif solver_name(model) == "Ipopt"
        set_optimizer_attribute(model, "warm_start_init_point", "yes")
    end
end

"""
    includeModelConstraints!(jumpModel::Model, yValues::Fl, T::Int, objectiveFunction::String, offset::Int) where Fl<:AbstractFloat

Includes the constraints in the JuMP model for the SARIMA model.

# Arguments
- `jumpModel::Model`: The JuMP model to which constraints will be included.
- `yValues::Fl`: The values of the time series.
- `T::Int`: The total number of observations.
- `objectiveFunction::String`: The objective function used for optimization.
- `offset::Int`: The offset value.
"""
function includeModelConstraints!(
    jumpModel::Model,
    yValues::AbstractVector,
    T::Int,
    objectiveFunction::String,
    offset::Int,
)
    # Every objective shares the same defining relation, eps = y - yhat. The MAE branch
    # only adds the split into non-negative parts that linearizes the absolute value.
    #
    # It used to omit that relation and pin eps through
    #     eps == eps_plus - eps_minus ;  yhat - y <= eps_plus ;  y - yhat <= eps_minus
    # which at the optimum yields eps = yhat - y, the OPPOSITE sign from every other
    # objective. The objective value stayed correct (eps_plus + eps_minus = |yhat - y|), so
    # the fit converged and nothing looked wrong -- but eps feeds the moving-average
    # recursion: theta was estimated against inverted innovations while `predict!` builds
    # forecasts with the standard convention, so the MA term entered the forecast with the
    # wrong sign. Only q > 0 or Q > 0 models were affected, and residual or sigma^2
    # diagnostics could not surface it because they square eps. Measured directly:
    # corr(eps, y - yhat) was exactly -1 under "mae" and +1 under "mse" and "ridge".
    @constraint(
        jumpModel,
        [t = offset:T],
        yValues[t] == jumpModel[:ŷ][t] + jumpModel[:ϵ][t]
    )
    if objectiveFunction == "mae"
        @variable(jumpModel, ϵ_plus[offset:T] >= 0)
        @variable(jumpModel, ϵ_minus[offset:T] >= 0)
        @constraint(jumpModel, [t = offset:T], jumpModel[:ϵ][t] == ϵ_plus[t] - ϵ_minus[t])
        # Pre-sample block in the SAME loss: |eps_pre| through the same non-negative pair.
        if haskey(object_dictionary(jumpModel), :ϵpre)
            pre = jumpModel[:ϵpre]
            lo = first(axes(pre, 1))
            @variable(jumpModel, ϵpre_plus[lo:0] >= 0)
            @variable(jumpModel, ϵpre_minus[lo:0] >= 0)
            @constraint(jumpModel, [t = lo:0], pre[t] == ϵpre_plus[t] - ϵpre_minus[t])
        end
    end
end

"""
    presampleSquares(jumpModel, penalizado) -> sum of pre-sample squares, or 0

The FIT TERM of every objective in this file measures error over the residuals. Under
`:penalized`/`:innovations` the pre-sample block is made of residuals like any other, only at
index `t <= 0`, so **the fit sum starts at `t = epsLo`, not at `t = lb`**.

This does not touch the regularizing part. The `lambda` of `ridge` penalizes COEFFICIENTS,
the tail structure of `stable` concerns the distribution of the residuals, and `elastic_net`
shrinks COEFFICIENTS through its penalty. None of them changes: what changes is where the fit
error starts being counted.

Returns `0.0` when there is no block (fixed-block modes, or no pre-sample indices), which
makes the call safe in every objective.
"""
function presampleSquares(jumpModel::Model, penalizado::Bool)
    (penalizado && haskey(object_dictionary(jumpModel), :ϵpre)) || return 0.0
    pre = jumpModel[:ϵpre]
    lo = first(axes(pre, 1))
    lo > 0 && return 0.0
    return sum(pre[t]^2 for t = lo:0)
end

"""
    objectiveFunctionDefinition!(
        jumpModel::Model,
        model::SARIMAModel,
        objectiveFunction::String,
        T::Int
    )

Defines the objective function for optimization in the SARIMA model.

# Arguments
- `jumpModel::Model`: The JuMP model.
- `model::SARIMAModel`: The SARIMA model to be optimized.
- `objectiveFunction::String`: The objective function to be defined.
- `T::Int`: The total number of observations.
- `cvarLevel::AbstractFloat`: Confidence level of the `"stable"` (CVaR) objective.

"""
function objectiveFunctionDefinition!(
    jumpModel::Model,
    model::SARIMAModel,
    objectiveFunction::String,
    T::Int,
    lb::Int,
    cvarLevel::AbstractFloat = DEFAULT_CVAR_LEVEL,
    yValues::Union{Nothing,AbstractVector} = nothing,
    penalizado::Bool = false,
    yLo::Int = 1,
    epsLo::Int = 1;
    # KEYWORD: this argument list is positional, so a new positional would shift the
    # slots after it.
    penaltyTarget::Symbol = :all,
)
    parametersVector::Vector{Symbol} = getParametersVector(model)
    parametersVectorExtended::Vector{VariableRef} =
        length(parametersVector) == 0 ? [] :
        reduce(vcat, [Vector{VariableRef}([jumpModel[el]...]) for el in parametersVector])
    if objectiveFunction == "mse" && penalizado
        # `:penalized` — the pre-sample values enter the objective paying THEIR prior,
        # instead of being free variables at no cost as under `:free`.
        #
        #   pre-sample innovations: a priori iid N(0, sigma^2), so the negative log-prior
        #   is eps^2/2sigma^2 — adding them to the sum of squares suffices. Exact, and it
        #   holds for MA and seasonal terms with no new machinery.
        #
        #   yback (AR side): the prior is the quadratic form y'Gamma^-1 y, which in the
        #   reflection-coefficient parameterization has a closed form validated against R's
        #   `arima(method="ML")` (median 3e-5 in phi): sum of e_k^2 * w_k with
        #   w_k = prod_{j=k..p}(1 - kappa_j^2), plus the log-determinant.
        #
        # The log-determinant is not decorative: without it the optimizer would drive
        # kappa -> +-1 to annul w_k and zero the penalty for free. It diverges at the
        # boundary and prevents that — the same barrier as in `ml_exact`.
        #
        # SCOPE: the `yback` block has p + s*P positions and the Levinson form covers the
        # first p. With P > 0 the remaining s*P stay free and are still charged in
        # `nPresampleFree`. Exact for P = 0.
        nEpsPre = max(0, 1 - epsLo)
        nYback = max(0, 1 - yLo)
        temEps = nEpsPre > 0 && haskey(object_dictionary(jumpModel), :ϵpre)
        temK = model.p > 0 && nYback > 0 && haskey(object_dictionary(jumpModel), :κAR) &&
               haskey(object_dictionary(jumpModel), :yback)
        S = sum(jumpModel[:ϵ] .^ 2)
        temEps && (S = S + sum(jumpModel[:ϵpre][t0]^2 for t0 = epsLo:0))
        nEf = T + (temEps ? nEpsPre : 0)
        if temK
            κ = jumpModel[:κAR]
            yb = jumpModel[:yback]
            estagios = reflectionToARStages([κ[i] for i = 1:model.p])
            nb = min(model.p, nYback)
            for k = 1:nb
                idx = yLo + k - 1
                pred = k == 1 ? 0.0 : sum(estagios[k-1][j] * yb[idx-j] for j = 1:(k-1))
                S = S + (yb[idx] - pred)^2 * prod([(1 - κ[j]^2) for j = k:model.p])
            end
            nEf += nb
            # SIMPLIFIED form: the log cancels against the Gaussian exponential.
            #
            #   -2logL = nEf*log(sigma^2) + log|Omega| + S/sigma^2
            # concentrating sigma^2 = S/nEf turns S/sigma^2 into the constant nEf, leaving
            #   -2logL = nEf*log(S) + log|Omega| + const
            # with log|Omega| = -sum_j j*log(1 - kappa_j^2). Since
            #   nEf*log(S) + log|Omega| = nEf*log( S * prod_j (1-kappa_j^2)^(-j/nEf) )
            # and nEf*log(.) is monotone increasing, the argmin is that of
            #   S * prod_j (1 - kappa_j^2)^(-j/nEf)
            # i.e. the sum of squares (with the initial terms weighted) times a scalar.
            #
            # This drops no term: the weights w_k and the determinant factor come from the
            # covariance |Omega|, not from the exponential, and both remain. Only the `log`
            # goes, and with it the auxiliary variable that existed solely to avoid log(0)
            # at the starting point, where all variables start at zero.
            #
            # The exponent divides by the dimension of the DENSITY (`T` observations), not
            # by the number of squared terms `S` decomposes into: profiling pre-sample
            # positions does not increase the dimension of the data. Verified against the
            # constructed theoretical covariance — with `1/T` the argmin reproduces that of
            # the exact likelihood, with `1/nEf` it does not.
            fator = prod([(1 - κ[j]^2)^(-j / T) for j = 1:model.p])
        else
            # With no AR block there is no determinant on that side, but `fator` must still
            # exist: the MA block below multiplies it, and `@objective` is emitted ONCE at
            # the end. Emitting the objective inside this branch would leave the MA term out
            # of it.
            fator = 1.0
        end
        # ---- MA block: the same determinant term the AR block already carries ----
        #
        # Adding the pre-sample innovations to the sum of squares is not enough on its own:
        # PROFILING (minimizing) over the pre-sample values is not the same as INTEGRATING
        # them out, and the difference is exactly the log-determinant. Without it the fit
        # is not the maximum-likelihood point — measured against R's `arima(method="ML")`,
        # a pure MA(2) is off by ~2e-3 in the coefficients, against ~2e-5 for a pure AR(2),
        # which does carry the term.
        #
        # The form matches the AR side, with the MA reflection coefficients:
        #     log|Omega| = -sum_j j*log(1 - kappa_j^2)
        # Verified against the determinant of a directly constructed MA(q) theoretical
        # covariance: error ~1e-14 at theta = [0.5], [0.7], [0.4,0.2], [0.6,-0.3],
        # [0.5,0.3,0.2]. The seasonal MA block gains the multiplicity `s`, by the same
        # factorization into phase chains.
        #
        # The MA reflection coefficients do NOT require the invertible parameterization:
        # the inverse Levinson-Durbin recursion produces them from `theta` itself, so the
        # determinant term holds with `theta` FREE. It also makes the hard constraint
        # redundant — `-sum_j j*log(1 - kappa_j^2)` diverges as |kappa| -> 1, so the
        # determinant is itself the invertibility barrier, repelling the boundary without
        # excluding any point by decree.
        if model.q > 0
            κMA = maToReflectionExpr(jumpModel[:θ], model.q)
            isnothing(κMA) ||
                (fator = fator * prod([(1 - κMA[j]^2)^(-j / T) for j = 1:model.q]))
        end
        if model.Q > 0
            κSMA = maToReflectionExpr(jumpModel[:Θ], model.Q)
            isnothing(κSMA) || (fator =
                fator *
                prod([(1 - κSMA[j]^2)^(-model.seasonality * j / T) for j = 1:model.Q]))
        end
        @objective(jumpModel, Min, S * fator)

    elseif objectiveFunction == "mse"
        @objective(jumpModel, Min, sum(jumpModel[:ϵ] .^ 2))
    elseif objectiveFunction == "ml_exact"
        # Likelihood that is exact in its TREATMENT OF THE INITIAL OBSERVATIONS, written in
        # closed form rather than obtained from a Kalman filter.
        #
        # CSS discards the first `lb-1` observations; the exact likelihood uses them,
        # weighted by the inverse conditional variance. For a stationary AR(p) those
        # quantities are a closed function of the partial autocorrelations, which this
        # package already carries as decision variables when `stationary = true`. With
        #     v_t = Var(y_t | y_1..y_{t-1}) / sigma^2 = 1 / prod_{j=t..p} (1 - kappa_j^2)
        # the weight is the INVERSE,
        #     w_t = prod_{j=t..p} (1 - kappa_j^2),
        # a POLYNOMIAL in the kappa: no matrix is assembled, nothing is inverted, and
        # `y_t - yhat_t` uses the AR(t-1) fit from the Levinson stages.
        #
        # Concentrating sigma^2 analytically, the objective is `n*log(S) + log|Gamma*|`
        # with log|Gamma*| = -sum_j j*log(1 - kappa_j^2). That term diverges to +infinity
        # as kappa -> +-1, acting as a BARRIER that repels the non-stationarity boundary,
        # which suits an interior-point solver.
        #
        # SCOPE: exact for P = q = Q = 0. With MA or seasonal terms the joint initial block
        # does not factor this way, and what applies is a partial correction (the
        # non-seasonal autoregressive part only); the remaining residuals stay in CSS.
        temK = haskey(object_dictionary(jumpModel), :κAR) && model.p > 0
        # Without `κAR` (that is, `stationary = false`) or without an AR part (`p = 0`) the
        # initial-value correction cannot be written and the objective degenerates to pure
        # CSS: the caller asked for the exact likelihood and gets the conditioned `mse`.
        # Verified: an AR(2) with `stationary = false` produces coefficients identical to
        # `mse`. Warn only on TOTAL degradation; the partial coverage under ARMA/seasonal
        # terms is declared scope in the documentation above, not a surprise.
        #
        # This is also why `stationary = false` cannot become a search default without
        # `ml_exact` silently becoming CSS.
        if !temK || isnothing(yValues)
            @warn "objectiveFunction = \"ml_exact\" degrades to plain CSS here: it needs " *
                  "the reflection parameterization (`stationary = true`) and a non-seasonal " *
                  "AR part (`p > 0`). Got " *
                  "stationary=$(haskey(object_dictionary(jumpModel), :κAR)), p=$(model.p)." maxlog = 1
        end
        if !temK || isnothing(yValues)
            @objective(jumpModel, Min,
                sum(jumpModel[:ϵ][t]^2 for t = lb:T) + presampleSquares(jumpModel, penalizado))
        else
            κ = jumpModel[:κAR]
            pAR = model.p
            estagios = reflectionToARStages([κ[i] for i = 1:pAR])
            nIni = min(pAR, lb - 1, length(yValues))
            termos = Any[]
            for t = 1:nIni
                # forecast of y_t from the AR(t-1) fit; at t = 1 there is no regressor
                pred = t == 1 ? 0.0 :
                       sum(estagios[t-1][j] * yValues[t-j] for j = 1:(t-1))
                e_t = yValues[t] - pred
                w_t = prod([(1 - κ[j]^2) for j = t:pAR])
                push!(termos, e_t^2 * w_t)
            end
            S = sum(jumpModel[:ϵ][t]^2 for t = lb:T) + presampleSquares(jumpModel, penalizado)
            isempty(termos) || (S = S + sum(termos))
            nEf = (T - lb + 1) + nIni
            logDet = -sum(j * log(1 - κ[j]^2) for j = 1:pAR)
            @objective(jumpModel, Min, nEf * log(S) + logDet)
        end
    elseif objectiveFunction == "huber"
        # Huber: quadratic near zero, linear in the tail.
        #     L(e) = e^2/2                 if |e| <= delta
        #          = delta*(|e| - delta/2) otherwise
        # The classical M-estimator: it keeps the efficiency of "mse" under Gaussian errors
        # and bounds the INFLUENCE of an outlier. Note the contrast with "stable" (CVaR),
        # which minimizes the mean of the tail and therefore CHASES the outlier instead of
        # discounting it — opposite objectives, despite both looking at the tail.
        #
        # Infimal-convolution formulation, which keeps it in the same quadratic regime as
        # "mse" (cheaper than "stable", which adds T variables AND T constraints):
        #     Huber(e) = min { u^2/2 + delta*|v| : u + v = e }
        # with |v| split into non-negative parts. The relation eps = y - yhat is already
        # imposed in includeModelConstraints! and is NOT restated here.
        δh = DEFAULT_HUBER_DELTA
        # Leaving `u` free is deliberate. At the optimum of
        # min{u^2/2 + delta*|v| : u + v = e} one always has |u| <= delta, so bounding it
        # looks exact and free — and measured on the M4 monthly it made things WORSE:
        # LOCALLY_INFEASIBLE went from 2 to 3 of the six diagnosed series and one that had
        # solved cleanly started returning forecasts of -6.5e5. Adding 2*(T-lb+1) active
        # box constraints costs Ipopt more than the degenerate direction it removes.
        @variable(jumpModel, uH[lb:T])
        @variable(jumpModel, vH_plus[lb:T] >= 0)
        @variable(jumpModel, vH_minus[lb:T] >= 0)
        @constraint(
            jumpModel,
            [t = lb:T],
            jumpModel[:ϵ][t] == uH[t] + vH_plus[t] - vH_minus[t]
        )
        # Under `:penalized`/`:innovations` the pre-sample block enters THE SAME LOSS: each
        # `eps_pre` gets its own Huber decomposition. Same rationale as `mae` — a single
        # loss applied to every residual — and likewise WITHOUT the determinant factor,
        # which is Gaussian algebra.
        hubObj = sum(0.5 * uH[t]^2 + δh * (vH_plus[t] + vH_minus[t]) for t = lb:T)
        if penalizado && haskey(object_dictionary(jumpModel), :ϵpre)
            preH = jumpModel[:ϵpre]
            loH = first(axes(preH, 1))
            @variable(jumpModel, uHpre[loH:0])
            @variable(jumpModel, vHpre_plus[loH:0] >= 0)
            @variable(jumpModel, vHpre_minus[loH:0] >= 0)
            @constraint(
                jumpModel,
                [t = loH:0],
                preH[t] == uHpre[t] + vHpre_plus[t] - vHpre_minus[t]
            )
            hubObj = hubObj +
                     sum(0.5 * uHpre[t]^2 + δh * (vHpre_plus[t] + vHpre_minus[t])
                         for t = loH:0)
        end
        @objective(jumpModel, Min, hubObj)
    elseif objectiveFunction == "ridge"
        # Ridge differs from "elastic_net" only in that lambda is fixed a priori here and
        # the penalty is a plain L2 term over the AR/MA blocks; `elastic_net` honours a
        # caller-supplied lambda and mixes L1 and L2 through alpha.
        #
        # SCALE OF LAMBDA. The heuristic is lambda = 1/sqrt(n) when the loss is the MEAN
        # squared error. This objective is written as a SUM, so the equivalent value is
        # lambda = sqrt(n) — multiplying (1/n)*RSS + l*||b||^2 by n gives RSS + n*l*||b||^2
        # and n/sqrt(n) = sqrt(n). Using 1/sqrt(n) on a sum-form objective would make the
        # penalty about n times too weak, i.e. no regularization at all.
        #
        # n IS THE EFFECTIVE SAMPLE, not the series length: observations before `lb` are
        # conditioned out and contribute nothing. Under `auto` with :zeroed, `lb` carries
        # the common conditioning length (29 by default at s = 12), so on a short series
        # the effective sample is far smaller than T.
        #
        # WHAT IS PENALIZED. Only the AR/MA coefficients: they are dimensionless and the
        # endogenous data is standardized internally, so the L2 term is scale-free without
        # further rescaling. The intercept and drift are deliberately left out (penalizing
        # the level has no shrinkage interpretation here). Exogenous coefficients are also
        # left out: they carry the units of their own regressor, which this package does
        # NOT standardize, so an L2 term over them would not be scale-invariant.
        nEff = T - lb + 1
        λ = sqrt(max(nEff, 1))
        shrunk = Symbol[]
        model.p > 0 && push!(shrunk, :ϕ)
        model.q > 0 && push!(shrunk, :θ)
        model.P > 0 && push!(shrunk, :Φ)
        model.Q > 0 && push!(shrunk, :Θ)
        if isempty(shrunk)
            @objective(jumpModel, Min,
                sum(jumpModel[:ϵ] .^ 2) + presampleSquares(jumpModel, penalizado))
        else
            coefs = reduce(vcat, [Vector{VariableRef}([jumpModel[el]...]) for el in shrunk])
            @objective(
                jumpModel,
                Min,
                sum(jumpModel[:ϵ] .^ 2) + presampleSquares(jumpModel, penalizado) +
                λ * sum(coefs .^ 2)
            )
        end
    elseif objectiveFunction == "mae"
        # Under `:penalized`/`:innovations` the pre-sample block enters THE SAME LOSS as the
        # data. That is what makes the extension unambiguous: there are no two losses to
        # mix scales between, only one applied to every residual from the pre-sample block
        # onwards.
        #
        # Deliberately WITHOUT the determinant factor. The `mse` factor comes from
        # concentrating sigma^2 in `T*log(S) + log|Omega|`, a step that requires a
        # quadratic loss. Here the mode moves only along the pre-sample axis.
        maeObj = sum(jumpModel[:ϵ_plus] + jumpModel[:ϵ_minus])
        if penalizado && haskey(object_dictionary(jumpModel), :ϵpre_plus)
            maeObj = maeObj +
                     sum(jumpModel[:ϵpre_plus] + jumpModel[:ϵpre_minus])
        end
        @objective(jumpModel, Min, maeObj)
    elseif objectiveFunction == "bilevel"
        # `bilevel` is not a different loss but a SOLUTION STRATEGY, with the MA
        # coefficients taken out of JuMP and optimized in an outer loop. The inner problem
        # is a sum of squares, so the extension holds by inheritance.
        @objective(
            jumpModel,
            Min,
            sum(jumpModel[:ϵ] .^ 2) + presampleSquares(jumpModel, penalizado)
        )
        set_time_limit_sec(jumpModel, 1.0)
    elseif objectiveFunction == "stable"
        # Conditional Value-at-Risk of the squared residuals, in the Rockafellar-Uryasev
        # form: CVaR_α = min_δ  δ + 1/((1-α)·n) · Σ max(ℓ_t - δ, 0), with δ standing for
        # the Value-at-Risk and u_t for the excess above it. The 1/((1-α)·n) factor is
        # what pins the level: without it the objective `0.7·δ + Σu` normalises to
        # `δ + Σu/0.7`, i.e. (1-α)·n = 0.7, so the effective level is α = 1 - 0.7/n —
        # about 99.7% for a 200-point series, and *different for every sample size*.
        # That is a min-max fit in disguise, which is why it chases outliers instead of
        # tolerating them (measured: with one 10σ outlier it is the only objective that
        # shrinks that residual, at the cost of a 5x worse median residual).
        #
        # The pre-sample block joins the SAME set of losses: those are residuals like the
        # others, only at index t <= 0. This does not change the tail structure of the
        # CVaR — it changes where the fit error starts being counted, and `nObs` follows.
        temPre = penalizado && haskey(object_dictionary(jumpModel), :ϵpre)
        preLo = temPre ? first(axes(jumpModel[:ϵpre], 1)) : 1
        temPre = temPre && preLo <= 0
        nPre = temPre ? (1 - preLo) : 0
        nObs = (T - lb + 1) + nPre
        @variable(jumpModel, δ >= 0)
        @variable(jumpModel, u[lb:T] >= 0)
        @constraint(jumpModel, [t = lb:T], δ + u[t] >= jumpModel[:ϵ][t]^2)
        cvarSum = sum(u[t] for t = lb:T)
        if temPre
            pre = jumpModel[:ϵpre]
            @variable(jumpModel, upre[preLo:0] >= 0)
            @constraint(jumpModel, [t = preLo:0], δ + upre[t] >= pre[t]^2)
            cvarSum = cvarSum + sum(upre[t] for t = preLo:0)
        end
        @objective(
            jumpModel,
            Min,
            δ + cvarSum / ((1 - cvarLevel) * nObs)
        )
    elseif objectiveFunction == "elastic_net"
        # Penalized elastic net: the selected residual loss plus
        # lambda * [alpha * L1 + (1 - alpha)/2 * L2] over the shrunk coefficient blocks.
        # The L1 term is linearized with the usual auxiliary variables.
        #
        # The intercept and the drift are always left out: penalizing the level has no
        # shrinkage interpretation here. `penaltyTarget` selects which of the remaining
        # blocks are shrunk -- `:all` (default), `:dynamics` for the AR/MA blocks only, or
        # `:exogenous` for the regressors only, which is the usual choice when the point of
        # the fit is selecting among regressors. Exogenous coefficients carry the units of
        # their own regressor, which the package does not standardize, so scale-comparable
        # regressors are the caller's responsibility.
        shrunk = Symbol[]
        if penaltyTarget in (:all, :dynamics)
            model.p > 0 && push!(shrunk, :ϕ)
            model.q > 0 && push!(shrunk, :θ)
            model.P > 0 && push!(shrunk, :Φ)
            model.Q > 0 && push!(shrunk, :Θ)
        end
        if penaltyTarget in (:all, :exogenous)
            isnothing(model.exog) || push!(shrunk, :β)
        end

        fitLoss = sum(jumpModel[:ϵ] .^ 2) + presampleSquares(jumpModel, penalizado)

        if isempty(shrunk)
            @objective(jumpModel, Min, fitLoss)
        else
            shrunkCoefs =
                reduce(vcat, [Vector{VariableRef}([jumpModel[el]...]) for el in shrunk])
            # Same lambda scale as the "ridge" objective: the 1/sqrt(n) heuristic is
            # stated for a MEAN squared loss, and this objective is a SUM, so the
            # equivalent default is sqrt(n) over the effective sample.
            nEff = T - lb + 1
            λ = isnothing(model.lambda) ? sqrt(max(nEff, 1)) : model.lambda
            α = isnothing(model.alpha) ? 0.5 : model.alpha
            @variable(jumpModel, absShrunk[i = 1:length(shrunkCoefs)] >= 0)
            @constraints(
                jumpModel,
                begin
                    [i = 1:length(shrunkCoefs)], absShrunk[i] >= shrunkCoefs[i]
                    [i = 1:length(shrunkCoefs)], absShrunk[i] >= -shrunkCoefs[i]
                end
            )
            @objective(
                jumpModel,
                Min,
                fitLoss +
                λ * (α * sum(absShrunk) + (1 - α) / 2 * sum(shrunkCoefs .^ 2))
            )
        end
    elseif objectiveFunction == "ml"
        # Concentrated conditional (CSS) Gaussian likelihood: profiling sigma out
        # analytically (sigma2 = RSS/n) makes maximizing the likelihood equivalent
        # to minimizing the sum of squared residuals over the effective sample.
        # Keeping sigma as a decision variable is degenerate when RSS -> 0
        # (noise-free data drives sigma -> 0 and the objective to +Inf).
        @objective(
            jumpModel,
            Min,
            sum(jumpModel[:ϵ][t]^2 for t = lb:T) + presampleSquares(jumpModel, penalizado)
        )
    end
end

"""
    checkSolverStatus(jumpModel::Model)

Checks the termination status of a solved JuMP model and warns if it does not indicate success.
"""
function checkSolverStatus(jumpModel::Model)
    st = termination_status(jumpModel)
    ok = st in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
    ok || @warn("Solver finished with a non-success termination status; estimates may be unreliable.", status = st)
    return st
end

"""
    optimizeModel!(jumpModel::Model, model::SARIMAModel, objectiveFunction::String)

Optimizes the SARIMA model using the specified objective function.

# Arguments
- `jumpModel::Model`: The JuMP model to be optimized.
- `model::SARIMAModel`: The SARIMA model to be optimized.
- `objectiveFunction::String`: The objective function used for optimization.

"""
function optimizeModel!(jumpModel::Model, model::SARIMAModel, objectiveFunction::String)
    JuMP.optimize!(jumpModel)
    checkSolverStatus(jumpModel)

    if objectiveFunction == "bilevel"

        function optimizeMA(coefficients)
            maCoefficients = coefficients[1:model.q]
            smaCoefficients = coefficients[model.q+1:end]
            set_parameter_value.(jumpModel[:θ], maCoefficients)
            set_parameter_value.(jumpModel[:Θ], smaCoefficients)
            JuMP.optimize!(jumpModel)
            return objective_value(jumpModel)
        end

        if model.q + model.Q > 0
            ma_lower_bound = -1 .* ones(model.q + model.Q)
            ma_upper_bound = ones(model.q + model.Q)
            initialCoefficients = zeros(model.q + model.Q)# vcat(parameter_value.(θ),parameter_value.(Θ))#
            # Fminbox's default inner solver (LBFGS + Hager-Zhang line search) can
            # throw an AssertionError from LineSearches on nearly-flat objective
            # surfaces (a known LineSearches.jl edge case), rather than returning
            # a non-converged result. That exception must be caught here too -
            # not just non-convergence - or it escapes uncaught and crashes fit!
            # entirely, before ever reaching the NelderMead fallback below.
            results = try
                Optim.optimize(optimizeMA, ma_lower_bound, ma_upper_bound, initialCoefficients)
            catch e
                @warn "The gradient-based optimizer failed" exception = e
                nothing
            end
            if isnothing(results) || !Optim.converged(results)
                isnothing(results) || @warn("The optimization did not converge")
                @warn("Trying another method")
                results =
                    Optim.optimize(optimizeMA, initialCoefficients, Optim.NelderMead())
                Optim.converged(results) || @warn("The optimization did not converge")
            end

            # Put the outer minimizer back into the model and re-solve. Without this the
            # JuMP model is left holding whatever `optimizeMA` probed LAST, which is not
            # the minimizer: a line search ends wherever it stopped, and Nelder-Mead's
            # final evaluation can be a reflection point that was rejected. Everything
            # downstream reads the coefficients off this model, so the fit returned was
            # not the solution of the problem the outer loop had just solved.
            if !isnothing(results)
                best = Optim.minimizer(results)
                model.q > 0 && set_parameter_value.(jumpModel[:θ], best[1:model.q])
                model.Q > 0 && set_parameter_value.(jumpModel[:Θ], best[model.q+1:end])
                JuMP.optimize!(jumpModel)
                checkSolverStatus(jumpModel)
            end
        end
    end
end

"""
    computeSARIMAModelVariance(model::Model, lb::Int, objectiveFunction::String, nParameters::Int, offset::Int)

Computes the variance of the SARIMA model's errors.

# Arguments
- `model::Model`: The SARIMA model.
- `objectiveFunction::String`: The objective function used for fitting the model.
- `nParameters::Int`: The number of parameters in the model.
- `offset::Int`: The offset value.

# Returns
- `AbstractFloat`: The computed variance.

"""
function computeSARIMAModelVariance(
    model::JuMP.Model,
    objectiveFunction::String,
    nParameters::Int,
    offset::Int;
    missingMask::Union{Nothing,AbstractVector{Bool}} = nothing,
)
    resid = value.(model[:ϵ])[offset:end]
    # Interpolated residuals at missing indices are not real innovations and
    # must not enter σ², the log-likelihood, or the effective sample size.
    isnothing(missingMask) || (resid = resid[.!missingMask])
    nstar = length(resid)
    rss = sum(resid .^ 2)
    if objectiveFunction == "ml"
        # ML convention: sigma2 = RSS / n (sigma was concentrated out)
        return rss / nstar
    end
    return rss / (nstar - nParameters + 1)
end


"""
    completeCoefficientsVector(model::SARIMAModel)

Complete the coefficient vectors for AR and MA parts of a SARIMA model.

# Arguments
- `model::SARIMAModel`: The SARIMA model containing the AR and MA coefficients, seasonal orders, and other model parameters.

# Returns
- `arCoefficients`: A vector of AR coefficients, extended to include seasonal AR coefficients.
- `maCoefficients`: A vector of MA coefficients, extended to include seasonal MA coefficients.

The function handles the seasonal components by zero-padding the coefficient vectors and placing the seasonal coefficients at the appropriate positions.
"""
function completeCoefficientsVector(model::SARIMAModel)
    ModelFl = typeofModelElements(model)
    s = model.seasonality
    phiV = isnothing(model.ϕ) ? ModelFl[] : model.ϕ
    thetaV = isnothing(model.θ) ? ModelFl[] : model.θ
    PhiV = isnothing(model.Φ) ? ModelFl[] : model.Φ
    ThetaV = isnothing(model.Θ) ? ModelFl[] : model.Θ

    if modelSeasonalForm(model) === :multiplicative
        # Full polynomials via the products phi(B)*Phi(B^s) and theta(B)*Theta(B^s),
        # including the cross-lag coefficients.
        arNS = vcat(one(ModelFl), -phiV)
        arS = zeros(ModelFl, length(PhiV) * s + 1)
        arS[1] = one(ModelFl)
        for k = 1:length(PhiV)
            arS[k*s+1] = -PhiV[k]
        end
        maNS = vcat(one(ModelFl), thetaV)
        maS = zeros(ModelFl, length(ThetaV) * s + 1)
        maS[1] = one(ModelFl)
        for w = 1:length(ThetaV)
            maS[w*s+1] = ThetaV[w]
        end
        arCoefficients = -polynomialMultiplication(arNS, arS)[2:end]
        maCoefficients = polynomialMultiplication(maNS, maS)[2:end]
        return arCoefficients, maCoefficients
    end

    # Additive form: zero-padded merge without cross terms (pre-v0.3 behavior).
    (model.p >= s && model.P > 0 || model.q >= s && model.Q > 0) && @warn(
        "Additive seasonal form with p >= s (or q >= s): seasonal coefficients " *
        "overwrite non-seasonal ones at the colliding lags."
    )
    maCoefficients = thetaV
    if model.Q > 0
        sizeMA = (model.Q * s > model.q) ? model.Q * s : model.q
        maCoefficients = zeros(ModelFl, sizeMA)
        maCoefficients[1:model.q] = thetaV
        for i = 1:model.Q
            maCoefficients[s*i] = ThetaV[i]
        end
    end

    arCoefficients = phiV
    if model.P > 0
        arCoefficients = zeros(ModelFl, model.P * s)
        arCoefficients[1:model.p] = phiV
        for i = 1:model.P
            arCoefficients[s*i] = PhiV[i]
        end
    end

    return arCoefficients, maCoefficients
end

"""
    to_ma(model::SARIMAModel, maxLags::Int=12)

    Convert a SARIMA model to a Moving Average (MA) model.

    # Arguments
    - `model::SARIMAModel`: The SARIMA model to convert.
    - `maxLags::Int=12`: The maximum number of lags to include in the MA model.

    # Returns
    - `MAmodel::MAModel`: The coefficients of the lagged errors in the MA model.

    # References
    - Brockwell, P. J., & Davis, R. A. Time Series: Theory and Methods (page 92). Springer(2009)
"""
function to_ma(model::SARIMAModel, maxLags::Int = 12)
    arCoefficients, maCoefficients = completeCoefficientsVector(model)
    p = isnothing(arCoefficients) ? 0 : length(arCoefficients)
    q = isnothing(maCoefficients) ? 0 : length(maCoefficients)
    ψ = zeros(maxLags)

    for i = 1:maxLags
        tmp = (i <= q) ? maCoefficients[i] : 0.0
        for j = 1:min(i, p)
            tmp += arCoefficients[j] * ((i - j > 0) ? ψ[i-j] : 1.0)
        end
        ψ[i] = tmp
    end
    return ψ
end


"""
    polynomialMultiplication(a::Vector{Fl}, b::Vector{Fl}) where Fl<:AbstractFloat

Multiplies two polynomials given by their coefficient vectors (constant term first).
"""
function polynomialMultiplication(a::Vector{Fl}, b::Vector{Fl}) where {Fl<:AbstractFloat}
    c = zeros(Fl, length(a) + length(b) - 1)
    for i in eachindex(a), j in eachindex(b)
        c[i+j-1] += a[i] * b[j]
    end
    return c
end

"""
    psiWeights(ar::Vector{Fl}, ma::Vector{Fl}, maxLags::Int) where Fl<:AbstractFloat

ψ-weights (MA(∞) representation) of an ARMA process with AR coefficients `ar`
and MA coefficients `ma`, via the standard recursion (Brockwell & Davis, 2009).
"""
function psiWeights(ar::Vector{Fl}, ma::Vector{Fl}, maxLags::Int) where {Fl<:AbstractFloat}
    p = length(ar)
    q = length(ma)
    ψ = zeros(Fl, maxLags)
    for i = 1:maxLags
        tmp = (i <= q) ? ma[i] : zero(Fl)
        for j = 1:min(i, p)
            tmp += ar[j] * ((i - j > 0) ? ψ[i-j] : one(Fl))
        end
        ψ[i] = tmp
    end
    return ψ
end

"""
    forecastErrors(model::SARIMAModel, maxLags::Int=12)

Computes the h-step-ahead forecast VARIANCES on the original (integrated)
scale: the ψ-weights are derived from the AR polynomial composed with the
differencing operator (1-B)^d (1-B^s)^D, so the uncertainty accumulated by
re-integration is propagated (e.g. ARIMA(0,1,0) yields σ²·h).

# References
- Brockwell, P. J., & Davis, R. A. Time Series: Theory and Methods (page 92). Springer(2009)
"""
function forecastErrors(model::SARIMAModel, maxLags::Int = 12)
    Fl = typeofModelElements(model)
    ar, ma = completeCoefficientsVector(model)
    arVec::Vector{Fl} = isnothing(ar) ? Fl[] : ar
    maVec::Vector{Fl} = isnothing(ma) ? Fl[] : ma
    # The forecast is reported on the ORIGINAL (integrated) scale, so the
    # ψ-weights must come from ϕ*(B) = ϕ(B)·(1-B)^d·(1-B^s)^D — the AR
    # polynomial composed with the differencing operator. Without this, the
    # variance of an integrated series (e.g. σ²·h for ARIMA(0,1,0)) is lost.
    arPoly = vcat(one(Fl), -arVec)
    diffPoly = differentiated_coefficients(model.d, model.D, model.seasonality, Fl)
    fullPoly = polynomialMultiplication(arPoly, diffPoly)
    φ = -fullPoly[2:end]
    ψ = psiWeights(φ, maVec, maxLags)
    computedForecastErrors = zeros(Fl, maxLags)
    computedForecastErrors[1] = model.σ²
    for lag = 2:maxLags
        computedForecastErrors[lag] = model.σ² * (1 + sum(abs2, ψ[1:lag-1]))
    end
    return computedForecastErrors
end

"""
    predict!(
        model::SARIMAModel;
        stepsAhead::Int = 1
        seed::Int = 1234,
        isSimulation::Bool = false,
        displayConfidenceIntervals::Bool = false,
        confidenceLevel::Fl = 0.95
        automaticExogDifferentiation::Bool=false
    ) where Fl<:AbstractFloat

Predicts the SARIMA model for the next `stepsAhead` periods.
The resulting forecast is stored within the model in the `forecast` field.

# Arguments
- `model::SARIMAModel`: The SARIMA model to make predictions.
- `stepsAhead::Int`: The number of periods ahead to forecast (default: 1).
- `seed::Int`: Seed for random number generation when simulating forecasts (default: 1234).
- `isSimulation::Bool`: Whether to perform a simulation-based forecast (default: false).
- `displayConfidenceIntervals::Bool`: Whether to display confidence intervals (default: false).
- `confidenceLevel::Fl`: The confidence level for the confidence intervals (default: 0.95).
- `automaticExogDifferentiation::Bool`: Whether to automatically differentiate the exogenous variables. Default is `false`.

# Example
```julia
julia> airPassengers = load_dataset(AIR_PASSENGERS)

julia> model = SARIMA(airPassengers, 0, 1, 1; seasonality=12, P=0, D=1, Q=1)

julia> fit!(model)

julia> predict!(model; stepsAhead=12)
"""
function predict!(
    model::SARIMAModel;
    stepsAhead::Int = 1,
    seed::Int = 1234,
    isSimulation::Bool = false,
    displayConfidenceIntervals::Bool = false,
    confidenceLevel::Fl = 0.95,
    automaticExogDifferentiation::Bool = false,
) where {Fl<:AbstractFloat}
    ModelFl = typeofModelElements(model)
    Random.seed!(seed)
    forecastValues::Vector{ModelFl} =
        predict(model, stepsAhead, isSimulation, automaticExogDifferentiation)
    forecastTimestamps::Vector{TimeType} = build_datetimes(
        timestamp(model.y)[end],
        getproperty(Dates, model.metadata["granularity"])(model.metadata["frequency"]),
        model.metadata["weekDaysOnly"],
        stepsAhead,
    )
    if displayConfidenceIntervals
        α::ModelFl = 1 - confidenceLevel
        computedForecastErrors::Vector{ModelFl} = forecastErrors(model, stepsAhead)
        zValue::ModelFl = quantile(Normal(0, 1), 1 - α / 2)
        lowerConfidenceInterval::Vector{ModelFl} = [
            forecastValues[i] - zValue * sqrt(computedForecastErrors[i]) for
            i = 1:stepsAhead
        ]
        upperConfidenceInterval::Vector{ModelFl} = [
            forecastValues[i] + zValue * sqrt(computedForecastErrors[i]) for
            i = 1:stepsAhead
        ]
        data = (
            datetime = forecastTimestamps,
            forecast = forecastValues,
            lower = lowerConfidenceInterval,
            upper = upperConfidenceInterval,
        )
        model.forecast = TimeArray(data; timestamp = :datetime)
    else
        model.forecast = TimeArray(forecastTimestamps, forecastValues, ["forecast"])
    end
end


"""
    predict(
        model::SARIMAModel,
        stepsAhead::Int = 1,
        isSimulation::Bool = true,
        automaticExogDifferentiation::Bool=false
    )

Predicts the SARIMA model for the next `stepsAhead` periods assuming the model's estimated σ² in case of a simulation.
Returns the forecasted values.

# Arguments
- `model::SARIMAModel`: The SARIMA model to make predictions.
- `stepsAhead::Int`: The number of periods ahead to forecast (default: 1).
- `isSimulation::Bool`: Whether to perform a simulation-based forecast (default: true).
- `automaticExogDifferentiation::Bool`: Whether to automatically differentiate the exogenous variables. Default is `false`.

# Example
```jldoctest
julia> airPassengers = load_dataset(AIR_PASSENGERS)

julia> model = SARIMA(airPassengers, 0, 1, 1; seasonality=12, P=0, D=1, Q=1)

julia> fit!(model)

julia> forecastedValues = predict(model, stepsAhead=12)
````
"""
function predict(
    model::SARIMAModel,
    stepsAhead::Int = 1,
    isSimulation::Bool = false,
    automaticExogDifferentiation::Bool = false,
)
    !isFitted(model) && throw(ModelNotFitted())
    ModelFl = typeofModelElements(model)

    diffY = differentiate(model.y, model.d, model.D, model.seasonality)
    valuesExog = []
    if !isnothing(model.exog)
        if automaticExogDifferentiation
            diffExog, _ = automatic_differentiation(model.exog)
        else
            diffExog = model.exog
        end
        # Adjust start points
        start_date = min(timestamp(diffY)[1], timestamp(diffExog)[1])
        diffY = from(diffY, start_date)
        diffExog = from(diffExog, start_date)

        valuesExog = values(diffExog)
    end

    if !isnothing(model.exog) && all(startswith.(string.(colnames(model.exog)), "outlier"))
        nCols = size(valuesExog, 2)
        valuesExog = vcat(valuesExog, zeros(stepsAhead, nCols))
    end

    T = size(diffY, 1)
    exogT = isnothing(model.exog) ? 0 : size(valuesExog, 1)
    if !isnothing(model.exog) && T + stepsAhead > exogT
        throw(MissingExogenousData())
    end

    yValues::Vector{ModelFl} = deepcopy(values(diffY))

    # Mirrors the estimation semantics (see `arAcc` in `fit!`). Under :regression_errors the
    # state the AR terms recurse on is eta_t = y_t - X_t'b rather than y_t, and the
    # exogenous block re-enters only at the level, at the end of each step. Under :armax
    # `arValues` IS `yValues` (the same object), so the `push!` feeds both.
    #
    # The fallback is `"armax"` because a model reaching here without the metadata key was
    # fitted by an earlier version, whose coefficients were estimated under ARMAX
    # semantics; forecasting them under the other form would give a wrong forecast from
    # correct coefficients. The fallback preserves the semantics that produced them.
    regErr::Bool =
        !isnothing(model.exog) &&
        get(model.metadata, "exogDynamics", "armax") == "regression_errors"
    arValues::Vector{ModelFl} = if regErr
        yValues .- valuesExog[1:length(yValues), :] * model.exogCoefficients
    else
        yValues
    end

    driftFuture::Vector{ModelFl} = model.allowDrift ?
        differentiate(
            collect(ModelFl, 1:(length(values(model.y)) + stepsAhead)),
            model.d, model.D, model.seasonality,
        ) : ModelFl[]
    errors = deepcopy(model.ϵ)

    for step = 1:stepsAhead
        forecastedValue::ModelFl =
            model.c + (model.allowDrift ? model.trend * driftFuture[T+step] : model.trend)
        errorsLength = length(errors)
        yLength = length(arValues)
        if model.p > 0
            # ∑ϕᵢyₜ -i
            forecastedValue += sum(model.ϕ[i] * arValues[end-i+1] for i = 1:model.p)
        end
        if model.q > 0
            # ∑θᵢϵₜ-i
            forecastedValue += sum(
                model.θ[j] * errors[end-j+1] for
                j = 1:model.q if (errorsLength - j + 1 > 0)
            )
        end
        if model.P > 0
            # ∑Φₖyₜ-(s*k)
            forecastedValue += sum(
                model.Φ[k] * arValues[end-(model.seasonality*k)+1] for
                k = 1:model.P if (yLength - model.seasonality * k + 1 > 0)
            )
        end
        if model.Q > 0
            # ∑Θₖϵₜ-(s*k)
            forecastedValue += sum(
                model.Θ[w] * errors[end-(model.seasonality*w)+1] for
                w = 1:model.Q if (errorsLength - (model.seasonality * w) + 1 > 0)
            )
        end
        if modelSeasonalForm(model) === :multiplicative
            if model.p > 0 && model.P > 0
                for i = 1:model.p, k = 1:model.P
                    idx = yLength - i - model.seasonality * k + 1
                    idx > 0 && (forecastedValue -= model.ϕ[i] * model.Φ[k] * arValues[idx])
                end
            end
            if model.q > 0 && model.Q > 0
                for j = 1:model.q, w = 1:model.Q
                    idx = errorsLength - j - model.seasonality * w + 1
                    idx > 0 && (forecastedValue += model.θ[j] * model.Θ[w] * errors[idx])
                end
            end
        end
        exogTerm::ModelFl =
            isnothing(model.exog) ? zero(ModelFl) :
            valuesExog[T+step, :]'model.exogCoefficients
        regErr || (forecastedValue += exogTerm)

        ϵₜ = isSimulation ? rand(Normal(0, sqrt(model.σ²))) : 0
        forecastedValue += ϵₜ

        push!(errors, ϵₜ)
        if regErr
            push!(arValues, forecastedValue)          # eta previsto alimenta a recursao AR
            push!(yValues, forecastedValue + exogTerm)  # o nivel soma o bloco exogeno
        else
            push!(yValues, forecastedValue)           # arValues === yValues, alimentado junto
        end
    end
    initialValuesLength = model.d + model.D * model.seasonality
    initialValuesOffset = length(values(model.y)) - initialValuesLength + 1
    initialValues::Vector{ModelFl} = values(model.y)[initialValuesOffset:end]
    forecast_values = integrate(
        initialValues,
        yValues[end-stepsAhead+1:end],
        model.d,
        model.D,
        model.seasonality,
    )
    return forecast_values[initialValuesLength+1:end]
end


"""
    simulate(
        model::SARIMAModel,
        stepsAhead::Int = 1,
        numScenarios::Int = 200,
        seed::Int = 1234
    )

Simulates the SARIMA model for the next `stepsAhead` periods assuming that the model's estimated σ².
Returns a vector of `numScenarios` scenarios of the forecasted values.

# Arguments
- `model::SARIMAModel`: The SARIMA model to simulate.
- `stepsAhead::Int`: The number of periods ahead to simulate. Default is 1.
- `numScenarios::Int`: The number of simulation scenarios. Default is 200.
- `seed::Int`: The seed of the simulation. Default is 1234.

# Returns
- `Vector{Vector{AbstractFloat}}`: A vector of scenarios, each containing the forecasted values for the next `stepsAhead` periods.

# Example
```jldoctest
julia> airPassengers = load_dataset(AIR_PASSENGERS)

julia> model = SARIMA(airPassengers, 0, 1, 1; seasonality=12, P=0, D=1, Q=1)

julia> fit!(model)

julia> scenarios = simulate(model, stepsAhead=12, numScenarios=1000)
```
"""
function simulate(
    model::SARIMAModel,
    stepsAhead::Int = 1,
    numScenarios::Int = 200,
    seed::Int = 1234,
)
    !isFitted(model) && throw(ModelNotFitted())
    ModelFl = typeofModelElements(model)
    Random.seed!(seed)

    scenarios::Vector{Vector{ModelFl}} = []
    for _ = 1:numScenarios
        push!(scenarios, predict(model, stepsAhead, true))
    end
    return scenarios
end

"""
    auto(
        y::TimeArray;
        exog::Union{TimeArray,Nothing}=nothing,
        seasonality::Int=1,
        d::Int = -1,
        D::Int = -1,
        maxp::Int = 5,
        maxd::Int = 2,
        maxq::Int = 5,
        maxP::Int = 2,
        maxD::Int = 1,
        maxQ::Int = 2,
        maxOrder::Int = 5,
        informationCriteria::String = "aicc",
        allowMean:Union{Bool,Nothing} = nothing,
        allowDrift::Union{Bool,Nothing} = nothing,
        integrationTest::String = "kpss",
        seasonalIntegrationTest::String = "seas",
        objectiveFunction::String = "mse",
        assertStationarity::Bool = true,
        assertInvertibility::Bool = true,
        showLogs::Bool = false,
        outlierDetection::Bool = false
        searchMethod::String = "stepwise"
    )

Automatically fits the best SARIMA model according to the specified parameters.

# Arguments
- `y::TimeArray`: The time series data.
- `exog::Union{TimeArray,Nothing}`: Optional exogenous variables. If `Nothing`, no exogenous variables are used.
- `seasonality::Int`: The seasonality period. Default is 1 (non-seasonal).
- `d::Int`: The degree of differencing for the non-seasonal part. Default is -1 (auto-select).
- `D::Int`: The degree of differencing for the seasonal part. Default is -1 (auto-select).
- `maxp::Int`: The maximum autoregressive order for the non-seasonal part. Default is 5.
- `maxd::Int`: The maximum integration order for the non-seasonal part. Default is 2.
- `maxq::Int`: The maximum moving average order for the non-seasonal part. Default is 5.
- `maxP::Int`: The maximum autoregressive order for the seasonal part. Default is 2.
- `maxD::Int`: The maximum integration order for the seasonal part. Default is 1.
- `maxQ::Int`: The maximum moving average order for the seasonal part. Default is 2.
- `maxOrder::Int`: Cap on `p + q + P + Q`. Default is 5. **Applies to
  `searchMethod = "grid"` only.** The stepwise searches deliberately run with the cap
  disabled, to match `forecast`: there `max.order` lives inside `search.arima` (the
  non-stepwise path), and since `auto.arima` defaults to stepwise, R routinely selects
  orders whose sum exceeds 5. The consequence is that at the monthly defaults
  (`maxp = maxq = 5`, `maxP = maxQ = 2`) the grid reaches 96 of the 324 order combinations
  in the box while the stepwise search reaches all 324 — i.e. the exhaustive method
  searches a *smaller* space than the heuristic one, and can lose to it. Raise `maxOrder`
  (up to `maxp + maxq + maxP + maxQ`) to make `"grid"` genuinely exhaustive.
- `informationCriteria::String`: The information criteria to be used for model selection. Options are "aic", "aicc", or "bic". Default is "aicc".
- `allowMean::Union{Bool,Nothing}`: Whether to include a mean term in the model. Default is nothing.
- `allowDrift::Union{Bool,Nothing}`: Whether to include a drift term in the model. Default is nothing.
- `integrationTest::String`: The integration test to be used for determining the non-seasonal
  integration order. `"kpssShort"` (default) uses urca-style `lags = "short"`, matching the
  differencing decisions of R's `forecast::ndiffs`/`auto.arima`; `"kpss"` uses the Hobijn
  et al. automatic lag selection (statsmodels-compatible).
- `invertible::Bool`: Fit every candidate with the invertibility-by-construction MA
  parameterization (reflection coefficients). Default `false`: candidates are first fitted
  with free (box-bounded) MA — which finds better optima on most series — and only when a
  fit fails the 1.001 root-admissibility check is it refitted with the constrained
  parameterization and re-checked (`ensureAdmissible!`, margin 2e-3). Every accepted model
  is therefore stationary and invertible either way; `invertible = true` merely forces the
  constrained parameterization on every fit (not compatible with the `"bilevel"` objective).
- `invertibilityMargin::AbstractFloat`: Margin keeping MA reflection coefficients in
  `[-(1-m), 1-m]` when `invertible = true`. Default `DEFAULT_DOMAIN_MARGIN` (`1e-6`) — it
  opens the domain, it does not enforce admissibility (that is `rootMargin`'s job).
- `stationary::Bool`: Fit candidates with the stationarity-by-construction AR
  parameterization. Defaults to `assertStationarity` (empirically as accurate as the free
  AR fit and cheaper than relying on rejection).
- `stationarityMargin::AbstractFloat`: AR analogue of `invertibilityMargin`. Default
  `DEFAULT_DOMAIN_MARGIN` (`1e-6`).
- `optimizer::Union{DataType,MOI.OptimizerWithAttributes}`: JuMP optimizer used to fit
  every candidate. Default `Ipopt.Optimizer` (fast local solutions). Pass
  `SCIP.Optimizer` — or `optimizer_with_attributes(SCIP.Optimizer, "limits/gap" => …)`
  to control its tolerances — to solve each CSS problem to certified global optimality
  (global certificate). Certification is only practical on short series (roughly
  T ≲ 100); much slower, intended for experiments and final refits rather than
  large-scale runs.
- `warmStartFromBox::Bool`: Forwarded to [`fit!`](@ref) for every candidate: seeds each
  constrained fit from a cheap unconstrained solve with a tiered fallback. Default
  `false`.
- `maxTimeSeconds::Union{Nothing,Real}`: Per-fit budget forwarded to [`fit!`](@ref).
  Within the search, candidate fits are capped at `min(maxTimeSeconds, 10)` seconds —
  a candidate that cannot be solved quickly is not going to win — while the final
  refit of the selected model keeps the full budget. Default `nothing`.
- `seasonalIntegrationTest::String`: The integration test to be used for determining the seasonal integration order. Default is "seas".
- `objectiveFunction::String`: The objective function to be used for model selection.
- `parallel::Bool`: Fit candidate models across Julia threads (experimental; applies to
  the "grid" and "stepwiseNaive" searches; requires starting Julia with multiple
  threads). Default is `false`. Options are "mse", "ml", or "bilevel". Default is "mse".
- `assertStationarity::Bool`: Whether to assert stationarity of the fitted model. Default is true.
- `assertInvertibility::Bool`: Whether to assert invertibility of the fitted model. Default is true.
- `showLogs::Bool`: Whether to suppress output. Default is false.
- `outlierDetection::Bool`: Whether to perform outlier detection. Default is false.
- `searchMethod::String`: The search strategy: "stepwise" (Hyndman-Khandakar style, default),
  "stepwiseNaive", "grid" (exhaustive), or "sarimax" (no search: fits a single dense
  specification at the maximum orders, intended for regularized estimation).
- `requireTermsWhenOverDifferenced::Bool`: When `d + D >= 2`, drop the term-free order
  `(0,d,0)(0,D,0)` from the search. Default is false: `auto.arima` has no such rule, so
  enabling it is a deliberate divergence from the reference implementation.
- `requireMAWhenDoublyDifferenced::Bool`: When `d >= 2`, require `q >= 1` (and symmetrically
  `Q >= 1` when `D >= 2`). Second-differencing induces a unit MA root at the lag that was
  differenced; a candidate without an MA term there cannot represent it and compensates with
  AR persistence, which explodes once re-integrated twice. Measured on 144 M4 monthly series
  with `d = 2, D = 0`: OWA 1.047 (worse than Naive2) to 0.993, per-series median 0.901 to
  0.839. The guard is dimensional, not aggregate — a seasonal MA at lag `s` cannot damp a
  unit root at lag 1. `d = D = 1` is untouched: with one difference in each dimension theory
  does not say which one carries the term. Default is false.

# References
- Hyndman, RJ and Khandakar. "Automatic time series forecasting: The forecast package for R." Journal of Statistical Software, 26(3), 2008.
"""
function auto(
    y::TimeArray;
    exog::Union{TimeArray,Nothing} = nothing,
    seasonality::Int = 1,
    d::Int = -1,
    D::Int = -1,
    maxp::Int = 5,
    maxd::Int = 2,
    maxq::Int = 5,
    maxP::Int = 2,
    maxD::Int = 1,
    maxQ::Int = 2,
    maxOrder::Int = 5,
    informationCriteria::String = "aicc",
    allowMean::Union{Bool,Nothing} = nothing,
    allowDrift::Union{Bool,Nothing} = nothing,
    integrationTest::String = "kpssShort",
    seasonalIntegrationTest::String = "seas",
    objectiveFunction::String = "mse",
    assertStationarity::Bool = true,
    assertInvertibility::Bool = true,
    showLogs::Bool = false,
    outlierDetection::Bool = false,
    searchMethod::String = "stepwise",
    parallel::Bool = false,
    seasonalForm::Symbol = :multiplicative,
    exogDynamics::Symbol = :armax,
    penaltyTarget::Symbol = :all,
    initialization::Symbol = DEFAULT_INITIALIZATION,
    multistart::Bool = false,
    # INDEPENDENT of `assertStationarity`. Tying the two would conflate distinct decisions:
    # imposing stationarity BY CONSTRUCTION (changing the parameterization) and REJECTING
    # inadmissible candidates (a selection rule). R only does the second.
    stationary::Bool = true,
    stationarityMargin::AbstractFloat = DEFAULT_DOMAIN_MARGIN,
    invertible::Bool = false,
    invertibilityMargin::AbstractFloat = DEFAULT_DOMAIN_MARGIN,
    constrainedRefit::Bool = false,
    rootMargin::AbstractFloat = DEFAULT_ROOT_MARGIN,
    optimizer::Union{DataType,MOI.OptimizerWithAttributes} = Ipopt.Optimizer,
    warmStartFromBox::Bool = false,
    maxTimeSeconds::Union{Nothing,Real} = nothing,
    cvarLevel::AbstractFloat = DEFAULT_CVAR_LEVEL,
    lambda::Union{Float64,Nothing} = nothing,
    alpha::Union{Float64,Nothing} = nothing,
    requireTermsWhenOverDifferenced::Bool = false,
    requireMAWhenDoublyDifferenced::Bool = false,
)
    # Parameter validation
    any(isnan, values(y)) && throw(
        ArgumentError(
            "Automatic order selection does not support missing data. Fit a " *
            "specific stationary model with SARIMA(...) + fit! instead, or impute first.",
        ),
    )
    @assert seasonality >= 1 "seasonality must be greater than 1. Use 1 for non-seasonal models"
    @assert d >= -1
    @assert d <= maxd
    @assert D >= -1
    @assert D <= maxD
    @assert maxp >= 0
    @assert maxd >= 0
    @assert maxq >= 0
    @assert maxP >= 0
    @assert maxD >= 0
    @assert maxQ >= 0
    @assert isnothing(lambda) || (lambda > 0)
    @assert isnothing(alpha) || (alpha >= 0 && alpha <= 1)
    @assert informationCriteria ∈ ["aic", "aicc", "bic"]
    @assert integrationTest ∈ ["kpss", "kpssShort"]
    @assert seasonalIntegrationTest ∈ ["seas", "ch", "ocsb"]
    @assert objectiveFunction ∈ ["mae", "mse", "ml", "bilevel", "elastic_net", "stable", "ridge", "huber", "ml_exact"]
    @assert objectiveFunction == "elastic_net" || isnothing(lambda)
    @assert objectiveFunction == "elastic_net" || isnothing(alpha)
    @assert searchMethod ∈ ["stepwise", "stepwiseNaive", "grid", "sarimax"]
    @assert !(invertible && objectiveFunction == "bilevel") "invertible = true is not compatible with the bilevel objective"
    @assert seasonalForm in (:multiplicative, :additive) "seasonalForm must be :multiplicative or :additive (:free is planned)"
    @assert initialization in (:zeroed, :warmup, :free, :penalized, :innovations) "initialization must be :zeroed, :warmup, :free, :penalized or :innovations"

    ModelFl = eltype(values(y))
    informationCriteriaFunction = getInformationCriteriaFunction(informationCriteria)

    # Deal with constant series
    if isConstant(y)
        showLogs && @info("The series is constant")
        constant = isnothing(allowMean) ? true : allowMean
        return constantSeriesModelSpecification(y, exog, constant)
    end

    # Adjustments based on parameters
    if seasonality == 1
        D = 0
    end

    if D < 0
        D =
            (length(values(y)) < 2 * seasonality) ? 0 :
            selectSeasonalIntegrationOrder(
                deepcopy(values(y)),
                seasonality,
                seasonalIntegrationTest,
            )

        # Check if chosen D is viable given the data
        if D > 0 && !isnothing(exog)
            diffExog = differentiate(exog, 0, D, seasonality)
            if isConstant(diffExog)
                showLogs && @info(
                    "The exogenous variables are constant after seasonal differencing"
                )
                D -= 1
            end
        end

        if D > 0
            diffY = differentiate(y, 0, D, seasonality)
            if all(ismissing.(values(diffY)))
                showLogs && @info("The series is missing after seasonal differencing")
                D -= 1
            end
        end
    end

    if d < 0
        d = selectIntegrationOrder(
            deepcopy(values(y)),
            maxd,
            D,
            seasonality,
            integrationTest,
        )

        # Check if chosen d is viable given the data
        if d > 0 && !isnothing(exog)
            diffExog = differentiate(exog, d, D, seasonality)
            if isConstant(diffExog)
                showLogs && @info(
                    "The exogenous variables are constant after non-seasonal differencing."
                )
                d -= 1
            end
        end

        if d > 0
            diffY = differentiate(y, d, D, seasonality)
            if all(ismissing.(values(diffY)))
                showLogs && @info("The series is missing after non-seasonal differencing")
                d -= 1
            end
        end
    end

    fixConstant = !isnothing(allowMean) || !isnothing(allowDrift) || (d + D > 1)

    allowMean = isnothing(allowMean) ? (d + D == 0) : allowMean
    allowDrift = isnothing(allowDrift) ? (d + D == 1) : allowDrift


    # Deal with series constant after differencing
    if d + D > 0 && isConstant(differentiate(y, d, D, seasonality))
        showLogs && @info("The series is constant after differencing")
        return constantDiffSeriesModelSpecification(
            y,
            exog,
            d,
            D,
            seasonality,
            allowMean,
            allowDrift,
        )
    end

    # The search only needs information criteria that are comparable across candidates,
    # so cap its fits tighter than the final refit: a candidate that cannot be solved
    # quickly is not going to win, and paying the full budget on each of the dozens of
    # candidates is what pushed hard series past the per-series timeout.
    searchMaxTime =
        isnothing(maxTimeSeconds) ? nothing : min(Float64(maxTimeSeconds), 10.0)

    # Set maximum orders
    maxp = min(maxp, floor(Int, length(values(y)) / 3))
    maxp = (seasonality == 1) ? maxp : min(maxp, seasonality-1) # Avoid overlap with seasonal orders
    maxq = min(maxq, floor(Int, length(values(y)) / 3))
    maxq = (seasonality == 1) ? maxq : min(maxq, seasonality-1) # Avoid overlap with seasonal orders
    maxP =
        (seasonality == 1) ? 0 : min(maxP, floor(Int, length(values(y)) / 3 * seasonality))
    maxQ =
        (seasonality == 1) ? 0 : min(maxQ, floor(Int, length(values(y)) / 3 * seasonality))

    # All search candidates are conditioned on the same pre-sample length so that their CSS
    # objectives — and the CSS likelihoods of the criterion FALLBACK path — live on the same
    # effective sample. Since the criteria moved to the exact likelihood (evaluated on the
    # full differenced sample regardless of conditioning), this common `lb` no longer shapes
    # the primary criterion path; it still governs the estimation objective and keeps the
    # fallback comparisons consistent. Whether it should be removed altogether is an open
    # question that needs M4-scale measurement (it changes the estimation, not just scoring).
    # With :free initialization every candidate already scores on the full differenced
    # sample (pre-sample values are estimated), so no common conditioning is needed.
    searchLb =
        initialization in (:free, :penalized, :innovations) ? 0 :
        conditioningLags(maxp, maxq, maxP, maxQ, seasonality, seasonalForm)

    if outlierDetection
        exog = detectOutliers(y, exog, d, D, seasonality, showLogs)
    end

    if searchMethod == "stepwise"
        bestModel = stepwiseSearch(
            y,
            exog,
            d,
            D,
            seasonality,
            informationCriteriaFunction;
            maxp = maxp,
            maxq = maxq,
            maxP = maxP,
            maxQ = maxQ,
            # `maxOrder` does NOT apply in the stepwise search, matching R: in `forecast`,
            # `max.order` is imposed only inside `search.arima`, the non-stepwise path.
            # Since `auto.arima` defaults to stepwise, R routinely selects orders with
            # p+q+P+Q > 5 that this search would otherwise refuse by construction. Measured
            # in the tail, R's order fell outside our space on 4.5% of the bad series,
            # carrying 4.8% of the damage. The per-term boxes (maxp/maxq/maxP/maxQ) still
            # apply.
            maxOrder = maxp + maxq + maxP + maxQ,
            warmStartFromBox = warmStartFromBox,
            maxTimeSeconds = searchMaxTime,
            cvarLevel = cvarLevel, multistart = multistart,
            objectiveFunction = objectiveFunction,
            assertStationarity = assertStationarity,
            assertInvertibility = assertInvertibility,
            showLogs = showLogs,
            minConditioningObs = searchLb,
            seasonalForm = seasonalForm,
            exogDynamics = exogDynamics,
            penaltyTarget = penaltyTarget,
            initialization = initialization,
            stationary = stationary,
            stationarityMargin = stationarityMargin,
            invertible = invertible,
            invertibilityMargin = invertibilityMargin,
            constrainedRefit = constrainedRefit,
            requireTermsWhenOverDifferenced = requireTermsWhenOverDifferenced,
            requireMAWhenDoublyDifferenced = requireMAWhenDoublyDifferenced,
            optimizer = optimizer,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda,
            rootMargin = rootMargin
        )
    elseif searchMethod == "stepwiseNaive"
        bestModel = stepWiseSearchNaive(
            y,
            exog,
            d,
            D,
            seasonality,
            informationCriteriaFunction;
            maxp = maxp,
            maxq = maxq,
            maxP = maxP,
            maxQ = maxQ,
            # as in `stepwise`: a local search, and R does not impose `max.order` outside
            # `search.arima`
            maxOrder = maxp + maxq + maxP + maxQ,
            warmStartFromBox = warmStartFromBox,
            maxTimeSeconds = searchMaxTime,
            cvarLevel = cvarLevel, multistart = multistart,
            objectiveFunction = objectiveFunction,
            assertStationarity = assertStationarity,
            assertInvertibility = assertInvertibility,
            showLogs = showLogs,
            minConditioningObs = searchLb,
            seasonalForm = seasonalForm,
            exogDynamics = exogDynamics,
            penaltyTarget = penaltyTarget,
            initialization = initialization,
            stationary = stationary,
            stationarityMargin = stationarityMargin,
            invertible = invertible,
            invertibilityMargin = invertibilityMargin,
            constrainedRefit = constrainedRefit,
            rootMargin = rootMargin,
            optimizer = optimizer,
            parallel = parallel,
            allowMean = allowMean,
            allowDrift = allowDrift,
            fixConstant = fixConstant,
            alpha = alpha,
            lambda = lambda,
        )
    elseif searchMethod == "grid"
        bestModel = gridSearch(
            y,
            exog,
            d,
            D,
            seasonality,
            informationCriteriaFunction;
            maxp = maxp,
            maxq = maxq,
            maxP = maxP,
            maxQ = maxQ,
            maxOrder = maxOrder,
            warmStartFromBox = warmStartFromBox,
            maxTimeSeconds = searchMaxTime,
            cvarLevel = cvarLevel, multistart = multistart,
            objectiveFunction = objectiveFunction,
            assertStationarity = assertStationarity,
            assertInvertibility = assertInvertibility,
            showLogs = showLogs,
            minConditioningObs = searchLb,
            seasonalForm = seasonalForm,
            exogDynamics = exogDynamics,
            penaltyTarget = penaltyTarget,
            initialization = initialization,
            stationary = stationary,
            stationarityMargin = stationarityMargin,
            invertible = invertible,
            invertibilityMargin = invertibilityMargin,
            constrainedRefit = constrainedRefit,
            optimizer = optimizer,
            parallel = parallel,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda,
            rootMargin = rootMargin
        )
    elseif searchMethod == "sarimax"
        if isnothing(exog)
            bestModel = SARIMA(
                y,
                maxp,
                d,
                2;
                P = maxP,
                D = D,
                Q = 1,
                seasonality = seasonality,
                allowMean = allowMean,
                allowDrift = allowDrift,
                alpha = alpha
            )
        else
            bestModel = SARIMA(
                y,
                exog,
                maxp,
                d,
                maxq;
                P = maxP,
                D = D,
                Q = maxQ,
                seasonality = seasonality,
                allowMean = allowMean,
                allowDrift = allowDrift,
                alpha = alpha
            )
        end

        fit!(bestModel; objectiveFunction = objectiveFunction, alpha = alpha, silent = !showLogs, minConditioningObs = searchLb, seasonalForm = seasonalForm, exogDynamics = exogDynamics, penaltyTarget = penaltyTarget, initialization = initialization, stationary = stationary, stationarityMargin = stationarityMargin, invertible = invertible, invertibilityMargin = invertibilityMargin, optimizer = optimizer, warmStartFromBox = warmStartFromBox, maxTimeSeconds = maxTimeSeconds, cvarLevel = cvarLevel, multistart = multistart)
    end

    bestModel.exog = exog
    showLogs && @info("The best model found is $(getId(bestModel))")

    return bestModel
end


"""
    getInformationCriteriaFunction(informationCriteria)

Returns the SELECTION criterion function for the search: `aic`/`aicc`/`bic` wrapped by
[`searchCriterionFunction`](@ref), so that candidates whose criterion came from the CSS
fallback (no computable exact likelihood — typically roots at the boundary) are penalized
and can never outrank a candidate scored by the exact likelihood. The public `aic`/`aicc`/
`bic` accessors are NOT affected.

# Arguments
- `informationCriteria::String`: The name of the information criteria ("aic", "aicc", or "bic").

# Returns
- `Function`: The selection criterion function corresponding to the input.

# Throws
- `ArgumentError`: If the provided `informationCriteria` is not one of "aic", "aicc", or "bic".
"""
function getInformationCriteriaFunction(informationCriteria::String)
    if informationCriteria == "aic"
        return searchCriterionFunction(aic)
    elseif informationCriteria == "aicc"
        return searchCriterionFunction(aicc)
    elseif informationCriteria == "bic"
        return searchCriterionFunction(bic)
    end
    throw(ArgumentError("The information criteria '$informationCriteria' is not supported"))
end

"""
    constantSeriesModelSpecification(
        y::TimeArray,
        exog::Union{TimeArray,Nothing},
        allowMean::Bool
    )

Returns a SARIMA model for a series that is constant.

# Arguments
- `y::TimeArray`: The time series data.
- `exog::Union{TimeArray,Nothing}`: Optional exogenous variables. If `Nothing`, no exogenous variables are used.
- `allowMean::Bool`: Whether to include a mean term in the model.

# Returns
- `SARIMAModel`: The SARIMA model for the constant series.
"""
function constantSeriesModelSpecification(
    y::TimeArray,
    exog::Union{TimeArray,Nothing},
    allowMean::Bool,
)
    model = SARIMA(y, exog, 0, 0, 0; allowMean = allowMean)
    fit!(model)
    return model
end

"""
    constantDiffSeriesModelSpecification(
        y::TimeArray,
        exog::Union{TimeArray,Nothing},
        d::Int,
        D::Int,
        seasonality::Int,
        allowMean::Bool,
        allowDrift::Bool
    )

Returns a SARIMA model for a series that is constant after differencing.

# Arguments
- `y::TimeArray`: The time series data.
- `exog::Union{TimeArray,Nothing}`: Optional exogenous variables. If `Nothing`, no exogenous variables are used.
- `d::Int`: The degree of differencing.
- `D::Int`: The degree of seasonal differencing.
- `seasonality::Int`: The seasonality period.
- `allowMean::Bool`: Whether to include a mean term in the model.
- `allowDrift::Bool`: Whether to include a drift term in the model.

# Returns
- `SARIMAModel`: The SARIMA model for the series that is constant after differencing.

"""
function constantDiffSeriesModelSpecification(
    y::TimeArray,
    exog::Union{TimeArray,Nothing},
    d::Int,
    D::Int,
    seasonality::Int,
    allowMean::Bool,
    allowDrift::Bool,
)
    if isnothing(exog)
        if (D > 0 && d == 0)
            # TODO: Check if it is necessary to specify the intercept value
            # constant should be mean(dy) / seasonality
            model = SARIMA(
                y,
                0,
                d,
                0;
                P = 0,
                D = D,
                Q = 0,
                seasonality = seasonality,
                allowMean = false,
                allowDrift = true,
            )
        elseif (D > 0 && d > 0)
            model = SARIMA(
                y,
                0,
                d,
                0;
                P = 0,
                D = D,
                Q = 0,
                seasonality = seasonality,
                allowMean = false,
                allowDrift = false,
            )
        elseif (d == 2)
            model = SARIMA(y, 0, d, 0; allowMean = false, allowDrift = false)
        elseif (d < 2)
            # TODO: Check if it is necessary to specify the intercept value
            # constant should be mean(dy)
            model = SARIMA(y, 0, d, 0; allowMean = true, allowDrift = false)
        else
            throw(
                ArgumentError(
                    "Data follow a simple polynomial and are not suitable for ARIMA modelling.",
                ),
            )
        end
    else
        if (D > 0)
            model = SARIMA(
                y,
                exog,
                0,
                d,
                0;
                P = 0,
                D = D,
                Q = 0,
                seasonality = seasonality,
                allowMean = false,
                allowDrift = false,
            )
        else
            model = SARIMA(y, exog, 0, d, 0; allowMean = false, allowDrift = false)
        end
    end

    fit!(model)

    return model
end

"""
    detectOutliers(
        y::TimeArray,
        exog::Union{TimeArray,Nothing},
        d::Int,
        D::Int,
        seasonality::Int,
        showLogs::Bool
    )

Detects outliers in the time series data and adds them to the exogenous variables.

# Arguments

- `y::TimeArray`: The time series data.
- `exog::Union{TimeArray,Nothing}`: Optional exogenous variables. If `Nothing`, no exogenous variables are used.
- `d::Int`: The degree of differencing.
- `D::Int`: The degree of seasonal differencing.
- `seasonality::Int`: The seasonality period.
- `showLogs::Bool`: Whether to suppress output.

# Returns
- `Union{TimeArray,Nothing}`: The exogenous variables with the detected outliers.
"""
function detectOutliers(
    y::TimeArray,
    exog::Union{TimeArray,Nothing},
    d::Int,
    D::Int,
    seasonality::Int,
    showLogs::Bool,
)
    if D == 0
        model = Sarimax.SARIMA(y, exog, 0, d, 0; allowMean = true)
    else
        model = Sarimax.SARIMA(
            y,
            exog,
            0,
            d,
            0;
            P = 0,
            D = D,
            Q = 0,
            seasonality = seasonality,
            allowMean = true,
        )
    end
    fit!(model)
    residuals = map(x -> abs(x) < 1e-10 ? 0.0 : x, model.ϵ)

    # Detect outliers
    outliers = identifyOutliers(residuals)

    # check if all elements are false
    if all(outliers .== 0.0)
        showLogs && @info("No outliers detected")
        return exog
    end

    originalOffset = length(values(y)) - length(residuals)
    # println("Original Offset: ", originalOffset)
    outliersIndex = findall(outliers .== 1.0) .+ originalOffset
    showLogs && @info("Outliers detected at indices: $(outliersIndex)")

    # Add dummies to the exogenous variables
    if isnothing(exog)
        # Generate Dummies
        dummyDataFrame = createOutliersDummies((outliers .== 1.0), originalOffset)
        dummyDataFrame[!, :date] = copy(timestamp(y))
        dummyTimeArray = TimeArray(dummyDataFrame, timestamp = :date)
        exog = dummyTimeArray
    else
        startDate = min(timestamp(y)[1], timestamp(exog)[1])
        filterExogTimestamps = timestamp(exog)[timestamp(exog).>=startDate]
        estimationExogLength =
            length(filterExogTimestamps[filterExogTimestamps.<=timestamp(y)[end]])
        if estimationExogLength < length(outliers)
            # cut outliers initial values
            outliers = outliers[end-estimationExogLength+1:end]
        end
        initialOffset = estimationExogLength - length(outliers)
        endOffset = length(filterExogTimestamps[filterExogTimestamps.>timestamp(y)[end]])
        dummyDataFrame = createOutliersDummies((outliers .== 1.0), initialOffset, endOffset)
        dummyDataFrame[!, :date] = copy(filterExogTimestamps)
        dummyTimeArray = TimeArray(dummyDataFrame, timestamp = :date)
        mergeVector::Vector{TimeArray} = [exog, dummyTimeArray]
        exog = Sarimax.merge(mergeVector)
    end

    return exog
end

"""
    initialNonSeasonalModels!(
        models::Vector{SARIMAModel},
        y::TimeArray,
        exog::Union{TimeArray,Nothing},
        maxp::Int,
        d::Int,
        maxq::Int,
        allowMean::Bool,
        allowDrift::Bool
    )

Populates the `models` vector with initial non-seasonal SARIMA models based on the specified parameters.
The models added are:
- SARIMA(0, d, 0)
- SARIMA(1, d, 0)
- SARIMA(0, d, 1)
- SARIMA(2, d, 2)

# Arguments
- `models::Vector{SARIMAModel}`: A vector to which the initial SARIMA models will be appended.
- `y::TimeArray`: The time series data.
- `exog::Union{TimeArray,Nothing}`: Optional exogenous variables. If `Nothing`, no exogenous variables are used.
- `maxp::Int`: The maximum autoregressive order.
- `d::Int`: The degree of differencing.
- `maxq::Int`: The maximum moving average order.
- `allowMean::Bool`: Whether to include a mean term in the model.
- `allowDrift::Bool`: Whether to include a drift term in the model.
"""
function initialNonSeasonalModels!(
    models::Vector{SARIMAModel},
    y::TimeArray,
    exog::Union{TimeArray,Nothing},
    maxp::Int,
    d::Int,
    maxq::Int,
    allowMean::Bool,
    allowDrift::Bool,
    alpha::Union{Nothing,Float64} = nothing,
    lambda::Union{Nothing,Float64} = nothing,
)
    push!(models, SARIMA(y, exog, 0, d, 0; allowMean = false, allowDrift = false, alpha = alpha, lambda = lambda))
    push!(models, SARIMA(y, exog, 0, d, 0; allowMean = allowMean, allowDrift = allowDrift, alpha = alpha, lambda = lambda))
    (maxp >= 1) && push!(
        models,
        SARIMA(y, exog, 1, d, 0; allowMean = allowMean, allowDrift = allowDrift, alpha = alpha, lambda = lambda),
    )
    (maxq >= 1) && push!(
        models,
        SARIMA(y, exog, 0, d, 1; allowMean = allowMean, allowDrift = allowDrift, alpha = alpha, lambda = lambda),
    )
    (maxp >= 2 && maxq >= 2) && push!(
        models,
        SARIMA(y, exog, 2, d, 2; allowMean = allowMean, allowDrift = allowDrift, alpha = alpha, lambda = lambda),
    )
end

"""
    initialSeasonalModels!(
        models::Vector{SARIMAModel},
        y::TimeArray,
        exog::Union{TimeArray,Nothing},
        maxp::Int,
        d::Int,
        maxq::Int,
        maxP::Int,
        D::Int,
        maxQ::Int,
        seasonality::Int,
        allowMean::Bool,
        allowDrift::Bool
    )

Populates the `models` vector with initial seasonal SARIMA models based on the specified parameters.
The models added are:
- SARIMA(0, d, 0)(0, D, 0)
- SARIMA(1, d, 0)(1, D, 0)
- SARIMA(0, d, 1)(0, D, 1)
- SARIMA(2, d, 2)(1, D, 1)

# Arguments
- `models::Vector{SARIMAModel}`: A vector to which the initial SARIMA models will be appended.
- `y::TimeArray`: The time series data.
- `exog::Union{TimeArray,Nothing}`: Optional exogenous variables. If `Nothing`, no exogenous variables are used.
- `maxp::Int`: The maximum autoregressive order for non-seasonal part.
- `d::Int`: The degree of differencing for non-seasonal part.
- `maxq::Int`: The maximum moving average order for non-seasonal part.
- `maxP::Int`: The maximum autoregressive order for seasonal part.
- `D::Int`: The degree of differencing for seasonal part.
- `maxQ::Int`: The maximum moving average order for seasonal part.
- `seasonality::Int`: The seasonality period.
- `allowMean::Bool`: Whether to include a mean term in the model.
- `allowDrift::Bool`: Whether to include a drift term in the model.
"""
function initialSeasonalModels!(
    models::Vector{SARIMAModel},
    y::TimeArray,
    exog::Union{TimeArray,Nothing},
    maxp::Int,
    d::Int,
    maxq::Int,
    maxP::Int,
    D::Int,
    maxQ::Int,
    seasonality::Int,
    allowMean::Bool,
    allowDrift::Bool,
    alpha::Union{Nothing,Float64} = nothing,
    lambda::Union{Nothing,Float64} = nothing,
)
    push!(
        models,
        SARIMA(
            y,
            exog,
            0,
            d,
            0;
            seasonality = seasonality,
            P = 0,
            D = D,
            Q = 0,
            allowMean = false,
            allowDrift = false,
            alpha = alpha,
            lambda = lambda,
        ),
    )
    push!(
        models,
        SARIMA(
            y,
            exog,
            0,
            d,
            0;
            seasonality = seasonality,
            P = 0,
            D = D,
            Q = 0,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda,
        ),
    )

    # Add non-seasonal models
    (maxp >= 1) && push!(
        models,
        SARIMA(
            y,
            exog,
            1,
            d,
            0;
            seasonality = seasonality,
            P = 0,
            D = D,
            Q = 0,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda,
        ),
    )
    (maxq >= 1) && push!(
        models,
        SARIMA(
            y,
            exog,
            0,
            d,
            1;
            seasonality = seasonality,
            P = 0,
            D = D,
            Q = 0,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda,
        ),
    )
    (maxp >= 2 && maxq >= 2) && push!(
        models,
        SARIMA(
            y,
            exog,
            2,
            d,
            2;
            seasonality = seasonality,
            P = 0,
            D = D,
            Q = 0,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda,
        ),
    )

    # Add seasonal models
    (maxp >= 1 && maxP >= 1) && push!(
        models,
        SARIMA(
            y,
            exog,
            1,
            d,
            0;
            seasonality = seasonality,
            P = 1,
            D = D,
            Q = 0,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda,
        ),
    )
    (maxq >= 1 && maxQ >= 1) && push!(
        models,
        SARIMA(
            y,
            exog,
            0,
            d,
            1;
            seasonality = seasonality,
            P = 0,
            D = D,
            Q = 1,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda,
        ),
    )
    (maxp >= 2 && maxq >= 2 && maxP >= 1 && maxQ >= 1) && push!(
        models,
        SARIMA(
            y,
            exog,
            2,
            d,
            2;
            seasonality = seasonality,
            P = 1,
            D = D,
            Q = 1,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda,
        ),
    )
end

"""
    getId(model::SARIMAModel)

Returns a string representation of the SARIMA model.

# Arguments
- `model::SARIMAModel`: The SARIMA model.

# Returns
- `String`: A string representation of the SARIMA model.

# Example
```jldoctest

julia> model = SARIMA(1, 0, 1; P=1, D=0, Q=1, seasonality=12, allowMean=true, allowDrift=false)

julia> getId(model)  # Returns "SARIMA(1,0,1)(1,0,1 s=12, c=true, drift=false)"
```
"""
function getId(model::SARIMAModel)
    return "SARIMA($(model.p),$(model.d),$(model.q))($(model.P),$(model.D),$(model.Q) s=$(model.seasonality), c=$(model.allowMean), drift=$(model.allowDrift))"
end

"""
    isVisited(model::SARIMAModel, visitedModels::Dict{String,Dict{String,Any}})

Checks if a SARIMA model has been visited during the search process.

# Arguments
- `model::SARIMAModel`: The SARIMA model to check.
- `visitedModels::Dict{String,Dict{String,Any}}`: A dictionary containing visited SARIMA models.

# Returns
- `Bool`: `true` if the model has been visited, `false` otherwise.

# Example
```jldoctest
julia> model = SARIMA(1, 0, 1; P=1, D=0, Q=1, seasonality=12, allowMean=true, allowDrift=false)

julia> visitedModels = Dict{String,Dict{String,Any}}("SARIMA(1,0,1)(1,0,1 s=12, c=true, drift=false)" => Dict("criteria" => 123))

julia> isVisited(model, visitedModels)  # Returns true
```
"""
function isVisited(model::SARIMAModel, visitedModels::Dict{String,Dict{String,Any}})
    id = getId(model)
    return haskey(visitedModels, id)
end

"""
    maxInverseRootModulus(a::Vector{Fl}) where Fl

Largest modulus among the inverse roots of the polynomial `1 + a[1] z + a[2] z^2 + ...`,
computed as the eigenvalues of the companion matrix of `z^n + a[1] z^(n-1) + ... + a[n]`.
A polynomial has all roots outside the unit circle iff this value is < 1. Returns 0 for
an empty (or all-zero) coefficient vector.
"""
function maxInverseRootModulus(a::Vector{Fl}) where {Fl}
    n = findlast(!iszero, a)
    isnothing(n) && return zero(real(Fl))
    coeffs = a[1:n]
    companion = zeros(Fl, n, n)
    companion[1, :] .= .-coeffs
    for i = 2:n
        companion[i, i-1] = one(Fl)
    end
    return maximum(abs.(eigvals(companion)))
end

"""
    checkModelStationarityInvertibility(model::SARIMAModel, assertStationarity::Bool, assertInvertibility::Bool, showLogs::Bool; rootMargin=1e-3)

Checks if a SARIMA model is stationary and invertible.

Following R's `auto.arima` (`myarima`), fits whose expanded AR/MA polynomial roots lie
within `rootMargin` of the unit circle (modulus < 1 + rootMargin, default 1.001) are
also rejected: such near-boundary fits behave like (near-)unit-root processes and can
produce erratic long-horizon forecasts even though they are technically admissible.
This complements the `stationary = true` fitting constraint, which guarantees strict
stationarity during optimization but lets solutions approach the boundary.

# Arguments

- `model::SARIMAModel`: The SARIMA model to check.
- `showLogs::Bool`: Whether to suppress output.
- `assertStationarity::Bool`: Whether to assert stationarity of the fitted models. Default is false.
- `assertInvertibility::Bool`: Whether to assert invertibility of the fitted models. Default is false.
- `rootMargin::AbstractFloat`: Rejection margin around the unit circle. Default
  [`DEFAULT_ROOT_MARGIN`] (`1e-2`), matching `forecast::auto.arima`: its `myarima` sets
  `ic = Inf` for any candidate whose smallest root falls within 1% of the unit circle. This
  is a SELECTION rule, not an estimation constraint — it does not touch the reflection
  coefficient parameterization, whose domain is set by [`DEFAULT_DOMAIN_MARGIN`].

# Returns
- `Bool`: `true` if the model is stationary and invertible, `false` otherwise.

"""
function checkModelStationarityInvertibility(
    model::SARIMAModel,
    assertStationarity::Bool,
    assertInvertibility::Bool,
    showLogs::Bool;
    rootMargin::AbstractFloat = DEFAULT_ROOT_MARGIN,
)
    # Candidates whose solver did not succeed are never selected: their
    # "estimates" are whatever point the solver stopped at. (TIME_LIMIT is
    # accepted: the bilevel objective sets a deliberate inner time limit.)
    solverStatus = get(model.metadata, "solverStatus", "OPTIMAL")
    solverOK = solverStatus in
        ("OPTIMAL", "LOCALLY_SOLVED", "ALMOST_OPTIMAL", "ALMOST_LOCALLY_SOLVED", "TIME_LIMIT")
    if !solverOK
        showLogs && @info("The model $(getId(model)) is discarded: solver status $(solverStatus)")
        return false
    end

    arCoefficients, maCoefficients = completeCoefficientsVector(model)

    # Admissibility threshold: all polynomial roots must have modulus > 1 + rootMargin,
    # i.e. all inverse roots must have modulus < 1 / (1 + rootMargin).
    threshold = 1 / (1 + rootMargin)

    # MA polynomial: 1 + θ₁ z + θ₂ z² + ...
    invertible = !assertInvertibility || maxInverseRootModulus(maCoefficients) < threshold
    showLogs && (invertible || @info("The model $(getId(model)) is not invertible (roots within $(rootMargin) of the unit circle)"))

    # AR polynomial: 1 - φ₁ z - φ₂ z² - ...
    stationary = !assertStationarity || maxInverseRootModulus(-arCoefficients) < threshold
    showLogs && (stationary || @info("The model $(getId(model)) is not stationary (roots within $(rootMargin) of the unit circle)"))

    showLogs && (!invertible || !stationary) && @info("The model will not be considered")
    return stationary && invertible
end

"""
    ensureAdmissible!(model, assertStationarity, assertInvertibility, showLogs; kwargs...)

On-demand constrained refitting: checks admissibility (stationarity/invertibility with
the 1.001 root margin) and, when the unconstrained fit fails the check, refits the model
with the stationarity/invertibility-by-construction parameterizations (margins
`refitMargin`) and re-checks. Returns `true` iff the (possibly refitted) model is
admissible.

Rationale: unconstrained (box-bounded) CSS estimation finds better optima on most series,
but on some it piles the MA root up at the unit circle and pure rejection then wipes out
the MA model space; always-constrained fitting rescues those series but degrades the rest.
Fitting free first and constraining only when the admissibility check bites keeps the best
of both — every accepted model is stationary and invertible, at unconstrained quality
wherever the free optimum is already admissible.
"""
function ensureAdmissible!(
    model::SARIMAModel,
    assertStationarity::Bool,
    assertInvertibility::Bool,
    showLogs::Bool;
    objectiveFunction::String = "mse",
    minConditioningObs::Int = 0,
    seasonalForm::Symbol = :multiplicative,
    exogDynamics::Symbol = :armax,
    penaltyTarget::Symbol = :all,
    initialization::Symbol = DEFAULT_INITIALIZATION,
    multistart::Bool = false,
    refitMargin::AbstractFloat = 2e-3,
    refit::Bool = true,
    stationary::Bool = assertStationarity,
    invertible::Bool = assertInvertibility,
    # Admissibility margin for the roots. The default 1e-3 is this package's; R's `myarima`
    # uses 1e-2, i.e. it sets `ic = Inf` for any candidate whose smallest root lies within
    # 1% of the unit circle. This is a SELECTION rule, not an estimation constraint: it does
    # not touch the reflection-coefficient parameterization.
    rootMargin::AbstractFloat = DEFAULT_ROOT_MARGIN,
    optimizer::Union{DataType,MOI.OptimizerWithAttributes} = Ipopt.Optimizer,
)
    checkModelStationarityInvertibility(
        model,
        assertStationarity,
        assertInvertibility,
        showLogs;
        rootMargin = rootMargin,
    ) && return true
    refit || return false
    (assertStationarity || assertInvertibility) || return false
    isFitted(model) || return false
    # The invertibility-by-construction parameterization does not support the bilevel
    # objective; in that case keep the plain rejection semantics.
    showLogs && @info("Refitting $(getId(model)) with constrained parameterization (on demand)")
    try
        fit!(
            model;
            objectiveFunction = objectiveFunction,
            minConditioningObs = minConditioningObs,
            seasonalForm = seasonalForm,
            exogDynamics = exogDynamics,
            penaltyTarget = penaltyTarget,
            initialization = initialization,
            # The on-demand refit takes the parameterization from the caller, which holds
            # the assertion and the constraint decisions separately, rather than deriving
            # one from the other.
            stationary = stationary,
            stationarityMargin = refitMargin,
            invertible = invertible && objectiveFunction != "bilevel",
            invertibilityMargin = refitMargin,
            optimizer = optimizer,
        )
    catch e
        showLogs && @info("Constrained refit of $(getId(model)) failed: $(typeof(e))")
        return false
    end
    return checkModelStationarityInvertibility(
        model,
        assertStationarity,
        assertInvertibility,
        showLogs;
        rootMargin = rootMargin,
    )
end

"""
    localSearch!(
        candidateModels::Vector{SARIMAModel},
        visitedModels::Dict{String,Dict{String,Any}},
        informationCriteriaFunction::Function,
        objectiveFunction::String = "mse",
        assertStationarity::Bool = false,
        assertInvertibility::Bool = false,
        showLogs::Bool = false
        icOffset::Fl = 0.0
    )

Performs a local search to find the best SARIMA model among the candidate models.

# Arguments
- `candidateModels::Vector{SARIMAModel}`: A vector of candidate SARIMA models to search from.
- `visitedModels::Dict{String,Dict{String,Any}}`: A dictionary containing information about visited models.
- `informationCriteriaFunction::Function`: A function to calculate the information criteria for a SARIMA model.
- `objectiveFunction::String`: The objective function to be used for fitting models. Default is "mse".
- `assertStationarity::Bool`: Whether to assert stationarity of the fitted models. Default is false.
- `assertInvertibility::Bool`: Whether to assert invertibility of the fitted models. Default is false.
- `showLogs::Bool`: Whether to suppress output. Default is false.
- `icOffset::Fl`: The offset to be added to the information criteria. Default is 0.0.

# Returns
- `Tuple{AbstractFloat, Union{SARIMAModel, Nothing}}`: A tuple containing the best criteria value and the corresponding best model found.

# Example
```jldoctest
julia> candidateModels = [SARIMA(1, 0, 1), SARIMA(0, 1, 1)]

julia> visitedModels = Dict{String,Dict{String,Any}}()

julia> informationCriteriaFunction = aicc

julia> localSearch!(candidateModels, visitedModels, informationCriteriaFunction)
```
"""
function localSearch!(
    candidateModels::Vector{SARIMAModel},
    visitedModels::Dict{String,Dict{String,Any}},
    informationCriteriaFunction::Function,
    objectiveFunction::String = "mse",
    assertStationarity::Bool = false,
    assertInvertibility::Bool = false,
    showLogs::Bool = false,
    icOffset::Fl = 0.0,
    minConditioningObs::Int = 0,
    seasonalForm::Symbol = :multiplicative,
    initialization::Symbol = DEFAULT_INITIALIZATION,
    # R's default: `stats::arima` with `transform.pars = TRUE` parameterizes the AR part
    # through `tanh`, i.e. stationary BY CONSTRUCTION on an open domain. The MA part stays
    # free (see `invertible`), which is the other half of R's behaviour.
    stationary::Bool = true,
    stationarityMargin::AbstractFloat = DEFAULT_DOMAIN_MARGIN,
    parallel::Bool = false,
    invertible::Bool = false,
    invertibilityMargin::AbstractFloat = DEFAULT_DOMAIN_MARGIN,
    constrainedRefit::Bool = false,
    rootMargin::AbstractFloat = DEFAULT_ROOT_MARGIN,
    optimizer::Union{DataType,MOI.OptimizerWithAttributes} = Ipopt.Optimizer,
    warmStartFromBox::Bool = false,
    maxTimeSeconds::Union{Nothing,Real} = nothing,
    cvarLevel::AbstractFloat = DEFAULT_CVAR_LEVEL,
    # LAST positional, deliberately: any other position shifts `rootMargin`/`optimizer`
    # and reproduces the MethodError documented above.
    multistart::Bool = false;
    # KEYWORD, deliberately: this argument list is positional, so a new positional here
    # would shift every slot after it.
    exogDynamics::Symbol = :armax,
    penaltyTarget::Symbol = :all,
) where {Fl<:AbstractFloat}
    ModelFl = Fl
    localBestCriteria::ModelFl = Inf
    localBestModel = nothing
    toFit = filter(m -> !isFitted(m), candidateModels)
    if parallel
        Threads.@threads for model in toFit
            try
                fit!(model; objectiveFunction = objectiveFunction, minConditioningObs = minConditioningObs, seasonalForm = seasonalForm, exogDynamics = exogDynamics, penaltyTarget = penaltyTarget, initialization = initialization, stationary = stationary, stationarityMargin = stationarityMargin, invertible = invertible, invertibilityMargin = invertibilityMargin, optimizer = optimizer, warmStartFromBox = warmStartFromBox, maxTimeSeconds = maxTimeSeconds, cvarLevel = cvarLevel, multistart = multistart)
            catch e
                @warn "Parallel candidate fit failed" exception = e
            end
        end
    else
        foreach(model -> fit!(model; objectiveFunction = objectiveFunction, minConditioningObs = minConditioningObs, seasonalForm = seasonalForm, exogDynamics = exogDynamics, penaltyTarget = penaltyTarget, initialization = initialization, stationary = stationary, stationarityMargin = stationarityMargin, invertible = invertible, invertibilityMargin = invertibilityMargin, optimizer = optimizer, warmStartFromBox = warmStartFromBox, maxTimeSeconds = maxTimeSeconds, cvarLevel = cvarLevel, multistart = multistart), toFit)
    end
    for model in toFit
        isFitted(model) || continue
        # ensureAdmissible! may refit the model in place, so the information criterion
        # must be evaluated afterwards.
        admissible = ensureAdmissible!(
            model,
            assertStationarity,
            assertInvertibility,
            showLogs;
            objectiveFunction = objectiveFunction,
            minConditioningObs = minConditioningObs,
            seasonalForm = seasonalForm,
            exogDynamics = exogDynamics,
            penaltyTarget = penaltyTarget,
            initialization = initialization,
            refit = constrainedRefit,
            optimizer = optimizer,
            rootMargin = rootMargin,
            stationary = stationary,
            invertible = invertible,
        )
        criteria = informationCriteriaFunction(model; offset = icOffset)
        showLogs && @info("Fitted $(getId(model)) with $(criteria)")
        # Besides the criterion, record the cost and the shape of the candidate. That is what
        # allows attributing the total time of a search to "more candidates", "larger
        # models" or "harder solves", and it also makes the criterion fallback rate
        # observable.
        visitedModels[getId(model)] = Dict(
            "criteria" => criteria,
            "buildTimeSec" => get(model.metadata, "buildTimeSec", missing),
            "solveTimeSec" => get(model.metadata, "solveTimeSec", missing),
            "solverTimeSec" => get(model.metadata, "solverTimeSec", missing),
            "solverIterations" => get(model.metadata, "solverIterations", missing),
            "solverStatus" => get(model.metadata, "solverStatus", missing),
            "criterionFallback" => get(model.metadata, "criterionFallback", missing),
            "K" => get_hyperparameters_number(model),
            "order" => (model.p, model.d, model.q, model.P, model.D, model.Q),
            "admissible" => admissible,
        )
        if admissible && criteria < localBestCriteria
            localBestCriteria = criteria
            localBestModel = model
        end
    end
    return localBestCriteria, localBestModel
end

"""
    addNonSeasonalModels!(
        bestModel::SARIMAModel,
        candidateModels::Vector{SARIMAModel},
        visitedModels::Dict{String,Dict{String,Any}},
        maxp::Int,
        maxq::Int,
        maxOrder::Int,
        allowMean::Bool,
        allowDrift::Bool,
        fixConstant::Bool
    )

Adds non-seasonal SARIMA models to the candidate models vector based on the best SARIMA model found.

# Arguments
- `bestModel::SARIMAModel`: The best SARIMA model found so far.
- `candidateModels::Vector{SARIMAModel}`: A vector of candidate SARIMA models to add new models to.
- `visitedModels::Dict{String,Dict{String,Any}}`: A dictionary containing information about visited models.
- `maxp::Int`: The maximum autoregressive order for non-seasonal part.
- `maxq::Int`: The maximum moving average order for non-seasonal part.
- `maxOrder::Int`: The maximum order for the non-seasonal part.
- `allowMean::Bool`: Whether to include a mean term in the model.
- `allowDrift::Bool`: Whether to include a drift term in the model.
- `fixConstant::Bool`: Whether to fix the constant term.

"""
function addNonSeasonalModels!(
    bestModel::SARIMAModel,
    candidateModels::Vector{SARIMAModel},
    visitedModels::Dict{String,Dict{String,Any}},
    maxp::Int,
    maxq::Int,
    maxOrder::Int,
    allowMean::Bool,
    allowDrift::Bool,
    fixConstant::Bool,
    alpha::Union{Nothing,Float64} = nothing,
    lambda::Union{Nothing,Float64} = nothing,
)
    for p = -1:1, q = -1:1
        newp = bestModel.p + p
        newq = bestModel.q + q
        if newp < 0 || newq < 0 || newp > maxp || newq > maxq || newp + newq == 0
            continue
        end

        if newp + newq + bestModel.P + bestModel.Q > maxOrder
            continue
        end

        newModel = SARIMA(
            deepcopy(bestModel.y),
            deepcopy(bestModel.exog),
            newp,
            bestModel.d,
            newq;
            seasonality = bestModel.seasonality,
            P = bestModel.P,
            D = bestModel.D,
            Q = bestModel.Q,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda,
        )
        if !isVisited(newModel, visitedModels)
            push!(candidateModels, newModel)
            fixConstant || addChangedConstantModel!(
                newModel,
                candidateModels,
                visitedModels,
                newModel.d + newModel.D == 1,
            )
        end
    end
end

"""
    addSeasonalModels!(
        bestModel::SARIMAModel,
        candidateModels::Vector{SARIMAModel},
        visitedModels::Dict{String,Dict{String,Any}},
        maxP::Int,
        maxQ::Int,
        maxOrder::Int,
        allowMean::Bool,
        allowDrift::Bool,
        fixConstant::Bool
    )

Adds seasonal SARIMA models to the candidate models vector based on the best SARIMA model found.

# Arguments
- `bestModel::SARIMAModel`: The best SARIMA model found so far.
- `candidateModels::Vector{SARIMAModel}`: A vector of candidate SARIMA models to add new models to.
- `visitedModels::Dict{String,Dict{String,Any}}`: A dictionary containing information about visited models.
- `maxP::Int`: The maximum autoregressive order for the seasonal part.
- `maxQ::Int`: The maximum moving average order for the seasonal part.
- `maxOrder::Int`: The maximum order of the model.
- `allowMean::Bool`: Whether to include a mean term in the model.
- `allowDrift::Bool`: Whether to include a drift term in the model.
- `fixConstant::Bool`: Whether to fix the constant term.

"""
function addSeasonalModels!(
    bestModel::SARIMAModel,
    candidateModels::Vector{SARIMAModel},
    visitedModels::Dict{String,Dict{String,Any}},
    maxP::Int,
    maxQ::Int,
    maxOrder::Int,
    allowMean::Bool,
    allowDrift::Bool,
    fixConstant::Bool,
    alpha::Union{Nothing,Float64} = nothing,
    lambda::Union{Nothing,Float64} = nothing,
)
    for P = -1:1, Q = -1:1
        newP = bestModel.P + P
        newQ = bestModel.Q + Q
        modelOrder = bestModel.p + bestModel.q + newP + newQ
        (modelOrder > maxOrder) && continue

        if newP < 0 ||
           newQ < 0 ||
           newP > maxP ||
           newQ > maxQ ||
           newP + newQ == 0 ||
           newP + newQ > 2
            continue
        end

        newModel = SARIMA(
            deepcopy(bestModel.y),
            deepcopy(bestModel.exog),
            bestModel.p,
            bestModel.d,
            bestModel.q;
            seasonality = bestModel.seasonality,
            P = newP,
            D = bestModel.D,
            Q = newQ,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda,
        )
        if !isVisited(newModel, visitedModels)
            push!(candidateModels, newModel)
            fixConstant || addChangedConstantModel!(
                newModel,
                candidateModels,
                visitedModels,
                newModel.d + newModel.D == 1,
            )
        end
    end
end

"""
    addNonSeasonalAndSeasonalModels!(
        bestModel::SARIMAModel,
        candidateModels::Vector{SARIMAModel},
        visitedModels::Dict{String,Dict{String,Any}},
        maxp::Int,
        maxq::Int,
        maxP::Int,
        maxQ::Int,
        maxOrder::Int,
        allowMean::Bool,
        allowDrift::Bool,
        fixConstant::Bool
    )

Adds non-seasonal and seasonal SARIMA models variation to the candidate models vector based on the best SARIMA model found.

# Arguments
- `bestModel::SARIMAModel`: The best SARIMA model found so far.
- `candidateModels::Vector{SARIMAModel}`: A vector of candidate SARIMA models to add new models to.
- `visitedModels::Dict{String,Dict{String,Any}}`: A dictionary containing information about visited models.
- `maxp::Int`: The maximum autoregressive order for the non-seasonal part.
- `maxq::Int`: The maximum moving average order for the non-seasonal part.
- `maxP::Int`: The maximum autoregressive order for the seasonal part.
- `maxQ::Int`: The maximum moving average order for the seasonal part.
- `maxOrder::Int`: The maximum order of the model.
- `allowMean::Bool`: Whether to include a mean term in the model.
- `allowDrift::Bool`: Whether to include a drift term in the model.
- `fixConstant::Bool`: Whether to fix the constant term.
"""
function addNonSeasonalAndSeasonalModels!(
    bestModel::SARIMAModel,
    candidateModels::Vector{SARIMAModel},
    visitedModels::Dict{String,Dict{String,Any}},
    maxp::Int,
    maxq::Int,
    maxP::Int,
    maxQ::Int,
    maxOrder::Int,
    allowMean::Bool,
    allowDrift::Bool,
    fixConstant::Bool,
    alpha::Union{Nothing,Float64} = nothing,
    lambda::Union{Nothing,Float64} = nothing,
)
    for p in [-1, 1], q in [-1, 1], P in [-1, 1], Q in [-1, 1]
        newp = bestModel.p + p
        newq = bestModel.q + q
        newP = bestModel.P + P
        newQ = bestModel.Q + Q
        if newp < 0 || newq < 0 || newp > maxp || newq > maxq
            continue
        end

        if newP < 0 || newQ < 0 || newP > maxP || newQ > maxQ
            continue
        end

        if newP + newQ + newp + newq > maxOrder
            continue
        end

        newModel = SARIMA(
            deepcopy(bestModel.y),
            deepcopy(bestModel.exog),
            newp,
            bestModel.d,
            newq;
            seasonality = bestModel.seasonality,
            P = newP,
            D = bestModel.D,
            Q = newQ,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda,
        )
        if !isVisited(newModel, visitedModels)
            push!(candidateModels, newModel)
            fixConstant || addChangedConstantModel!(
                newModel,
                candidateModels,
                visitedModels,
                newModel.d + newModel.D == 1,
            )
        end
    end
end

"""
    addChangedConstantModel!(
        bestModel::SARIMAModel,
        candidateModels::Vector{SARIMAModel},
        visitedModels::Dict{String,Dict{String,Any}},
        drift::Bool = false
    )

    addChangedConstantModel!(
        bestModel::SARIMAModel,
        candidateModels::Vector{SARIMAModel},
        visitedModels::Dict{String,Dict{String,Any}},
        drift::Bool = false
    )

Adds a SARIMA model with a changed constant term to the candidate models vector based on the best SARIMA model found.

# Arguments
- `bestModel::SARIMAModel`: The best SARIMA model found so far.
- `candidateModels::Vector{SARIMAModel}`: A vector of candidate SARIMA models to add new models to.
- `visitedModels::Dict{String,Dict{String,Any}}`: A dictionary containing information about visited models.
- `drift::Bool`: Whether to change the drift term. Default is false.

"""
function addChangedConstantModel!(
    bestModel::SARIMAModel,
    candidateModels::Vector{SARIMAModel},
    visitedModels::Dict{String,Dict{String,Any}},
    drift::Bool = false,
    alpha::Union{Nothing,Float64} = nothing,
    lambda::Union{Nothing,Float64} = nothing,
)
    allowDrift = drift && !bestModel.allowDrift
    allowMean = !drift && !bestModel.allowMean
    newModel = SARIMA(
        deepcopy(bestModel.y),
        deepcopy(bestModel.exog),
        bestModel.p,
        bestModel.d,
        bestModel.q;
        seasonality = bestModel.seasonality,
        P = bestModel.P,
        D = bestModel.D,
        Q = bestModel.Q,
        allowMean = allowMean,
        allowDrift = allowDrift,
        alpha = alpha,
        lambda = lambda,
    )
    if !isVisited(newModel, visitedModels)
        push!(candidateModels, newModel)
    end
end

"""
    stepWiseSearchNaive(
        y::TimeArray,
        exog::Union{TimeArray,Nothing},
        d::Int,
        D::Int,
        seasonality::Int,
        informationCriteriaFunction::Function;
        maxp::Int=5,
        maxq::Int=5,
        maxP::Int=2,
        maxQ::Int=2,
        maxOrder::Int=5,
        objectiveFunction::String = "mse",
        assertStationarity::Bool = false,
        assertInvertibility::Bool = false,
        allowMean::Bool = true,
        allowDrift::Bool = false,
        showLogs::Bool = false,
        icOffset::Fl = 0.0
        fixConstant::Bool = false
    ) where Fl <: AbstractFloat

Performs a naive stepwise search to find the best SARIMA model based on the specified parameters.

# Arguments

- `y::TimeArray`: The time series data.
- `exog::Union{TimeArray,Nothing}`: Optional exogenous variables. If `Nothing`, no exogenous variables are used.
- `d::Int`: The degree of differencing.
- `D::Int`: The degree of seasonal differencing.
- `seasonality::Int`: The seasonality period.
- `informationCriteriaFunction::Function`: A function to calculate the information criteria for a SARIMA model.
- `maxp::Int`: The maximum autoregressive order for the non-seasonal part. Default is 5.
- `maxq::Int`: The maximum moving average order for the non-seasonal part. Default is 5.
- `maxP::Int`: The maximum autoregressive order for the seasonal part. Default is 2.
- `maxQ::Int`: The maximum moving average order for the seasonal part. Default is 2.
- `maxOrder::Int`: The maximum order of the model. Default is 5.
- `objectiveFunction::String`: The objective function to be used for fitting models. Default is "mse".
- `assertStationarity::Bool`: Whether to assert stationarity of the fitted models. Default is false.
- `assertInvertibility::Bool`: Whether to assert invertibility of the fitted models. Default is false.
- `allowMean::Bool`: Whether to include a mean term in the model. Default is true.
- `allowDrift::Bool`: Whether to include a drift term in the model. Default is false.
- `showLogs::Bool`: Whether to suppress output. Default is false.
- `icOffset::Fl`: The offset to be added to the information criteria. Default is 0.0.
- `fixConstant::Bool`: Whether to fix the constant term. Default is false.

# Returns

- `SARIMAModel`: The best SARIMA model found.
"""
function stepWiseSearchNaive(
    y::TimeArray,
    exog::Union{TimeArray,Nothing},
    d::Int,
    D::Int,
    seasonality::Int,
    informationCriteriaFunction::Function;
    maxp::Int = 5,
    maxq::Int = 5,
    maxP::Int = 2,
    maxQ::Int = 2,
    maxOrder::Int = 5,
    objectiveFunction::String = "mse",
    assertStationarity::Bool = false,
    assertInvertibility::Bool = false,
    allowMean::Bool = true,
    allowDrift::Bool = false,
    showLogs::Bool = false,
    icOffset::AbstractFloat = 0.0,
    minConditioningObs::Int = 0,
    seasonalForm::Symbol = :multiplicative,
    exogDynamics::Symbol = :armax,
    penaltyTarget::Symbol = :all,
    initialization::Symbol = DEFAULT_INITIALIZATION,
    multistart::Bool = false,
    # R's default: `stats::arima` with `transform.pars = TRUE` parameterizes the AR part
    # through `tanh`, i.e. stationary BY CONSTRUCTION on an open domain. The MA part stays
    # free (see `invertible`), which is the other half of R's behaviour.
    stationary::Bool = true,
    stationarityMargin::AbstractFloat = DEFAULT_DOMAIN_MARGIN,
    parallel::Bool = false,
    invertible::Bool = false,
    invertibilityMargin::AbstractFloat = DEFAULT_DOMAIN_MARGIN,
    constrainedRefit::Bool = false,
    # `rootMargin` must be declared here: the body forwards it to `localSearch!`, which
    # takes it as the 19th positional argument.
    rootMargin::AbstractFloat = DEFAULT_ROOT_MARGIN,
    # Ceiling on models visited, as in `stepwiseSearch`. See [`DEFAULT_NMODELS`].
    maxModels::Int = DEFAULT_NMODELS,
    optimizer::Union{DataType,MOI.OptimizerWithAttributes} = Ipopt.Optimizer,
    warmStartFromBox::Bool = false,
    maxTimeSeconds::Union{Nothing,Real} = nothing,
    cvarLevel::AbstractFloat = DEFAULT_CVAR_LEVEL,
    fixConstant::Bool = false,
    alpha::Union{Nothing,Float64} = nothing,
    lambda::Union{Nothing,Float64} = nothing,
)
    # Include initial models
    candidateModels = Vector{SARIMAModel}()
    visitedModels = Dict{String,Dict{String,Any}}()

    if seasonality == 1
        initialNonSeasonalModels!(
            candidateModels,
            y,
            exog,
            maxp,
            d,
            maxq,
            allowMean,
            allowDrift,
        )
    else
        initialSeasonalModels!(
            candidateModels,
            y,
            exog,
            maxp,
            d,
            maxq,
            maxP,
            D,
            maxQ,
            seasonality,
            allowMean,
            allowDrift,
        )
    end

    # Fit models
    bestCriteria, bestModel = localSearch!(
        candidateModels,
        visitedModels,
        informationCriteriaFunction,
        objectiveFunction,
        assertStationarity,
        assertInvertibility,
        showLogs,
        icOffset,
        minConditioningObs,
        seasonalForm,
        initialization,
        stationary,
        stationarityMargin,
        parallel,
        invertible,
        invertibilityMargin,
        constrainedRefit,
        # `rootMargin` is the 19th POSITIONAL argument of localSearch!, between
        # constrainedRefit and optimizer. It must be passed explicitly: omitting it makes
        # `optimizer` land in its slot and dispatch fails with a MethodError.
        rootMargin,
        optimizer,
        warmStartFromBox,
        maxTimeSeconds,
        cvarLevel,
        multistart;
        exogDynamics = exogDynamics,
        penaltyTarget = penaltyTarget,
    )

    if isnothing(bestModel)
        @warn "No stationary/invertible candidate found among initial models; selecting best by information criterion regardless."
        fittedModels = filter(isFitted, candidateModels)
        bestModel = argmin(
            m -> informationCriteriaFunction(m; offset = icOffset),
            fittedModels,
        )
        bestCriteria = informationCriteriaFunction(bestModel; offset = icOffset)
    end

    ITERATION_LIMIT = 100
    iterations = 1
    # Ceiling on MODELS visited, in the manner of `nmodels = 94` in `forecast::auto.arima`.
    #
    # The `ITERATION_LIMIT` above bounds hill-climb ITERATIONS, not models: each iteration
    # fits a whole neighbourhood. `stepwiseSearch`, the default method, already had this
    # ceiling under the name `maxModels`; this search did not, and the gap became visible
    # once `maxOrder` moved out of here to match R, which imposes it only in `search.arima`.
    while iterations <= ITERATION_LIMIT && length(visitedModels) < maxModels

        addNonSeasonalModels!(
            bestModel,
            candidateModels,
            visitedModels,
            maxp,
            maxq,
            maxOrder,
            allowMean,
            allowDrift,
            fixConstant,
        )
        (seasonality > 1) && addSeasonalModels!(
            bestModel,
            candidateModels,
            visitedModels,
            maxP,
            maxQ,
            maxOrder,
            allowMean,
            allowDrift,
            fixConstant,
        )
        # (seasonality > 1) && addNonSeasonalAndSeasonalModels!(bestModel, candidateModels, visitedModels, maxp, maxq, maxP, maxQ, maxOrder, allowMean, allowDrift, fixConstant)

        itBestCriteria, itBestModel = localSearch!(
            candidateModels,
            visitedModels,
            informationCriteriaFunction,
            objectiveFunction,
            assertStationarity,
            assertInvertibility,
            showLogs,
            icOffset,
            minConditioningObs,
            seasonalForm,
            initialization,
            stationary,
            stationarityMargin,
            parallel,
            invertible,
            invertibilityMargin,
            constrainedRefit,
            rootMargin,
            optimizer,
            warmStartFromBox,
            maxTimeSeconds,
            cvarLevel,
            multistart;
            exogDynamics = exogDynamics,
            penaltyTarget = penaltyTarget,
        )
        showLogs && !isnothing(itBestModel) && @info(
            "Iteration $(iterations): Best model found is $(getId(itBestModel)) with $(itBestCriteria) criteria"
        )

        (itBestCriteria > bestCriteria) && break
        bestCriteria = itBestCriteria
        bestModel = itBestModel

        iterations += 1
    end

    # `visitedModels` dies with this function, so per-candidate telemetry only escapes if it
    # travels with the selected model. It goes into the metadata (internal, no signature
    # changes) so that cost attribution — number of fits, build/solve split, distribution of
    # K, solver iterations, criterion fallback rate — is readable from outside.
    bestModel.metadata["searchTelemetry"] = visitedModels
    return bestModel
end

function newModel(
    results::Dict{String,SARIMAModel},
    p::Int,
    d::Int,
    q::Int,
    P::Int,
    D::Int,
    Q::Int,
    seasonality::Int,
    allowMean::Bool,
    allowDrift::Bool,
)
    id = "SARIMA($p,$d,$q)($P,$D,$Q s=$seasonality, c=$allowMean, drift=$allowDrift)"
    return !haskey(results, id)
end

"""
    stepwiseSearch(
        y::TimeArray,
        exog::Union{TimeArray,Nothing},
        d::Int,
        D::Int,
        seasonality::Int=1,
        informationCriteriaFunction::Function;
        startp::Int=2,
        startq::Int=2,
        startP::Int=1,
        startQ::Int=1,
        maxp::Int=5,
        maxq::Int=5,
        maxP::Int=2,
        maxQ::Int=2,
        maxOrder::Int=5,
        objectiveFunction::String = "mse",
        assertStationarity::Bool = false,
        assertInvertibility::Bool = false,
        allowMean::Bool = true,
        allowDrift::Bool = false,
        showLogs::Bool = false,
        icOffset::Fl = 0.0,
        maxModels::Int = 94
    ) where Fl <: AbstractFloat

Performs a stepwise search to find the best SARIMA model based on the specified parameters.

# Arguments

- `y::TimeArray`: The time series data.
- `exog::Union{TimeArray,Nothing}`: Optional exogenous variables. If `Nothing`, no exogenous variables are used.
- `d::Int`: The degree of differencing.
- `D::Int`: The degree of seasonal differencing.
- `seasonality::Int`: The seasonality period.
- `informationCriteriaFunction::Function`: A function to calculate the information criteria for a SARIMA model.
- `startp::Int`: The starting autoregressive order for the non-seasonal part. Default is 2.
- `startq::Int`: The starting moving average order for the non-seasonal part. Default is 2.
- `startP::Int`: The starting autoregressive order for the seasonal part. Default is 1.
- `startQ::Int`: The starting moving average order for the seasonal part. Default is 1.
- `maxp::Int`: The maximum autoregressive order for the non-seasonal part. Default is 5.
- `maxq::Int`: The maximum moving average order for the non-seasonal part. Default is 5.
- `maxP::Int`: The maximum autoregressive order for the seasonal part. Default is 2.
- `maxQ::Int`: The maximum moving average order for the seasonal part. Default is 2.
- `maxOrder::Int`: The maximum order of the model. Default is 5.
- `objectiveFunction::String`: The objective function to be used for fitting models. Default is "mse".
- `assertStationarity::Bool`: Whether to assert stationarity of the fitted models. Default is false.
- `assertInvertibility::Bool`: Whether to assert invertibility of the fitted models. Default is false.
- `allowMean::Bool`: Whether to include a mean term in the model. Default is true.
- `allowDrift::Bool`: Whether to include a drift term in the model. Default is false.
- `showLogs::Bool`: Whether to suppress output. Default is false.
- `icOffset::Fl`: The offset to be added to the information criteria. Default is 0.0.
- `maxModels::Int`: The maximum number of models to consider. Default is 94.

# Returns
- `SARIMAModel`: The best SARIMA model found.
"""
function stepwiseSearch(
    y::TimeArray,
    exog::Union{TimeArray,Nothing},
    d::Int,
    D::Int,
    seasonality::Int,
    informationCriteriaFunction::Function;
    startp::Int = 2,
    startq::Int = 2,
    startP::Int = 1,
    startQ::Int = 1,
    maxp::Int = 5,
    maxq::Int = 5,
    maxP::Int = 2,
    maxQ::Int = 2,
    maxOrder::Int = 5,
    objectiveFunction::String = "mse",
    assertStationarity::Bool = false,
    assertInvertibility::Bool = false,
    allowMean::Bool = true,
    allowDrift::Bool = false,
    showLogs::Bool = false,
    icOffset::AbstractFloat = 0.0,
    minConditioningObs::Int = 0,
    seasonalForm::Symbol = :multiplicative,
    exogDynamics::Symbol = :armax,
    penaltyTarget::Symbol = :all,
    initialization::Symbol = DEFAULT_INITIALIZATION,
    multistart::Bool = false,
    # R's default: `stats::arima` with `transform.pars = TRUE` parameterizes the AR part
    # through `tanh`, i.e. stationary BY CONSTRUCTION on an open domain. The MA part stays
    # free (see `invertible`), which is the other half of R's behaviour.
    stationary::Bool = true,
    stationarityMargin::AbstractFloat = DEFAULT_DOMAIN_MARGIN,
    invertible::Bool = false,
    invertibilityMargin::AbstractFloat = DEFAULT_DOMAIN_MARGIN,
    constrainedRefit::Bool = false,
    rootMargin::AbstractFloat = DEFAULT_ROOT_MARGIN,
    optimizer::Union{DataType,MOI.OptimizerWithAttributes} = Ipopt.Optimizer,
    warmStartFromBox::Bool = false,
    maxTimeSeconds::Union{Nothing,Real} = nothing,
    cvarLevel::AbstractFloat = DEFAULT_CVAR_LEVEL,
    maxModels::Int = DEFAULT_NMODELS,
    alpha::Union{Nothing,<:AbstractFloat} = nothing,
    lambda::Union{Nothing,<:AbstractFloat} = nothing,
    requireTermsWhenOverDifferenced::Bool = false,
    requireMAWhenDoublyDifferenced::Bool = false,
)
    constant = allowDrift || allowMean
    # With d + D >= 2 and no AR/MA term at all, the forecast is a pure extrapolation of
    # the locally fitted slope: nothing damps it, so it runs off in a straight line and,
    # on the strictly positive series of the M4, crosses zero — which pins sMAPE near its
    # 200 ceiling. Measured on the M4 monthly tail, the residual failures after every
    # other correction were exactly (0,2,0)(0,0,0) and (0,1,0)(0,1,0). This guard removes
    # that degenerate corner from the search while leaving every other order reachable.
    # Off by default: R's auto.arima has no such rule (it avoids the corner by scoring
    # candidates on the full sample instead), so enabling it is a deliberate divergence
    # from the reference implementation.
    overDifferenced = requireTermsWhenOverDifferenced && (d + D) >= 2
    # Second-differencing induces a unit MA root AT THE LAG THAT WAS DIFFERENCED. A candidate
    # with no MA term at that lag cannot represent it and compensates with AR persistence,
    # which explodes once re-integrated twice. Measured on the M4 monthly (144 series with
    # d = 2, D = 0): every one of the seven worst offenders has q = 0 — including (1,0,0,0),
    # which the "high AR order" reading does not cover. The seasonal counterpart (Q when
    # D >= 2) is vacuous on monthly M4 but stated for symmetry.
    #
    # The guard is DIMENSIONAL, not aggregate: `q + Q >= 1` under `d + D >= 2` was rejected
    # ex ante, because a seasonal MA at lag s cannot damp a unit root at lag 1 — (5,0,1,1)
    # has Q = 1 and blew up anyway. d = D = 1 stays untouched by construction: with one
    # difference in each dimension, theory does not say which one carries the insurance.
    #
    # This constrains representability, not magnitude: a genuinely q = 0 process stays nested
    # at theta = 0, costing one AICc parameter.
    requireNonSeasonalMA = requireMAWhenDoublyDifferenced && d >= 2
    requireSeasonalMA = requireMAWhenDoublyDifferenced && D >= 2
    orderAllowed(np::Int, nq::Int, nP::Int, nQ::Int) =
        !(overDifferenced && (np + nq + nP + nQ) == 0) &&
        !(requireNonSeasonalMA && nq == 0) &&
        !(requireSeasonalMA && nQ == 0)
    # The constant term must live in the slot that matches the differencing order:
    # mean when d + D == 0 (allowMean), drift when d + D == 1 (allowDrift). Using the
    # wrong slot adds a term that vanishes after differencing (not identifiable) while
    # the search never explores the identifiable one. curMean/curDrift track the constant
    # setting of the current best model (kept in sync by the toggle move below).
    useDrift = allowDrift
    curMean = allowMean
    curDrift = allowDrift
    p = min(startp, maxp)
    q = min(startq, maxq)
    P = min(startP, maxP)
    Q = min(startQ, maxQ)
    results = Dict{String,SARIMAModel}()

    bestModel = SARIMA(
        y,
        exog,
        p,
        d,
        q;
        P = P,
        D = D,
        Q = Q,
        seasonality = seasonality,
        allowMean = allowMean,
        allowDrift = allowDrift,
        alpha = alpha,
        lambda = lambda
    )
    fit!(bestModel; objectiveFunction = objectiveFunction, minConditioningObs = minConditioningObs, seasonalForm = seasonalForm, exogDynamics = exogDynamics, penaltyTarget = penaltyTarget, initialization = initialization, stationary = stationary, stationarityMargin = stationarityMargin, invertible = invertible, invertibilityMargin = invertibilityMargin, optimizer = optimizer, warmStartFromBox = warmStartFromBox, maxTimeSeconds = maxTimeSeconds, cvarLevel = cvarLevel, multistart = multistart)
    showLogs && @info(
        "Fitted $(getId(bestModel)) with $(informationCriteriaFunction(bestModel; offset=icOffset)) criteria"
    )

    results[getId(bestModel)] = bestModel

    considerModel = ensureAdmissible!(
        bestModel,
        assertStationarity,
        assertInvertibility,
        showLogs;
        objectiveFunction = objectiveFunction,
        minConditioningObs = minConditioningObs,
        seasonalForm = seasonalForm,
        exogDynamics = exogDynamics,
        penaltyTarget = penaltyTarget,
        initialization = initialization,
        refit = constrainedRefit,
        optimizer = optimizer,
    )

    # The "null" model is both a candidate and the safety net: the line below adopts it
    # unconditionally when the first candidate turns out inadmissible. Guarding only the
    # *adoption* therefore left the degenerate order reachable through that fallback,
    # which is why the guard measured as completely inert on the M4 tail. Under the guard
    # the safety net itself must carry a term, so the minimal admissible model takes its
    # place. If no lag is available at all the guard simply cannot be honoured, and the
    # plain null model stands.
    # Under the MA guard the safety net must carry the MA term itself: preferring an AR lag
    # here reintroduces exactly the (1,0,0,0) corner the guard exists to remove, through the
    # one path that adopts unconditionally.
    nullp, nullq = 0, 0
    if requireNonSeasonalMA && maxq > 0
        nullq = 1
    elseif overDifferenced
        if maxp > 0
            nullp = 1
        elseif maxq > 0
            nullq = 1
        end
    end
    fitModel = SARIMA(
        y,
        exog,
        nullp,
        d,
        nullq;
        P = 0,
        D = D,
        Q = 0,
        seasonality = seasonality,
        allowMean = allowMean,
        allowDrift = allowDrift,
        alpha = alpha,
        lambda = lambda
    )
    fit!(fitModel; objectiveFunction = objectiveFunction, minConditioningObs = minConditioningObs, seasonalForm = seasonalForm, exogDynamics = exogDynamics, penaltyTarget = penaltyTarget, initialization = initialization, stationary = stationary, stationarityMargin = stationarityMargin, invertible = invertible, invertibilityMargin = invertibilityMargin, optimizer = optimizer, warmStartFromBox = warmStartFromBox, maxTimeSeconds = maxTimeSeconds, cvarLevel = cvarLevel, multistart = multistart)
    showLogs && @info(
        "Fitted $(getId(fitModel)) with $(informationCriteriaFunction(fitModel; offset=icOffset)) criteria"
    )
    bestModel = considerModel ? bestModel : fitModel

    results[getId(fitModel)] = fitModel

    considerModel = ensureAdmissible!(
        fitModel,
        assertStationarity,
        assertInvertibility,
        showLogs;
        objectiveFunction = objectiveFunction,
        minConditioningObs = minConditioningObs,
        seasonalForm = seasonalForm,
        exogDynamics = exogDynamics,
        penaltyTarget = penaltyTarget,
        initialization = initialization,
        refit = constrainedRefit,
        optimizer = optimizer,
    )

    # The null model keeps its role as the safety net above (it is the fallback when the
    # first candidate is inadmissible), but under the guard it must not be *adopted* as
    # the best: (0,d,0)(0,D,0) with d + D >= 2 is precisely the degenerate corner.
    if considerModel &&
       orderAllowed(nullp, nullq, 0, 0) &&
       informationCriteriaFunction(bestModel; offset = icOffset) >
       informationCriteriaFunction(fitModel; offset = icOffset)
        bestModel = fitModel
        p = nullp
        q = nullq
        P = 0
        Q = 0
    end

    k = 2

    if (maxp > 0 || maxP > 0)
        auxp = (maxp > 0) ? 1 : 0
        auxP = (maxP > 0 && seasonality > 1) ? 1 : 0
        fitModel = SARIMA(
            y,
            exog,
            auxp,
            d,
            0;
            P = auxP,
            D = D,
            Q = 0,
            seasonality = seasonality,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda
        )
        fit!(fitModel; objectiveFunction = objectiveFunction, minConditioningObs = minConditioningObs, seasonalForm = seasonalForm, exogDynamics = exogDynamics, penaltyTarget = penaltyTarget, initialization = initialization, stationary = stationary, stationarityMargin = stationarityMargin, invertible = invertible, invertibilityMargin = invertibilityMargin, optimizer = optimizer, warmStartFromBox = warmStartFromBox, maxTimeSeconds = maxTimeSeconds, cvarLevel = cvarLevel, multistart = multistart)
        showLogs && @info(
            "Fitted $(getId(fitModel)) with $(informationCriteriaFunction(fitModel; offset=icOffset)) criteria"
        )
        considerModel = ensureAdmissible!(
            fitModel,
            assertStationarity,
            assertInvertibility,
            showLogs;
            objectiveFunction = objectiveFunction,
            minConditioningObs = minConditioningObs,
            seasonalForm = seasonalForm,
            exogDynamics = exogDynamics,
            penaltyTarget = penaltyTarget,
            initialization = initialization,
            refit = constrainedRefit,
            optimizer = optimizer,
            rootMargin = rootMargin,
            stationary = stationary,
            invertible = invertible,
        )
        results[getId(fitModel)] = fitModel
        if considerModel &&
           orderAllowed(auxp, 0, auxP, 0) &&
           informationCriteriaFunction(fitModel; offset = icOffset) <
           informationCriteriaFunction(bestModel; offset = icOffset)
            bestModel = fitModel
            p = auxp
            q = 0
            P = auxP
            Q = 0
        end
        k += 1
    end

    if (maxq > 0 || maxQ > 0)
        auxq = (maxq > 0) ? 1 : 0
        auxQ = (maxQ > 0 && seasonality > 1) ? 1 : 0
        fitModel = SARIMA(
            y,
            exog,
            0,
            d,
            auxq;
            P = 0,
            D = D,
            Q = auxQ,
            seasonality = seasonality,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda
        )
        fit!(fitModel; objectiveFunction = objectiveFunction, minConditioningObs = minConditioningObs, seasonalForm = seasonalForm, exogDynamics = exogDynamics, penaltyTarget = penaltyTarget, initialization = initialization, stationary = stationary, stationarityMargin = stationarityMargin, invertible = invertible, invertibilityMargin = invertibilityMargin, optimizer = optimizer, warmStartFromBox = warmStartFromBox, maxTimeSeconds = maxTimeSeconds, cvarLevel = cvarLevel, multistart = multistart)
        showLogs && @info(
            "Fitted $(getId(fitModel)) with $(informationCriteriaFunction(fitModel; offset=icOffset)) criteria"
        )
        considerModel = ensureAdmissible!(
            fitModel,
            assertStationarity,
            assertInvertibility,
            showLogs;
            objectiveFunction = objectiveFunction,
            minConditioningObs = minConditioningObs,
            seasonalForm = seasonalForm,
            exogDynamics = exogDynamics,
            penaltyTarget = penaltyTarget,
            initialization = initialization,
            refit = constrainedRefit,
            optimizer = optimizer,
            rootMargin = rootMargin,
            stationary = stationary,
            invertible = invertible,
        )
        results[getId(fitModel)] = fitModel
        if considerModel &&
           orderAllowed(0, auxq, 0, auxQ) &&
           informationCriteriaFunction(fitModel; offset = icOffset) <
           informationCriteriaFunction(bestModel; offset = icOffset)
            bestModel = fitModel
            p = 0
            q = auxq
            P = 0
            Q = auxQ
        end
        k += 1
    end

    if (allowMean || allowDrift)
        fitModel = SARIMA(
            y,
            exog,
            0,
            d,
            0;
            P = 0,
            D = D,
            Q = 0,
            seasonality = seasonality,
            allowMean = false,
            allowDrift = false,
            alpha = alpha,
            lambda = lambda
        )
        fit!(fitModel; objectiveFunction = objectiveFunction, minConditioningObs = minConditioningObs, seasonalForm = seasonalForm, exogDynamics = exogDynamics, penaltyTarget = penaltyTarget, initialization = initialization, stationary = stationary, stationarityMargin = stationarityMargin, invertible = invertible, invertibilityMargin = invertibilityMargin, optimizer = optimizer, warmStartFromBox = warmStartFromBox, maxTimeSeconds = maxTimeSeconds, cvarLevel = cvarLevel, multistart = multistart)
        showLogs && @info(
            "Fitted $(getId(fitModel)) with $(informationCriteriaFunction(fitModel; offset=icOffset)) criteria"
        )
        considerModel = ensureAdmissible!(
            fitModel,
            assertStationarity,
            assertInvertibility,
            showLogs;
            objectiveFunction = objectiveFunction,
            minConditioningObs = minConditioningObs,
            seasonalForm = seasonalForm,
            exogDynamics = exogDynamics,
            penaltyTarget = penaltyTarget,
            initialization = initialization,
            refit = constrainedRefit,
            optimizer = optimizer,
            rootMargin = rootMargin,
            stationary = stationary,
            invertible = invertible,
        )
        results[getId(fitModel)] = fitModel
        if considerModel &&
           orderAllowed(0, 0, 0, 0) &&
           informationCriteriaFunction(fitModel; offset = icOffset) <
           informationCriteriaFunction(bestModel; offset = icOffset)
            bestModel = fitModel
            p = 0
            q = 0
            P = 0
            Q = 0
        end
        k += 1
    end

    # Try one neighbour specification; when it improves the information
    # criterion and passes the admissibility checks, adopt it as the new best.
    function tryCandidate!(newp::Int, newq::Int, newP::Int, newQ::Int, cAllowMean::Bool, cAllowDrift::Bool)
        orderAllowed(newp, newq, newP, newQ) || return false
        newModel(results, newp, d, newq, newP, D, newQ, seasonality, cAllowMean, cAllowDrift) ||
            return false
        k += 1
        k > maxModels && return false
        fitModel = SARIMA(
            y,
            exog,
            newp,
            d,
            newq;
            P = newP,
            D = D,
            Q = newQ,
            seasonality = seasonality,
            allowMean = cAllowMean,
            allowDrift = cAllowDrift,
            alpha = alpha,
            lambda = lambda,
        )
        fit!(fitModel; objectiveFunction = objectiveFunction, minConditioningObs = minConditioningObs, seasonalForm = seasonalForm, exogDynamics = exogDynamics, penaltyTarget = penaltyTarget, initialization = initialization, stationary = stationary, stationarityMargin = stationarityMargin, invertible = invertible, invertibilityMargin = invertibilityMargin, optimizer = optimizer, warmStartFromBox = warmStartFromBox, maxTimeSeconds = maxTimeSeconds, cvarLevel = cvarLevel, multistart = multistart)
        showLogs && @info(
            "Fitted $(getId(fitModel)) with $(informationCriteriaFunction(fitModel; offset=icOffset)) criteria"
        )
        considerModel = ensureAdmissible!(
            fitModel,
            assertStationarity,
            assertInvertibility,
            showLogs;
            objectiveFunction = objectiveFunction,
            minConditioningObs = minConditioningObs,
            seasonalForm = seasonalForm,
            exogDynamics = exogDynamics,
            penaltyTarget = penaltyTarget,
            initialization = initialization,
            refit = constrainedRefit,
            optimizer = optimizer,
            rootMargin = rootMargin,
            stationary = stationary,
            invertible = invertible,
        )
        results[getId(fitModel)] = fitModel
        if considerModel &&
           informationCriteriaFunction(fitModel; offset = icOffset) <
           informationCriteriaFunction(bestModel; offset = icOffset)
            bestModel = fitModel
            return true
        end
        return false
    end

    # Hyndman-Khandakar neighbourhood scan. The move order matches the previous
    # (unrolled) implementation: seasonal singles, seasonal pairs, non-seasonal
    # singles, non-seasonal pairs; on the first improving move the scan restarts.
    moves = (
        (0, 0, -1, 0),
        (0, 0, 0, -1),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (0, 0, -1, -1),
        (0, 0, -1, 1),
        (0, 0, 1, -1),
        (0, 0, 1, 1),
        (-1, 0, 0, 0),
        (0, -1, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (-1, -1, 0, 0),
        (-1, 1, 0, 0),
        (1, -1, 0, 0),
        (1, 1, 0, 0),
    )

    startk = 0
    while (startk < k && k < maxModels)
        startk = k
        improved = false
        for (dp, dq, dP, dQ) in moves
            newp, newq, newP, newQ = p + dp, q + dq, P + dP, Q + dQ
            (0 <= newp <= maxp && 0 <= newq <= maxq && 0 <= newP <= maxP && 0 <= newQ <= maxQ) ||
                continue
            if tryCandidate!(newp, newq, newP, newQ, curMean, curDrift)
                p, q, P, Q = newp, newq, newP, newQ
                improved = true
                break
            end
        end
        improved && continue

        # Toggle the constant in the slot that matches the differencing order
        # (drift when d + D == 1, mean when d + D == 0), as in auto.arima.
        if allowDrift || allowMean
            newMean = useDrift ? false : !constant
            newDrift = useDrift ? !constant : false
            if tryCandidate!(p, q, P, Q, newMean, newDrift)
                constant = !constant
                curMean = newMean
                curDrift = newDrift
            end
        end
    end

    # `results` holds the fitted models, whose metadata already carries the cost measured
    # in `fit!`. Summarizing here, rather than attaching the whole of `results`, avoids a
    # reference cycle — `bestModel` is inside `results` — and keeps the metadata
    # serializable.
    bestModel.metadata["searchTelemetry"] = summarizeSearchCost(results)
    return bestModel
end

"""
    summarizeSearchCost(results) -> Dict{String,Dict{String,Any}}

Extracts, from a dictionary of fitted candidate models, the per-candidate summary that allows
ATTRIBUTING the cost of a search: `number of fits x cost per fit`, with the cost split into
building the JuMP problem and solving it, plus the shape of the candidate (`K`, order) and its
numerical difficulty (solver iterations, status).

Without it the three possible causes of a time regression — more candidates, larger models, or
harder solves — cannot be told apart, and they have different remedies. The criterion fallback
rate (`criterionFallback`) comes out of the same summary.
"""
function summarizeSearchCost(results::Dict{String,SARIMAModel})
    summary = Dict{String,Dict{String,Any}}()
    for (id, m) in results
        md = m.metadata
        summary[id] = Dict{String,Any}(
            # ...Total is the sum over ALL fits of the candidate (warm start does up to 3,
            # and `ensureAdmissible!` may refit on top). The unsuffixed fields hold only the
            # last fit and under-report on those paths.
            "buildTimeSec" => get(md, "buildTimeSecTotal", get(md, "buildTimeSec", missing)),
            "solveTimeSec" => get(md, "solveTimeSecTotal", get(md, "solveTimeSec", missing)),
            "fitCount" => get(md, "fitCount", missing),
            "solverTimeSec" => get(md, "solverTimeSec", missing),
            "solverIterations" => get(md, "solverIterations", missing),
            "solverStatus" => get(md, "solverStatus", missing),
            "criterionFallback" => get(md, "criterionFallback", missing),
            "K" => get_hyperparameters_number(m),
            "order" => (m.p, m.d, m.q, m.P, m.D, m.Q),
        )
    end
    return summary
end

"""
    gridSearch(
        y::TimeArray,
        exog::Union{TimeArray,Nothing},
        d::Int,
        D::Int,
        seasonality::Int,
        informationCriteriaFunction::Function;
        maxp::Int=5,
        maxq::Int=5,
        maxP::Int=2,
        maxQ::Int=2,
        maxOrder::Int=5,
        objectiveFunction::String = "mse",
        assertStationarity::Bool = false,
        assertInvertibility::Bool = false,
        allowMean::Bool = false,
        allowDrift::Bool = false,
        showLogs::Bool = false,
        icOffset::Fl = 0.0
    ) where Fl <: AbstractFloat

Performs a grid search to find the best SARIMA model based on the specified parameters.

# Arguments

- `y::TimeArray`: The time series data.
- `exog::Union{TimeArray,Nothing}`: Optional exogenous variables. If `Nothing`, no exogenous variables are used.
- `d::Int`: The degree of differencing.
- `D::Int`: The degree of seasonal differencing.
- `seasonality::Int`: The seasonality period.
- `informationCriteriaFunction::Function`: A function to calculate the information criteria for a SARIMA model.
- `maxp::Int`: The maximum autoregressive order for the non-seasonal part. Default is 5.
- `maxq::Int`: The maximum moving average order for the non-seasonal part. Default is 5.
- `maxP::Int`: The maximum autoregressive order for the seasonal part. Default is 2.
- `maxQ::Int`: The maximum moving average order for the seasonal part. Default is 2.
- `maxOrder::Int`: The maximum order of the model. Default is 5.
- `objectiveFunction::String`: The objective function to be used for fitting models. Default is "mse".
- `assertStationarity::Bool`: Whether to assert stationarity of the fitted models. Default is false.
- `assertInvertibility::Bool`: Whether to assert invertibility of the fitted models. Default is false.
- `allowMean::Bool`: Whether to include a mean term in the model. Default is false.
- `allowDrift::Bool`: Whether to include a drift term in the model. Default is false.
- `showLogs::Bool`: Whether to suppress output. Default is false.
- `icOffset::Fl`: The offset to be added to the information criteria. Default is 0.0.

# Returns
- `SARIMAModel`: The best SARIMA model found.
"""
function gridSearch(
    y::TimeArray,
    exog::Union{TimeArray,Nothing},
    d::Int,
    D::Int,
    seasonality::Int,
    informationCriteriaFunction::Function;
    maxp::Int = 5,
    maxq::Int = 5,
    maxP::Int = 2,
    maxQ::Int = 2,
    maxOrder::Int = 5,
    objectiveFunction::String = "mse",
    assertStationarity::Bool = false,
    assertInvertibility::Bool = false,
    allowMean::Bool = false,
    allowDrift::Bool = false,
    showLogs::Bool = false,
    icOffset::AbstractFloat = 0.0,
    minConditioningObs::Int = 0,
    seasonalForm::Symbol = :multiplicative,
    exogDynamics::Symbol = :armax,
    penaltyTarget::Symbol = :all,
    initialization::Symbol = DEFAULT_INITIALIZATION,
    multistart::Bool = false,
    # R's default: `stats::arima` with `transform.pars = TRUE` parameterizes the AR part
    # through `tanh`, i.e. stationary BY CONSTRUCTION on an open domain. The MA part stays
    # free (see `invertible`), which is the other half of R's behaviour.
    stationary::Bool = true,
    stationarityMargin::AbstractFloat = DEFAULT_DOMAIN_MARGIN,
    parallel::Bool = false,
    invertible::Bool = false,
    invertibilityMargin::AbstractFloat = DEFAULT_DOMAIN_MARGIN,
    constrainedRefit::Bool = false,
    rootMargin::AbstractFloat = DEFAULT_ROOT_MARGIN,
    optimizer::Union{DataType,MOI.OptimizerWithAttributes} = Ipopt.Optimizer,
    warmStartFromBox::Bool = false,
    maxTimeSeconds::Union{Nothing,Real} = nothing,
    cvarLevel::AbstractFloat = DEFAULT_CVAR_LEVEL,
    alpha::Union{Nothing,Float64} = nothing,
    lambda::Union{Nothing,Float64} = nothing,
)
    maxK = (allowMean || allowDrift) ? 1 : 0
    candidates = SARIMAModel[]
    push!(
        candidates,
        SARIMA(
            y,
            exog,
            0,
            d,
            0;
            P = 0,
            D = D,
            Q = 0,
            seasonality = seasonality,
            allowMean = allowMean,
            allowDrift = allowDrift,
            alpha = alpha,
            lambda = lambda,
            # WITHOUT `rootMargin`: it is a REJECTION margin, a property of selection rather
            # than of the model, and the `SARIMA` constructor has no such keyword. The
            # consumer is `ensureAdmissible!` further below, which receives it correctly.
        ),
    )
    for p = 0:maxp, q = 0:maxq, P = 0:maxP, Q = 0:maxQ, kc = 0:maxK
        p + q + P + Q > maxOrder && continue
        push!(
            candidates,
            SARIMA(
                y,
                exog,
                p,
                d,
                q;
                P = P,
                D = D,
                Q = Q,
                seasonality = seasonality,
                allowMean = (kc == 1),
                allowDrift = false,
                alpha = alpha,
                lambda = lambda,
            ),
        )
    end

    fitOne!(m) = fit!(m; objectiveFunction = objectiveFunction, minConditioningObs = minConditioningObs, seasonalForm = seasonalForm, exogDynamics = exogDynamics, penaltyTarget = penaltyTarget, initialization = initialization, stationary = stationary, stationarityMargin = stationarityMargin, invertible = invertible, invertibilityMargin = invertibilityMargin, optimizer = optimizer, warmStartFromBox = warmStartFromBox, maxTimeSeconds = maxTimeSeconds, cvarLevel = cvarLevel, multistart = multistart)
    if parallel
        Threads.@threads for m in candidates
            try
                fitOne!(m)
            catch e
                @warn "Parallel candidate fit failed" exception = e
            end
        end
    else
        foreach(fitOne!, candidates)
    end

    bestModel = nothing
    bestIC = Inf
    for m in candidates
        isFitted(m) || continue
        considerModel = ensureAdmissible!(
            m,
            assertStationarity,
            assertInvertibility,
            showLogs;
            objectiveFunction = objectiveFunction,
            minConditioningObs = minConditioningObs,
            seasonalForm = seasonalForm,
            exogDynamics = exogDynamics,
            penaltyTarget = penaltyTarget,
            initialization = initialization,
            refit = constrainedRefit,
            optimizer = optimizer,
            rootMargin = rootMargin,
            stationary = stationary,
            invertible = invertible,
        )
        ic = informationCriteriaFunction(m; offset = icOffset)
        showLogs && @info("Fitted $(getId(m)) with $(ic) criteria")
        if considerModel && ic < bestIC
            bestModel = m
            bestIC = ic
        end
    end
    isnothing(bestModel) && (bestModel = candidates[1])
    return bestModel
end

"""
    maToReflectionExpr(θ, q)

Reflection coefficients of the MA block as JuMP EXPRESSIONS, through the inverse
Levinson-Durbin recursion — the same one as [`maToReflection`](@ref), written for decision
variables rather than numbers. It exists so that the MA log-determinant term can be assembled
without requiring `invertible = true`: the invertible parameterization creates the `kappa` as
bounded variables, but the determinant needs only their VALUE, which is a rational function of
the `theta`.

Returns `nothing` when the order is zero. The divisions by `1 - kappa^2` are the same
denominators as in the numerical version; there is no zero guard here because the determinant
term itself diverges there, which is the intended behaviour — it repels the boundary instead
of needing a constraint that excludes it.
"""
function maToReflectionExpr(θ, q::Int)
    q == 0 && return nothing
    a = Any[θ[i] for i = 1:q]
    κ = Vector{Any}(undef, q)
    for m = q:-1:1
        κ[m] = a[m]
        if m > 1
            d = 1 - κ[m]^2
            aprev = Vector{Any}(undef, m - 1)
            for i = 1:(m-1)
                aprev[i] = (a[i] - κ[m] * a[m-i]) / d
            end
            for i = 1:(m-1)
                a[i] = aprev[i]
            end
        end
    end
    κ
end
