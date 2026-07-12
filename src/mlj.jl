# Minimal MLJ integration (deterministic forecaster).
#
# Design notes: MLJ has no first-class time-series forecasting API, so the
# wrapper adopts the pragmatic convention used by other forecasters:
# `fit(spec, verbosity, X, y)` trains on the target vector `y` (equally spaced;
# `X` is currently ignored — exogenous support via MLJ is not yet wired), and
# `predict(spec, fitresult, Xnew)` forecasts `nrows(Xnew)` steps ahead.

"""
    SARIMAForecaster(; p = 0, d = 0, q = 0, P = 0, D = 0, Q = 0, seasonality = 1,
                     allowMean = true, allowDrift = false,
                     objectiveFunction = "mse", seasonalForm = :multiplicative,
                     initialization = :zeroed)

MLJ-compatible deterministic forecaster wrapping [`SARIMA`](@ref)/[`fit!`](@ref).
The target `y` is a real vector sampled at equal intervals; features `X` are
ignored at present (exogenous variables via MLJ are not yet supported — use the
native `SARIMA`/`auto` API for SARIMAX). `predict(mach, Xnew)` returns the
`nrows(Xnew)`-step-ahead forecast.
"""
MLJModelInterface.@mlj_model mutable struct SARIMAForecaster <:
                                            MLJModelInterface.Deterministic
    p::Int = 0::(_ >= 0)
    d::Int = 0::(_ >= 0)
    q::Int = 0::(_ >= 0)
    P::Int = 0::(_ >= 0)
    D::Int = 0::(_ >= 0)
    Q::Int = 0::(_ >= 0)
    seasonality::Int = 1::(_ >= 1)
    allowMean::Bool = true
    allowDrift::Bool = false
    objectiveFunction::String = "mse"
    seasonalForm::Symbol = :multiplicative
    initialization::Symbol = :zeroed
end

function MLJModelInterface.fit(spec::SARIMAForecaster, verbosity::Int, X, y)
    yVector = collect(float.(y))
    n = length(yVector)
    syntheticDates = Date(2000, 1, 1) .+ Day.(0:n-1)
    series = TimeArray(syntheticDates, yVector)
    model = SARIMA(
        series,
        spec.p,
        spec.d,
        spec.q;
        P = spec.P,
        D = spec.D,
        Q = spec.Q,
        seasonality = spec.seasonality,
        allowMean = spec.allowMean,
        allowDrift = spec.allowDrift,
    )
    fit!(
        model;
        objectiveFunction = spec.objectiveFunction,
        seasonalForm = spec.seasonalForm,
        initialization = spec.initialization,
        silent = verbosity <= 0,
    )
    cache = nothing
    report = (
        aic = aic(model),
        aicc = aicc(model),
        bic = bic(model),
        loglik = loglike(model),
        coefficients = StatsAPI.coef(model),
        coefficient_names = StatsAPI.coefnames(model),
    )
    return model, cache, report
end

function MLJModelInterface.predict(::SARIMAForecaster, model::SARIMAModel, Xnew)
    # MLJModelInterface.nrows requires the full MLJ data front-end; count rows
    # with Tables directly so the light interface works standalone. An integer
    # Xnew is accepted as an explicit horizon.
    stepsAhead = if Xnew isa Integer
        Int(Xnew)
    elseif Tables.istable(Xnew)
        cols = Tables.columns(Xnew)
        names = Tables.columnnames(cols)
        isempty(names) ? 0 : length(Tables.getcolumn(cols, names[1]))
    else
        length(Xnew)
    end
    return predict(model, stepsAhead)
end

MLJModelInterface.metadata_pkg(
    SARIMAForecaster;
    package_name = "Sarimax",
    package_uuid = "32cb7113-d955-45ac-bb89-2e20d5d4d2f9",
    package_url = "https://github.com/LAMPSPUC/Sarimax.jl",
    is_pure_julia = false,
    package_license = "MIT",
)

MLJModelInterface.metadata_model(
    SARIMAForecaster;
    input_scitype = MLJModelInterface.Table,
    target_scitype = AbstractVector{<:MLJModelInterface.Continuous},
    load_path = "Sarimax.SARIMAForecaster",
)
