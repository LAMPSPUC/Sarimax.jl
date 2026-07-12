# Rolling-origin (expanding window) temporal cross-validation.

"""
    cross_validation(y::TimeArray; initialTrainSize, stepsAhead = 1, step = 1,
                     fitFunction = train -> auto(train), showLogs = false)

Rolling-origin evaluation with an expanding window (Hyndman's `tsCV` scheme):
for each origin `o = initialTrainSize, initialTrainSize+step, …` a model is
fitted on `y[1:o]` by `fitFunction` and forecast `stepsAhead` periods; the
out-of-sample errors `y[o+h] − ŷ[o+h]` are collected.

`fitFunction` receives the training `TimeArray` and must return a fitted model
supporting `predict!` (e.g. `train -> auto(train; seasonality = 12)`, or a
closure that fits a fixed specification). Exogenous variables can be handled by
closing over them and slicing inside `fitFunction`.

# Returns
A named tuple:
- `errors::Matrix`  — `(origin, horizon)` forecast errors (`NaN` where a fit failed);
- `origins::Vector` — the training sizes used;
- `mae`, `rmse`     — per-horizon aggregates over origins (skipping `NaN`s).

# Example
```julia
cv = cross_validation(y; initialTrainSize = 100, stepsAhead = 12,
                      fitFunction = train -> auto(train; seasonality = 12))
cv.rmse   # 12-vector: RMSE by horizon
```
"""
function cross_validation(
    y::TimeArray;
    initialTrainSize::Int,
    stepsAhead::Int = 1,
    step::Int = 1,
    fitFunction::Function = train -> auto(train),
    showLogs::Bool = false,
)
    n = length(y)
    initialTrainSize >= 1 || throw(ArgumentError("initialTrainSize must be positive"))
    initialTrainSize + stepsAhead <= n ||
        throw(ArgumentError("Series too short for the requested initialTrainSize and stepsAhead"))
    step >= 1 || throw(ArgumentError("step must be positive"))

    origins = collect(initialTrainSize:step:(n-stepsAhead))
    errors = fill(NaN, length(origins), stepsAhead)

    for (oi, o) in enumerate(origins)
        train = TimeArray(timestamp(y)[1:o], values(y)[1:o])
        try
            model = fitFunction(train)
            predict!(model; stepsAhead = stepsAhead)
            forecastValues = values(model.forecast)[:, 1]
            actual = values(y)[o+1:o+stepsAhead]
            errors[oi, :] = actual .- forecastValues
            showLogs && @info "cross_validation origin $o done"
        catch e
            showLogs && @warn "cross_validation origin $o failed" exception = e
        end
    end

    mae = [mean(abs.(filter(!isnan, errors[:, h]))) for h = 1:stepsAhead]
    rmse = [sqrt(mean(abs2.(filter(!isnan, errors[:, h])))) for h = 1:stepsAhead]
    return (errors = errors, origins = origins, mae = mae, rmse = rmse)
end
