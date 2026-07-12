# Ecosystem integrations: Tables.jl input and a Plots.jl recipe (via RecipesBase).

"""
    load_dataset(table; timestampColumn = :date, showLogs = false)

Load any Tables.jl-compatible source (named tuple of vectors, CSV.File,
Arrow table, …) as a `TimeArray`. If `timestampColumn` is present it is used
as the time index; otherwise a synthetic index is created (see the `DataFrame`
method).
"""
function load_dataset(table; timestampColumn::Symbol = :date, showLogs::Bool = false)
    Tables.istable(table) || throw(
        ArgumentError(
            "Input is neither a known dataset nor a Tables.jl-compatible table",
        ),
    )
    df = DataFrame(table)
    if timestampColumn != :date && timestampColumn in propertynames(df)
        DataFrames.rename!(df, timestampColumn => :date)
    end
    return load_dataset(df, showLogs)
end

# Plots.jl recipe: plot(model) draws the observed series, the in-sample fit and,
# when available, the forecast with its confidence band.
RecipesBase.@recipe function plotSARIMA(model::SARIMAModel)
    legend --> :topleft

    RecipesBase.@series begin
        label --> "observed"
        timestamp(model.y), values(model.y)
    end

    if !isnothing(model.fitInSample)
        RecipesBase.@series begin
            label --> "fitted"
            linestyle --> :dash
            timestamp(model.fitInSample), values(model.fitInSample)
        end
    end

    if !isnothing(model.forecast)
        forecastTimestamps = timestamp(model.forecast)
        forecastValues = values(model.forecast)
        RecipesBase.@series begin
            label --> "forecast"
            if size(forecastValues, 2) == 3
                # columns: forecast, lower, upper (from displayConfidenceIntervals)
                ribbon := (
                    forecastValues[:, 1] .- forecastValues[:, 2],
                    forecastValues[:, 3] .- forecastValues[:, 1],
                )
            end
            forecastTimestamps, forecastValues[:, 1]
        end
    end
end
