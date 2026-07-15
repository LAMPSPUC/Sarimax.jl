# Shared helpers for SARIMAX.jl benchmark scripts.
# JSON Lines raw records; no fabricated values; failures recorded as status="failed".

using Sarimax
using TimeSeries, DataFrames, Dates, Random, Statistics, JSON

const RAW = joinpath(@__DIR__, "..", "results", "raw")

"Append one run record (a Dict) as a JSON line."
function append_record(path::AbstractString, rec::AbstractDict)
    open(path, "a") do io
        println(io, JSON.json(rec))
    end
end

"Safely extract fitted coefficients from a SARIMAModel into a plain Dict."
function extract_estimates(m)
    est = Dict{String,Any}()
    for (name, f) in (("phi", :ϕ), ("theta", :θ), ("Phi", :Φ), ("Theta", :Θ),
                      ("exog", :exogCoefficients))
        v = getfield(m, f)
        est[name] = v === nothing ? nothing : collect(v)
    end
    est["mean"] = m.c
    est["trend"] = m.trend
    est["sigma2"] = m.σ²
    return est
end

"Try metric functions, returning nothing on failure (definitions may not align)."
trymetric(f, m) = try f(m) catch; nothing end

"Build a TimeArray from a numeric vector with a monthly date index (for simulation)."
function as_monthly(values::AbstractVector; start::Date = Date(2000, 1, 1))
    ts = [start + Month(i - 1) for i in 1:length(values)]
    return TimeArray(ts, collect(float.(values)), ["value"])
end

# --- Forecast accuracy metrics (actual vs forecast over horizon) ---
mae(a, f) = mean(abs.(a .- f))
rmse(a, f) = sqrt(mean((a .- f) .^ 2))
smape(a, f) = mean(2 .* abs.(a .- f) ./ (abs.(a) .+ abs.(f) .+ eps())) * 100
"MASE scaled by in-sample seasonal-naive MAE of the training series."
function mase(a, f, train, m::Int)
    denom = m < length(train) ? mean(abs.(train[(m+1):end] .- train[1:(end-m)])) : NaN
    return (denom == 0 || isnan(denom)) ? NaN : mean(abs.(a .- f)) / denom
end
"Seasonal-naive forecast: repeat last season of training over the horizon."
function seasonal_naive(train::AbstractVector, h::Int, m::Int)
    m = max(m, 1)
    return [train[end - m + 1 + ((i - 1) % m)] for i in 1:h]
end

"Expanding-window rolling origins: train-end indices k with k+H <= n."
function rolling_origins(n::Int; init_frac::Float64 = 0.7, H::Int = 12, step::Int = 12)
    k0 = floor(Int, init_frac * n)
    return [k for k in k0:step:(n - H)]
end
