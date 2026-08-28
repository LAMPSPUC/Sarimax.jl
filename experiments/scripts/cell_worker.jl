# Worker for one M4 cell: fit `Sarimax.auto` on one series, forecast, score.
#
# The harness of the parent ForecastTester repository is deliberately NOT loaded. That
# module does `using RCall` and `src/models/Naive.jl` calls into R at include time, so
# loading it requires a working R installation. This worker does not need one: the metrics
# are computed here and the reference forecasts (auto.arima and Naive) are read from files.
#
# The sMAPE and MASE definitions below are the ones in the parent harness
# (`src/metrics.jl`, lines 11 and 30), copied term by term. Diverging from them would make
# the numbers incomparable with the stored auto.arima baseline.
#
# This file is a separate include target rather than an `@everywhere begin ... end` block
# because `@everywhere` only reaches workers that exist when it runs: any replacement
# worker spawned later would come up without `runSeries` and fail every job it received.

import Pkg
Pkg.activate(get(ENV, "SARIMAX_PROJECT", joinpath(@__DIR__, "..", "..")))

using LinearAlgebra
# One BLAS thread per worker. This machine has 12 physical cores and BLAS defaults to 6
# threads; ten worker processes at that default would request 60 threads on 12 cores.
# Oversubscription of that kind has produced NaNs and spurious timeouts in this project.
# Parallelism is between processes, each serial inside.
BLAS.set_num_threads(1)

using CSV, DataFrames, TimeSeries, Statistics
include(joinpath(get(ENV, "SARIMAX_SRC", joinpath(@__DIR__, "..", "..")), "src", "Sarimax.jl"))
const Sx = Main.Sarimax

# Seasonality, horizon and horizon blocks per frequency, taken from GRANULARITY_DICT and
# WINDOWS_HORIZON_DICT of the parent harness (`src/ForecastTester.jl`, lines 22 and 31).
# These are the official M4 definitions; changing them would break comparability with the
# stored baseline.
const FREQS = Dict(
    "monthly" => (s = 12, H = 18,
        blocks = ["short" => 1:6, "medium" => 7:12, "long" => 13:18, "total" => 1:18]),
    "quarterly" => (s = 4, H = 8,
        blocks = ["short" => 1:2, "medium" => 3:5, "long" => 6:8, "total" => 1:8]),
    "yearly" => (s = 1, H = 6,
        blocks = ["short" => 1:2, "medium" => 3:4, "long" => 5:6, "total" => 1:6]),
    "weekly" => (s = 1, H = 13,
        blocks = ["short" => 1:4, "medium" => 5:9, "long" => 10:13, "total" => 1:13]),
    "daily" => (s = 1, H = 14,
        blocks = ["short" => 1:4, "medium" => 5:9, "long" => 10:14, "total" => 1:14]),
    "hourly" => (s = 24, H = 48,
        blocks = ["short" => 1:16, "medium" => 17:32, "long" => 33:48, "total" => 1:48]),
)

# src/metrics.jl:11 - (200/H) * sum(|a-f| / (|a|+|f|))
sMAPE(a, f) =
    (200 / length(a)) * sum(abs(a[i] - f[i]) / (abs(a[i]) + abs(f[i])) for i in eachindex(a))

# src/metrics.jl:30 - numerator (1/H)*sum|a-f|; denominator the seasonal naive of the
# TRAINING window, (1/(T-s))*sum_{j=s+1}^{T} |y_j - y_{j-s}|. The denominator does not
# depend on the block: it is the same for short/medium/long/total, as in the harness.
function maseDenominator(y, s)
    T = length(y)
    return (1 / (T - s)) * sum(abs(y[j] - y[j-s]) for j = (s+1):T)
end
MASE(a, f, den) = ((1 / length(a)) * sum(abs(a[i] - f[i]) for i in eachindex(a))) / den

# The cell configuration travels inside the job tuple, not in a global or in the process
# environment. `pmapWithTimeout` replaces workers mid-run, and a replacement that came up
# without the configuration would silently run its series under a different objective -
# a run with mixed cells that nothing in the output would reveal.
function runSeries(args)
    (sid, y, yTest, objective, granularity, initialization, capSeconds, requireTerms) = args
    cfg = FREQS[granularity]
    s, H, BLOCKS = cfg.s, cfg.H, cfg.blocks
    t0 = time()
    # `capSeconds` is the CLI encoding, not the value `auto` takes: negative selects the
    # production rule (a 120 s cap on short series only), zero means no cap at all, and a
    # positive value is used as given. The distinction matters at the boundary: passing 0.0
    # straight through makes Ipopt reject `max_wall_time = 0.0` and every series fails.
    searchLb = 5 + 2 * s
    isShort = length(y) <= 5 * searchLb
    cap = capSeconds < 0 ? (isShort ? 120.0 : nothing) :
          (capSeconds == 0 ? nothing : Float64(capSeconds))
    try
        # EVERY argument of `auto` that affects estimation is passed explicitly, including
        # those whose value equals the current default. The defaults of this package have
        # moved between releases (`initialization` went from :zeroed to :innovations;
        # `exogDynamics` moved and moved back), so a script that inherits a default does
        # not denote a fixed computation. Two arguments are deliberately absent:
        #   * `lambda`/`alpha` - under `objectiveFunction = "ridge"` the package REJECTS a
        #     caller-supplied `lambda`, fixing it at sqrt(effective sample size) by
        #     construction. Passing it raises ArgumentError. They belong to `elastic_net`.
        #   * `exog`/`exogDynamics` - these campaigns have no exogenous regressors, and
        #     `exogDynamics` did not exist in the package at the campaign commits.
        m = Sx.auto(
            Sx.loadDataset(DataFrame(y = y));
            seasonality = s,
            d = -1,
            D = -1,
            maxp = 5,
            maxd = 2,
            maxq = 5,
            maxP = 2,
            maxD = 1,
            maxQ = 2,
            maxOrder = 5,
            informationCriteria = "aicc",
            allowMean = nothing,
            allowDrift = nothing,
            integrationTest = "kpssShort",
            seasonalIntegrationTest = "seas",
            objectiveFunction = objective,
            assertStationarity = true,
            assertInvertibility = true,
            showLogs = false,
            outlierDetection = false,
            searchMethod = "stepwise",
            parallel = false,
            seasonalForm = :multiplicative,
            initialization = initialization,
            multistart = false,
            # R's scheme: estimate FREE and REJECT the boundary candidate, rather than
            # constraining the domain. `stationary`/`invertible` choose the
            # parameterisation; `assert*` with `rootMargin` is the rejection rule.
            stationary = false,
            stationarityMargin = 1e-6,
            invertible = false,
            invertibilityMargin = 1e-6,
            constrainedRefit = false,
            rootMargin = 1e-2,
            optimizer = Sx.Ipopt.Optimizer,
            warmStartFromBox = true,
            maxTimeSeconds = cap,
            cvarLevel = 0.9,
            requireTermsWhenOverDifferenced = requireTerms,
            requireMAWhenDoublyDifferenced = false,
        )
        Sx.predict!(m; stepsAhead = H)
        f = Float64.(TimeSeries.values(m.forecast))
        h = min(H, length(yTest))
        den = maseDenominator(y, s)
        met = Float64[]
        for (_, range) in BLOCKS
            idx = [i for i in range if i <= h]
            if isempty(idx)
                append!(met, [NaN, NaN])
            else
                append!(met, [round(sMAPE(yTest[idx], f[idx]), digits = 6),
                    round(MASE(yTest[idx], f[idx], den), digits = 6)])
            end
        end
        Any[sid, length(y), m.p, m.d, m.q, m.P, m.D, m.Q,
            isnothing(m.c) ? 0 : (abs(m.c) > 1e-10 ? 1 : 0),
            isnothing(m.trend) ? 0 : (abs(m.trend) > 1e-10 ? 1 : 0),
            met...,
            round(den, sigdigits = 10),
            haskey(m.metadata, "huberFallback") ? (m.metadata["huberFallback"] ? 1 : 0) : -1,
            round(time() - t0, digits = 2), "OK",
            String(get(m.metadata, "solverStatus", "")),
            # Per-series forecast. Any metric at any horizon cut can be recomputed later
            # without re-running. The realised values are not duplicated here: they come
            # from datasets/<Freq>-test.csv, which is deterministic.
            join(map(x -> string(round(x, digits = 6)), f), ";")]
    catch e
        # The failure row keeps the SAME width as a success row. A short row shifts every
        # later field of that line, which turns a recorded failure into silently corrupt
        # data for whoever parses the file.
        Any[sid, length(y), -1, -1, -1, -1, -1, -1, -1, -1,
            NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, -1,
            round(time() - t0, digits = 2),
            "ERROR:" * first(replace(sprint(showerror, e), '\n' => ' ', ',' => ' '), 60),
            "", ""]
    end
end
