# Two questions in one run, on M4 monthly, RANDOM sample of the 48k population.
#
# See REPRODUCE.md, campaign B.
#
# P1 (the one that needs large n): under the package code, what is the WIN RATE against
#    auto.arima? Separating the competing estimates requires n: at 4,000 series the
#    standard error of the rate falls to ~0.8 points.
#
# P2: do ridge, mae and huber improve the forecast?
#
# CONFOUNDER AVOIDED: the three alternative objectives run under `:zeroed` and a
# `mse_zeroed` control cell exists, so the BETWEEN-OBJECTIVE comparison is made at fixed
# initialization. The `mse` cell (production, `:penalized`) is reserved for P1.
#
# ############################################################################
# # BEHAVIOUR CHANGE — THE `_pen` CELLS DO NOT REPRODUCE. READ BEFORE CITING. #
# ############################################################################
#
# The P3 cells below (`ridge_pen`, `mae_pen`, `huber_pen`) pair a non-mse objective with
# `initialization = :penalized`. What that combination MEANS changed in the package after
# this campaign ran, without this file changing at all:
#
#   at the campaign commit  the pre-sample treatment was implemented for `mse` only. The
#                           package emitted a `@warn` and the fit fell through to the
#                           ordinary branch, i.e. it silently became `:free`. Under a
#                           10-worker run the warning is invisible in the log. ALL THREE
#                           `_pen` CELLS THEREFORE MEASURED `:free`, not `:penalized`.
#
#   under the current code  the pre-sample block is implemented for nine objectives,
#                           ridge/mae/huber among them. The same three cells now really do
#                           get the penalized block, and the guard no longer fires.
#
# So re-running this script today produces DIFFERENT NUMBERS FOR THE SAME CELL NAMES, and
# the difference is a change of estimator, not of default. The published `_pen` rows
# describe `:free`; the header comment they were run under claimed `:penalized`. Any table
# legend derived from them must say `:free`, or the rows must be regenerated and relabelled.
#
# The `mse`, `mse_zeroed`, `ridge`, `mae` and `huber` cells are unaffected: none of them
# pairs a non-mse objective with `:penalized`.
#
#   julia --project=<harness> run_objectives.jl <N> <NOBJ> <workers> [out] [cells]
const R_HOME = get(ENV, "R_HOME", "")
isempty(R_HOME) && error("set R_HOME to the R installation directory (RCall needs it)")

const N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4000
const NOBJ  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1500
const NW    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
const OUT   = length(ARGS) >= 4 ? ARGS[4] : "objectives_monthly.csv"
const CELLS = length(ARGS) >= 5 ? String.(split(ARGS[5], ',')) :
              ["mse_zeroed", "ridge", "mae", "huber"]
const SEED  = 20260819

import Pkg; Pkg.activate(".")
include("provenance.jl"); using .Provenance
requireCleanTrees()
using Distributed, Random
addprocs(NW; exeflags = ["--project=."])
@everywhere begin
    ENV["R_HOME"] = $R_HOME
    import Pkg; Pkg.activate(".")
    using LinearAlgebra; BLAS.set_num_threads(1)
    using ForecastTester
    include("src/ForecastTester.jl")
    using DataFrames, TimeSeries, Statistics
    const Sxw = Main.ForecastTester.Sarimax
    smape(f, a) = mean(2 .* abs.(f .- a) ./ (abs.(f) .+ abs.(a) .+ 1e-12)) * 100
    function mase(f, a, y, m)
        den = m < length(y) ? mean(abs.(y[m+1:end] .- y[1:end-m])) : mean(abs.(diff(y)))
        den <= 0 && (den = 1e-9); mean(abs.(f .- a)) / den
    end
    function maxCoef(m)
        mx = 0.0
        for fl in (:ϕ, :θ, :Φ, :Θ)
            v = getfield(m, fl); (isnothing(v) || isempty([v...])) && continue
            mx = max(mx, maximum(abs, [v...]))
        end
        mx
    end

    # cell -> (objective, initialization)
    const CONFIG = Dict(
        "mse"        => ("mse",   :penalized),   # production: this is the cell answering P1
        "mse_zeroed" => ("mse",   :zeroed),      # control for P2
        "ridge"      => ("ridge", :zeroed),
        "mae"        => ("mae",   :zeroed),
        "huber"      => ("huber", :zeroed),
        # P3 — see the behaviour-change banner at the top of this file.
        "ridge_pen"  => ("ridge", :penalized),
        "mae_pen"    => ("mae",   :penalized),
        "huber_pen"  => ("huber", :penalized),
    )

    """
        objectivesConfig(seasonality, objective, initialization) -> Dict{Symbol,Any}

    Campaign B, argument by argument. The 120 s per-fit cap is part of the design here (it
    was not in campaign A) and is stated rather than inherited.
    """
    function objectivesConfig(s::Int, obj::String, ini::Symbol)
        Dict{Symbol,Any}(
            :seasonality                     => s,
            :objectiveFunction               => obj,
            :initialization                  => ini,
            :seasonalForm                    => :multiplicative,
            :stationary                      => true,
            :stationarityMargin              => 1e-6,
            :invertible                      => false,
            :invertibilityMargin             => 1e-6,
            :assertStationarity              => true,
            :assertInvertibility             => true,
            :rootMargin                      => 1e-2,
            :constrainedRefit                => false,
            :searchMethod                    => "stepwise",
            :informationCriteria             => "aicc",
            :integrationTest                 => "kpssShort",
            :seasonalIntegrationTest         => "seas",
            :d                               => -1,
            :D                               => -1,
            :maxd                            => 2,
            :maxD                            => 1,
            :maxp                            => 5,
            :maxq                            => 5,
            :maxP                            => 2,
            :maxQ                            => 2,
            :maxOrder                        => 5,
            :multistart                      => false,
            :warmStartFromBox                => false,
            :maxTimeSeconds                  => 120.0,
            :parallel                        => false,
            :cvarLevel                       => 0.9,
            :outlierDetection                => false,
            :requireTermsWhenOverDifferenced => false,
            :requireMAWhenDoublyDifferenced  => false,
            # `lambda` and `alpha` stay unset: the package rejects them for every objective
            # except elastic_net, which this campaign does not use. The ridge penalty
            # weight is chosen internally.
        )
    end

    function runOne(job, chan)
        (sid, cell, y, yt, s, H, seed, prov) = job
        (obj, ini) = CONFIG[cell]
        h = min(H, length(yt)); t0 = time()
        row = try
            m = Sxw.auto(Sxw.loadDataset(DataFrame(y = y)); objectivesConfig(s, obj, ini)...)
            Sxw.predict!(m; stepsAhead = H)
            f = Float64.(TimeSeries.values(m.forecast))
            vcat([sid, length(y), cell, obj, string(ini), m.p, m.d, m.q, m.P, m.D, m.Q,
                  round(smape(f[1:h], yt[1:h]), digits = 4),
                  round(mase(f[1:h], yt[1:h], y, s), digits = 4),
                  round(maxCoef(m), digits = 4),
                  minimum(f) < 0 ? 1 : 0, round(time() - t0, digits = 2),
                  get(m.metadata, "solverStatus", "unknown"), "OK", seed], prov)
        catch e
            vcat([sid, length(y), cell, obj, string(ini), -1,-1,-1,-1,-1,-1, NaN, NaN, NaN,
                  -1, round(time() - t0, digits = 2), "-",
                  "ERROR:" * first(replace(sprint(showerror, e), '\n'=>' ', ','=>' '), 30),
                  seed], prov)
        end
        put!(chan, row)
        nothing
    end
end

const FT = Main.ForecastTester
const PROV_KEYS = [k for (k, _) in provenanceFields()]
const PROV_VALS = [v for (_, v) in provenanceFields()]

dd = FT.build_train_test_dict(FT.read_dataframes("monthly")...)
s = FT.GRANULARITY_DICT["monthly"]["s"]; H = FT.GRANULARITY_DICT["monthly"]["H"]

# reproducible random sample of the whole population
all_ids = sort(collect(keys(dd)))
Random.seed!(SEED)
ids = shuffle(all_ids)[1:min(N, length(all_ids))]
# nested subsample for the non-mse cells: still random, just smaller
idsObj = ids[1:min(NOBJ, length(ids))]

done = Set{Tuple{Int,String}}()
if isfile(OUT)
    for l in eachline(OUT)
        startswith(l, "#") && continue
        v = split(l, ','); length(v) < 3 && continue
        n = tryparse(Int, v[1]); isnothing(n) || push!(done, (n, strip(v[3], '"')))
    end
    println("resuming: $(length(done)) (series, cell) pairs already present")
else
    open(OUT, "w") do io
        stamp(io)
        println(io, join(vcat(["series","T","cell","objective","initialization","p","d","q",
                               "P","D","Q","smape","mase","max_coef","forecast_negative",
                               "seconds","solver_status","status","sample_seed"],
                              PROV_KEYS), ','))
    end
end

jobs = Any[]
if "mse" in CELLS
    for i in ids
        haskey(dd, i) || continue
        (i, "mse") in done ||
            push!(jobs, (i, "mse", Float64.(dd[i]["train"]), Float64.(dd[i]["test"]), s, H, SEED, PROV_VALS))
    end
end
for i in idsObj
    haskey(dd, i) || continue
    for cell in CELLS
        (i, cell) in done ||
            push!(jobs, (i, cell, Float64.(dd[i]["train"]), Float64.(dd[i]["test"]), s, H, SEED, PROV_VALS))
    end
end
# Shuffle the JOBS as well: mixes cheap and expensive cells through the run, which improves
# balance and keeps any prefix a valid sample.
shuffle!(jobs)
println("N=$N series (mse cell) | NOBJ=$NOBJ (other cells) | $(length(jobs)) jobs | $NW workers")
flush(stdout)

chan = RemoteChannel(() -> Channel{Any}(4096))
writer = @async begin
    n = 0; t0 = time()
    open(OUT, "a") do io
        while true
            r = take!(chan); r === nothing && break
            println(io, join(map(x -> x isa AbstractString ? "\"$x\"" : x, r), ","))
            n += 1
            if n % 50 == 0
                flush(io)
                el = time() - t0
                println("progress: $n/$(length(jobs)) jobs | $(round(el/60, digits=1)) min | " *
                        "eta $(round(el/n*(length(jobs)-n)/60, digits=1)) min")
                flush(stdout)
            end
        end
        flush(io)
    end
    n
end

# batch_size = 1 -> dynamic scheduling, no per-block barrier
pmap(j -> runOne(j, chan), jobs; batch_size = 1, on_error = e -> nothing)
put!(chan, nothing)
n = fetch(writer)
println("OBJECTIVES_DONE: $n jobs written -> $OUT")
