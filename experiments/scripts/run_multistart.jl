# Does deterministic multistart pay? See REPRODUCE.md, campaign C.
#
# `fit!(...; multistart = true)` fits from {zero, CSS seed} and keeps the better candidate
# BY THE INFORMATION CRITERION (which already uses the exact likelihood), not by SSE. The
# zero start remains a candidate, so the criterion can never get worse — that invariant is
# the design, and the runner records which start won so it can be checked.
#
# Paired design: both arms on the same series, so the delta is within-series.
#
#   julia --project=<harness> run_multistart.jl <N> <workers> [out]
const R_HOME = get(ENV, "R_HOME", "")
isempty(R_HOME) && error("set R_HOME to the R installation directory (RCall needs it)")

const N    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 400
const NW   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
const OUT  = length(ARGS) >= 3 ? ARGS[3] : "multistart_random.csv"
const SEED = 20260818

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

    """
        multistartConfig(seasonality, multistart) -> Dict{Symbol,Any}

    Campaign C, argument by argument. `multistart` is the only axis; everything else is
    held fixed, including the 120 s per-fit cap, which is part of this design.
    """
    function multistartConfig(s::Int, multistart::Bool)
        Dict{Symbol,Any}(
            :seasonality                     => s,
            :objectiveFunction               => "mse",
            :initialization                  => :penalized,
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
            :warmStartFromBox                => false,
            :maxTimeSeconds                  => 120.0,
            :parallel                        => false,
            :cvarLevel                       => 0.9,
            :outlierDetection                => false,
            :requireTermsWhenOverDifferenced => false,
            :requireMAWhenDoublyDifferenced  => false,
            # --- the treatment axis ---
            :multistart                      => multistart,
        )
    end

    function runSeries(job)
        (sid, y, yt, s, H, seed, prov) = job
        h = min(H, length(yt))
        rows = Any[]
        for (cell, ms) in [("base", false), ("multi", true)]
            t0 = time()
            try
                m = Sxw.auto(Sxw.loadDataset(DataFrame(y = y)); multistartConfig(s, ms)...)
                Sxw.predict!(m; stepsAhead = H)
                f = Float64.(TimeSeries.values(m.forecast))
                push!(rows, vcat([sid, length(y), cell, ms, m.p, m.d, m.q, m.P, m.D, m.Q,
                                  round(smape(f[1:h], yt[1:h]), digits = 4),
                                  round(mase(f[1:h], yt[1:h], y, s), digits = 4),
                                  round(maxCoef(m), digits = 4),
                                  minimum(f) < 0 ? 1 : 0,
                                  get(m.metadata, "multistartVenceuCSS", false) === true ? 1 : 0,
                                  round(time() - t0, digits = 2),
                                  get(m.metadata, "solverStatus", "unknown"), "OK", seed], prov))
            catch e
                push!(rows, vcat([sid, length(y), cell, ms, -1,-1,-1,-1,-1,-1, NaN, NaN, NaN,
                                  -1, -1, round(time() - t0, digits = 2), "-",
                                  "ERROR:" * first(replace(sprint(showerror, e), '\n'=>' ', ','=>' '), 30),
                                  seed], prov))
            end
        end
        rows
    end
end

const FT = Main.ForecastTester
const PROV_KEYS = [k for (k, _) in provenanceFields()]
const PROV_VALS = [v for (_, v) in provenanceFields()]

dd = FT.build_train_test_dict(FT.read_dataframes("monthly")...)
s = FT.GRANULARITY_DICT["monthly"]["s"]; H = FT.GRANULARITY_DICT["monthly"]["H"]
all_ids = sort(collect(keys(dd)))
Random.seed!(SEED)
ids = shuffle(all_ids)[1:min(N, length(all_ids))]

function main()
    open(OUT, "w") do io
        stamp(io)
        println(io, join(vcat(["series","T","cell","multistart","p","d","q","P","D","Q",
                               "smape","mase","max_coef","forecast_negative",
                               "multistart_beat_zero","seconds","solver_status","status",
                               "sample_seed"], PROV_KEYS), ','))
    end
    jobs = [(sid, Float64.(dd[sid]["train"]), Float64.(dd[sid]["test"]), s, H, SEED, PROV_VALS)
            for sid in ids]
    println("multistart | $(length(jobs)) series x 2 arms | $NW workers"); flush(stdout)
    n = 0
    open(OUT, "a") do io
        for rows in pmap(runSeries, jobs; batch_size = 1, on_error = e -> nothing)
            isnothing(rows) && continue
            for r in rows
                println(io, join(map(x -> x isa AbstractString ? "\"$x\"" : x, r), ","))
                n += 1
            end
            flush(io)
            n % 100 == 0 && (println("  $n rows"); flush(stdout))
        end
    end
    println("MULTISTART_DONE: $n rows -> $OUT")
end
main()
