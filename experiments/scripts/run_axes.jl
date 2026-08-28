# 2x2 over the two wrapper axes that were left unexplained. See REPRODUCE.md, campaign E.
#
# `requireTermsWhenOverDifferenced` was measured to HURT weekly, and the production wrapper
# turns it on UNCONDITIONALLY. If that axis only ever costs, the production configuration
# should lose by it — and it is the production configuration that wins. So another axis
# must be compensating with room to spare.
#
# The visible candidate is the pair of differencing tests the wrapper substitutes:
#   integrationTest = "kpssShort", seasonalIntegrationTest = "seas"
#
# This 2x2 measures the two axes alone and together, ALL at the same commit, same machine,
# same day — the only way for the comparison not to carry version drift. Three weeks of
# package code were measured to move weekly OWA by 0.0320, which is larger than most of the
# effects being compared; cross-campaign differences are therefore not interpretable.
#
# NOTE ON THE `base` AND `req` CELLS. They deliberately do NOT pass `integrationTest` /
# `seasonalIntegrationTest`, because the axis under test is precisely "wrapper tests vs
# package defaults". Those two keywords are the ONLY arguments in this file left to the
# package default, and that is the experimental treatment rather than an oversight. The
# values they take are pinned in REPRODUCE.md so the cell remains identifiable if the
# defaults move.
#
#   julia --project=<harness> run_axes.jl <freq> <workers> [out]
const R_HOME = get(ENV, "R_HOME", "")
isempty(R_HOME) && error("set R_HOME to the R installation directory (RCall needs it)")

const FREQ = length(ARGS) >= 1 ? ARGS[1] : "weekly"
const NW   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
const OUT  = length(ARGS) >= 3 ? ARGS[3] : "axes_$(FREQ).csv"

import Pkg; Pkg.activate(".")
include("provenance.jl"); using .Provenance
requireCleanTrees()
using Distributed
addprocs(NW; exeflags = ["--project=."])
@everywhere begin
    ENV["R_HOME"] = $R_HOME
    import Pkg; Pkg.activate(".")
    using LinearAlgebra; BLAS.set_num_threads(1)
    using ForecastTester
    include("src/ForecastTester.jl")
    using DataFrames, TimeSeries, Statistics
    const Sxw = Main.ForecastTester.Sarimax
    const FTw = Main.ForecastTester
    const CACHE = Dict{String,Any}()
    data(f) = get!(CACHE, f) do
        FTw.build_train_test_dict(FTw.read_dataframes(f)...)
    end
    smape(f, a) = mean(2 .* abs.(f .- a) ./ (abs.(f) .+ abs.(a) .+ 1e-12)) * 100
    function mase(f, a, y, m)
        den = m < length(y) ? mean(abs.(y[m+1:end] .- y[1:end-m])) : mean(abs.(diff(y)))
        den <= 0 && (den = 1e-9); mean(abs.(f .- a)) / den
    end

    """
        axesConfig(seasonality, requireTerms, wrapperTests) -> Dict{Symbol,Any}

    Campaign E, argument by argument. Everything is fixed except the two treatment axes.
    """
    function axesConfig(s::Int, requireTerms::Bool, wrapperTests::Bool)
        cfg = Dict{Symbol,Any}(
            :seasonality                     => s,
            :objectiveFunction               => "mse",
            :initialization                  => :innovations,
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
            :maxTimeSeconds                  => nothing,
            :parallel                        => false,
            :cvarLevel                       => 0.9,
            :outlierDetection                => false,
            :requireMAWhenDoublyDifferenced  => false,
            # --- treatment axis 1 ---
            :requireTermsWhenOverDifferenced => requireTerms,
        )
        # --- treatment axis 2: the wrapper's differencing tests, or the package defaults ---
        if wrapperTests
            cfg[:integrationTest] = "kpssShort"
            cfg[:seasonalIntegrationTest] = "seas"
        end
        cfg
    end

    function runOne(job)
        (freq, sid, cell, requireTerms, wrapperTests, prov) = job
        s = FTw.GRANULARITY_DICT[freq]["s"]; H = FTw.GRANULARITY_DICT[freq]["H"]
        dd = data(freq)
        y = Float64.(dd[sid]["train"]); yt = Float64.(dd[sid]["test"])
        h = min(H, length(yt)); t0 = time()
        try
            m = Sxw.auto(Sxw.loadDataset(DataFrame(y = y));
                         axesConfig(s, requireTerms, wrapperTests)...)
            Sxw.predict!(m; stepsAhead = H)
            f = Float64.(TimeSeries.values(m.forecast))
            vcat([cell, requireTerms, wrapperTests, freq, sid, m.p, m.d, m.q, m.P, m.D, m.Q,
                  round(smape(f[1:h], yt[1:h]), digits = 4),
                  round(mase(f[1:h], yt[1:h], y, s), digits = 4),
                  round(time() - t0, digits = 2),
                  get(m.metadata, "solverStatus", "unknown"), "OK"], prov)
        catch e
            vcat([cell, requireTerms, wrapperTests, freq, sid, -1,-1,-1,-1,-1,-1, NaN, NaN,
                  round(time() - t0, digits = 2), "-",
                  "ERROR:" * first(replace(sprint(showerror, e), '\n'=>' ', ','=>' '), 30)], prov)
        end
    end
end
const FT = Main.ForecastTester
const PROV_KEYS = [k for (k, _) in provenanceFields()]
const PROV_VALS = [v for (_, v) in provenanceFields()]

const CELLS = [("base", false, false), ("req", true, false),
               ("tests", false, true), ("req+tests", true, true)]

# Everything inside a function: `n += 1` inside `open(...) do io` does not reach a
# top-level variable under soft scope, and that has silently broken runners here before.
function main()
    dd = FT.build_train_test_dict(FT.read_dataframes(FREQ)...)
    ids = sort(collect(keys(dd))); dd = nothing; GC.gc()
    open(OUT, "w") do io
        stamp(io)
        println(io, join(vcat(["cell","require_terms","wrapper_tests","freq","series","p","d",
                               "q","P","D","Q","smape","mase","seconds","solver_status",
                               "status"], PROV_KEYS), ','))
    end
    jobs = [(FREQ, sid, cell, rt, wt, PROV_VALS) for (cell, rt, wt) in CELLS for sid in ids]
    println("axes | $FREQ: $(length(ids)) series x $(length(CELLS)) cells = $(length(jobs)) fits | $NW workers")
    flush(stdout)
    n = 0
    open(OUT, "a") do io
        for r in pmap(runOne, jobs; batch_size = 1, on_error = e -> nothing)
            isnothing(r) && continue
            println(io, join(map(x -> x isa AbstractString ? "\"$x\"" : x, r), ","))
            flush(io); n += 1
            n % 200 == 0 && (println("  $n/$(length(jobs))"); flush(stdout))
        end
    end
    println("AXES_DONE: $n rows -> $OUT")
end
main()
