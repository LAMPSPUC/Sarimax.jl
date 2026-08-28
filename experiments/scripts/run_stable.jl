# Does the `stable` objective (CVaR of the squared residuals) beat `mse`?
#
# Three arms PAIRED on the same series: mse, stable a=0.9, stable a=0.5. See REPRODUCE.md,
# campaign F.
#
# RECONSTRUCTED ARM — READ BEFORE CITING. The published rows (results/raw/stable_m4.csv)
# contain three arms: `mse`, `stable_090`, `stable_050`. The runner as it stood in the
# harness working tree at consolidation time declared only two:
#
#     const ARMS = [("mse","mse",0.9), ("stable_050","stable",0.5)]
#
# The `stable_090` arm had been removed from the runner AFTER the run. The file on disk
# therefore could not regenerate its own results, and the companion analyser still required
# the third arm. The arm below is RESTORED from three independent attestations: the arm
# label present in the data, the analyser's required arm list, and the runner's own header
# and log line ("3 arms"). The cvarLevel value 0.9 is inferred from the label `stable_090`
# and from DEFAULT_CVAR_LEVEL at the campaign commit; it was NOT recorded per row in the
# original output, and this is flagged in REPRODUCE.md as an unverified reconstruction.
#
# The reconstruction is stated here rather than silently applied, because a runner that
# quietly regrows a deleted arm is exactly as unauditable as one that quietly loses it.
#
#   julia --project=<harness> run_stable.jl <freqs> <workers> [out] [sampleSize]
const R_HOME = get(ENV, "R_HOME", "")
isempty(R_HOME) && error("set R_HOME to the R installation directory (RCall needs it)")

const FREQS = String.(split(length(ARGS) >= 1 ? ARGS[1] : "weekly", ','))
const NW    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
const OUT   = length(ARGS) >= 3 ? ARGS[3] : "stable_m4.csv"
# SAMPLE per frequency (0 = whole population, which is what the published run used).
#
# THE SIZE IS NOT ARBITRARY. The effect of `stable` on yearly is small — a sMAPE delta of
# -0.16 with a CI half-width of 0.078 over 23,000 series. The half-width scales as
# 1/sqrt(n): at 1,500 series it becomes ~0.31 and the interval would INCLUDE zero, i.e. the
# sample would lack the power to detect an effect the size of the one already measured.
const SAMPLE  = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 0
# Sampling is seeded and the seed is written to every row: an unseeded sample is not
# reproducible, and a sample taken in id order is not random.
const SEED    = 20260826

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
    const FTw = Main.ForecastTester
    const CACHE = Dict{String,Any}()
    data(f) = get!(CACHE, f) do
        empty!(CACHE); FTw.build_train_test_dict(FTw.read_dataframes(f)...)
    end
    smape(f, a) = mean(2 .* abs.(f .- a) ./ (abs.(f) .+ abs.(a) .+ 1e-12)) * 100
    function maseDen(y, m)
        den = m < length(y) ? mean(abs.(y[m+1:end] .- y[1:end-m])) : mean(abs.(diff(y)))
        den <= 0 ? 1e-9 : den
    end
    vec2str(v) = join((string(round(x, digits = 6)) for x in v), ';')

    """
        stableConfig(seasonality, objective, cvarLevel) -> Dict{Symbol,Any}

    Campaign F, argument by argument. Identical to campaign A's configuration except for
    `objectiveFunction` and `cvarLevel` — which is the point: the arms differ in the
    objective and in nothing else, so any difference in the result is attributable.
    """
    function stableConfig(s::Int, obj::String, level::Float64)
        Dict{Symbol,Any}(
            :seasonality                     => s,
            :objectiveFunction               => obj,
            :cvarLevel                       => level,
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
            :maxTimeSeconds                  => nothing,
            :parallel                        => false,
            :outlierDetection                => false,
            :requireTermsWhenOverDifferenced => false,
            :requireMAWhenDoublyDifferenced  => false,
        )
    end

    function runOne(job)
        (freq, sid, arm, obj, level, seed, prov) = job
        s = FTw.GRANULARITY_DICT[freq]["s"]; H = FTw.GRANULARITY_DICT[freq]["H"]
        dd = data(freq)
        y = Float64.(dd[sid]["train"]); yt = Float64.(dd[sid]["test"])
        h = min(H, length(yt)); t0 = time()
        try
            m = Sxw.auto(Sxw.loadDataset(DataFrame(y = y)); stableConfig(s, obj, level)...)
            Sxw.predict!(m; stepsAhead = H)
            f = Float64.(TimeSeries.values(m.forecast))
            vcat([arm, obj, level, freq, sid, m.p, m.d, m.q, m.P, m.D, m.Q,
                  round(smape(f[1:h], yt[1:h]), digits = 4),
                  round(mean(abs.(f[1:h] .- yt[1:h])) / maseDen(y, s), digits = 4),
                  round(maseDen(y, s), digits = 6), round(time() - t0, digits = 2),
                  get(m.metadata, "solverStatus", "unknown"), "OK", seed,
                  vec2str(f[1:h]), vec2str(yt[1:h])], prov)
        catch e
            vcat([arm, obj, level, freq, sid, -1,-1,-1,-1,-1,-1, NaN, NaN, NaN,
                  round(time() - t0, digits = 2), "-",
                  "ERROR:" * first(replace(sprint(showerror, e), '\n'=>' ', ','=>' '), 30),
                  seed, "", ""], prov)
        end
    end
end
const FT = Main.ForecastTester
const PROV_KEYS = [k for (k, _) in provenanceFields()]
const PROV_VALS = [v for (_, v) in provenanceFields()]

# `cvarLevel` is carried as a COLUMN, not only in the arm label. The original output
# recorded the arm name alone, which made the level inferable but not verifiable.
const ARMS = [("mse", "mse", 0.9), ("stable_090", "stable", 0.9), ("stable_050", "stable", 0.5)]

function main()
    open(OUT, "w") do io
        stamp(io)
        println(io, join(vcat(["arm","objective","cvar_level","freq","series","p","d","q",
                               "P","D","Q","smape","mase","mase_den","seconds",
                               "solver_status","status","sample_seed","forecast","actual"],
                              PROV_KEYS), ','))
    end
    println("stable | $(join(FREQS, ", ")) | $(length(ARMS)) arms | $NW workers"); flush(stdout)
    for freq in FREQS
        dd = FT.build_train_test_dict(FT.read_dataframes(freq)...)
        ids = sort(collect(keys(dd))); dd = nothing; GC.gc()
        if SAMPLE > 0 && length(ids) > SAMPLE
            Random.seed!(SEED)
            ids = sort(Random.shuffle(ids)[1:SAMPLE])
            println("  seeded sample: $(length(ids)) series (seed $SEED)")
        end
        jobs = [(freq, sid, arm, obj, lv, SEED, PROV_VALS)
                for (arm, obj, lv) in ARMS for sid in ids]
        println("--- $freq: $(length(ids)) series x $(length(ARMS)) arms = $(length(jobs)) fits ---")
        flush(stdout)
        n = 0
        open(OUT, "a") do io
            for r in pmap(runOne, jobs; batch_size = 1, on_error = e -> nothing)
                isnothing(r) && continue
                println(io, join(map(x -> x isa AbstractString ? "\"$x\"" : x, r), ","))
                flush(io); n += 1
                n % 200 == 0 && (println("  $freq: $n/$(length(jobs))"); flush(stdout))
            end
        end
        println("$freq: $n rows"); flush(stdout); GC.gc()
    end
    println("STABLE_DONE -> $OUT")
end
main()
