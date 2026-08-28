# Same-commit isolation: the production configuration against `:innovations`, both arms
# built from the same package tree. See REPRODUCE.md, campaign D.
#
# WHY THIS EXISTS. The distance between the production arm and the `:innovations` arm had
# only ever been measured across campaigns run three weeks apart — and three weeks of
# package code were measured to move weekly OWA by 0.0320, which is LARGER than the
# distance being estimated. Any such cross-campaign difference is therefore uninterpretable:
# it confounds the treatment with version drift.
#
# The fix is to run both ends at the same commit, on the same machine, on the same day.
#
# The production arm calls the frozen wrapper in `wrapper_v0_6.jl` DIRECTLY rather than
# restating its keywords, because the wrapper is the only faithful definition of what that
# arm means. Read the banner in that file before citing this campaign: the wrapper was
# reconstructed from an uncommitted working tree.
#
#   julia --project=<harness> run_isolation.jl <freqs> <workers> [out]
const R_HOME = get(ENV, "R_HOME", "")
isempty(R_HOME) && error("set R_HOME to the R installation directory (RCall needs it)")

const FREQS = String.(split(length(ARGS) >= 1 ? ARGS[1] : "weekly,yearly", ','))
const NW    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
const OUT   = length(ARGS) >= 3 ? ARGS[3] : "isolation.csv"
const WRAPPER = joinpath(@__DIR__, "wrapper_v0_6.jl")

# `include` resolves relative to THIS file, not the working directory, so the harness
# source has to be named absolutely: these runners live in a subdirectory of the package
# repository, not at the harness root.
const HARNESS_SRC = joinpath(get(ENV, "REPLICATION_HARNESS_REPO", pwd()), "src", "ForecastTester.jl")
isfile(HARNESS_SRC) || error("harness source not found at $HARNESS_SRC; set REPLICATION_HARNESS_REPO")
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
    include($HARNESS_SRC)
    using DataFrames, TimeSeries, Statistics
    const Sxw = Main.ForecastTester.Sarimax
    const FTw = Main.ForecastTester
    include($WRAPPER); using .WrapperV06
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
        innovationsConfig(seasonality) -> Dict{Symbol,Any}

    The `:innovations` arm, argument by argument. Identical to campaign A's configuration,
    deliberately: this arm is meant to be the same estimator measured in the headline
    table, so the two campaigns can be read against each other.
    """
    function innovationsConfig(s::Int)
        Dict{Symbol,Any}(
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
            :cvarLevel                       => 0.9,
            :outlierDetection                => false,
            :requireTermsWhenOverDifferenced => false,
            :requireMAWhenDoublyDifferenced  => false,
        )
    end

    function runOne(job)
        (freq, sid, arm, prov) = job
        s = FTw.GRANULARITY_DICT[freq]["s"]; H = FTw.GRANULARITY_DICT[freq]["H"]
        dd = data(freq)
        y = Float64.(dd[sid]["train"]); yt = Float64.(dd[sid]["test"])
        h = min(H, length(yt)); t0 = time()
        try
            f = if arm == "production_v0_6"
                # S = 1: the wrapper hcats the scenarios and S = 0 would break it. The
                # scenarios are discarded; only the point forecast is compared.
                first(forecastV06(y, s, H, 1))
            else
                m = Sxw.auto(Sxw.loadDataset(DataFrame(y = y)); innovationsConfig(s)...)
                Sxw.predict!(m; stepsAhead = H)
                Float64.(TimeSeries.values(m.forecast))
            end
            vcat([arm, freq, sid, round(smape(f[1:h], yt[1:h]), digits = 4),
                  round(mean(abs.(f[1:h] .- yt[1:h])) / maseDen(y, s), digits = 4),
                  round(maseDen(y, s), digits = 6), round(time() - t0, digits = 2), "OK",
                  vec2str(f[1:h]), vec2str(yt[1:h])], prov)
        catch e
            vcat([arm, freq, sid, NaN, NaN, NaN, round(time() - t0, digits = 2),
                  "ERROR:" * first(replace(sprint(showerror, e), '\n'=>' ', ','=>' '), 30),
                  "", ""], prov)
        end
    end
end
const FT = Main.ForecastTester
const PROV_KEYS = [k for (k, _) in provenanceFields()]
const PROV_VALS = [v for (_, v) in provenanceFields()]

function main()
    open(OUT, "w") do io
        stamp(io)
        println(io, join(vcat(["arm","freq","series","smape","mase","mase_den","seconds",
                               "status","forecast","actual"], PROV_KEYS), ','))
    end
    println("isolation at a single commit | $(join(FREQS, ", ")) | $NW workers"); flush(stdout)
    for freq in FREQS
        dd = FT.build_train_test_dict(FT.read_dataframes(freq)...)
        ids = sort(collect(keys(dd))); dd = nothing; GC.gc()
        jobs = [(freq, sid, arm, PROV_VALS)
                for arm in ("production_v0_6", "innovations") for sid in ids]
        println("--- $freq: $(length(ids)) series x 2 arms ---"); flush(stdout)
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
    println("ISOLATION_DONE -> $OUT")
end
main()
