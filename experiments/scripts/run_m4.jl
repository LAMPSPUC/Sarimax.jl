# M4 benchmark, ONE FREQUENCY AT A TIME.
#
# Generates the raw rows behind the headline accuracy table (per frequency, per horizon
# block). See REPRODUCE.md, campaign A.
#
# WHY ONE FREQUENCY AT A TIME. An earlier runner built the whole queue with the DATA
# EMBEDDED: 100,000 train/test series in the job array, plus six data dictionaries held on
# the master. At daily (T ~ 2,940) that exhausted memory, the machine started paging, and
# throughput collapsed to 44.6 series/hour against the ~2,450 projected. The fix is
# structural, not a tuning knob:
#   - process one frequency at a time and release its dictionary before the next;
#   - the queue carries ONLY the series ids; each worker reads its own slice from a local
#     cache, so the master never holds 100,000 series.
#
#   julia --project=<harness> run_m4.jl <init> <freqs> <workers> [out] [capSeconds] [tailLen]
#
# Example (the invocation that produced the published rows, per frequency):
#   julia --project=. run_m4.jl innovations monthly 10 m4_innov_monthly.csv 0 0
#
# NO ARGUMENT IS LEFT TO A PACKAGE DEFAULT. Every keyword that can move the estimate is
# stated below, even where the value coincides with today's default. The defaults of this
# package have changed between campaigns (`initialization` went from :zeroed to
# :innovations); a script that inherits them silently reports a different number when
# re-run later.
const R_HOME = get(ENV, "R_HOME", "")
isempty(R_HOME) && error("set R_HOME to the R installation directory (RCall needs it)")

const INIT  = Symbol(length(ARGS) >= 1 ? ARGS[1] : "innovations")
const FREQS = String.(split(length(ARGS) >= 2 ? ARGS[2] : "weekly", ','))
const NW    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
const OUT   = length(ARGS) >= 4 ? ARGS[4] : string("m4_", INIT, ".csv")
# Per-FIT wall-clock cap. Pass 0 for NO CAP, which is what the published runs used.
# Measured: with a 120 s cap, 35% of weekly series truncate and the truncated ones carry
# 86% of the gap against auto.arima — the number becomes a property of the clock rather
# than of the method. The cap is a MEASUREMENT parameter and must be reported with the
# result.
const CAP   = length(ARGS) >= 5 ? (parse(Float64, ARGS[5]) > 0 ? parse(Float64, ARGS[5]) : nothing) : nothing
# HISTORY TRUNCATION: fit on the LAST `TAIL` observations only. 0 = whole series.
#
# The MASE denominator still comes from the COMPLETE series, deliberately: if it shrank
# too, MASE would move because the metric changed rather than because the forecast did,
# and the comparison against the untruncated run would stop isolating the fit.
const TAIL  = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 0

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
    # One process per core, each SERIAL inside. Letting BLAS open its default thread pool
    # under N worker processes oversubscribes the machine and has produced NaNs and false
    # timeouts in this project.
    using LinearAlgebra; BLAS.set_num_threads(1)
    using ForecastTester
    include($HARNESS_SRC)
    using DataFrames, TimeSeries, Statistics
    const Sxw = Main.ForecastTester.Sarimax
    const FTw = Main.ForecastTester

    # Per-worker cache of the current frequency. This is what keeps the data out of the
    # queue: the master sends only the id and the worker resolves it locally.
    const CACHE = Dict{String,Any}()
    function data(freq::String)
        get!(CACHE, freq) do
            empty!(CACHE)   # one frequency alive at a time, on the worker too
            FTw.build_train_test_dict(FTw.read_dataframes(freq)...)
        end
    end

    smape(f, a) = mean(2 .* abs.(f .- a) ./ (abs.(f) .+ abs.(a) .+ 1e-12)) * 100
    function maseDen(y, m)
        den = m < length(y) ? mean(abs.(y[m+1:end] .- y[1:end-m])) : mean(abs.(diff(y)))
        den <= 0 ? 1e-9 : den
    end
    mase(f, a, y, m) = mean(abs.(f .- a)) / maseDen(y, m)
    vec2str(v) = join((string(round(x, digits = 6)) for x in v), ';')

    """
        m4Config(seasonality, initialization, cap) -> Dict{Symbol,Any}

    THE configuration of campaign A, stated argument by argument. Nothing here is
    inherited. Values that happen to equal the current package default are still written
    out, because "equal to the default" is a fact about today's package, not about the
    run.
    """
    function m4Config(s::Int, ini::Symbol, cap)
        Dict{Symbol,Any}(
            # --- what is being estimated ---
            :seasonality                     => s,
            :objectiveFunction               => "mse",
            :initialization                  => ini,
            :seasonalForm                    => :multiplicative,
            # --- admissibility: constrain by construction, reject by rule ---
            :stationary                      => true,
            :stationarityMargin              => 1e-6,   # DEFAULT_DOMAIN_MARGIN
            :invertible                      => false,
            :invertibilityMargin             => 1e-6,   # DEFAULT_DOMAIN_MARGIN
            :assertStationarity              => true,
            :assertInvertibility             => true,
            :rootMargin                      => 1e-2,   # DEFAULT_ROOT_MARGIN
            :constrainedRefit                => false,
            # --- search ---
            :searchMethod                    => "stepwise",
            :informationCriteria             => "aicc",
            :integrationTest                 => "kpssShort",
            :seasonalIntegrationTest         => "seas",
            :d                               => -1,     # -1 = select by test
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
            # --- solver / budget ---
            :maxTimeSeconds                  => cap,
            :parallel                        => false,
            # --- objective-specific knobs, inert under "mse" but pinned anyway ---
            :cvarLevel                       => 0.9,    # DEFAULT_CVAR_LEVEL
            # --- guards that are OFF in this campaign ---
            :outlierDetection                => false,
            :requireTermsWhenOverDifferenced => false,
            :requireMAWhenDoublyDifferenced  => false,
        )
        # NOTE ON `exogDynamics`: not applicable. The M4 series are univariate and no
        # `exog` is passed, so the exogenous-equation semantics never engage. It is
        # therefore deliberately absent rather than pinned to a meaningless value.
        # NOTE ON `optimizer`: left at the package default (Ipopt.Optimizer). Naming the
        # type here would pin the constructor but not the solver build; the Ipopt version
        # is pinned by env/Manifest.toml and recorded per row instead.
    end

    function runSeries(job)
        (freq, sid, ini, cap, tail, prov) = job
        s = FTw.GRANULARITY_DICT[freq]["s"]; H = FTw.GRANULARITY_DICT[freq]["H"]
        dd = data(freq)
        yFull = Float64.(dd[sid]["train"]); yt = Float64.(dd[sid]["test"])
        # the fit sees only the tail; the metric stays anchored on the whole series
        y = (tail > 0 && length(yFull) > tail) ? yFull[(end-tail+1):end] : yFull
        h = min(H, length(yt)); t0 = time()
        try
            m = Sxw.auto(Sxw.loadDataset(DataFrame(y = y)); m4Config(s, ini, cap)...)
            Sxw.predict!(m; stepsAhead = H)
            f = Float64.(TimeSeries.values(m.forecast))
            # THE VECTORS ARE THE POINT. With forecast + actual + MASE denominator, any
            # metric at any horizon (short/medium/long, or h by h) stays recomputable
            # WITHOUT RE-RUNNING — including metrics not yet chosen, and OWA, which only
            # needs the Naive2 reference. An earlier run stored sMAPE alone, ended up
            # without MASE, could not produce OWA, and had to be redone in full.
            vcat([freq, sid, length(y), m.p, m.d, m.q, m.P, m.D, m.Q,
                  round(smape(f[1:h], yt[1:h]), digits = 4),
                  round(mase(f[1:h], yt[1:h], yFull, s), digits = 4),
                  round(maseDen(yFull, s), digits = 6),
                  minimum(f) < 0 ? 1 : 0, round(time() - t0, digits = 2),
                  get(m.metadata, "solverStatus", "unknown"), "OK",
                  vec2str(f[1:h]), vec2str(yt[1:h])], prov)
        catch e
            vcat([freq, sid, length(y), -1,-1,-1,-1,-1,-1, NaN, NaN, NaN, -1,
                  round(time() - t0, digits = 2), "-",
                  "ERROR:" * first(replace(sprint(showerror, e), '\n'=>' ', ','=>' '), 30),
                  "", ""], prov)
        end
    end
end

const FT = Main.ForecastTester
const PROV_KEYS = [k for (k, _) in provenanceFields()]
const PROV_VALS = [v for (_, v) in provenanceFields()]

done = Set{Tuple{String,Int}}()
if isfile(OUT)
    for l in eachline(OUT)
        startswith(l, "#") && continue
        v = split(l, ','); length(v) < 2 && continue
        n = tryparse(Int, v[2]); isnothing(n) || push!(done, (strip(v[1], '"'), n))
    end
    println("resuming: $(length(done)) series already present")
else
    open(OUT, "w") do io
        stamp(io)
        # Provenance is repeated as COLUMNS on every row, not only in the header comment:
        # a row that is copied into a table, or concatenated with rows from another
        # campaign, still carries the commit and the solver stack it came from.
        println(io, join(vcat(["freq","series","T","p","d","q","P","D","Q","smape","mase",
                               "mase_den","forecast_negative","seconds","solver_status",
                               "status","forecast","actual"], PROV_KEYS), ','))
    end
end

println("tail = $(TAIL == 0 ? "whole series" : "last $(TAIL)") | " *
        "initialization = :$(INIT) | cap = $(isnothing(CAP) ? "NONE" : "$(CAP)s") | " *
        "frequencies = $(join(FREQS, ", ")) | $NW workers")
flush(stdout)

total = 0
for freq in FREQS
    dd = FT.build_train_test_dict(FT.read_dataframes(freq)...)
    ids = [sid for sid in sort(collect(keys(dd))) if !((freq, sid) in done)]
    # ORDER BY LENGTH, shortest first. Cost grows steeply with T, so ascending order
    # maximises series per unit time and makes an early stop INTERPRETABLE: instead of
    # "60% at random" it is "every series up to T = X".
    #
    # THE PRICE, which must be declared alongside any partial: an early stop is a sample
    # BIASED TOWARDS THE EASY SERIES. On a frequency with continuous T, a partial does not
    # represent the population.
    lengths = Dict(sid => length(dd[sid]["train"]) for sid in ids)
    dd = nothing; GC.gc()
    isempty(ids) && (println("$freq: nothing to do"); continue)
    sort!(ids; by = sid -> lengths[sid])
    println("--- $freq: $(length(ids)) series ---"); flush(stdout)

    chan = RemoteChannel(() -> Channel{Any}(4096))
    writer = @async begin
        n = 0; t0 = time()
        open(OUT, "a") do io
            while true
                r = take!(chan); r === nothing && break
                println(io, join(map(x -> x isa AbstractString ? "\"$x\"" : x, r), ","))
                flush(io)   # per row: without this there is no way to tell slow from dead
                n += 1
                if n % 100 == 0
                    el = time() - t0
                    println("$freq: $n/$(length(ids)) | " *
                            "$(round(n/el*3600, digits=0)) series/h | " *
                            "eta $(round(el/n*(length(ids)-n)/60, digits=1)) min")
                    flush(stdout)
                end
            end
        end
        n
    end
    pmap(sid -> (put!(chan, runSeries((freq, sid, INIT, CAP, TAIL, PROV_VALS))); nothing),
         ids; batch_size = 1, on_error = e -> nothing)
    put!(chan, nothing)
    n = fetch(writer); global total += n
    println("$freq: $n series written"); flush(stdout)
    GC.gc()
end
println("M4_DONE: $total series -> $OUT")
