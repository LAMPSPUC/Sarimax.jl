# Does the script, with every argument explicit, reproduce the archived numbers under the
# current package tree?
#
# This does NOT re-run a campaign. It draws a seeded sample of series already present in an
# archived result file, re-fits them under the campaign's explicit configuration, and
# compares sMAPE and selected order against the stored values. A campaign that takes hours
# is answered in minutes.
#
# Run this against any release before publishing material alongside it: the verification
# recorded in REPRODUCE.md was made against one specific commit, and changes merged
# afterwards are not covered by it.
#
#   julia --project=<harness> verify_reproduction.jl <campaign> <n> <workers> [rawDir]
#
# campaign in {m4, objectives, multistart, axes, stable}
const R_HOME = get(ENV, "R_HOME", "")
isempty(R_HOME) && error("set R_HOME to the R installation directory (RCall needs it)")

const CAMP = length(ARGS) >= 1 ? ARGS[1] : "m4"
const NSAMP = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 25
const NW    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 6
const RAW   = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "..", "results", "raw")
# Fixed so that a re-check draws the same series and the answer is comparable over time.
const SEED  = 20260828

import Pkg; Pkg.activate(".")
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

    # The configurations below MUST stay identical to those in the runners. They are
    # restated rather than imported because the point of this check is to catch a drift
    # between the recorded number and the declared configuration; sharing the definition
    # would let a single edit move both sides at once and the check would pass vacuously.
    function baseConfig(s)
        Dict{Symbol,Any}(:seasonality=>s, :objectiveFunction=>"mse",
            :initialization=>:innovations, :seasonalForm=>:multiplicative,
            :stationary=>true, :stationarityMargin=>1e-6,
            :invertible=>false, :invertibilityMargin=>1e-6,
            :assertStationarity=>true, :assertInvertibility=>true,
            :rootMargin=>1e-2, :constrainedRefit=>false,
            :searchMethod=>"stepwise", :informationCriteria=>"aicc",
            :integrationTest=>"kpssShort", :seasonalIntegrationTest=>"seas",
            :d=>-1, :D=>-1, :maxd=>2, :maxD=>1,
            :maxp=>5, :maxq=>5, :maxP=>2, :maxQ=>2, :maxOrder=>5,
            :multistart=>false, :warmStartFromBox=>false, :maxTimeSeconds=>nothing,
            :parallel=>false, :cvarLevel=>0.9, :outlierDetection=>false,
            :requireTermsWhenOverDifferenced=>false,
            :requireMAWhenDoublyDifferenced=>false)
    end
    stableCfg(s, obj, lv) = (c = baseConfig(s); c[:objectiveFunction] = obj; c[:cvarLevel] = lv; c)
    objectivesCfg(s, obj, ini) = (c = baseConfig(s); c[:objectiveFunction] = obj;
                                  c[:initialization] = ini; c[:maxTimeSeconds] = 120.0; c)
    multistartCfg(s, ms) = (c = baseConfig(s); c[:initialization] = :penalized;
                            c[:maxTimeSeconds] = 120.0; c[:multistart] = ms; c)
    function axesCfg(s, req, tests)
        c = baseConfig(s); c[:requireTermsWhenOverDifferenced] = req
        if !tests
            delete!(c, :integrationTest); delete!(c, :seasonalIntegrationTest)
        end
        c
    end

    function refit(job)
        (freq, sid, arm, cfg) = job
        H = FTw.GRANULARITY_DICT[freq]["H"]
        dd = data(freq)
        y = Float64.(dd[sid]["train"]); yt = Float64.(dd[sid]["test"])
        h = min(H, length(yt))
        try
            m = Sxw.auto(Sxw.loadDataset(DataFrame(y = y)); cfg...)
            Sxw.predict!(m; stepsAhead = H)
            f = Float64.(TimeSeries.values(m.forecast))
            (freq, sid, arm, round(smape(f[1:h], yt[1:h]), digits = 4),
             string(m.p, ",", m.d, ",", m.q, ",", m.P, ",", m.D, ",", m.Q), "OK")
        catch e
            (freq, sid, arm, NaN, "-",
             string("ERROR:", first(replace(sprint(showerror, e), '\n'=>' ', ','=>' '), 40)))
        end
    end
end
const FT = Main.ForecastTester

"""
    readArchive(path, armCol, freqCol, fixedFreq) -> Dict{(arm,freq,series) => (smape, order)}

Reads an archived result file, skipping the `#` provenance block.
"""
function readArchive(path, armCol, freqCol, fixedFreq = "")
    out = Dict{Tuple{String,String,Int},Tuple{Float64,String}}()
    hdr = nothing
    for line in eachline(path)
        startswith(line, "#") && continue
        v = split(strip(line), ',')
        if hdr === nothing; hdr = v; continue; end
        ix = Dict(hdr[i] => i for i in eachindex(hdr))
        (haskey(ix, "status") && length(v) >= length(hdr)) || continue
        startswith(strip(v[ix["status"]], '"'), "OK") || continue
        sid = tryparse(Int, v[ix["series"]]); isnothing(sid) && continue
        sm = tryparse(Float64, v[ix["smape"]]); isnothing(sm) && continue
        arm = isempty(armCol) ? "-" : strip(v[ix[armCol]], '"')
        fr = isempty(freqCol) ? fixedFreq : strip(v[ix[freqCol]], '"')
        order = all(k -> haskey(ix, k), ("p","d","q","P","D","Q")) ?
                join([strip(v[ix[k]], '"') for k in ("p","d","q","P","D","Q")], ",") : "-"
        out[(arm, fr, sid)] = (sm, order)
    end
    out
end

jobs = Any[]
archive = Dict{Tuple{String,String,Int},Tuple{Float64,String}}()
Random.seed!(SEED)
pick(pool) = shuffle(pool)[1:min(NSAMP, length(pool))]

if CAMP == "m4"
    for freq in ("yearly","quarterly","weekly","hourly","daily","monthly")
        path = joinpath(RAW, "m4_innovations_$(freq).csv")
        isfile(path) || continue
        d = readArchive(path, "", "freq"); merge!(archive, d)
        s = FT.GRANULARITY_DICT[freq]["s"]
        for sid in pick([x for (a,f,x) in keys(d) if f == freq])
            push!(jobs, (freq, sid, "-", baseConfig(s)))
        end
    end
elseif CAMP == "stable"
    d = readArchive(joinpath(RAW, "stable_weekly_yearly.csv"), "arm", "freq")
    merge!(archive, d)
    for freq in ("weekly","yearly")
        s = FT.GRANULARITY_DICT[freq]["s"]
        ids = pick(unique([x for (a,f,x) in keys(d) if f == freq]))
        for (arm, obj, lv) in [("mse","mse",0.9), ("stable_090","stable",0.9),
                               ("stable_050","stable",0.5)]
            for sid in ids; push!(jobs, (freq, sid, arm, stableCfg(s, obj, lv))); end
        end
    end
elseif CAMP == "objectives"
    for (file, cells) in [("objectives_monthly.csv",
                           [("mse","mse",:penalized), ("mse_zeroed","mse",:zeroed),
                            ("ridge","ridge",:zeroed), ("mae","mae",:zeroed),
                            ("huber","huber",:zeroed)]),
                          ("objectives_monthly_penalized.csv",
                           [("ridge_pen","ridge",:penalized), ("mae_pen","mae",:penalized),
                            ("huber_pen","huber",:penalized)])]
        path = joinpath(RAW, file); isfile(path) || continue
        d = readArchive(path, "cell", "", "monthly"); merge!(archive, d)
        s = FT.GRANULARITY_DICT["monthly"]["s"]
        for (arm, obj, ini) in cells
            pool = [x for (a,f,x) in keys(d) if a == arm]
            isempty(pool) && continue
            for sid in pick(pool); push!(jobs, ("monthly", sid, arm, objectivesCfg(s, obj, ini))); end
        end
    end
elseif CAMP == "multistart"
    d = readArchive(joinpath(RAW, "multistart_random.csv"), "cell", "", "monthly")
    merge!(archive, d)
    s = FT.GRANULARITY_DICT["monthly"]["s"]
    for arm in unique([a for (a,f,x) in keys(d)])
        pool = [x for (a,f,x) in keys(d) if a == arm]
        for sid in pick(pool); push!(jobs, ("monthly", sid, arm, multistartCfg(s, arm != "base"))); end
    end
elseif CAMP == "axes"
    d = readArchive(joinpath(RAW, "axes_weekly.csv"), "cell", "freq"); merge!(archive, d)
    s = FT.GRANULARITY_DICT["weekly"]["s"]
    for (arm, req, tests) in [("base",false,false), ("req",true,false),
                              ("tests",false,true), ("req+tests",true,true)]
        pool = [x for (a,f,x) in keys(d) if a == arm]
        isempty(pool) && continue
        for sid in pick(pool); push!(jobs, ("weekly", sid, arm, axesCfg(s, req, tests))); end
    end
else
    error("unknown campaign: $CAMP")
end

println("VERIFY $CAMP: $(length(jobs)) refits | $NW workers | seed $SEED"); flush(stdout)
res = pmap(refit, jobs; batch_size = 1, on_error = e -> nothing)

out = joinpath(RAW, "..", "verify_$(CAMP).csv")
total = 0; differing = 0
open(out, "w") do io
    println(io, "freq,arm,series,smape_archived,smape_refitted,delta,order_archived,order_refitted,status")
    for r in res
        isnothing(r) && continue
        (freq, sid, arm, sm, order, st) = r
        k = (arm, freq, sid); haskey(archive, k) || continue
        (smA, orderA) = archive[k]
        delta = isnan(sm) ? NaN : round(sm - smA, digits = 6)
        total += 1
        (isnan(sm) || abs(delta) > 1e-9) && (differing += 1)
        println(io, join([freq, arm, sid, smA, sm, delta,
                          string('"', orderA, '"'), string('"', order, '"'),
                          string('"', st, '"')], ","))
    end
end
println("compared: $total | identical: $(total - differing) | differing: $differing")
println(differing == 0 ? "REPRODUCES" : "DOES NOT REPRODUCE — inspect $out")
println("VERIFY_DONE -> ", out)
