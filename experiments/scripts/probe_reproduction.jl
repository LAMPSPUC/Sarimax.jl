# Reproduction probe: runs every table cell of the M4 monthly benchmark against an
# ARBITRARY Sarimax.jl checkout, with all estimation arguments declared explicitly, and
# records the selected order and sMAPE per series.
#
# Diffing its output against the stored campaign rows answers directly: "does the script
# with explicit arguments, under this build of the package, reproduce the table?" That is
# what section 7 of REPRODUCE.md reports, and re-running this is how to re-establish it if
# the package moves.
#
# Usage: julia --project=<pkgroot> probe_reproduction.jl <pkgroot> <nseries> <out.csv>
#
# The environment must be held fixed for the comparison to isolate the package: same host,
# same Julia, same Manifest. Only the checkout under test should differ.
PKG = ARGS[1]
NSER = parse(Int, ARGS[2])
OUT = ARGS[3]
NW = 8

import Pkg
Pkg.activate(PKG)
using Distributed
addprocs(NW; exeflags = ["--project=" * PKG])

@everywhere begin
    PKG = $PKG
    import Pkg
    Pkg.activate(PKG)
    using LinearAlgebra
    BLAS.set_num_threads(1)
    using CSV, DataFrames, TimeSeries, Statistics
    include(joinpath(PKG, "src", "Sarimax.jl"))
    const Sx = Main.Sarimax

    sMAPE(a, f) =
        (200 / length(a)) * sum(abs(a[i] - f[i]) / (abs(a[i]) + abs(f[i])) for i in eachindex(a))

    # Every argument that affects estimation, declared. The values are the ones in force in
    # this host's campaigns, checked against the `auto` signature at commits 87f7bfb and
    # 5b2ec6b, which are identical to each other except for the `initialization` default.
    # `lambda`/`alpha` are deliberately absent: under `ridge` the package rejects `lambda`.
    # `exogDynamics`/`penaltyTarget` are absent because they did not exist at the campaign
    # commits; passing them would test a different configuration than the one that ran.
    function probeSeries(job)
        (sid, y, yTest, objective, initialization, cap) = job
        s, H = 12, 18
        try
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
                requireTermsWhenOverDifferenced = true,
                requireMAWhenDoublyDifferenced = false,
            )
            Sx.predict!(m; stepsAhead = H)
            f = Float64.(TimeSeries.values(m.forecast))
            h = min(H, length(yTest))
            Any[sid, objective, String(initialization), m.p, m.d, m.q, m.P, m.D, m.Q,
                round(sMAPE(yTest[1:h], f[1:h]), digits = 6), "OK"]
        catch e
            Any[sid, objective, String(initialization), -1, -1, -1, -1, -1, -1, NaN,
                "ERROR:" * first(replace(sprint(showerror, e), '\n' => ' ', ',' => ' '), 80)]
        end
    end
end

const DATA = get(ENV, "M4_DATASETS", "datasets")
dfTrain = CSV.read(joinpath(DATA, "Monthly-train.csv"), DataFrame)
dfTest = CSV.read(joinpath(DATA, "Monthly-test.csv"), DataFrame)

# (objective, initialization, cap) - mirrors the two campaign families on this host:
# `cel_*_zeroed_*` ran with a 600 s model cap; `obj_*_innov_*` ran with no cap.
CELLS = [("mse", :zeroed, 600.0), ("huber", :zeroed, 600.0), ("mae", :zeroed, 600.0),
    ("mse", :innovations, nothing), ("huber", :innovations, nothing),
    ("mae", :innovations, nothing), ("ridge", :innovations, nothing)]

jobs = Any[]
for i = 1:NSER
    raw = Vector(dfTrain[i, :])[2:end]
    yTrain = Float64.(raw[1:findlast(x -> !ismissing(x), raw)])
    yTest = Float64.(Vector(dfTest[i, :])[2:end])
    for (o, ini, cap) in CELLS
        push!(jobs, (i, yTrain, yTest, o, ini, cap))
    end
end

println("probe | package=$PKG | $(NSER) series x $(length(CELLS)) cells = $(length(jobs)) fits")
flush(stdout)

res = pmap(probeSeries, jobs)

open(OUT, "w") do io
    println(io, "series,objective,initialization,p,d,q,P,D,Q,smape_total,status")
    for r in res
        println(io, join(map(x -> x isa AbstractString ? "\"$x\"" : x, r), ","))
    end
end
println("PROBE_DONE -> $OUT")
