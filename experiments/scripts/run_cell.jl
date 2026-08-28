# Runs one cell of the M4 benchmark: one objective function x one initialization x one
# frequency, over the series of that frequency.
#
# Usage:
#   julia --project=<sarimax-root> run_cell.jl <nseries> <workers> <out.csv> \
#         <objective> <frequency> <orchestrationLimit> <initialization> <cap> <requireTerms>
#
#   nseries        0 = every series of the frequency
#   cap            -1 = production rule (short ? 120.0 : nothing); 0 = no cap; >0 = that value
#   requireTerms   true|false -> requireTermsWhenOverDifferenced
#
# Writes incrementally and is resumable: re-running skips series already present in the
# output file.
#
# `ridge` with `initialization = :innovations` requires a package build where the penalized
# pre-sample block covers that objective. On builds where it does not, `auto` raises
# ArgumentError rather than degrading silently - see REPRODUCE.md.

NSER = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
NW = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
OUT = length(ARGS) >= 3 ? ARGS[3] : "cell.csv"
OBJ = length(ARGS) >= 4 ? ARGS[4] : "mse"
GRAN = length(ARGS) >= 5 ? ARGS[5] : "monthly"
# Orchestration deadline, NOT a cap on the model. The model cap is `maxTimeSeconds`, passed
# to `auto` in the worker. This limit must exceed the legitimate running time of the
# slowest series, otherwise the runner kills a worker that is merely slow.
LIM = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 600.0
INI = Symbol(length(ARGS) >= 7 ? ARGS[7] : "innovations")
CAP = length(ARGS) >= 8 ? parse(Float64, ARGS[8]) : -1.0
# `requireTermsWhenOverDifferenced` is an AXIS, not a constant. The guard does not act by
# rejecting the winner: it removes (0,d,0)(0,D,0) from the candidate set, which is one of
# the stepwise seeds, and so changes the TRAJECTORY of the search.
REQT = length(ARGS) >= 9 ? (lowercase(ARGS[9]) in ("1", "true")) : true

const PKG = get(ENV, "SARIMAX_SRC", abspath(joinpath(@__DIR__, "..", "..")))
ENV["SARIMAX_PROJECT"] = get(ENV, "SARIMAX_PROJECT", PKG)

import Pkg
Pkg.activate(ENV["SARIMAX_PROJECT"])
using Distributed
addprocs(NW; exeflags = ["--project=" * ENV["SARIMAX_PROJECT"]])
@everywhere include(joinpath(@__DIR__, "cell_worker.jl"))

# ---------------------------------------------------------------------------------------
# Provenance stamp. Every output row carries it.
#
# Measured reason this exists: three weeks of package changes moved the weekly OWA by
# 0.0320 - more than the effect the experiment was isolating - and changed the sMAPE of 62%
# of the series. Without the commit on the row, a number from one week and a number from
# the next are not comparable, and the mismatch is later misread as "machine" or
# "configuration". A dirty tree is stamped too: a result from a modified tree is not
# reproducible from the commit alone.
# ---------------------------------------------------------------------------------------
const COMMIT = try
    sha = strip(read(`git -C $(PKG) rev-parse --short HEAD`, String))
    dirty = !isempty(strip(read(`git -C $(PKG) status --porcelain`, String)))
    # When HEAD sits exactly on a tag, record the tag next to the SHA. A run from a release
    # should say which release; the SHA alone makes the reader look it up, and a tag can be
    # moved, so neither identifier is sufficient alone.
    tag = try
        strip(read(`git -C $(PKG) describe --tags --exact-match`, String))
    catch
        ""
    end
    stamp = isempty(tag) ? sha : "$(tag) ($(sha))"
    stamp * (dirty ? "-dirty" : "")
catch
    "unknown"
end

function depVersion(name::AbstractString)
    for (_, p) in Pkg.dependencies()
        p.name == name && return string(something(p.version, "unknown"))
    end
    return "unknown"
end

const JULIA_V = string(VERSION)
const JUMP_V = depVersion("JuMP")
const MOI_V = depVersion("MathOptInterface")
const IPOPT_V = depVersion("Ipopt")
const SCIP_V = depVersion("SCIP")
const OSARCH = string(Sys.KERNEL) * " " * string(Sys.MACHINE)
const HOST = gethostname()
# No random number generator is reached on the fit/forecast path: `Random.seed!` and `rand`
# occur only in Sarimax's simulation entry points, which this script never calls. There is
# therefore no seed to fix; the value below records that fact rather than a number.
const SEED = "n/a-deterministic"

println("package commit: $COMMIT | julia $JULIA_V | JuMP $JUMP_V | MOI $MOI_V | Ipopt $IPOPT_V")

function reinitWorker(w::Int)
    Distributed.remotecall_eval(Main, [w], :(include(joinpath($(@__DIR__), "cell_worker.jl"))))
    # Confirm the replacement is usable. Without this a badly initialised worker goes back
    # into the queue and fails every series it receives, contaminating the whole run.
    remotecall_fetch(() -> isdefined(Main, :runSeries), w) ||
        error("worker $w reinitialised without runSeries")
    return nothing
end

# pmap with a per-task deadline. `istaskdone` is a LOCAL check on the master: a worker stuck
# in native code never blocks the loop, unlike `isready(::Future)`.
function pmapWithTimeout(f::Function, inputs::Vector, limit::Float64, ncols::Int)
    n = length(inputs)
    out = Vector{Any}(undef, n)
    free = collect(workers())
    ospid = Dict(w => remotecall_fetch(getpid, w) for w in free)
    running = Dict{Int,NamedTuple{(:idx, :task, :t0),Tuple{Int,Task,Float64}}}()
    nextJob = 1
    # 10 header columns (series, T, the six orders, has_mean, has_drift), then the 8 metrics
    # plus mase_den as NaN, then fallback, time, status, solver, forecast. `ncols - 15`
    # comes from 10 + X + 5 = ncols; getting it wrong shifts the entire CSV row.
    failureRow(e, elapsed, reason) =
        Any[e[1], length(e[2]), fill(-1, 8)..., fill(NaN, ncols - 15)..., elapsed, reason, "", ""]

    # Rebuild a dead worker and REFUND the stall to the series still in flight: `addprocs`
    # and the package include block the master for one to two minutes, and without the
    # refund several in-flight series blow their deadline for that reason alone, each one
    # triggering another rebuild in a cascade.
    function replace!(w, reason)
        tRe = time()
        pid = get(ospid, w, 0)
        delete!(ospid, w)
        pid > 0 && (try
            run(pipeline(`taskkill /F /T /PID $(pid)`; stdout = devnull, stderr = devnull))
        catch
        end)
        @async try
            rmprocs(w; waitfor = 0)
        catch
        end
        fresh = nothing
        for _ = 1:3
            try
                cand = addprocs(1; exeflags = ["--project=" * ENV["SARIMAX_PROJECT"]])[1]
                reinitWorker(cand)
                ospid[cand] = remotecall_fetch(getpid, cand)
                fresh = cand
                break
            catch
            end
        end
        fresh === nothing ? printstyled("  no replacement ($reason)\n"; color = :red) :
        push!(free, fresh)
        stalled = time() - tRe
        if stalled > 1.0
            for (w2, i2) in collect(running)
                running[w2] = (idx = i2.idx, task = i2.task, t0 = i2.t0 + stalled)
            end
        end
        isempty(free) && isempty(running) && error("no worker available")
    end

    while nextJob <= n || !isempty(running)
        while !isempty(free) && nextJob <= n
            w = pop!(free)
            idx = nextJob
            t = let w = w, idx = idx
                @async try
                    (:ok, remotecall_fetch(f, w, inputs[idx]))
                catch e
                    (:err, e)
                end
            end
            running[w] = (idx = idx, task = t, t0 = time())
            nextJob += 1
        end
        for (w, info) in collect(running)
            if istaskdone(info.task)
                st, val = fetch(info.task)
                delete!(running, w)
                if st === :ok
                    out[info.idx] = val
                    push!(free, w)
                else
                    # The reason is RECORDED and the worker is not recycled: a remote
                    # failure labelled "TIMEOUT" hides the exception from whoever reads the
                    # CSV, and a dead worker back in the queue fails everything after it.
                    msg = first(replace(sprint(showerror, val), '\n' => ' ', ',' => ' '), 60)
                    out[info.idx] = failureRow(inputs[info.idx],
                        round(time() - info.t0, digits = 2), "REMOTE:" * msg)
                    printstyled("REMOTE FAILURE series $(inputs[info.idx][1]): $(msg)\n";
                        color = :red)
                    flush(stdout)
                    replace!(w, "remote failure")
                end
            elseif time() - info.t0 > limit
                printstyled("TIMEOUT series $(inputs[info.idx][1]) after $(round(Int, time() - info.t0))s\n";
                    color = :red)
                flush(stdout)
                out[info.idx] = failureRow(inputs[info.idx], limit, "TIMEOUT")
                delete!(running, w)
                replace!(w, "timeout")
            end
        end
        (isempty(free) || nextJob > n) && sleep(0.5)
    end
    return out
end

# Datasets are read directly, without the harness (see the worker header). The ROW index is
# the series id, matching `build_train_test_dict` (src/preparedata.jl) of the parent
# repository and matching the baseline CSVs under results/baselines/.
const FILE = uppercase(GRAN[1]) * GRAN[2:end]
const DATA = get(ENV, "M4_DATASETS", "datasets")
dfTrain = CSV.read(joinpath(DATA, "$(FILE)-train.csv"), DataFrame)
dfTest = CSV.read(joinpath(DATA, "$(FILE)-test.csv"), DataFrame)

function series(i)
    raw = Vector(dfTrain[i, :])[2:end]
    yTrain = Float64.(raw[1:findlast(x -> !ismissing(x), raw)])
    return (i, yTrain, Float64.(Vector(dfTest[i, :])[2:end]), OBJ, GRAN, INI, CAP, REQT)
end

const PROV = ["sarimax_commit", "julia_version", "jump_version", "moi_version",
    "ipopt_version", "scip_version", "os_arch", "host", "objective", "initialization",
    "max_time_seconds", "require_terms_over_differenced", "frequency", "seed"]
const HDR = "series,T,p,d,q,P,D,Q,has_mean,has_drift," *
            "smape_short,mase_short,smape_medium,mase_medium,smape_long,mase_long," *
            "smape_total,mase_total,mase_den,fallback,time,status,solver,forecast," *
            join(PROV, ",")
const NCOLS = 24   # columns produced by the worker, before the provenance block

capLabel = CAP < 0 ? "production-rule" : (CAP == 0 ? "none" : string(CAP))
provRow = [COMMIT, JULIA_V, JUMP_V, MOI_V, IPOPT_V, SCIP_V, OSARCH, HOST,
    OBJ, string(INI), capLabel, string(REQT), GRAN, SEED]

done = Set{Int}()
if isfile(OUT)
    for (i, l) in enumerate(eachline(OUT))
        i == 1 && continue
        v = split(l, ',')
        isempty(v[1]) || push!(done, parse(Int, v[1]))
    end
    println("resuming: $(length(done)) rows already present")
else
    open(OUT, "w") do io
        println(io, HDR)
    end
end

target = NSER <= 0 ? nrow(dfTrain) : min(NSER, nrow(dfTrain))
ids = [i for i = 1:target if !(i in done)]
println("cell | objective=$OBJ | init=:$INI | requireTerms=$REQT | cap=$capLabel | " *
        "$GRAN | $(length(ids)) series | workers=$NW | orchestrationLimit=$(LIM)s")
flush(stdout)

completed = 0
# Batches of 20: the file is written at the end of a batch, so a large batch means hours
# with no visibility into what is happening inside it.
for chunk in Iterators.partition([series(i) for i in ids], 20)
    res = pmapWithTimeout(runSeries, collect(chunk), LIM, NCOLS)
    open(OUT, "a") do io
        for r in res
            row = vcat(r, provRow)
            println(io, join(map(x -> x isa AbstractString ? "\"$x\"" : x, row), ","))
        end
    end
    global completed += length(chunk)
    println("progress: $completed/$(length(ids))")
    flush(stdout)
end
println("CELL_DONE -> $OUT")
