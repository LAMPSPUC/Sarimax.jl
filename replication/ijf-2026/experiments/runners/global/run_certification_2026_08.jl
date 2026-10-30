# Re-run of the August 2026 certification batch (Table S5, Section 5.6) with its own harness.
#
# The model is the batch's: the harness archived in results/historical/certification-2026-08/
# received/harness/scip_prog_lote2_22-08_1435.jl is loaded verbatim (casos, monta, the Ipopt and
# SCIP options), so the conditional least-squares problem, the lifted products in boxes +-B^2,
# the innovation bounds |eps_t| <= sqrt(f_Ipopt)(1 + 1e-9), the warm start and the 600 s limit
# are the ones that ran. Only these are replaced:
#   * the data: the synthetic series are read from experiments/configs/
#     certification_2026_08_instances.csv (written by certification_2026_08_instances.jl under
#     Julia >= 1.11, as the batch's Julia 1.12.3 drew them), the real M4 series from the M4
#     monthly training file of the package's data directory (IJF_M4_DIR), as the harness did;
#   * SCIP's display level, 0 in the batch, is 4 here, so that every case leaves a log (the
#     display level does not change the solve);
#   * what is recorded: besides the batch's columns, Ipopt's status, time and coefficients, SCIP's
#     raw status, primal status, node count, solve time and coefficients, and one log per case;
#   * the external clock: every case runs in its own process, killed after --hard-limit seconds
#     (900, as the batch's scip_wd.sh for ids 34 and 35, which ignored SCIP's limits/time).
#
# Usage (through `./reproduce.sh full --experiment global_optimization_grid --jobs N`):
#   julia run_certification_2026_08.jl INSTANCES.csv [--only 1,2,3] [--hard-limit 900] [--ipopt-only]
#   julia run_certification_2026_08.jl INSTANCES.csv --case ID [--ipopt-only]     (one case; internal)
# Output: $IJF_RUN_DIR/raw/certification-2026-08/records.csv (one row per case, in id order),
# cases/<id>.csv and logs/<id>.log. --ipopt-only skips SCIP (checks the data on any platform; SCIP
# crashes on native Windows).

using Printf

const REPL = normpath(joinpath(@__DIR__, "..", "..", ".."))
const HARNESS = joinpath(REPL, "results", "historical", "certification-2026-08", "received", "harness",
                         "scip_prog_lote2_22-08_1435.jl")
const COLUMNS = ["id", "cel", "rotulo", "P", "Q", "T", "rep", "modo", "sid",
                 # the batch's columns
                 "obj_ipopt", "obj_scip", "dual_scip", "status", "t_scip", "gap_rel", "gap_certificado", "piso_ok",
                 # recorded here, not in the batch
                 "ipopt_status", "t_ipopt", "ipopt_phi", "ipopt_theta", "ipopt_Phi", "ipopt_Theta",
                 "scip_raw_status", "scip_primal_status", "scip_nodes", "scip_solve_time",
                 "scip_phi", "scip_theta", "scip_Phi", "scip_Theta", "scip_version", "julia_version", "note"]

function parse_args(args)
    o = Dict{String,Any}("file" => nothing, "only" => Int[], "hard" => 900.0, "case" => 0, "ipopt_only" => false)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--only";           o["only"] = parse.(Int, split(args[i+1], ',')); i += 2
        elseif a == "--hard-limit"; o["hard"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--case";       o["case"] = parse(Int, args[i+1]); i += 2
        elseif a == "--ipopt-only"; o["ipopt_only"] = true; i += 1
        else                        o["file"] = abspath(a); i += 1
        end
    end
    o["file"] === nothing && error("usage: run_certification_2026_08.jl INSTANCES.csv [--only IDS] [--hard-limit S] [--ipopt-only]")
    return o
end

const OPTS = parse_args(ARGS)
const RUN_DIR = let d = get(ENV, "IJF_RUN_DIR", "")
    isempty(d) && error("IJF_RUN_DIR is not set; run through ./reproduce.sh full --experiment global_optimization_grid")
    abspath(d)
end
const OUT = joinpath(RUN_DIR, "raw", "certification-2026-08")

"Instances file: id => (row fields, series)."
function read_instances(path)
    lines = readlines(path)
    head = split(lines[1], ',')
    rows = Dict{Int,Dict{String,String}}()
    for l in lines[2:end]
        f = split(l, ',')
        r = Dict(String(k) => String(v) for (k, v) in zip(head, f))
        rows[parse(Int, r["id"])] = r
    end
    return rows
end

csvfield(x) = x isa AbstractString ? (occursin(r"[,\"\n]", x) ? "\"" * replace(x, "\"" => "\"\"") * "\"" : x) :
              x isa AbstractFloat ? (isnan(x) ? "" : repr(x)) : string(x)

# =============================================================================================
# One case, in its own process
# =============================================================================================
"The harness, verbatim but for its driver and SCIP's display level (loaded at top level)."
function harness_source()
    src = read(HARNESS, String)
    src = replace(src, r"\nmain\(\)\s*$" => "\n")                       # the harness's driver
    occursin("\"display/verblevel\" => 0", src) || error("harness: SCIP options not found")
    return replace(src, "\"display/verblevel\" => 0" => "\"display/verblevel\" => 4")
end

function run_case(id::Int)
    inst = read_instances(OPTS["file"])
    c = only(filter(x -> x.id == id, Main.casos()))
    r = inst[id]
    (r["cell"], r["label"]) == (c.cel, c.rotulo) || error("instance $id: file and harness disagree")
    if c.sid > 0
        m4 = joinpath(get(ENV, "IJF_M4_DIR", ""), "Monthly-train.csv")
        isfile(m4) || error("instance $id is the M4 monthly series $(c.sid): $m4 not found (./reproduce.sh fetch m4)")
        df = Main.CSV.read(m4, Main.DataFrame)
        v = Float64[x for x in df[c.sid, 2:end] if !ismissing(x)]
        y = (v .- Main.mean(v)) ./ Main.std(v)                            # as serie_real() does
    else
        y = parse.(Float64, split(r["y"], ';'))
        length(y) == c.T || error("instance $id: series length $(length(y)) != T = $(c.T)")
    end
    cc = c.sid > 0 ? Main.Caso(c.id, c.cel, c.P, c.Q, length(y), c.rep, c.modo, c.rotulo, c.par, c.sid) : c

    # roda() of the harness, with the extra fields read from the same solved model
    function solve(opt; ub = nothing, warm = nothing)
        m = Main.monta(y, cc, opt; ub = ub, warm = warm); t0 = time()
        Main.optimize!(m); st = string(Main.termination_status(m)); el = time() - t0
        hv = Main.has_values(m)
        co = hv ? [Main.value(m[:ϕ]), Main.value(m[:θ]), cc.P > 0 ? Main.value(m[:Φ]) : 0.0,
                   cc.Q > 0 ? Main.value(m[:Θ]) : 0.0] : fill(NaN, 4)
        tryget(f, default) = try f() catch; default end
        ob = tryget(() -> Main.objective_bound(m), NaN)
        ev = hv ? [Main.value(m[:ε][t]) for t = 1:length(y)] : Float64[]
        e0v = (hv && cc.modo === :innovations) ? Main.value(m[:e0]) : NaN
        extra = (raw = string(tryget(() -> Main.MOI.get(m, Main.MOI.RawStatusString()), "")),
                 primal = string(Main.primal_status(m)),
                 nodes = tryget(() -> Int(Main.MOI.get(m, Main.MOI.NodeCount())), -1),
                 solvetime = tryget(() -> Float64(Main.MOI.get(m, Main.MOI.SolveTimeSec())), NaN))
        return (hv ? Main.objective_value(m) : NaN, st, el, co, ob, ev, e0v, extra)
    end

    println("case $id ($(cc.cel) $(cc.rotulo), T = $(length(y))): Ipopt"); flush(stdout)
    oi = solve(Main.IP())
    @printf("Ipopt %s in %.2f s, objective %.6f\n", oi[2], oi[3], oi[1]); flush(stdout)
    os = OPTS["ipopt_only"] ? (NaN, "NOT_RUN", NaN, fill(NaN, 4), NaN, Float64[], NaN,
                               (raw = "", primal = "", nodes = -1, solvetime = NaN)) :
         solve(Main.SC(Main.TL); ub = oi[1], warm = (oi[4], oi[6], oi[7]))
    piso = 0.25 * (length(y) - 1)
    piso_ok = isnan(os[1]) || os[1] >= piso
    g = (isnan(oi[1]) || isnan(os[1])) ? NaN : (oi[1] - os[1]) / max(abs(os[1]), 1e-12)
    gc = (isnan(os[1]) || isnan(os[5])) ? NaN : (os[1] - os[5]) / max(abs(os[1]), 1e-12)
    scipv = OPTS["ipopt_only"] ? "" : string(Main.SCIP.SCIP_versionnumber())
    row = Any[cc.id, cc.cel, cc.rotulo, cc.P, cc.Q, length(y), cc.rep, string(cc.modo), cc.sid,
              oi[1], os[1], os[5], os[2], os[3], g, gc, piso_ok,
              oi[2], oi[3], oi[4]..., os[8].raw, os[8].primal, os[8].nodes, os[8].solvetime, os[4]...,
              scipv, string(VERSION), ""]
    open(joinpath(OUT, "cases", "$id.csv"), "w") do io
        println(io, join(COLUMNS, ","))
        println(io, join(csvfield.(row), ","))
    end
    @printf("SCIP %s in %.1f s, primal %.6f, dual %.6f, gap %.3e, nodes %d\n", os[2], os[3], os[1], os[5], gc, os[8].nodes)
end

# =============================================================================================
# Driver: one process per case, at most IJF_JOBS at a time, each killed after the hard limit
# =============================================================================================
function drive()
    inst = read_instances(OPTS["file"])
    ids = sort(collect(keys(inst)))
    isempty(OPTS["only"]) || (ids = [i for i in ids if i in OPTS["only"]])
    jobs = clamp(parse(Int, get(ENV, "IJF_JOBS", "1")), 1, max(1, Sys.CPU_THREADS))
    mkpath(joinpath(OUT, "cases")); mkpath(joinpath(OUT, "logs"))
    julia = Base.julia_cmd()
    proj = dirname(Base.active_project())
    killed = Dict{Int,Float64}()
    queue = copy(ids); running = Dict{Int,Tuple{Base.Process,Float64,IOStream}}()
    println("certification batch 2026-08: $(length(ids)) cases, $jobs at a time, hard limit $(OPTS["hard"]) s",
            OPTS["ipopt_only"] ? ", Ipopt only" : "")
    while !isempty(queue) || !isempty(running)
        while !isempty(queue) && length(running) < jobs
            id = popfirst!(queue)
            done = joinpath(OUT, "cases", "$id.csv")
            isfile(done) && continue                                     # resumed run: case already recorded
            cmd = `$julia --project=$proj --startup-file=no $(@__FILE__) $(OPTS["file"]) --case $id`
            OPTS["ipopt_only"] && (cmd = `$cmd --ipopt-only`)
            log = open(joinpath(OUT, "logs", "$id.log"), "w")
            p = run(pipeline(ignorestatus(cmd); stdout = log, stderr = log); wait = false)
            running[id] = (p, time(), log)
        end
        sleep(1)
        for (id, (p, t0, log)) in collect(running)
            if !process_running(p)
                close(log); delete!(running, id)
                success(p) || @warn "case $id exited with code $(p.exitcode); see logs/$id.log"
                println("case $id done in $(round(Int, time() - t0)) s"); flush(stdout)
            elseif time() - t0 > OPTS["hard"]
                kill(p); wait(p); close(log); delete!(running, id); killed[id] = time() - t0
                println("case $id killed by the external clock after $(round(Int, killed[id])) s"); flush(stdout)
            end
        end
    end
    # consolidate, in id order; a killed or failed case keeps a row with its reason
    open(joinpath(OUT, "records.csv"), "w") do io
        println(io, join(COLUMNS, ","))
        for id in ids
            f = joinpath(OUT, "cases", "$id.csv")
            if isfile(f)
                println(io, readlines(f)[2])
            else
                r = inst[id]
                why = haskey(killed, id) ? "KILLED_EXTERNAL_CLOCK_$(round(Int, OPTS["hard"]))s" : "FAILED (see logs/$id.log)"
                row = Any[id, r["cell"], r["label"], r["P"], r["Q"], r["T"], r["draw"], r["mode"], r["sid"],
                          NaN, NaN, NaN, why, NaN, NaN, NaN, "", "", NaN, NaN, NaN, NaN, NaN,
                          "", "", -1, NaN, NaN, NaN, NaN, NaN, "", string(VERSION), why]
                println(io, join(csvfield.(row), ","))
            end
        end
    end
    println("records: ", joinpath(OUT, "records.csv"))
end

if OPTS["case"] > 0
    include_string(Main, harness_source(), basename(HARNESS))
    run_case(OPTS["case"])
else
    drive()
end
