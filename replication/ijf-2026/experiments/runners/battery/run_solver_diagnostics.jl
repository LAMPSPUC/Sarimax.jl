# Block 4A/4B - Solver modularity at the PACKAGE level (via fit!).
#
# Local baseline: Ipopt (default optimizer). Global backend: SCIP, passed directly as
# fit!(optimizer = SCIP.Optimizer) — SCIP solves the nonconvex (bilinear) MA model natively and
# certifies global optimality, so no decomposition layer is needed. The package also wires the
# Alpine decomposition workflow (optimizer = Alpine.Optimizer, with a `mipSolver` sub-solver
# argument); its configuration guard is exercised below at config level: requesting HiGHS as the
# MIP sub-solver with a quadratic objective emits a warning (HiGHS cannot solve MIQP relaxations).
# HiGHS is loaded here from the benchmark env only; it is not a package dependency.
#
# Port of original-scripts/battery-2026-07/scripts/run_solver_diagnostics.jl to the release API.
# Same instance (sim_arma[1:20]), models, objective, repeats, solvers and warning check.
# Adaptations:
#   * every fit! receives initialization = :zeroed and stationary = false (the v0.3.0 defaults
#     the archived script relied on; the release defaults are :innovations and true).
#   * "termination" is what the solver returned, never a hard-coded label. The original wrote
#     "locally_solved" for the Ipopt rows without checking, and "global optimum (matches certified
#     JuMP-level solve)" for the SCIP row, a claim the script never verified. The port records
#     the release's metadata["solverStatus"] (all three repeats, summarised), and for SCIP the
#     MOI termination status, so these Termination cells differ from the archived table by
#     construction.
#   * the SCIP solve inside fit! is read back through the optimizer instance fit! created (the
#     `optimizer` argument is an MOI.OptimizerWithAttributes whose constructor keeps a handle on
#     it; same solver, same default attributes). Recorded: termination / raw / primal status,
#     objective value and dual bound (in fit!'s internal units: the release solves on the series
#     divided by its standard deviation s, so the SSE is objective * s^2; both are recorded,
#     with "objective_scale" = s^2), relative gap (scale-free) and the SCIP version.
#   * Alpine is not a dependency of the replication environment; the check uses the copy the
#     package itself loads (Sarimax.Alpine, version pinned by the environment's Manifest).
#     includeSolverParameters! and its warning are unchanged in the release.
#   * warning_text keeps the first 400 characters (the original sliced 400 bytes, which throws
#     when the cut falls inside a multi-byte character of the log frame).
#   * load_dataset instead of loadDataset; input regenerated in this run by
#     setup/gen_sim_data.py; output $IJF_RUN_DIR/raw/battery-2026-07/solver/julia_results.jsonl.
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV, JuMP, Ipopt, SCIP, HiGHS, Logging
const Alpine = Sarimax.Alpine

const OUT = outfile("solver", "julia_results.jsonl")
isfile(OUT) && rm(OUT)

const V030 = (initialization = V030_INITIALIZATION, stationary = V030_STATIONARY)

# Small instance: first 20 obs of the simulated ARMA series.
full = vec(values(load_dataset(CSV.read(datafile("validation", "sim_arma.csv";
                                                 producer = "experiments/runners/battery/setup/gen_sim_data.py"), DataFrame))))
ydata = full[1:20]
ts = [Date(2000, 1, 1) + Month(i - 1) for i = 1:length(ydata)]
ta = TimeArray(ts, ydata, ["value"])
const N = length(ydata)

rss(m) = sum(abs2, m.ϵ)
emit(rec) = (append_record(OUT, rec); println(rec["dataset"], " / ", rec["solver"], " (", rec["setting"], ") -> ", rec["status"]))
base(mid, solver, setting, obj) = Dict{String,Any}(
    "block" => "solver", "dataset" => mid, "solver" => solver, "setting" => setting,
    "implementation" => "SARIMAX.jl(fit!)", "objective" => obj, "instance" => "sim_arma[1:$N]",
    "seed" => 1234)

# JIT warm-up (UNTIMED): compile the fit! path so reported Ipopt runtimes are warm (JIT excluded),
# consistent with the convention used throughout the artifact.
try
    let mw = SARIMA(ta, 0, 0, 1); fit!(mw; optimizer = Ipopt.Optimizer, objectiveFunction = "mse", V030...); end
catch
end

# --- Ipopt baseline (deterministic high-level starts), 3 repeats ---
for (p, q, mid) in [(0, 1, "ARIMA(0,0,1)"), (1, 1, "ARIMA(1,0,1)")]
    try
        objs = Float64[]; rts = Float64[]; statuses = String[]
        for _ = 1:3
            m = SARIMA(ta, p, 0, q)
            t = @elapsed fit!(m; optimizer = Ipopt.Optimizer, objectiveFunction = "mse", V030...)
            push!(objs, rss(m)); push!(rts, t); push!(statuses, fit_status(m))
        end
        r = base(mid, "Ipopt", "local x3 (deterministic starts)", "mse")
        r["status"] = "ok"; r["obj_value"] = minimum(objs)
        r["obj_spread"] = maximum(objs) - minimum(objs); r["runtime_s"] = sum(rts) / 3
        r["termination"] = status_summary(statuses); emit(r)
    catch e
        r = base(mid, "Ipopt", "local", "mse"); r["status"] = "failed"
        r["termination"] = "error"; r["error"] = sprint(showerror, e); emit(r)
    end
end

# --- SCIP via fit! (global backend through the same package interface) ---
let mid = "ARIMA(0,0,1)"
    handle = Ref{Any}(nothing)
    scip = JuMP.optimizer_with_attributes(() -> (o = SCIP.Optimizer(); handle[] = o; o))
    try
        m = SARIMA(ta, 0, 0, 1)
        t = @elapsed fit!(m; optimizer = scip, objectiveFunction = "mse", V030...)
        r = base(mid, "SCIP", "global, direct (fit!)", "mse")
        r["status"] = "ok"; r["obj_value"] = rss(m); r["runtime_s"] = t
        r["estimates"] = extract_estimates(m)
        diag = handle[] === nothing ? Dict{String,Any}() : moi_diagnostics(handle[])
        r["termination"] = something(get(diag, "termination_status", nothing), fit_status(m))
        r["package_solver_status"] = fit_status(m)
        # fit! solves on y / s with s = std(differenced y) (here d = 0): SSE = objective * s^2
        s2 = Statistics.std(ydata)^2
        r["objective_scale"] = s2
        r["solver_objective_value"] = get(diag, "solver_objective_value", nothing)
        r["objective_bound_solver_units"] = get(diag, "objective_bound", nothing)
        ob = get(diag, "objective_bound", nothing)
        r["objective_bound"] = ob isa Real ? ob * s2 : nothing
        for k in ("relative_gap", "raw_status", "primal_status", "result_count", "solver_time_s",
                  "node_count", "solver_version")
            r[k] = get(diag, k, nothing)
        end
        emit(r)
    catch e
        r = base(mid, "SCIP", "global, direct (fit!)", "mse")
        r["status"] = "failed"; r["termination"] = "error"; r["error"] = sprint(showerror, e)
        r["solver_version"] = try MOI.get(SCIP.Optimizer(), MOI.SolverVersion()) catch; nothing end
        emit(r)
    end
end

# --- 4B HiGHS warning behavior for the Alpine workflow (configuration-level, no solve) ---
# The package warns when HiGHS is requested as Alpine's MIP sub-solver with a non-"mae"
# (quadratic -> MIQP) objective. Tested by invoking includeSolverParameters! directly.
for obj in ("mse", "mae")
    buf = IOBuffer()
    with_logger(SimpleLogger(buf, Logging.Warn)) do
        m = Model(Alpine.Optimizer)
        Sarimax.includeSolverParameters!(m, true; mipSolver = HiGHS.Optimizer, objectiveFunction = obj)
    end
    warntext = String(take!(buf))
    emitted = occursin("HiGHS MIP sub-solver cannot", warntext)
    r = base("ARIMA(0,0,1)", "Alpine+HiGHS", "warning check, config-level (no solve)", obj)
    r["status"] = "ok"; r["termination"] = "config-only"
    r["warning_emitted"] = emitted
    # first 400 characters (the original sliced 400 BYTES, which can split a multi-byte
    # character of the log frame and throw; identical for ASCII text)
    r["warning_text"] = emitted ? first(strip(replace(warntext, r"\s+" => " ")), 400) : ""
    emit(r)
end

println("solver (package-level 4A/4B) DONE -> ", OUT)
