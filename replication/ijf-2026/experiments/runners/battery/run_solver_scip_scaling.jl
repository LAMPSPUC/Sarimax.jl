# Block 4 (scaling) - How far does DIRECT SCIP certify the nonconvex MA(1) SSE model?
#
# Companion to run_solver_scip_cert.jl (which covers T=8..40). Here we remove the small-instance
# restriction and push the sample size up until SCIP can no longer close the optimality gap within
# a generous per-instance time budget. This characterizes the ACTUAL scalability frontier of exact
# open-source global certification for this problem, rather than asserting one.
#
# Protocol: increasing T on the simulated ARMA series, raw model (no warm start, no tuning), a
# generous wall-clock budget per solve. We escalate T until two consecutive instances fail to
# certify (TIME_LIMIT with gap > tol), so the run terminates near the frontier instead of grinding
# through every large size. Each certified optimum is cross-checked against a brute-force profile
# search over (theta, c), which is exact up to grid refinement because the residuals follow a
# deterministic recursion for fixed coefficients.
#
# Port of original-scripts/battery-2026-07/scripts/run_solver_scip_scaling.jl. Model, SCIP
# options, default budget (600 s) and size ladder (40, 60, 80, 100, 120, 150, 200, 250, 300),
# stop rule (two consecutive non-certified sizes), brute-force audit (rtol 1e-3, certified sizes
# only) and record layout unchanged; no package estimation API is involved. Adaptations:
#   * budget and sizes are command-line arguments (`--budget 600 --sizes 40,60,...`; defaults =
#     the archived protocol) instead of the environment variables SCIP_SCALING_BUDGET /
#     SCIP_SCALING_SIZES, so the configuration states them; the append mode
#     (SCIP_SCALING_APPEND) is dropped: a run always starts a fresh file. The archived file was
#     appended over three invocations (T=40..120; T=150 after a NaN-serialisation crash; T=200
#     after a scoping crash, logs scip_scaling_run{,2,3}.log); with the default ladder and the
#     stop rule one invocation covers the same sizes when the frontier falls at the same place.
#   * each record also carries what SCIP reported (moi_diagnostics): raw status, primal status,
#     result count, solver objective value, dual bound ("objective_bound"), relative gap
#     ("relative_gap"; the original "rel_gap" is kept), solver time, node count, SCIP version.
#   * theta is read only when SCIP returned a primal solution (the original called value()
#     unconditionally, which throws without one).
#   * load_dataset instead of loadDataset; input regenerated in this run by
#     setup/gen_sim_data.py; output $IJF_RUN_DIR/raw/battery-2026-07/solver/scip_scaling_results.jsonl.
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV, JuMP, SCIP

function parse_args(args)
    budget = 600.0
    sizes = [40, 60, 80, 100, 120, 150, 200, 250, 300]
    i = 1
    while i <= length(args)
        if args[i] == "--budget" && i < length(args)
            budget = parse(Float64, args[i+1]); i += 2
        elseif args[i] == "--sizes" && i < length(args)
            sizes = parse.(Int, split(args[i+1], ",")); i += 2
        else
            error("unknown argument $(args[i]); usage: run_solver_scip_scaling.jl [--budget S] [--sizes T1,T2,...]")
        end
    end
    return budget, sizes
end
const TIME_BUDGET, SIZES = parse_args(ARGS)
println("SCIP scaling protocol: budget $(TIME_BUDGET) s per instance, sizes $(SIZES), stop after two consecutive non-certified sizes")

const OUT = outfile("solver", "scip_scaling_results.jsonl")
isfile(OUT) && rm(OUT)

# JSON.json errors on NaN/Inf; store non-finite numbers as null so time-limited rows still record.
naclean(x) = (x isa Real && !isfinite(x)) ? nothing : x

full = vec(values(load_dataset(CSV.read(datafile("validation", "sim_arma.csv";
                                                 producer = "experiments/runners/battery/setup/gen_sim_data.py"), DataFrame))))
length(full) >= maximum(SIZES) || error("sim_arma has $(length(full)) observations, fewer than the largest size $(maximum(SIZES))")

"Brute-force global SSE bound over (θ, c) with local refinement (audit, not a solver)."
function brute_force_sse(y)
    sse(θ, c) = begin
        ε = 0.0; s = 0.0
        for t in 2:length(y)
            ε = y[t] - c - θ * ε
            s += ε^2
        end
        s
    end
    best = (Inf, NaN, NaN)
    for θ in range(-1, 1; length = 4001), c in range(-3, 3; length = 1201)
        v = sse(θ, c); v < best[1] && (best = (v, θ, c))
    end
    θ0, c0 = best[2], best[3]
    for θ in range(θ0 - 0.002, θ0 + 0.002; length = 801),
        c in range(c0 - 0.02, c0 + 0.02; length = 801)
        v = sse(θ, c); v < best[1] && (best = (v, θ, c))
    end
    return best
end

consecutive_fail = 0
for T in SIZES
    y = full[1:T]
    m = Model(optimizer_with_attributes(SCIP.Optimizer,
        "display/verblevel" => 0, "limits/time" => TIME_BUDGET))
    @variable(m, c); @variable(m, -1 <= θ <= 1); @variable(m, ε[1:T]); fix(ε[1], 0.0)
    @constraint(m, [t = 2:T], y[t] == c + θ * ε[t-1] + ε[t])
    @objective(m, Min, sum(ε[t]^2 for t = 1:T))
    rt = @elapsed optimize!(m)
    term = termination_status(m)
    o = has_values(m) ? objective_value(m) : NaN
    gap = try relative_gap(m) catch; NaN end
    certified = term == MOI.OPTIMAL
    # brute-force audit only when certified (and only up to a size where the grid is meaningful)
    bf, bf_agrees = NaN, nothing
    if certified
        bfval, _, _ = brute_force_sse(y)
        bf = bfval; bf_agrees = isapprox(o, bfval; rtol = 1e-3)
    end
    rec = Dict{String,Any}("block" => "solver", "experiment" => "scip_scaling",
        "dataset" => "MA(1) T=$T", "solver" => "SCIP", "objective" => "mse",
        "setting" => "global, direct (raw model, $(Int(TIME_BUDGET))s budget)",
        "implementation" => "JuMP-direct", "seed" => 1234, "n" => T,
        "obj_value" => naclean(o), "rel_gap" => naclean(gap), "termination" => string(term),
        "runtime_s" => rt, "theta" => has_values(m) ? naclean(value(θ)) : nothing,
        "certified" => certified, "brute_force_check" => naclean(bf), "brute_force_agrees" => bf_agrees,
        "time_budget_s" => TIME_BUDGET,
        "status" => certified ? "ok" : "not_certified")
    merge!(rec, moi_diagnostics(m))
    append_record(OUT, rec)
    println("T=$T -> ", term, "  obj=", round(o, digits = 4), "  bound=", rec["objective_bound"],
            "  gap=", gap, "  rt=", round(rt, digits = 1), "s",
            certified ? "  (brute-force agrees: $(bf_agrees))" : "  [not certified]",
            "  SCIP ", rec["solver_version"])
    flush(stdout)
    global consecutive_fail = certified ? 0 : consecutive_fail + 1
    if consecutive_fail >= 2
        println("Two consecutive non-certified instances; frontier reached, stopping escalation.")
        break
    end
end
println("scip_scaling DONE -> ", OUT)
