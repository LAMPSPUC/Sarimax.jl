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
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV, JuMP, SCIP

const OUT = joinpath(RAW, "solver", "scip_scaling_results.jsonl")
# Fresh run truncates; set SCIP_SCALING_APPEND=1 to add sizes to an existing file.
(isfile(OUT) && !haskey(ENV, "SCIP_SCALING_APPEND")) && rm(OUT)

const TIME_BUDGET = get(ENV, "SCIP_SCALING_BUDGET", "600") |> x -> parse(Float64, x)
const SIZES = haskey(ENV, "SCIP_SCALING_SIZES") ?
    parse.(Int, split(ENV["SCIP_SCALING_SIZES"], ",")) :
    [40, 60, 80, 100, 120, 150, 200, 250, 300]

# JSON.json errors on NaN/Inf; store non-finite numbers as null so time-limited rows still record.
naclean(x) = (x isa Real && !isfinite(x)) ? nothing : x

full = vec(values(loadDataset(CSV.read(joinpath(RAW, "validation", "data", "sim_arma.csv"), DataFrame))))

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
        "runtime_s" => rt, "theta" => naclean(value(θ)),
        "certified" => certified, "brute_force_check" => naclean(bf), "brute_force_agrees" => bf_agrees,
        "time_budget_s" => TIME_BUDGET,
        "status" => certified ? "ok" : "not_certified")
    append_record(OUT, rec)
    println("T=$T -> ", term, "  obj=", round(o, digits = 4), "  gap=", gap,
            "  rt=", round(rt, digits = 1), "s",
            certified ? "  (brute-force agrees: $(bf_agrees))" : "  [not certified]")
    flush(stdout)
    global consecutive_fail = certified ? 0 : consecutive_fail + 1
    if consecutive_fail >= 2
        println("Two consecutive non-certified instances; frontier reached, stopping escalation.")
        break
    end
end
println("scip_scaling DONE -> ", OUT)
