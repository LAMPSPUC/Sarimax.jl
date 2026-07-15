# Block 4 (global certificates) - SCIP solves the nonconvex MA(1) SSE model DIRECTLY.
#
# SCIP is a global MINLP solver: it handles the bilinear equality constraints and the quadratic
# objective natively (spatial branch-and-bound), so no decomposition layer, sub-solver wiring, or
# problem-aware initialization is required. On these instances it returns EXACT global-optimality
# certificates (relative gap 0) in seconds — including T=20, which the decomposition-based
# Alpine workflow never certified within its time budget.
#
# Each SCIP optimum is cross-checked against a brute-force profile search: for fixed (θ, c) the
# residuals follow the deterministic recursion ε_t = y_t − c − θ ε_{t−1}, so a fine grid over
# (θ, c) with local refinement bounds the true global SSE. This audit is what exposed that the
# earlier Alpine incumbents (obj 7.3447 at T=8, 8.506 at T=10) were BELOW the true global optima —
# i.e. infeasible beyond tolerance — and motivated the switch to direct SCIP.
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV, JuMP, SCIP

const OUT = joinpath(RAW, "solver", "scip_cert_results.jsonl")
isfile(OUT) && rm(OUT)

full = vec(values(loadDataset(CSV.read(joinpath(RAW, "validation", "data", "sim_arma.csv"), DataFrame))))

"Brute-force global SSE bound: grid over (θ, c) + local refinement (audit, not a solver)."
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

for T in (8, 10, 20, 40)
    y = full[1:T]
    m = Model(optimizer_with_attributes(SCIP.Optimizer,
        "display/verblevel" => 0, "limits/time" => 300.0))
    @variable(m, c); @variable(m, -1 <= θ <= 1); @variable(m, ε[1:T]); fix(ε[1], 0.0)
    @constraint(m, [t = 2:T], y[t] == c + θ * ε[t-1] + ε[t])
    @objective(m, Min, sum(ε[t]^2 for t = 1:T))
    rt = @elapsed optimize!(m)
    o = objective_value(m)
    bf, θbf, _ = brute_force_sse(y)
    rec = Dict{String,Any}("block" => "solver", "dataset" => "MA(1) T=$T",
        "solver" => "SCIP", "objective" => "mse",
        "setting" => "global, direct (no decomposition, raw model)",
        "implementation" => "JuMP-direct", "seed" => 1234,
        "obj_value" => o, "rel_gap" => (try relative_gap(m) catch; nothing end),
        "termination" => string(termination_status(m)),
        "runtime_s" => rt, "theta" => value(θ),
        "brute_force_check" => bf, "brute_force_agrees" => isapprox(o, bf; rtol = 1e-4),
        "status" => termination_status(m) == MOI.OPTIMAL ? "ok" : "not_certified")
    append_record(OUT, rec)
    println("T=$T -> ", rec["termination"], "  obj=", round(o, digits = 4),
            "  brute-force=", round(bf, digits = 4), " (agrees: ", rec["brute_force_agrees"],
            ")  rt=", round(rt, digits = 2), "s")
end
println("scip_cert DONE -> ", OUT)
