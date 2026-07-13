# OPTIONAL - Gurobi global solve of the nonconvex MA(1) SSE model (NOT in the default pipeline).
#
# Gurobi (>= 9.0) solves nonconvex bilinear QCQPs to global optimality when NonConvex = 2 is set;
# the package's includeSolverParameters! sets this automatically when the solver is Gurobi, so
# fit!(optimizer = Gurobi.Optimizer) works end-to-end. Gurobi is COMMERCIAL software: it is not a
# package dependency and this script only runs if Gurobi.jl and a valid license are available.
# The paper's open-source claims do not depend on it; this script exists so that users with a
# license can reproduce/compare (expected: OPTIMAL with certified gap, typically fast).
#
# Usage:  julia +1.11 --project=experiments/env -e 'using Pkg; Pkg.add("Gurobi")'   # needs license
#         julia +1.11 --project=experiments/env experiments/scripts/run_solver_gurobi.jl
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV, JuMP

const OUT = joinpath(RAW, "solver", "gurobi_results.jsonl")

ok = try
    @eval using Gurobi
    true
catch
    println("Gurobi.jl not available (not installed or unlicensed); skipping. ",
            "This optional check requires a Gurobi license.")
    false
end

if ok
    isfile(OUT) && rm(OUT)
    full = vec(values(loadDataset(CSV.read(joinpath(RAW, "validation", "data", "sim_arma.csv"), DataFrame))))
    for T in (8, 10, 20, 40)
        y = full[1:T]
        m = Model(optimizer_with_attributes(Gurobi.Optimizer,
            "NonConvex" => 2, "TimeLimit" => 300.0, "Threads" => 1, "Seed" => 1234,
            "OutputFlag" => 0))
        @variable(m, c); @variable(m, -1 <= θ <= 1); @variable(m, ε[1:T]); fix(ε[1], 0.0)
        @constraint(m, [t = 2:T], y[t] == c + θ * ε[t-1] + ε[t])
        @objective(m, Min, sum(ε[t]^2 for t = 1:T))
        rt = @elapsed optimize!(m)
        rec = Dict{String,Any}("block" => "solver", "dataset" => "MA(1) T=$T",
            "solver" => "Gurobi", "objective" => "mse",
            "setting" => "global, direct (NonConvex=2)", "implementation" => "JuMP-direct",
            "seed" => 1234, "obj_value" => objective_value(m),
            "rel_gap" => (try relative_gap(m) catch; nothing end),
            "termination" => string(termination_status(m)), "runtime_s" => rt,
            "status" => termination_status(m) == MOI.OPTIMAL ? "ok" : "not_certified")
        append_record(OUT, rec)
        println("T=$T -> ", rec["termination"], "  obj=", round(rec["obj_value"], digits = 4),
                "  rt=", round(rt, digits = 2), "s")
    end
    println("gurobi DONE -> ", OUT)
end
