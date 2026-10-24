# Block 4C (JuMP-level) - genuine randomized multistart on the MA(1) SSE model.
#
# The package's high-level fit! uses deterministic start values, so it cannot do randomized
# multistart. Here we reconstruct SARIMAX.jl's MA(1) SSE model directly in JuMP and run 25 random
# Ipopt starts. The single optimum found (34.83 on T=40) is independently CERTIFIED as the global
# optimum by the direct SCIP solve in run_solver_scip_cert.jl (OPTIMAL, gap 0, T=40).
#
# Port of original-scripts/battery-2026-07/scripts/run_solver_jump_diagnostics.jl. The model,
# instance (sim_arma[1:40]), seed (1234), 25 random starts, Ipopt options, convergence filter and
# rounding of distinct optima are unchanged; no package estimation API is involved. Adaptations:
#   * "termination" is what Ipopt returned over the 25 starts (summarised, with per-status
#     counts in "termination_counts"), not the original's hard-coded
#     "locally_solved (optimum certified global by SCIP)": that the optimum is the global one is
#     what run_solver_scip_cert.jl checks, in its own record.
#   * load_dataset instead of loadDataset; input regenerated in this run by
#     setup/gen_sim_data.py; output $IJF_RUN_DIR/raw/battery-2026-07/solver/jump_results.jsonl.
#   * Random.seed!(1234) drives Julia's default RNG; its stream for rand()/randn() is the same in
#     Julia 1.10 (this environment) and 1.11 (the archived run).
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV, JuMP, Ipopt

const OUT = outfile("solver", "jump_results.jsonl")
isfile(OUT) && rm(OUT)

full = vec(values(load_dataset(CSV.read(datafile("validation", "sim_arma.csv";
                                                 producer = "experiments/runners/battery/setup/gen_sim_data.py"), DataFrame))))

# Build SARIMAX.jl's MA(1) SSE model: y_t = c + θ ε_{t-1} + ε_t, minimize Σ ε_t^2.
function build_ma1(optimizer, y)
    T = length(y)
    m = Model(optimizer)
    @variable(m, c)
    @variable(m, -1 <= θ <= 1)
    @variable(m, ε[1:T])
    fix(ε[1], 0.0)
    @constraint(m, [t = 2:T], y[t] == c + θ * ε[t-1] + ε[t])
    @objective(m, Min, sum(ε[t]^2 for t = 1:T))
    return m
end

let
    y = full[1:40]; T = length(y)
    # JIT warm-up (UNTIMED) so the multistart total is warm (JIT excluded).
    let mw = build_ma1(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0), y)
        optimize!(mw)
    end
    objs = Float64[]; statuses = String[]; Random.seed!(1234); nstart = 25
    rt = @elapsed for s in 1:nstart
        m = build_ma1(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0), y)
        set_start_value(m[:θ], rand() * 1.98 - 0.99)   # random θ0 in (-0.99, 0.99)
        set_start_value(m[:c], randn())
        optimize!(m)
        push!(statuses, string(termination_status(m)))
        if termination_status(m) in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL)
            push!(objs, objective_value(m))
        end
    end
    rounded = unique(round.(objs; digits=4))
    rec = Dict("block"=>"solver", "dataset"=>"MA(1) T=$T", "solver"=>"Ipopt",
        "setting"=>"multistart x$nstart (random theta,c)", "implementation"=>"JuMP-direct",
        "seed"=>1234, "status"=>isempty(objs) ? "failed" : "ok",
        "obj_value"=> isempty(objs) ? nothing : minimum(objs),
        "obj_spread"=> isempty(objs) ? nothing : maximum(objs) - minimum(objs),
        "n_distinct_optima"=>length(rounded), "n_converged"=>length(objs),
        "termination"=>status_summary(statuses),
        "termination_counts"=>Dict(s => count(==(s), statuses) for s in unique(statuses)),
        "runtime_s"=>rt)
    append_record(OUT, rec)
    println("multistart -> ", rec["status"], "  distinct=", length(rounded),
            "  obj=", rec["obj_value"], "  rt=", round(rt, digits=2), "s")
end
println("solver_jump (4C) DONE -> ", OUT)
