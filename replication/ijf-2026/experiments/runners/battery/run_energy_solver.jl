# Energy dataset (PJME) - solver experiment on a REDUCED MA(1)-containing instance.
# Ipopt (local, via fit!) vs SCIP (global, JuMP level, direct solve of the nonconvex model with an
# explicit time limit). SCIP handles the bilinear MA structure natively and returns an exact
# global-optimality certificate (relative gap 0) on this reduced instance.
#
# Port of original-scripts/battery-2026-07/scripts/run_energy_solver.jl to the release API.
# Instance (last 24 hours of the recent hourly slice, centred, divided by 1000), MA(1) model,
# solvers, SCIP options (display/verblevel 0, limits/time 300 s) and record layout unchanged.
# No table reads these records (the energy table keeps only block = "energy_forecasting"), so no
# configuration runs this script; run it by hand after the first two steps of
# experiments/configs/battery_energy.toml (same environment variables; needs SCIP) to re-create
# the archived energy/solver_results.jsonl. Adaptations:
#   * fit! receives initialization = :zeroed and stationary = false (the v0.3.0 defaults the
#     archived script relied on; the release defaults are :innovations and true).
#   * "termination" of the Ipopt row is the release's metadata["solverStatus"], not the
#     original's hard-coded "locally_solved".
#   * the SCIP record also carries what SCIP reported (moi_diagnostics): raw status, primal
#     status, result count, solver objective value, dual bound ("objective_bound"), relative gap
#     ("relative_gap"; the original "rel_gap" is kept), solver time, node count, SCIP version.
#     objective_value / value(θ) are read only when SCIP returned a primal solution (the
#     original read them unconditionally, which throws without one).
#   * input: pjme_hourly_recent.csv derived in this run by setup/gen_energy_data.py; output
#     $IJF_RUN_DIR/raw/battery-2026-07/energy/solver_results.jsonl.
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV, JuMP, Ipopt, SCIP

const OUT = outfile("energy", "solver_results.jsonl")
isfile(OUT) && rm(OUT)

# Reduced instance: last 24 hourly observations from the recent slice (MA-containing, small).
hourly = Float64.(CSV.read(datafile("energy", "pjme_hourly_recent.csv";
                                    producer = "experiments/runners/battery/setup/gen_energy_data.py (needs ./reproduce.sh fetch pjme)"),
                           DataFrame).value)
yv = hourly[end-24+1:end]
# center/scale to keep the optimization well-conditioned (load is ~3e4 MW)
yv = (yv .- mean(yv)) ./ 1000.0
T = length(yv)
ts = [Date(2000,1,1)+Day(i-1) for i in 1:T]
ta = TimeArray(ts, yv, ["value"])

emit(rec) = (append_record(OUT, rec); println(rec["solver"], " (", rec["setting"], ") -> ", rec["status"]))

# Ipopt local via fit! (MA(1))
let
    rec = Dict{String,Any}("block"=>"energy_solver", "dataset"=>"PJME reduced MA(1) T=$T",
        "solver"=>"Ipopt", "setting"=>"local (fit!)", "implementation"=>"SARIMAX.jl(fit!)",
        "objective"=>"mse", "seed"=>1234)
    try
        m = SARIMA(ta, 0, 0, 1)
        t = @elapsed fit!(m; optimizer=Ipopt.Optimizer, objectiveFunction="mse",
                          initialization=V030_INITIALIZATION, stationary=V030_STATIONARY)
        rec["status"]="ok"; rec["obj_value"]=sum(abs2, m.ϵ); rec["runtime_s"]=t
        rec["termination"]=fit_status(m); rec["theta"]=m.θ[1]
    catch e
        rec["status"]="failed"; rec["termination"]="error"; rec["error"]=sprint(showerror,e)
    end
    emit(rec)
end

# SCIP global (JuMP level, direct), explicit 300s time limit -> expect OPTIMAL, gap 0.
let
    rec = Dict{String,Any}("block"=>"energy_solver", "dataset"=>"PJME reduced MA(1) T=$T",
        "solver"=>"SCIP", "setting"=>"global, direct (300s limit)",
        "implementation"=>"JuMP-direct", "objective"=>"mse", "seed"=>1234)
    try
        m = Model(optimizer_with_attributes(SCIP.Optimizer,
            "display/verblevel"=>0, "limits/time"=>300.0))
        @variable(m, c); @variable(m, -1 <= θ <= 1); @variable(m, ε[1:T]); fix(ε[1], 0.0)
        @constraint(m, [t=2:T], yv[t] == c + θ*ε[t-1] + ε[t])
        @objective(m, Min, sum(ε[t]^2 for t=1:T))
        rt = @elapsed optimize!(m)
        rec["status"]="ok"; rec["runtime_s"]=rt
        rec["termination"]=string(termination_status(m))
        if has_values(m)
            rec["obj_value"]=objective_value(m); rec["theta"]=value(θ)
        end
        try; rec["rel_gap"]=relative_gap(m); catch; end
        merge!(rec, moi_diagnostics(m))
    catch e
        rec["status"]="failed"; rec["termination"]="error"; rec["error"]=sprint(showerror,e)
    end
    emit(rec)
end

println("energy_solver DONE -> ", OUT)
