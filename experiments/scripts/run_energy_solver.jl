# Energy dataset (PJME) - solver experiment on a REDUCED MA(1)-containing instance.
# Ipopt (local, via fit!) vs SCIP (global, JuMP level, direct solve of the nonconvex model with an
# explicit time limit). SCIP handles the bilinear MA structure natively and returns an exact
# global-optimality certificate (relative gap 0) on this reduced instance.
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV, JuMP, Ipopt, SCIP

const OUT = joinpath(RAW, "energy", "solver_results.jsonl")
isfile(OUT) && rm(OUT)

# Reduced instance: last 24 hourly observations from the recent slice (MA-containing, small).
hourly = Float64.(CSV.read(joinpath(RAW, "energy", "data", "pjme_hourly_recent.csv"), DataFrame).value)
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
        t = @elapsed fit!(m; optimizer=Ipopt.Optimizer, objectiveFunction="mse")
        rec["status"]="ok"; rec["obj_value"]=sum(abs2, m.ϵ); rec["runtime_s"]=t
        rec["termination"]="locally_solved"; rec["theta"]=m.θ[1]
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
        rec["status"]="ok"; rec["runtime_s"]=rt; rec["obj_value"]=objective_value(m)
        rec["termination"]=string(termination_status(m)); rec["theta"]=value(θ)
        try; rec["rel_gap"]=relative_gap(m); catch; end
    catch e
        rec["status"]="failed"; rec["termination"]="error"; rec["error"]=sprint(showerror,e)
    end
    emit(rec)
end

println("energy_solver DONE -> ", OUT)
