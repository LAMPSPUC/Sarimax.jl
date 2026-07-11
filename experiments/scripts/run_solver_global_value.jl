# Block 4d - Does certified global optimality matter for forecasting?
#
# For each instance we estimate the SAME specification twice — Ipopt (local) and direct SCIP
# (global, certified) — and compare: final objective, estimated coefficients, out-of-sample
# forecast errors over a held-out window, and wall time. The point is NOT to benchmark solvers but
# to quantify, on instances where an exact certificate is attainable, how much the guarantee
# changes the statistical output. Either outcome is informative: a gap would show the practical
# value of certification; identical solutions show that certification VERIFIES (at computational
# cost) what the local solver already delivered — the architecture makes this cost-benefit
# measurable rather than assumed.
#
# Conventions match fit!: presample residuals fixed at 0 (conditional baseline), phi/theta in
# [-1,1], free intercept. Forecasts use the standard ARMA recursion with future innovations at 0.
# Instances are restricted to sizes where SCIP certifies within the budget (see the scaling study).
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV, JuMP, Ipopt, SCIP

const OUT = joinpath(RAW, "solver", "global_value_results.jsonl")
isfile(OUT) && rm(OUT)
const H = 12                 # forecast horizon
const NTRAIN = 80            # certifiable per the scaling study (T=80 ~ 1 min for SCIP)
const SCIP_LIMIT = 900.0

simarma = vec(values(loadDataset(CSV.read(joinpath(RAW, "validation", "data", "sim_arma.csv"), DataFrame))))
pjdaily = Float64.(CSV.read(joinpath(RAW, "energy", "data", "pjme_daily.csv"), DataFrame).value)
pjdaily = (pjdaily .- mean(pjdaily)) ./ 1000.0   # scale as in the energy solver experiment

# Build the ARMA(p<=1, q=1) SSE model exactly as fit! does (conditional baseline).
function build_arma(optimizer, y, p)
    T = length(y)
    m = Model(optimizer)
    @variable(m, c)
    p == 1 && @variable(m, -1 <= φ <= 1)
    @variable(m, -1 <= θ <= 1)
    @variable(m, ε[1:T])
    lb = max(p, 1) + 1
    for t in 1:lb-1; fix(ε[t], 0.0); end
    if p == 1
        @constraint(m, [t = lb:T], y[t] == c + φ * y[t-1] + θ * ε[t-1] + ε[t])
    else
        @constraint(m, [t = lb:T], y[t] == c + θ * ε[t-1] + ε[t])
    end
    @objective(m, Min, sum(ε[t]^2 for t = 1:T))
    return m
end

# h-step forecasts with future innovations at 0 (standard ARMA point forecast).
function forecast_arma(y, c, φ, θ, εT, h)
    fc = Float64[]
    yprev = y[end]; eprev = εT
    for i in 1:h
        ŷ = c + φ * yprev + θ * eprev
        push!(fc, ŷ)
        yprev = ŷ; eprev = 0.0
    end
    return fc
end

rmse(a, b) = sqrt(mean((a .- b) .^ 2))
mae(a, b) = mean(abs.(a .- b))

instances = [
    ("sim_arma", simarma, 0),   # MA(1)
    ("sim_arma", simarma, 1),   # ARMA(1,0,1)
    ("pjme_daily(scaled)", pjdaily, 1),  # ARMA(1,0,1), reduced real-data diagnostic
]

for (name, series, p) in instances
    ytr = series[1:NTRAIN]; yte = series[NTRAIN+1:NTRAIN+H]
    spec = p == 1 ? "ARMA(1,0,1)" : "MA(1)"
    for (solver, opt) in [
        ("Ipopt", optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)),
        ("SCIP", optimizer_with_attributes(SCIP.Optimizer,
            "display/verblevel" => 0, "limits/time" => SCIP_LIMIT)),
    ]
        m = build_arma(opt, ytr, p)
        rt = @elapsed optimize!(m)
        term = termination_status(m)
        certified = term == MOI.OPTIMAL
        ok = has_values(m)
        rec = Dict{String,Any}("block" => "solver", "experiment" => "global_value",
            "dataset" => "$name $spec T=$NTRAIN", "solver" => solver, "objective" => "mse",
            "setting" => solver == "SCIP" ? "global, direct" : "local",
            "implementation" => "JuMP-direct (fit! conventions)", "seed" => 1234,
            "n_train" => NTRAIN, "horizon" => H,
            "termination" => string(term), "certified" => certified, "runtime_s" => rt)
        if ok
            cv = value(m[:c]); θv = value(m[:θ]); φv = p == 1 ? value(m[:φ]) : 0.0
            εT = value(m[:ε][NTRAIN])
            fc = forecast_arma(ytr, cv, φv, θv, εT, H)
            rec["obj_value"] = objective_value(m)
            rec["rel_gap"] = (try g = relative_gap(m); isfinite(g) ? g : nothing catch; nothing end)
            rec["c"] = cv; rec["theta"] = θv; p == 1 && (rec["phi"] = φv)
            rec["rmse_oos"] = rmse(fc, yte); rec["mae_oos"] = mae(fc, yte)
            rec["status"] = "ok"
        else
            rec["status"] = "failed"
        end
        append_record(OUT, rec)
        println(rec["dataset"], " / ", solver, " -> ", term,
                ok ? "  obj=$(round(rec["obj_value"],digits=4)) rmse_oos=$(round(rec["rmse_oos"],digits=4)) rt=$(round(rt,digits=1))s" : "")
        flush(stdout)
    end
end
println("global_value DONE -> ", OUT)
