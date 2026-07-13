# Block 1 — Validation (SARIMAX.jl side).
# Fits classical models on shared simulated data + bundled real series; writes JSONL.
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV

const OUT = joinpath(RAW, "validation", "julia_results.jsonl")
isfile(OUT) && rm(OUT)

datadir = joinpath(RAW, "validation", "data")
sim_arma = loadDataset(CSV.read(joinpath(datadir, "sim_arma.csv"), DataFrame))
sim_x_df = CSV.read(joinpath(datadir, "sim_sarimax.csv"), DataFrame)
airp = loadDataset(AIR_PASSENGERS)

function run_one(id, y; p, d, q, P=0, D=0, Q=0, seasonality=1, exog=nothing, orderlabel=nothing,
                 initialization=:zeroed, implementation="SARIMAX.jl")
    rec = Dict{String,Any}("block"=>"validation", "implementation"=>implementation,
        "dataset"=>id,
        "order"=> orderlabel === nothing ? "($p,$d,$q)($P,$D,$Q)_$seasonality" : orderlabel,
        "objective"=>"mse", "solver"=>"Ipopt", "seed"=>1234)
    exog === nothing || (rec["model_family"] = "ARX")
    try
        m = exog === nothing ? SARIMA(y, p, d, q; seasonality=seasonality, P=P, D=D, Q=Q) :
                               SARIMA(y, exog, p, d, q; seasonality=seasonality, P=P, D=D, Q=Q)
        t = @elapsed fit!(m; objectiveFunction="mse", initialization=initialization)
        rec["status"] = "ok"
        rec["runtime_s"] = t
        rec["estimates"] = extract_estimates(m)
        rec["loglike"] = trymetric(Sarimax.loglikelihood, m)
        rec["aic"] = trymetric(Sarimax.aic, m)
        rec["bic"] = trymetric(Sarimax.bic, m)
        res = m.ϵ
        rec["rss"] = res === nothing ? nothing : sum(abs2, res)
    catch e
        rec["status"] = "failed"
        rec["error"] = sprint(showerror, e)
    end
    append_record(OUT, rec)
    println(rec["dataset"], " ", rec["order"], " -> ", rec["status"])
end

run_one("sim_arma", sim_arma; p=1, d=0, q=0)
run_one("sim_arma", sim_arma; p=0, d=0, q=1)
run_one("sim_arma", sim_arma; p=1, d=0, q=1)
run_one("airpassengers", airp; p=1, d=0, q=1)
run_one("airpassengers", airp; p=1, d=0, q=1, P=1, D=0, Q=1, seasonality=12)

# CSS-matched convention (initialization=:warmup reproduces R arima(method="CSS"))
for spec in [(1,0,0,0,0,0,1), (0,0,1,0,0,0,1), (1,0,1,0,0,0,1)]
    run_one("sim_arma", sim_arma; p=spec[1], d=spec[2], q=spec[3], P=spec[4], D=spec[5],
            Q=spec[6], seasonality=spec[7], initialization=:warmup,
            implementation="SARIMAX.jl (CSS-warmup)")
end
run_one("airpassengers", airp; p=1, d=0, q=1, initialization=:warmup,
        implementation="SARIMAX.jl (CSS-warmup)")
run_one("airpassengers", airp; p=1, d=0, q=1, P=1, D=0, Q=1, seasonality=12,
        initialization=:warmup, implementation="SARIMAX.jl (CSS-warmup)")

# SARIMAX with two exogenous regressors
let
    ts = sim_x_df.date
    y = TimeArray(ts, Float64.(sim_x_df.value), ["value"])
    exog = TimeArray(ts, hcat(Float64.(sim_x_df.x1), Float64.(sim_x_df.x2)), ["x1","x2"])
    run_one("sim_sarimax", y; p=1, d=0, q=0, exog=exog, orderlabel="ARX(1)+2exog")
end

println("validation_julia DONE -> ", OUT)
