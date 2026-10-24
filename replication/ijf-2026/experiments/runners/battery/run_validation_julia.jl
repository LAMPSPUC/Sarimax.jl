# Block 1 — Validation (SARIMAX.jl side).
# Fits classical models on shared simulated data + bundled real series; writes JSONL.
#
# Port of original-scripts/battery-2026-07/scripts/run_validation_julia.jl to the release API.
# Same models, orders, objective ("mse"), initialization (:zeroed, and :warmup for the
# CSS-matched rows), data and record layout. Adaptations:
#   * fit! also receives stationary = false: v0.3.0's default; the release defaults to the
#     reflection (stationary-by-construction) AR parameterisation. initialization was already
#     explicit here (:zeroed / :warmup), so the release's new :innovations default is not used.
#   * load_dataset instead of the deprecated loadDataset alias (same function).
#   * inputs: the simulated CSVs regenerated in this run by setup/gen_sim_data.py
#     ($IJF_RUN_DIR/raw/battery-2026-07/validation/data/); output:
#     $IJF_RUN_DIR/raw/battery-2026-07/validation/julia_results.jsonl.
#   * "aic"/"bic" are the release's Sarimax.aic/bic as before, but the release scores the EXACT
#     Gaussian likelihood when it is computable (v0.3.0 used the conditional CSS likelihood), so
#     the AIC/BIC columns differ by definition. The v0.3.0 values are recorded alongside as
#     "aic_css"/"bic_css" (from css_criteria); "loglike" is unchanged in definition (CSS).
#   * Extra fields: "solver_status" (the release's metadata["solverStatus"]), "initialization".
#   * Release-internal changes no keyword can undo (see PORTING.md): fit! solves on the series
#     divided by its standard deviation and maps the estimates back.
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV

const OUT = outfile("validation", "julia_results.jsonl")
isfile(OUT) && rm(OUT)

const PRODUCER = "experiments/runners/battery/setup/gen_sim_data.py"
sim_arma = load_dataset(CSV.read(datafile("validation", "sim_arma.csv"; producer = PRODUCER), DataFrame))
sim_x_df = CSV.read(datafile("validation", "sim_sarimax.csv"; producer = PRODUCER), DataFrame)
airp = load_dataset(AIR_PASSENGERS)

function run_one(id, y; p, d, q, P=0, D=0, Q=0, seasonality=1, exog=nothing, orderlabel=nothing,
                 initialization=:zeroed, implementation="SARIMAX.jl")
    rec = Dict{String,Any}("block"=>"validation", "implementation"=>implementation,
        "dataset"=>id,
        "order"=> orderlabel === nothing ? "($p,$d,$q)($P,$D,$Q)_$seasonality" : orderlabel,
        "objective"=>"mse", "solver"=>"Ipopt", "seed"=>1234,
        "initialization"=>String(initialization))
    exog === nothing || (rec["model_family"] = "ARX")
    try
        m = exog === nothing ? SARIMA(y, p, d, q; seasonality=seasonality, P=P, D=D, Q=Q) :
                               SARIMA(y, exog, p, d, q; seasonality=seasonality, P=P, D=D, Q=Q)
        t = @elapsed fit!(m; objectiveFunction="mse", initialization=initialization,
                          stationary=V030_STATIONARY)
        rec["status"] = "ok"
        rec["runtime_s"] = t
        rec["solver_status"] = fit_status(m)
        rec["estimates"] = extract_estimates(m)
        rec["loglike"] = trymetric(Sarimax.loglikelihood, m)
        rec["aic"] = trymetric(Sarimax.aic, m)
        rec["bic"] = trymetric(Sarimax.bic, m)
        crit = trymetric(css_criteria, m)
        crit === nothing || (rec["aic_css"] = crit["aic_css"]; rec["bic_css"] = crit["bic_css"])
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
