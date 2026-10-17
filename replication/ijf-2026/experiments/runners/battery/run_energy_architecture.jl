# Energy dataset (PJME daily) - architecture/extensibility: same SARIMA specification,
# only the optimization objective changes (MSE, MAE, Ridge, Elastic Net).
#
# Port of original-scripts/battery-2026-07/scripts/run_energy_architecture.jl to the release
# API. Window (last 360 days), specification SARIMA(1,1,1)(1,0,1)_7, the four settings and the
# record layout unchanged. No table reads these records (the energy table keeps only
# block = "energy_forecasting"), so no configuration runs this script; run it by hand after the
# first two steps of experiments/configs/battery_energy.toml (same environment variables) to
# re-create the archived energy/architecture_results.jsonl. Adaptations:
#   * every fit! receives initialization = :zeroed and stationary = false (the v0.3.0 defaults
#     the archived script relied on; the release defaults are :innovations and true).
#   * "elastic_net" is a DIFFERENT ESTIMATOR in the release (single-stage penalty with
#     lambda = sqrt(effective sample size) instead of v0.3.0's two-stage, RSS-tolerance-
#     constrained adaptive norm; CHANGELOG 1.0.0); called with the same alpha and
#     penaltyTarget = :all (its default, written out).
#   * "mae" with MA terms is a different fit in the release: v0.3.0 tied the residual to
#     yhat - y (opposite sign) in the MAE branch, so its MA terms were estimated against
#     inverted innovations; the release fixed the sign (CHANGELOG, includeModelConstraints!).
#   * input: pjme_daily.csv derived in this run by setup/gen_energy_data.py; output
#     $IJF_RUN_DIR/raw/battery-2026-07/energy/architecture_results.jsonl; load_dataset instead
#     of loadDataset. Extra fields: "solver_status", "penaltyTarget".
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV, LinearAlgebra

const OUT = outfile("energy", "architecture_results.jsonl")
isfile(OUT) && rm(OUT)

daily_full = load_dataset(CSV.read(datafile("energy", "pjme_daily.csv";
                                            producer = "experiments/runners/battery/setup/gen_energy_data.py (needs ./reproduce.sh fetch pjme)"),
                                   DataFrame))
ta = daily_full[end-360+1:end]               # bounded window
const SPEC = (p=1, d=1, q=1, s=7, P=1, D=0, Q=1)
const V030 = (initialization = V030_INITIALIZATION, stationary = V030_STATIONARY)
coefnorm(m) = norm(vcat(m.ϕ === nothing ? Float64[] : m.ϕ,
                        m.θ === nothing ? Float64[] : m.θ,
                        m.Φ === nothing ? Float64[] : m.Φ,
                        m.Θ === nothing ? Float64[] : m.Θ))

function emit(setting, objective, m, rt; alpha=nothing, status="ok", err=nothing)
    rec = Dict{String,Any}("block"=>"energy_architecture", "experiment"=>"objective/regularization",
        "setting"=>setting, "objective"=>objective, "implementation"=>"SARIMAX.jl",
        "dataset"=>"PJME_daily", "order"=>"(1,1,1)(1,0,1)_7", "alpha"=>alpha, "status"=>status,
        "runtime_s"=>rt)
    alpha === nothing || (rec["penaltyTarget"] = "all")
    if status == "ok"
        rec["rss"]=sum(abs2, m.ϵ); rec["resid_mae"]=mean(abs.(m.ϵ)); rec["coef_norm"]=coefnorm(m)
        rec["solver_status"]=fit_status(m)
    else
        rec["error"]=err
    end
    append_record(OUT, rec); println(setting, " -> ", status)
end

mk() = SARIMA(ta, SPEC.p, SPEC.d, SPEC.q; seasonality=SPEC.s, P=SPEC.P, D=SPEC.D, Q=SPEC.Q)

for (setting, obj, alpha) in [("squared-error (MSE)", "mse", nothing),
                              ("absolute-error (MAE)", "mae", nothing),
                              ("ridge (L2)", "elastic_net", 0.0),
                              ("elastic-net", "elastic_net", 0.5)]
    try
        m = mk()
        rt = @elapsed (alpha === nothing ? fit!(m; objectiveFunction=obj, V030...) :
                                           fit!(m; objectiveFunction=obj, alpha=alpha,
                                                penaltyTarget=:all, V030...))
        emit(setting, obj, m, rt; alpha=alpha)
    catch e
        emit(setting, obj, nothing, nothing; alpha=alpha, status="failed", err=sprint(showerror, e))
    end
end

println("energy_architecture DONE -> ", OUT)
