# Energy dataset (PJME daily) - architecture/extensibility: same SARIMA specification,
# only the optimization objective changes (MSE, MAE, Ridge, Elastic Net).
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV, LinearAlgebra

const OUT = joinpath(RAW, "energy", "architecture_results.jsonl")
isfile(OUT) && rm(OUT)

daily_full = loadDataset(CSV.read(joinpath(RAW, "energy", "data", "pjme_daily.csv"), DataFrame))
ta = daily_full[end-360+1:end]               # bounded window
const SPEC = (p=1, d=1, q=1, s=7, P=1, D=0, Q=1)
coefnorm(m) = norm(vcat(m.ϕ === nothing ? Float64[] : m.ϕ,
                        m.θ === nothing ? Float64[] : m.θ,
                        m.Φ === nothing ? Float64[] : m.Φ,
                        m.Θ === nothing ? Float64[] : m.Θ))

function emit(setting, objective, m, rt; alpha=nothing, status="ok", err=nothing)
    rec = Dict{String,Any}("block"=>"energy_architecture", "experiment"=>"objective/regularization",
        "setting"=>setting, "objective"=>objective, "implementation"=>"SARIMAX.jl",
        "dataset"=>"PJME_daily", "order"=>"(1,1,1)(1,0,1)_7", "alpha"=>alpha, "status"=>status,
        "runtime_s"=>rt)
    if status == "ok"
        rec["rss"]=sum(abs2, m.ϵ); rec["resid_mae"]=mean(abs.(m.ϵ)); rec["coef_norm"]=coefnorm(m)
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
        rt = @elapsed (alpha === nothing ? fit!(m; objectiveFunction=obj) :
                                           fit!(m; objectiveFunction=obj, alpha=alpha))
        emit(setting, obj, m, rt; alpha=alpha)
    catch e
        emit(setting, obj, nothing, nothing; alpha=alpha, status="failed", err=sprint(showerror, e))
    end
end

println("energy_architecture DONE -> ", OUT)
