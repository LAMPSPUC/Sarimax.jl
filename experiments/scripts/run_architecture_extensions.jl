# Block 3 - Architecture & extensibility (SARIMAX.jl only).
# Same dynamic SARIMAX specification reused while objective / regularization are varied.
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV, LinearAlgebra

const OUT = joinpath(RAW, "architecture", "julia_results.jsonl")
isfile(OUT) && rm(OUT)

datadir = joinpath(RAW, "architecture", "data")
outliers = loadDataset(CSV.read(joinpath(datadir, "sim_outliers.csv"), DataFrame))
regdf = CSV.read(joinpath(datadir, "sim_regularization.csv"), DataFrame)

resid_summary(m) = (rss = sum(abs2, m.ϵ), resid_mae = mean(abs.(m.ϵ)))

function emit(experiment, setting, objective, m; rt, status="ok", extra=Dict(), err=nothing)
    rec = Dict{String,Any}("block"=>"architecture", "experiment"=>experiment,
        "setting"=>setting, "objective"=>objective, "implementation"=>"SARIMAX.jl",
        "dataset"=>experiment, "solver"=>"Ipopt", "seed"=>1234,
        "status"=>status, "runtime_s"=>rt)
    if status == "ok"
        s = resid_summary(m)
        rec["rss"] = s.rss; rec["resid_mae"] = s.resid_mae
        rec["estimates"] = extract_estimates(m)
        ec = m.exogCoefficients
        rec["coef_norm"] = ec === nothing ? nothing : norm(ec)
    else
        rec["error"] = err
    end
    merge!(rec, extra)
    append_record(OUT, rec)
    println(experiment, " / ", setting, " -> ", status)
end

# --- Experiment 1: objective-function swap on outlier-contaminated series ---
# AR-only specification (no MA terms), so both problems are convex: the squared-error fit is a
# convex QP (global RSS minimizer) and the absolute-error fit is an LP (global sum-|eps| minimizer).
# This makes the swap interpretable: MSE attains the lower RSS, MAE the lower residual MAE.
for obj in ("mse", "mae")
    try
        m = SARIMA(outliers, 1, 0, 0)
        rt = @elapsed fit!(m; objectiveFunction = obj)
        emit("objective_swap", obj == "mse" ? "squared-error" : "absolute-error", obj, m;
             rt=rt, extra = Dict("order" => "(1,0,0)"))
    catch e
        emit("objective_swap", obj, obj, nothing; rt=nothing, status="failed",
             err=sprint(showerror, e))
    end
end

# --- Experiment 2: regularization with many exogenous regressors ---
let
    ts = regdf.date
    y = TimeArray(ts, Float64.(regdf.value), ["value"])
    xcols = [c for c in names(regdf) if startswith(String(c), "x")]
    X = Matrix{Float64}(regdf[:, xcols])
    exog = TimeArray(ts, X, String.(xcols))
    settings = [("unregularized", "mse", nothing),
                ("ridge", "elastic_net", 0.0),
                ("lasso", "elastic_net", 1.0),
                ("elastic_net", "elastic_net", 0.5)]
    for (name, obj, alpha) in settings
        try
            m = SARIMA(y, exog, 1, 0, 0)
            rt = @elapsed begin
                if alpha === nothing
                    fit!(m; objectiveFunction = obj)
                else
                    fit!(m; objectiveFunction = obj, alpha = alpha)
                end
            end
            emit("regularization", name, obj, m; rt=rt,
                 extra = Dict("alpha" => alpha))
        catch e
            emit("regularization", name, obj, nothing; rt=nothing, status="failed",
                 err=sprint(showerror, e), extra = Dict("alpha" => alpha))
        end
    end
end

# --- Experiment 3: constraint / admissibility (auto with assert flags on vs off) ---
# SARIMAX.jl does NOT enforce stationarity/invertibility as optimization constraints;
# fit! imposes per-coefficient box bounds [-1,1]. `auto` post-filters candidate models by
# checking the AR/MA polynomial roots (StateSpaceModels.assert_stationarity/invertibility).
# So "constrained vs unconstrained" = the admissibility filter in the model search.
let
    airp = loadDataset(AIR_PASSENGERS)
    sim_arma2 = loadDataset(CSV.read(joinpath(RAW, "validation", "data", "sim_arma.csv"), DataFrame))
    order_of(m) = "($(m.p),$(m.d),$(m.q))($(m.P),$(m.D),$(m.Q))_$(m.seasonality)"

    # 3a. Direct fit! admissibility checks: fit! only imposes box bounds [-1,1], so a fit
    # may sit on the boundary (unit root => non-stationary). Demonstrates fit! does NOT
    # enforce stationarity/invertibility.
    for (lbl, data, p, d, q) in (("fit! ARIMA(1,0,1) airpassengers", airp, 1, 0, 1),
                                 ("fit! ARIMA(1,0,1) sim_arma", sim_arma2, 1, 0, 1))
        rec = Dict{String,Any}("block"=>"architecture", "experiment"=>"admissibility",
            "setting"=>lbl, "objective"=>"mse", "implementation"=>"SARIMAX.jl",
            "dataset"=>"mixed", "solver"=>"Ipopt", "seed"=>1234)
        try
            local m, rt
            rt = @elapsed (m = SARIMA(data, p, d, q); fit!(m; objectiveFunction="mse"))
            rec["status"]="ok"; rec["runtime_s"]=rt; rec["order"]=order_of(m)
            rec["rss"]=sum(abs2, m.ϵ); rec["resid_mae"]=mean(abs.(m.ϵ))
            rec["stationary"]=Sarimax.checkModelStationarityInvertibility(m, true, false, false)
            rec["invertible"]=Sarimax.checkModelStationarityInvertibility(m, false, true, false)
        catch e
            rec["status"]="failed"; rec["error"]=sprint(showerror, e)
        end
        append_record(OUT, rec)
        println("admissibility / ", lbl, " -> ", rec["status"])
    end

    for (name, assertSI) in (("admissible-only (assert on)", true),
                             ("unconstrained search (assert off)", false))
        rec = Dict{String,Any}("block"=>"architecture", "experiment"=>"admissibility",
            "setting"=>name, "objective"=>"mse", "implementation"=>"SARIMAX.jl",
            "dataset"=>"airpassengers", "solver"=>"Ipopt", "seed"=>1234)
        try
            local m, rt
            rt = @elapsed m = auto(airp; seasonality=12, objectiveFunction="mse",
                assertStationarity=assertSI, assertInvertibility=assertSI,
                maxp=2, maxq=2, maxP=1, maxQ=1, showLogs=false)
            stat = Sarimax.checkModelStationarityInvertibility(m, true, false, false)
            inv  = Sarimax.checkModelStationarityInvertibility(m, false, true, false)
            rec["status"] = "ok"; rec["runtime_s"] = rt
            rec["order"] = order_of(m)
            rec["ic"] = trymetric(Sarimax.aicc, m)
            rec["rss"] = m.ϵ === nothing ? nothing : sum(abs2, m.ϵ)
            rec["resid_mae"] = m.ϵ === nothing ? nothing : mean(abs.(m.ϵ))
            rec["stationary"] = stat; rec["invertible"] = inv
        catch e
            rec["status"] = "failed"; rec["error"] = sprint(showerror, e)
        end
        append_record(OUT, rec)
        println("admissibility / ", name, " -> ", rec["status"])
    end
end

# --- Experiment 4: corrected invertibility parameterization (invertible=true) ---
# fit! default imposes only per-coefficient box bounds [-1,1] (NOT the invertibility region
# for q>=2, and the box boundary is a unit MA root). invertible=true generates the MA
# coefficients from bounded reflection coefficients κ (|κ|<=1-ρ), guaranteeing an invertible
# MA polynomial by construction. Demonstrated on the airline model, where the box fit piles
# θ up on the unit-root boundary.
let
    airp = loadDataset(AIR_PASSENGERS)
    function emit_inv(setting, m, rt; extra = Dict())
        rec = Dict{String,Any}("block"=>"architecture", "experiment"=>"invertibility",
            "setting"=>setting, "objective"=>"mse", "implementation"=>"SARIMAX.jl",
            "dataset"=>"airpassengers", "solver"=>"Ipopt", "seed"=>1234, "status"=>"ok",
            "runtime_s"=>rt, "order"=>"(0,1,1)(0,1,1)_12")
        rec["rss"] = sum(abs2, m.ϵ); rec["resid_mae"] = mean(abs.(m.ϵ))
        rec["detail"] = "theta=$(round.(m.θ,digits=3)), Theta=$(round.(m.Θ,digits=3))"
        rec["invertible"] = Sarimax.checkModelStationarityInvertibility(m, false, true, false)
        rec["stationary"] = Sarimax.checkModelStationarityInvertibility(m, true, false, false)
        merge!(rec, extra)
        append_record(OUT, rec)
        println("invertibility / ", setting, " -> ok  (", rec["detail"], ")")
    end
    mb = SARIMA(airp, 0, 1, 1; seasonality=12, P=0, D=1, Q=1)
    tb = @elapsed fit!(mb; objectiveFunction="mse")
    emit_inv("fit! default (box bounds)", mb, tb)
    mr = SARIMA(airp, 0, 1, 1; seasonality=12, P=0, D=1, Q=1)
    tr = @elapsed fit!(mr; objectiveFunction="mse", invertible=true, invertibilityMargin=0.05)
    emit_inv("fit! invertible=true (rho=0.05)", mr, tr; extra=Dict("invertibilityMargin"=>0.05))
end

println("architecture DONE -> ", OUT)
