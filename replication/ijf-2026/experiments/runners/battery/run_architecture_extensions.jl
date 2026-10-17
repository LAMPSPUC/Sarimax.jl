# Block 3 - Architecture & extensibility (SARIMAX.jl only).
# Same dynamic SARIMAX specification reused while objective / regularization are varied.
#
# Port of original-scripts/battery-2026-07/scripts/run_architecture_extensions.jl to the release
# API. Same four experiments, data, orders, objectives, alphas, margins and record layout.
# Adaptations:
#   * every fit! receives initialization = :zeroed and stationary = false (the v0.3.0 defaults
#     the archived script relied on; the release defaults are :innovations and true).
#   * "elastic_net" is a DIFFERENT ESTIMATOR in the release and no keyword restores the old
#     one. v0.3.0 (with lambda unset) solved least squares, then minimized an adaptively
#     weighted alpha-L1/(1-alpha)-L2 norm of ALL coefficients (intercept included) subject to
#     RSS <= RSS* + sqrt(2 nu) sigma2. The release minimizes RSS + lambda*[alpha*L1 +
#     (1-alpha)/2*L2] in one stage with lambda = sqrt(effective sample size), never penalizing
#     the intercept (CHANGELOG 1.0.0, "Changed (breaking)"). The port calls the release's
#     "elastic_net" with the same alpha and penaltyTarget = :all (its default, written out:
#     the AR/MA and exogenous blocks, the closest set to v0.3.0's), so the ridge / lasso /
#     elastic_net rows measure the release's estimator.
#   * admissibility flags use the release's checkModelStationarityInvertibility with
#     rootMargin = 0.0: v0.3.0 rejected only roots on/inside the unit circle
#     (StateSpaceModels.assert_*), the release defaults to a 1% margin.
#   * auto(): every keyword is passed explicitly at its v0.3.0 (3f68017) value; the ones the
#     release changed are integrationTest ("kpss", release "kpssShort"), seasonalIntegrationTest
#     ("ocsb", release "seas"), initialization (:zeroed), stationary (false); release-only
#     keywords are set to the values that reproduce the v0.3.0 search (rootMargin = 0.0,
#     constrainedRefit = false, multistart = false, ...). Search internals that changed with no
#     keyword to restore them (exact-likelihood criterion, maxOrder not applied by the stepwise
#     search, order-2 coefficient boxes [-2, 2], internal scaling) are listed in PORTING.md.
#   * "ic" is the release's Sarimax.aicc (exact Gaussian likelihood when computable); the
#     v0.3.0 CSS value is recorded alongside as "ic_css".
#   * load_dataset instead of loadDataset; inputs regenerated in this run by
#     setup/gen_sim_data.py; output $IJF_RUN_DIR/raw/battery-2026-07/architecture/julia_results.jsonl.
#   * Extra fields: "solver_status", "penaltyTarget" (regularization rows).
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV, LinearAlgebra

const OUT = outfile("architecture", "julia_results.jsonl")
isfile(OUT) && rm(OUT)

const PRODUCER = "experiments/runners/battery/setup/gen_sim_data.py"
outliers = load_dataset(CSV.read(datafile("architecture", "sim_outliers.csv"; producer = PRODUCER), DataFrame))
regdf = CSV.read(datafile("architecture", "sim_regularization.csv"; producer = PRODUCER), DataFrame)

const V030 = (initialization = V030_INITIALIZATION, stationary = V030_STATIONARY)

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
        rec["solver_status"] = fit_status(m)
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
        rt = @elapsed fit!(m; objectiveFunction = obj, V030...)
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
        extra = Dict{String,Any}("alpha" => alpha)
        alpha === nothing || (extra["penaltyTarget"] = "all")
        try
            m = SARIMA(y, exog, 1, 0, 0)
            rt = @elapsed begin
                if alpha === nothing
                    fit!(m; objectiveFunction = obj, V030...)
                else
                    fit!(m; objectiveFunction = obj, alpha = alpha, penaltyTarget = :all, V030...)
                end
            end
            emit("regularization", name, obj, m; rt=rt, extra = extra)
        catch e
            emit("regularization", name, obj, nothing; rt=nothing, status="failed",
                 err=sprint(showerror, e), extra = extra)
        end
    end
end

# --- Experiment 3: constraint / admissibility (auto with assert flags on vs off) ---
# SARIMAX.jl does NOT enforce stationarity/invertibility as optimization constraints;
# fit! imposes per-coefficient box bounds [-1,1]. `auto` post-filters candidate models by
# checking the AR/MA polynomial roots (StateSpaceModels.assert_stationarity/invertibility).
# So "constrained vs unconstrained" = the admissibility filter in the model search.
# (Port: with stationary = false and orders <= 1 the release's boxes are the same [-1, 1]; the
# root check is the release's, at rootMargin = 0.0, i.e. the v0.3.0 strict unit-circle test.)
let
    airp = load_dataset(AIR_PASSENGERS)
    sim_arma2 = load_dataset(CSV.read(datafile("validation", "sim_arma.csv"; producer = PRODUCER), DataFrame))
    order_of(m) = "($(m.p),$(m.d),$(m.q))($(m.P),$(m.D),$(m.Q))_$(m.seasonality)"
    is_stationary(m) = Sarimax.checkModelStationarityInvertibility(m, true, false, false; rootMargin = 0.0)
    is_invertible(m) = Sarimax.checkModelStationarityInvertibility(m, false, true, false; rootMargin = 0.0)

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
            rt = @elapsed (m = SARIMA(data, p, d, q); fit!(m; objectiveFunction="mse", V030...))
            rec["status"]="ok"; rec["runtime_s"]=rt; rec["order"]=order_of(m)
            rec["rss"]=sum(abs2, m.ϵ); rec["resid_mae"]=mean(abs.(m.ϵ))
            rec["stationary"]=is_stationary(m)
            rec["invertible"]=is_invertible(m)
            rec["solver_status"]=fit_status(m)
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
                maxp=2, maxq=2, maxP=1, maxQ=1, showLogs=false,
                # every other keyword at its v0.3.0 (3f68017) value, written out
                d=-1, D=-1, maxd=2, maxD=1, maxOrder=5, informationCriteria="aicc",
                allowMean=nothing, allowDrift=nothing,
                integrationTest="kpss", seasonalIntegrationTest="ocsb",
                outlierDetection=false, searchMethod="stepwise", parallel=false,
                seasonalForm=:multiplicative, initialization=V030_INITIALIZATION,
                stationary=V030_STATIONARY, stationarityMargin=0.0,
                # release-only keywords, at the values that reproduce the v0.3.0 search
                rootMargin=0.0, exogDynamics=:armax, penalty=:none, multistart=false,
                invertible=false, constrainedRefit=false, warmStartFromBox=false,
                maxTimeSeconds=nothing, requireTermsWhenOverDifferenced=false,
                requireMAWhenDoublyDifferenced=false, optimizer=Sarimax.Ipopt.Optimizer)
            stat = is_stationary(m)
            inv  = is_invertible(m)
            rec["status"] = "ok"; rec["runtime_s"] = rt
            rec["order"] = order_of(m)
            rec["ic"] = trymetric(Sarimax.aicc, m)
            crit = trymetric(css_criteria, m)
            crit === nothing || (rec["ic_css"] = crit["aicc_css"])
            rec["rss"] = m.ϵ === nothing ? nothing : sum(abs2, m.ϵ)
            rec["resid_mae"] = m.ϵ === nothing ? nothing : mean(abs.(m.ϵ))
            rec["stationary"] = stat; rec["invertible"] = inv
            rec["solver_status"] = fit_status(m)
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
    airp = load_dataset(AIR_PASSENGERS)
    function emit_inv(setting, m, rt; extra = Dict())
        rec = Dict{String,Any}("block"=>"architecture", "experiment"=>"invertibility",
            "setting"=>setting, "objective"=>"mse", "implementation"=>"SARIMAX.jl",
            "dataset"=>"airpassengers", "solver"=>"Ipopt", "seed"=>1234, "status"=>"ok",
            "runtime_s"=>rt, "order"=>"(0,1,1)(0,1,1)_12")
        rec["rss"] = sum(abs2, m.ϵ); rec["resid_mae"] = mean(abs.(m.ϵ))
        rec["detail"] = "theta=$(round.(m.θ,digits=3)), Theta=$(round.(m.Θ,digits=3))"
        rec["invertible"] = Sarimax.checkModelStationarityInvertibility(m, false, true, false; rootMargin = 0.0)
        rec["stationary"] = Sarimax.checkModelStationarityInvertibility(m, true, false, false; rootMargin = 0.0)
        rec["solver_status"] = fit_status(m)
        merge!(rec, extra)
        append_record(OUT, rec)
        println("invertibility / ", setting, " -> ok  (", rec["detail"], ")")
    end
    mb = SARIMA(airp, 0, 1, 1; seasonality=12, P=0, D=1, Q=1)
    tb = @elapsed fit!(mb; objectiveFunction="mse", V030...)
    emit_inv("fit! default (multiplicative, box bounds)", mb, tb)
    # Legacy additive form: the unit-root pile-up that motivated invertible=true is a
    # misspecification artifact of the additive seasonal polynomial (see manuscript).
    ma = SARIMA(airp, 0, 1, 1; seasonality=12, P=0, D=1, Q=1)
    ta = @elapsed fit!(ma; objectiveFunction="mse", seasonalForm=:additive, V030...)
    emit_inv("fit! additive form (box bounds)", ma, ta; extra=Dict("seasonalForm"=>"additive"))
    mr = SARIMA(airp, 0, 1, 1; seasonality=12, P=0, D=1, Q=1)
    tr = @elapsed fit!(mr; objectiveFunction="mse", invertible=true, invertibilityMargin=0.05, V030...)
    emit_inv("fit! invertible=true (rho=0.05)", mr, tr; extra=Dict("invertibilityMargin"=>0.05))
end

println("architecture DONE -> ", OUT)
