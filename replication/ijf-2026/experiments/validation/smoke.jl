# Smoke test, Julia side: small but real estimation problems through the package's own code
# paths. Prints one JSON object per measured quantity:
#   {"engine": "julia", "test": ..., "quantity": ..., "value": ...}
# The orchestrator (scripts/ijflib/smoke.py) compares them with smoke_expected.json.
#
# Usage: julia --project=<replication env> smoke.jl [test-name ...]   (default: all tests)

using Sarimax
using JuMP, Ipopt, SCIP
using CSV, DataFrames, Dates, JSON, LinearAlgebra, Statistics, TimeSeries
const MOI = JuMP.MOI

BLAS.set_num_threads(1)
const FIXTURES = normpath(joinpath(@__DIR__, "..", "..", "data", "fixtures"))
const ONLY = isempty(ARGS) ? nothing : Set(ARGS)

function emit(test, quantity, value; note = "")
    v = value isa AbstractFloat && !isfinite(value) ? string(value) : value
    println(JSON.json(Dict("engine" => "julia", "test" => test, "quantity" => quantity,
                           "value" => v, "note" => note)))
    flush(stdout)
end

# Runs one test; an exception is reported as a quantity, so the orchestrator sees a FAIL
# rather than a silently missing test.
function runtest(f, name)
    (ONLY === nothing || name in ONLY) || return
    t = @elapsed try
        f()
    catch err
        emit(name, "error", sprint(showerror, err))
    end
    emit(name, "seconds", round(t, digits = 2))
end

simArma() = load_dataset(CSV.read(joinpath(FIXTURES, "sim_arma.csv"), DataFrame))

# Deterministic white noise: a 64-bit LCG (Knuth's MMIX constants) and Box-Muller. The integer
# arithmetic wraps identically on every platform, so the series does not depend on the Julia
# version's random number generator.
function whitenoise(n; seed = 20261017)
    s = UInt64(seed)
    next() = (s = s * 0x5851f42d4c957f2d + 0x14057b7ef767814f; ((s >> 11) + 0.5) / 2.0^53)
    return [sqrt(-2log(next())) * cos(2pi * next()) for _ = 1:n]
end
ta(y) = TimeArray(collect(Date(2000, 1, 1):Month(1):(Date(2000, 1, 1) + Month(length(y) - 1))), y)
smape(a, f) = 200 / length(a) * sum(abs(a[i] - f[i]) / (abs(a[i]) + abs(f[i])) for i in eachindex(a))
maseDen(y, s) = sum(abs(y[j] - y[j-s]) for j = (s+1):length(y)) / (length(y) - s)
rootsAR2(ϕ1, ϕ2) = (Δ = complex(ϕ1^2 + 4ϕ2); [(-ϕ1 + sqrt(Δ)) / (2ϕ2), (-ϕ1 - sqrt(Δ)) / (2ϕ2)])

# 1. Seasonal ARIMA with Ipopt, forecast and forecast evaluation ------------------------------
# The "airline" model on the log of the package's AIR_PASSENGERS dataset, last 12 months held
# out. (The file datasets/airpassengers.csv holds 204 monthly observations, 1991-2008; it is
# not the 144-observation Box-Jenkins airline series. See LIMITATIONS.md.)
runtest("airline_ipopt") do
    airp = load_dataset(AIR_PASSENGERS)
    y = log.(values(airp)[:, 1])
    train, test = y[1:end-12], y[end-11:end]
    m = SARIMA(ta(train), 0, 1, 1; seasonality = 12, P = 0, D = 1, Q = 1)
    fit!(m; objectiveFunction = "mse")
    emit("airline_ipopt", "solver_status", m.metadata["solverStatus"])
    emit("airline_ipopt", "theta1", m.θ[1])
    emit("airline_ipopt", "Theta1", m.Θ[1])
    emit("airline_ipopt", "sigma2", m.σ²)
    predict!(m; stepsAhead = 12)
    f = values(m.forecast)[:, 1]
    emit("airline_ipopt", "forecast_h1", f[1])
    emit("airline_ipopt", "forecast_h12", f[12])
    a, g = exp.(test), exp.(f)
    emit("airline_ipopt", "smape_12", smape(a, g))
    emit("airline_ipopt", "mase_12", mean(abs.(a .- g)) / maseDen(exp.(train), 12))
end

# 2. Conditional least squares against R's CSS (archived July 2026 values) -----------------
# initialization = :warmup reproduces R `arima(method = "CSS")`; R reports the process mean
# mu, the package the intercept c = mu * (1 - phi).
runtest("ar1_css") do
    m = SARIMA(simArma(), 1, 0, 0)
    fit!(m; objectiveFunction = "mse", initialization = :warmup)
    emit("ar1_css", "solver_status", m.metadata["solverStatus"])
    emit("ar1_css", "phi1", m.ϕ[1])
    emit("ar1_css", "mu", m.c / (1 - m.ϕ[1]))
    emit("ar1_css", "rss", sum(abs2, m.ϵ))
end

# 3. Non-default objectives: each estimate is optimal for its own criterion ------------------
# AR(2) residuals are affine in the parameters, so mse (a QP) and mae (an LP) are convex and
# Ipopt's solutions are global: the mse fit must have the smaller sum of squares and the mae
# fit the smaller sum of absolute residuals, on the same residual definition (:zeroed).
runtest("objectives") do
    fits = Dict{String,Any}()
    for obj in ("mse", "mae", "huber")
        m = SARIMA(simArma(), 2, 0, 0)
        fit!(m; objectiveFunction = obj, initialization = :zeroed)
        fits[obj] = m
        emit("objectives", "$(obj)_solver_status", m.metadata["solverStatus"])
        emit("objectives", "$(obj)_phi1", m.ϕ[1])
        emit("objectives", "$(obj)_phi2", m.ϕ[2])
    end
    rss(m) = sum(abs2, m.ϵ)
    sad(m) = sum(abs, m.ϵ)
    emit("objectives", "mse_has_smallest_rss", rss(fits["mse"]) <= rss(fits["mae"]) * (1 + 1e-9))
    emit("objectives", "mae_has_smallest_sad", sad(fits["mae"]) <= sad(fits["mse"]) * (1 + 1e-9))
    emit("objectives", "mae_differs_from_mse", abs(fits["mae"].ϕ[1] - fits["mse"].ϕ[1]) > 1e-4)
end

# 4. Admissibility constraints --------------------------------------------------------------
# (a) Differencing white noise over-differences it: the MA(1) of the result has its root on
#     the unit circle (theta = -1). With invertible = true and a 0.05 margin the estimate
#     must stay at |theta| <= 0.95, i.e. the MA root modulus must be >= 1/0.95.
# (b) A random walk fitted with an AR(2) and no differencing: stationary = true must keep
#     both AR roots outside the unit circle.
runtest("admissibility") do
    x = whitenoise(300) .+ 10.0
    m = SARIMA(ta(x), 0, 1, 1; allowMean = false)
    fit!(m; objectiveFunction = "mse", invertible = true, invertibilityMargin = 0.05,
         initialization = :zeroed)
    emit("admissibility", "invertible_solver_status", m.metadata["solverStatus"])
    emit("admissibility", "invertible_theta1", m.θ[1])
    emit("admissibility", "invertible_ma_root_modulus", 1 / abs(m.θ[1]))
    emit("admissibility", "invertible_within_margin", abs(m.θ[1]) <= 0.95 + 1e-7)
    free = SARIMA(ta(x), 0, 1, 1; allowMean = false)
    fit!(free; objectiveFunction = "mse", invertible = false, initialization = :zeroed)
    emit("admissibility", "free_theta1", free.θ[1], note = "unconstrained estimate, at the unit circle")

    z = cumsum(whitenoise(300)) .+ 50.0
    s = SARIMA(ta(z), 2, 0, 0)
    fit!(s; objectiveFunction = "mse", stationary = true, initialization = :zeroed)
    r = rootsAR2(s.ϕ[1], s.ϕ[2])
    emit("admissibility", "stationary_solver_status", s.metadata["solverStatus"])
    emit("admissibility", "stationary_phi1", s.ϕ[1])
    emit("admissibility", "stationary_phi2", s.ϕ[2])
    emit("admissibility", "stationary_min_root_modulus", minimum(abs.(r)))
    emit("admissibility", "stationary_roots_outside_unit_circle", minimum(abs.(r)) > 1)
end

# 5. Global optimisation with SCIP, with the certificate data -------------------------------
# MA(1) by conditional least squares on 20 observations of y_t = e_t + 0.5 e_{t-1}: with
# eps_0 = 0 the objective depends on theta alone, so a dense grid over [-1, 1] (independent
# of any solver) locates the global minimum, which here is interior and unique
# (theta* = 0.675838, f* = 15.2302802). SCIP must close the gap, its value must not exceed the
# grid's or Ipopt's, and Sarimax's own fit! with SCIP must agree. The package fits use
# `initialization = :warmup`, which is this objective (eps_0 = 0, every observation kept, as in
# R's CSS); `:zeroed` would instead drop the first observation and fix eps_1 = 0, a different
# problem whose optimum is theta = 0.7786.
ma1Data() = (e = whitenoise(21); [e[t+1] + 0.5 * e[t] for t = 1:20])
function ma1Model(y, optimizer)
    T = length(y)
    M = 10 * maximum(abs, y)
    jm = Model(optimizer)
    set_silent(jm)
    @variable(jm, -1 <= θ <= 1)
    @variable(jm, -M <= ε[1:T] <= M)
    @constraint(jm, ε[1] == y[1])
    @constraint(jm, [t = 2:T], ε[t] + θ * ε[t-1] == y[t])
    @objective(jm, Min, sum(ε[t]^2 for t = 1:T))
    return jm
end
function cssMA1(y, θ)
    e, s = 0.0, 0.0
    for t in eachindex(y)
        e = y[t] - θ * e
        s += e^2
    end
    return s
end

runtest("scip_global") do
    y = ma1Data()
    scip = optimizer_with_attributes(SCIP.Optimizer, "limits/time" => 120.0, "limits/gap" => 0.0,
                                     "display/verblevel" => 0, "randomization/randomseedshift" => 0)
    jm = ma1Model(y, scip)
    optimize!(jm)
    emit("scip_global", "termination_status", string(termination_status(jm)))
    emit("scip_global", "primal_status", string(primal_status(jm)))
    emit("scip_global", "objective_value", objective_value(jm))
    emit("scip_global", "objective_bound", objective_bound(jm))
    emit("scip_global", "relative_gap", relative_gap(jm))
    emit("scip_global", "theta", value(jm[:θ]))
    emit("scip_global", "scip_version", MOI.get(jm, MOI.SolverVersion()))
    emit("scip_global", "solve_time_s", solve_time(jm))
    grid = minimum(cssMA1(y, θ) for θ in range(-1, 1; length = 200_001))
    emit("scip_global", "grid_upper_bound", grid)
    emit("scip_global", "scip_not_above_grid", objective_value(jm) <= grid + 1e-6)
    ip = ma1Model(y, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    set_start_value(ip[:θ], 0.0)
    optimize!(ip)
    emit("scip_global", "ipopt_status", string(termination_status(ip)))
    emit("scip_global", "ipopt_objective", objective_value(ip))
    emit("scip_global", "scip_not_above_ipopt", objective_value(jm) <= objective_value(ip) + 1e-6)

    mS = SARIMA(ta(y), 0, 0, 1; allowMean = false)
    fit!(mS; optimizer = scip, initialization = :warmup)
    mI = SARIMA(ta(y), 0, 0, 1; allowMean = false)
    fit!(mI; initialization = :warmup)
    emit("scip_global", "package_scip_status", mS.metadata["solverStatus"])
    emit("scip_global", "package_scip_theta", mS.θ[1])
    emit("scip_global", "package_scip_rss_not_above_ipopt",
         sum(abs2, mS.ϵ) <= sum(abs2, mI.ϵ) + 1e-6)
end
