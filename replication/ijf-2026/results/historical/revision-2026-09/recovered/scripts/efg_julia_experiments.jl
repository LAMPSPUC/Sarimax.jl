# Parts E, F, G - controlled experiments run against the CURRENT package (master f96feac).
#
#   E  quantile loss: functional demonstration on a series with asymmetric innovations
#   F  coefficient-specific penalty weights: scalar vs heterogeneous lambda on a known model
#   G  local-solver sensitivity to the starting point: K random starts per case
#
# Every keyword that can move an estimate is stated explicitly (no package default is
# inherited), the convention of the replication package. Output: plain CSV in ../data/.
#
#   julia +lts --project=<SARIMAX.jl> efg_julia_experiments.jl [E] [F] [G]
using Sarimax, TimeSeries, Dates, Random, Statistics, LinearAlgebra, Printf

const OUT = joinpath(@__DIR__, "..", "data")
const SEED = 20260911
dates(n) = collect(Date(2000, 1, 1):Month(1):(Date(2000, 1, 1) + Month(n - 1)))
rho(τ, e) = τ * max(e, 0.0) + (1 - τ) * max(-e, 0.0)
fvec(m) = (v = values(m.forecast); ndims(v) == 1 ? collect(v) : collect(v[:, 1]))

function writecsv(path, header::Vector{String}, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join([x isa AbstractString ? "\"$x\"" : string(x) for x in r], ","))
        end
    end
end

# ---------------------------------------------------------------------------------------------
# E. Quantile loss. AR(1) around a level, innovations e_t = Exp(1) - 1: mean zero, right
# skewed, so the 0.1 / 0.5 / 0.9 conditional quantiles are far apart and unequally spaced.
# ---------------------------------------------------------------------------------------------
function simulate_ar1_skew(rng, n; μ = 10.0, φ = 0.6, σ = 1.0)
    y = zeros(n); y[1] = μ
    for t = 2:n
        y[t] = μ + φ * (y[t-1] - μ) + σ * (-log(rand(rng)) - 1.0)
    end
    y
end

function fit_quantile(ytrain, τ)
    m = SARIMA(TimeArray(dates(length(ytrain)), ytrain), 1, 0, 0;
               seasonality = 1, allowMean = true, allowDrift = false, silent = true)
    fit!(m; objectiveFunction = "quantile", quantileLevel = τ, initialization = :zeroed,
         stationary = true, invertible = false, seasonalForm = :multiplicative, silent = true)
    m
end

function part_E()
    taus = (0.1, 0.5, 0.9); n = 240; H = 24; R = 200
    rng = MersenneTwister(SEED)
    # (1) one representative replication, for the figure
    y = simulate_ar1_skew(rng, n)
    ytr, yte = y[1:n-H], y[n-H+1:end]
    rows = Any[]
    for τ in taus
        m = fit_quantile(ytr, τ)
        fit_ = collect(values(m.fitInSample)); off = length(ytr) - length(fit_)
        for t in eachindex(ytr)
            push!(rows, (τ, t, "train", ytr[t], t > off ? fit_[t - off] : NaN))
        end
        predict!(m; stepsAhead = H)
        f = fvec(m)
        for h = 1:H
            push!(rows, (τ, n - H + h, "test", yte[h], f[h]))
        end
    end
    writecsv(joinpath(OUT, "quantile_example_paths.csv"),
             ["tau", "t", "split", "y", "path"], rows)

    # (2) R replications: coverage and the pinball loss each fit is designed to minimize
    cov_in = Dict(τ => Float64[] for τ in taus)
    cov_h1 = Dict(τ => Float64[] for τ in taus)
    cov_all = Dict(τ => Float64[] for τ in taus)
    pin = Dict((τ, τe) => Float64[] for τ in taus, τe in taus)
    coef = Dict(τ => Tuple{Float64,Float64}[] for τ in taus)
    order_ok = 0
    for r = 1:R
        y = simulate_ar1_skew(rng, n)
        ytr, yte = y[1:n-H], y[n-H+1:end]
        fits = Dict(τ => fit_quantile(ytr, τ) for τ in taus)
        locs = Dict{Float64,Float64}()
        for τ in taus
            m = fits[τ]
            push!(cov_in[τ], mean(m.ϵ .<= 1e-9))
            predict!(m; stepsAhead = H); f = fvec(m)
            push!(cov_h1[τ], yte[1] <= f[1] ? 1.0 : 0.0)
            push!(cov_all[τ], mean(yte .<= f))
            for τe in taus
                push!(pin[(τ, τe)], mean(rho.(τe, yte .- f)))
            end
            push!(coef[τ], (m.ϕ[1], m.c))
            locs[τ] = mean(collect(values(m.fitInSample)))
        end
        order_ok += (locs[0.1] <= locs[0.5] <= locs[0.9]) ? 1 : 0
    end
    srows = Any[]
    for τ in taus
        push!(srows, (τ, R, mean(cov_in[τ]), mean(cov_h1[τ]), mean(cov_all[τ]),
                      mean(first.(coef[τ])), mean(last.(coef[τ])),
                      [mean(pin[(τ, τe)]) for τe in taus]...))
    end
    writecsv(joinpath(OUT, "quantile_example_summary.csv"),
             ["tau_fit", "replications", "coverage_in_sample", "coverage_holdout_h1",
              "coverage_holdout_h1to24", "mean_phi", "mean_intercept",
              "holdout_pinball_at_0.1", "holdout_pinball_at_0.5", "holdout_pinball_at_0.9"], srows)
    open(joinpath(OUT, "quantile_example_ordering.txt"), "w") do io
        println(io, "replications = $R")
        println(io, "fitted in-sample location ordered q0.1 <= q0.5 <= q0.9 in $order_ok of $R")
    end
    println("E done: ordering held in $order_ok / $R replications")
end

# ---------------------------------------------------------------------------------------------
# F. Coefficient-specific penalty weights. Regression with AR(1) errors and six standardized
#    regressors, three active and three null. Scalar lambda vs heterogeneous lambda_j.
# ---------------------------------------------------------------------------------------------
function part_F()
    rng = MersenneTwister(SEED + 1)
    n = 200; K = 6; βtrue = [2.0, -1.5, 1.0, 0.0, 0.0, 0.0]
    X = randn(rng, n, K); X = (X .- mean(X; dims = 1)) ./ std(X; dims = 1)
    η = zeros(n); e = randn(rng, n)
    for t = 2:n; η[t] = 0.5 * η[t-1] + e[t]; end
    y = 5.0 .+ X * βtrue .+ η
    ta = TimeArray(dates(n), y); xa = TimeArray(dates(n), X, [Symbol("x$j") for j = 1:K])
    base_kw = (; initialization = :zeroed, stationary = true, invertible = false,
               seasonalForm = :multiplicative, exogDynamics = :regression_errors, silent = true)
    mk() = SARIMA(ta, xa, 1, 0, 0; seasonality = 1, allowMean = true, allowDrift = false,
                  silent = true)

    m0 = mk(); fit!(m0; objectiveFunction = "mse", base_kw...)
    βt = collect(m0.exogCoefficients)
    w = 1.0 ./ max.(abs.(βt), 1e-6)                      # adaptive-Lasso weights, gamma = 1
    schemes(L) = [
        ("scalar", fill(L, K)),
        ("adaptive", L .* w),
        ("known_active_unpenalized", [0.0, 0.0, 0.0, L, L, L]),
    ]
    fitpen(λ) = (m = mk(); fit!(m; objectiveFunction = "mse", penalty = :elastic_net,
                                alpha = 1.0, lambda = λ, penaltyTarget = :exogenous,
                                base_kw...); m)
    grid = exp.(range(log(0.5), log(400.0); length = 16))
    prow = Any[]
    for L in grid, (name, λ) in schemes(L)
        m = fitpen(λ)
        for j = 1:K
            push!(prow, (name, L, j, λ[j], m.exogCoefficients[j], βtrue[j]))
        end
    end
    writecsv(joinpath(OUT, "weighted_penalty_path.csv"),
             ["scheme", "L", "j", "lambda_j", "beta_hat", "beta_true"], prow)
    # Reported L: the smallest grid value at which the SCALAR lasso sets every null coefficient
    # to zero - chosen by rule, from the scalar scheme alone, not by comparing schemes.
    Lstar = nothing
    for L in grid
        m = fitpen(fill(L, K))
        if all(abs.(m.exogCoefficients[4:6]) .< 1e-4); Lstar = L; break; end
    end
    Lstar === nothing && (Lstar = last(grid))
    trow = Any[]
    fits = Dict(name => fitpen(λ) for (name, λ) in schemes(Lstar))
    for j = 1:K
        push!(trow, ("x$j", βtrue[j], βt[j], w[j],
                     [x for (name, λ) in schemes(Lstar) for x in (λ[j], fits[name].exogCoefficients[j])]...))
    end
    writecsv(joinpath(OUT, "weighted_penalty_table.csv"),
             ["coef", "beta_true", "beta_unpenalized", "adaptive_weight",
              "lambda_scalar", "beta_scalar", "lambda_adaptive", "beta_adaptive",
              "lambda_known_active_unpenalized", "beta_known_active_unpenalized"], trow)
    open(joinpath(OUT, "weighted_penalty_meta.txt"), "w") do io
        println(io, "L_star = $Lstar  (smallest grid value zeroing all null coefficients under the scalar lasso)")
        println(io, "penaltyCoefficientNames = ", penaltyCoefficientNames(mk(); penaltyTarget = :exogenous))
        println(io, "phi_hat (unpenalized) = ", m0.ϕ[1])
    end
    println("F done: L* = $Lstar")
end

# ---------------------------------------------------------------------------------------------
# G. Local-solver sensitivity to the starting point. For each case: the zero start plus K-1
#    random starts drawn in REFLECTION coordinates (so every start is stationary/invertible),
#    under two initialization conventions.
# ---------------------------------------------------------------------------------------------
function simulate_arma(rng, n, φ, θ; burn = 200)
    p, q = length(φ), length(θ); N = n + burn
    e = randn(rng, N); y = zeros(N)
    for t = (max(p, q) + 1):N
        y[t] = sum(φ[i] * y[t-i] for i = 1:p; init = 0.0) +
               e[t] + sum(θ[j] * e[t-j] for j = 1:q; init = 0.0)
    end
    y[burn+1:end]
end

function part_G()
    rng = MersenneTwister(SEED + 2)
    K = 30
    airp = log.(values(load_dataset(AIR_PASSENGERS)))
    cases = [
        ("AR(2), easy",                 simulate_arma(rng, 200, [0.5, -0.3], Float64[]), (2, 0, 0, 1, 0, 0, 0)),
        ("ARMA(1,1), mixed",            simulate_arma(rng, 200, [0.7], [-0.4]),          (1, 0, 1, 1, 0, 0, 0)),
        ("ARMA(2,1), mixed",            simulate_arma(rng, 200, [1.2, -0.5], [0.3]),     (2, 0, 1, 1, 0, 0, 0)),
        ("ARMA(1,1) on white noise",    simulate_arma(rng, 200, Float64[], Float64[]),   (1, 0, 1, 1, 0, 0, 0)),
        ("ARMA(2,2), near-cancelling",  simulate_arma(rng, 200, [0.6], [-0.5]),          (2, 0, 2, 1, 0, 0, 0)),
        ("Airline, log AirPassengers",  airp,                                            (0, 1, 1, 12, 0, 1, 1)),
    ]
    rows = Any[]
    for (name, y, (p, d, q, s, P, D, Q)) in cases, init in (:zeroed, :innovations)
        ta = TimeArray(dates(length(y)), y)
        mk() = SARIMA(ta, p, d, q; seasonality = s, P = P, D = D, Q = Q,
                      allowMean = (d + D == 0), allowDrift = false, silent = true)
        for k = 1:K
            ws = nothing
            s0 = Float64[]
            if k > 1
                ws = mk()
                ws.ϕ = p > 0 ? Float64.(Sarimax.reflectionToAR(rand(rng, p) .* 1.9 .- 0.95)) : nothing
                ws.θ = q > 0 ? Float64.(Sarimax.reflectionToMA(rand(rng, q) .* 1.9 .- 0.95)) : nothing
                ws.Φ = P > 0 ? Float64.(Sarimax.reflectionToAR(rand(rng, P) .* 1.9 .- 0.95)) : nothing
                ws.Θ = Q > 0 ? Float64.(Sarimax.reflectionToMA(rand(rng, Q) .* 1.9 .- 0.95)) : nothing
                s0 = vcat(something(ws.ϕ, Float64[]), something(ws.θ, Float64[]),
                          something(ws.Φ, Float64[]), something(ws.Θ, Float64[]))
            end
            m = mk()
            t0 = time()
            fit!(m; objectiveFunction = "mse", initialization = init, stationary = true,
                 invertible = false, seasonalForm = :multiplicative, warmStart = ws,
                 multistart = false, silent = true)
            el = time() - t0
            c = vcat(something(m.ϕ, Float64[]), something(m.θ, Float64[]),
                     something(m.Φ, Float64[]), something(m.Θ, Float64[]))
            push!(rows, (name, String(init), k, k == 1 ? "zero" : "random",
                         join(round.(s0; digits = 6), ";"),
                         get(m.metadata, "objectiveValue", NaN),
                         join(round.(c; digits = 8), ";"),
                         get(m.metadata, "solverStatus", "?"),
                         get(m.metadata, "solverIterations", -1), round(el; digits = 3)))
        end
        println("G: $name / $init done")
    end
    writecsv(joinpath(OUT, "multistart_kstart.csv"),
             ["case", "initialization", "start", "start_kind", "start_coefficients",
              "objective", "coefficients", "solver_status", "iterations", "seconds"], rows)
end

which = isempty(ARGS) ? ["E", "F", "G"] : ARGS
"E" in which && part_E()
"F" in which && part_F()
"G" in which && part_G()
