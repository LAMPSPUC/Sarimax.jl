# Effect of the exact-likelihood fix on models with regressors: run with the code under test
# (src/ of the checkout) and write one CSV row per case. Usage: julia effects_exog.jl OUT.csv SIMSARIMAX.csv
using Sarimax, TimeSeries, Dates, Random, Statistics, CSV, DataFrames, Printf

function simulate(kind; d = 0, nreg = 2, T = 240, burn = 200, extra = 12, seed = 1)
    rng = MersenneTwister(seed)
    N = T + burn + extra
    X = zeros(N, 2); u = randn(rng, N, 2)
    for t = 2:N
        X[t, 1] = 5 + 0.7 * (X[t-1, 1] - 5) + u[t, 1]
        X[t, 2] = -2 + 0.4 * (X[t-1, 2] + 2) + u[t, 2]
    end
    b = [2.0, -1.5][1:nreg]
    e, z, η = randn(rng, N), zeros(N), zeros(N)
    for t = 2:N
        xb = sum(X[t, j] * b[j] for j = 1:nreg)
        if kind === :armax
            z[t] = 0.6 * z[t-1] + xb + e[t] + 0.3 * e[t-1]
        else
            η[t] = 0.6 * η[t-1] + e[t] + 0.3 * e[t-1]
            z[t] = xb + η[t]
        end
    end
    y = d == 0 ? z : cumsum(z)
    ts = collect(Date(2000, 1, 1):Day(1):(Date(2000, 1, 1)+Day(T + extra - 1)))
    TimeArray(ts[1:T], y[(burn+1):(burn+T)], [:y]),
    TimeArray(ts, X[(burn+1):end, 1:nreg], [:x1, :x2][1:nreg]),
    y[(burn+T+1):end]
end
order(m) = "($(m.p),$(m.d),$(m.q))"
rows = DataFrame()
for kind in (:armax, :regression_errors), nreg in (1, 2), d in (0, 1), seed in 1:10
    y, X, future = simulate(kind; d = d, nreg = nreg, seed = seed)
    for sem in (:armax, :regression_errors)
        m = auto(y; exog = X, seasonality = 1, exogDynamics = sem)
        predict!(m; stepsAhead = 12)
        fc = Float64.(vec(values(m.forecast)))
        push!(rows, (case = "sim", dgp = String(kind), nreg = nreg, d_true = d, seed = seed,
                     semantics = String(sem), order = order(m), aicc = aicc(m), aic = aic(m), bic = bic(m),
                     fallback = get(m.metadata, "criterionFallback", missing),
                     rmse12 = sqrt(mean((fc .- future) .^ 2)), fc1 = fc[1], fc12 = fc[12]); promote = true)
    end
    flush(stdout)
end
# The battery's ARX row (validation, sim_sarimax): fixed order ARX(1) with two regressors.
df = CSV.read(ARGS[2], DataFrame)
ts = Date.(df.date)
yb = TimeArray(ts, Float64.(df.value), [:value])
Xb = TimeArray(ts, hcat(Float64.(df.x1), Float64.(df.x2)), [:x1, :x2])
mb = SARIMA(yb, Xb, 1, 0, 0)
fit!(mb; objectiveFunction = "mse", initialization = :zeroed, stationary = false)
@printf("battery ARX(1)+2exog: aic %.2f bic %.2f exactLoglike %s CSS loglike %.4f\n", aic(mb), bic(mb),
        string(Sarimax.exactLoglike(mb)), Sarimax.loglike(mb))
CSV.write(ARGS[1], rows)
