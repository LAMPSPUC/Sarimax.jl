# What the information criteria score when `auto` ranks candidates.
#
# Each testset compares the package's ranking likelihood (`exactLoglike`, through
# `criterionLoglikeAndN`) with a likelihood computed here from the fitted model's OWN equation:
# the mean it implies is read from its forecasts, and the residual process from the
# exogenous semantics it was fitted under. Where the two disagree the property is recorded as
# `@test_broken`, and the current value is pinned with a plain `@test` beside it, so that a
# change in either direction is seen. They are recorded rather than fixed because a fix moves
# model selection (every `d + D = 1` search with AR terms, and every `:armax` search with
# regressors), which needs review before it lands.
@testset "criterion scoring: the likelihood that ranks candidates" begin
    dates(n) = collect(Date(2000, 1, 1):Day(1):(Date(2000, 1, 1)+Day(n - 1)))

    @testset "drift under AR terms: the level the exact likelihood removes" begin
        # dz_t = b + phi dz_{t-1} + e_t, the equation the package fits with `allowDrift`:
        # `trend * driftValues[t]` enters yhat beside the AR terms (sarima.jl, in-sample
        # expression), exactly as `c` does. The differenced series therefore has mean
        # b / (1 - phi), not b.
        rng = MersenneTwister(20261016)
        n, b, φ = 400, 0.5, 0.6
        e = randn(rng, n)
        dz = zeros(n)
        for t = 2:n
            dz[t] = b + φ * dz[t-1] + e[t]
        end
        m = SARIMA(TimeArray(dates(n), cumsum(dz) .+ 100.0), 1, 1, 0;
                   allowMean = false, allowDrift = true)
        fit!(m; objectiveFunction = "mse", initialization = :zeroed, stationary = true)
        φ̂, b̂ = Float64(m.ϕ[1]), Float64(m.trend)
        level = b̂ / (1 - φ̂)

        # The fitted model's own mean: its long-horizon forecast increments converge to it.
        predict!(m; stepsAhead = 400)
        f = Float64.(values(m.forecast))
        @test isapprox(f[end] - f[end-1], level; rtol = 1e-6)

        z = Float64.(values(differentiate(m.y, 1, 0, 1)))
        llLevel = Sarimax.exactGaussianLogLikelihood(z .- level, [φ̂], Float64[])
        llTrend = Sarimax.exactGaussianLogLikelihood(z .- b̂, [φ̂], Float64[])
        @test llLevel - llTrend > 10            # premise: the two differ, and by a lot
        @test_broken Sarimax.exactLoglike(m) ≈ llLevel
        @test Sarimax.exactLoglike(m) ≈ llTrend  # current behaviour (see deterministic_term.jl)
        # The constant, by contrast, is converted to its level (deterministic_term.jl, first
        # testset): the two deterministic terms are treated by different conventions.
    end

    @testset "regressors under :armax: the model the exact likelihood evaluates" begin
        # Same coefficients, two exogenous semantics. They are different models (their
        # conditional residuals differ), so their exact likelihoods should differ too.
        rng = MersenneTwister(7)
        n, β, φ = 300, 2.0, 0.7
        x = randn(rng, n)
        e = randn(rng, n)
        y = zeros(n)
        for t = 2:n
            y[t] = β * x[t] + φ * y[t-1] + e[t]      # :armax data-generating process
        end
        ta = TimeArray(dates(n), y)
        X = TimeArray(dates(n), reshape(x, :, 1), [:x])
        fixedFit(dyn) = begin
            m = SARIMA(ta, X, 1, 0, 0; allowMean = false)
            m.ϕ = [φ]
            m.exogCoefficients = [β]
            m.keepProvidedCoefficients = true
            fit!(m; objectiveFunction = "mse", initialization = :zeroed, stationary = false,
                 exogDynamics = dyn)
            m
        end
        mA, mR = fixedFit(:armax), fixedFit(:regression_errors)
        @test !isapprox(Sarimax.loglike(mA), Sarimax.loglike(mR); rtol = 1e-3)   # different models
        # What is computed is the regression-with-ARMA-errors likelihood of (y - x*beta):
        llReg = Sarimax.exactGaussianLogLikelihood(y .- β .* x, [φ], Float64[])
        @test Sarimax.exactLoglike(mR) ≈ llReg
        @test_broken !(Sarimax.exactLoglike(mA) ≈ Sarimax.exactLoglike(mR))
        @test Sarimax.exactLoglike(mA) ≈ llReg   # current behaviour: :armax scored as :regression_errors
    end

    @testset "parameter count K: conventions" begin
        rng = MersenneTwister(11)
        n = 240
        z = zeros(n)
        e = randn(rng, n)
        for t = 3:n
            z[t] = 0.5 * z[t-1] - 0.2 * z[t-2] + e[t] + 0.3 * e[t-1]
        end
        ta = TimeArray(dates(n), z .+ 10.0)
        # ridge shrinks without zeroing; K stays the nominal count (p + q + mean + sigma^2),
        # i.e. no effective-degrees-of-freedom reduction is applied
        mr = SARIMA(ta, 2, 0, 1; allowMean = true)
        fit!(mr; objectiveFunction = "ridge", initialization = :zeroed)
        @test get_hyperparameters_number(mr) == 2 + 1 + 1 + 1
        # coefficients FIXED by keepProvidedCoefficients are still counted: K does not depend
        # on whether a coefficient was estimated (stats::arima's `fixed` would not count them)
        mk = SARIMA(ta, 2, 0, 1; allowMean = false)
        mk.ϕ = [0.5, -0.2]
        mk.θ = [0.3]
        mk.keepProvidedCoefficients = true
        fit!(mk; objectiveFunction = "mse", initialization = :zeroed, stationary = false)
        @test get_hyperparameters_number(mk) == 2 + 1 + 1
        # n of the exact path is the differenced length, the same for every candidate of a
        # search because d and D are fixed before it starts
        ll, nn, exact = Sarimax.criterionLoglikeAndN(mk)
        @test exact
        @test nn == n
    end
end
