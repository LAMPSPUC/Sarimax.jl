@testset "Alternative objective functions" begin
    # The optimization formulation is what lets the objective be swapped — robust
    # (mae), tail-averse (stable/CVaR), regularized (elastic_net) and bilevel fits all
    # reuse the same model. These guards came out of a stability sweep across the six
    # M4 frequencies, which is also where the elastic-net edge case below surfaced.

    airp = Float64.(values(loadDataset(AIR_PASSENGERS)))
    dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(length(airp) - 1))
    series = TimeArray(collect(dates), airp)

    @testset "each objective produces a usable fit" begin
        for obj in ("mse", "mae", "ml", "stable")
            model = SARIMA(series, 1, 1, 1; seasonality = 12, P = 0, D = 1, Q = 1)
            fit!(model; objectiveFunction = obj)
            @test Sarimax.isFitted(model)
            @test all(isfinite, model.ϕ)
            @test all(isfinite, model.θ)
            @test isfinite(model.σ²) && model.σ² > 0
            predict!(model; stepsAhead = 12)
            @test all(isfinite, values(model.forecast))
        end
    end

    @testset "bilevel rejects the invertible parameterization explicitly" begin
        # The MA coefficients are outer parameters there, so they cannot also be
        # generated from reflection coefficients. The clear failure is the contract.
        model = SARIMA(series, 1, 1, 1; seasonality = 12, P = 0, D = 1, Q = 1)
        @test_throws AssertionError fit!(model; objectiveFunction = "bilevel",
                                         invertible = true)
    end

    @testset "CVaR level controls how hard the tail is chased" begin
        # The "stable" objective is CVaR of the squared residuals in Rockafellar-Uryasev
        # form. Its normalization used to be missing: `0.7δ + Σu` is `δ + Σu/0.7`, i.e.
        # (1-α)·n = 0.7, so the effective level was α = 1 - 0.7/n — ~99% for a typical
        # sample and, worse, *dependent on the sample size*. That is min-max in
        # disguise: it equalizes residuals and chases the outlier instead of tolerating
        # it. With the normalization in place the level behaves monotonically, which is
        # what this test pins.
        Random.seed!(6001)
        n = 120
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        y = 50.0 .+ cumsum(randn(n) .* 0.5)
        y[60] += 10 * std(diff(y))                     # a single 10σ additive outlier
        ta = TimeArray(collect(dates), y)

        function tailStats(level)
            m = SARIMA(ta, 1, 1, 1; seasonality = 1)
            fit!(m; objectiveFunction = "stable", cvarLevel = level)
            r = abs.(m.ϵ)
            (maximum(r), median(r))
        end

        loMax, loMed = tailStats(0.5)
        hiMax, hiMed = tailStats(0.99)

        # A high level concentrates on the worst residuals: it shrinks the outlier's
        # residual and pays for it in the body of the sample.
        @test hiMax < loMax
        @test hiMed > loMed

        @test_throws AssertionError fit!(SARIMA(ta, 1, 1, 1; seasonality = 1);
                                         objectiveFunction = "stable", cvarLevel = 1.0)
        @test_throws AssertionError fit!(SARIMA(ta, 1, 1, 1; seasonality = 1);
                                         objectiveFunction = "stable", cvarLevel = 0.0)
    end

    @testset "elastic net survives a saturated model" begin
        # Regression: with few residual degrees of freedom after conditioning
        # (ν = n - lb - K + 1 ≤ 0) the tolerance refinement used to evaluate
        # sqrt of a negative number and throw, discarding a valid first-stage fit.
        # A short seasonal series reproduces it — seasonal conditioning consumes the
        # sample faster than a longer non-seasonal one of similar length.
        Random.seed!(5001)
        n = 17
        shortDates = Date(2000, 1, 1):Quarter(1):(Date(2000, 1, 1)+Quarter(n - 1))
        shortSeries = TimeArray(collect(shortDates), 100.0 .+ cumsum(randn(n)))
        model = SARIMA(shortSeries, 1, 1, 1; seasonality = 4, P = 1, D = 0, Q = 1)

        # A warning is expected here; an exception is not.
        @test (fit!(model; objectiveFunction = "elastic_net", alpha = 0.5); true)
        @test Sarimax.isFitted(model)
        @test all(isfinite, model.ϕ)
        @test isfinite(model.σ²)
    end
end
