@testset "Numerical conditioning (data scaling)" begin
    # The CSS objective is quadratic in the data, so an unscaled series in the 1e4-1e6
    # range yields an objective near 1e8-1e12 and Ipopt can spend a whole iteration in
    # its restoration phase — long enough that neither a time nor an iteration cap can
    # bound the fit (both are only checked *between* iterations). `fit!` therefore
    # solves in units of the differenced series' standard deviation and maps the
    # scale-dependent estimates back. These tests pin both halves of that contract.

    @testset "AR/MA coefficients are invariant to a change of units" begin
        # φ and θ carry no units, so rescaling the data must leave them untouched.
        # This is the guard that catches an incorrect un-scaling: if any factor were
        # applied to the wrong quantity, the two fits would disagree here.
        Random.seed!(2026)
        n = 120
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        base = 100.0 .+ cumsum(randn(n))

        for factor in (1e3, 1e-3)
            m1 = SARIMA(TimeArray(collect(dates), base), 1, 1, 1; allowMean = false)
            m2 = SARIMA(TimeArray(collect(dates), base .* factor), 1, 1, 1; allowMean = false)
            fit!(m1)
            fit!(m2)

            @test isapprox(m1.ϕ, m2.ϕ; atol = 1e-6)
            @test isapprox(m1.θ, m2.θ; atol = 1e-6)

            # Scale-dependent quantities carry the factor exactly.
            predict!(m1; stepsAhead = 6)
            predict!(m2; stepsAhead = 6)
            f1 = values(m1.forecast) .* factor
            f2 = values(m2.forecast)
            @test maximum(abs.((f1 .- f2) ./ max.(abs.(f2), 1e-9))) < 1e-8
            @test isapprox(m2.σ² / m1.σ², factor^2; rtol = 1e-6)
        end
    end

    @testset "large-magnitude series converges" begin
        # Regression: on M4 daily, a series with mean ~1e4 took 208s and returned
        # TIME_LIMIT before the data were scaled; scaled, the same fit converges in
        # about a second. A budget an order of magnitude above the scaled runtime
        # fails loudly if the conditioning ever regresses.
        Random.seed!(2027)
        n = 300
        dates = Date(2000, 1, 1):Day(1):(Date(2000, 1, 1)+Day(n - 1))
        level = 9_000.0 .+ 3_000.0 .* cumsum(randn(n)) ./ sqrt(n)
        model = SARIMA(TimeArray(collect(dates), level), 2, 1, 2; allowMean = false)

        elapsed = @elapsed fit!(model; maxTimeSeconds = 60.0)
        @test model.metadata["solverStatus"] in
              ("LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED")
        @test elapsed < 60.0
        @test all(isfinite, model.ϕ)
        @test all(isfinite, model.θ)
        @test isfinite(model.σ²) && model.σ² > 0
    end

    @testset "provided coefficients survive the internal rescaling" begin
        # Coefficients supplied by the caller are expressed in the user's units and are
        # held fixed, so they must be converted into the internal units before being
        # fixed and converted back afterwards. Regression: the scale-dependent ones
        # (c, trend, β) came back multiplied by the scale factor, silently returning a
        # different model than the one requested, while the dimensionless φ/θ/Φ/Θ
        # looked fine — which is what makes this failure easy to miss.
        Random.seed!(2029)
        n = 60
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        y = 800.0 .+ cumsum(randn(n) .* 4.0)
        exog = TimeArray(collect(dates), collect(1.0:n))

        model = SARIMA(TimeArray(collect(dates), y); exog = exog, arCoefficients = [0.5],
                       exogCoefficients = [0.5], seasonality = 1)
        fit!(model)

        @test model.exogCoefficients[1] ≈ 0.5 atol = 1e-8
        @test model.ϕ[1] ≈ 0.5 atol = 1e-8
    end

    @testset "fitted values and residuals come back in the original units" begin
        # The un-scaling has to happen before the fitted values are re-integrated
        # against the untouched observed series; if it did not, the in-sample fit
        # would sit at a completely different level from the data.
        Random.seed!(2028)
        n = 100
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        y = 50_000.0 .+ 500.0 .* cumsum(randn(n))
        model = SARIMA(TimeArray(collect(dates), y), 1, 1, 0; allowMean = false)
        fit!(model)

        fitted = values(model.fitInSample)
        @test all(isfinite, fitted)
        # in-sample fit must live on the same scale as the data, not on the internal one
        @test 0.5 < mean(fitted) / mean(y) < 2.0
        # residuals are differences on the original scale, so their spread must be
        # commensurate with the differenced data rather than with the scaled units
        @test std(model.ϵ) > 1.0
    end
end
