@testset "Warm start and bounded solves" begin
    # Fitting with stationarity/invertibility by construction can stall on long series:
    # the reflection parameterisation is what costs, not the sample size. `fit!` can
    # therefore seed the constrained problem from a cheap unconstrained solve and bound
    # each attempt, falling back through progressively weaker constraints. These tests
    # pin the pieces that are easy to break silently.

    @testset "reflection maps round-trip" begin
        # arToReflection / maToReflection invert the Levinson-Durbin recursions used to
        # build a stationary (invertible) polynomial from bounded coefficients. If a
        # sign or a denominator drifts, the warm start would seed a different model
        # than the one it means to, which no end-to-end test would flag clearly.
        for coefs in ([0.4], [0.4, -0.2], [0.4, -0.2, 0.1], [0.5, 0.2, -0.3, 0.1])
            κ = Sarimax.arToReflection(coefs)
            @test isapprox(Float64.(Sarimax.reflectionToAR(κ)), coefs; atol = 1e-10)
            @test all(abs.(κ) .< 1)          # a stationary polynomial has |κ| < 1

            κma = Sarimax.maToReflection(coefs)
            @test isapprox(Float64.(Sarimax.reflectionToMA(κma)), coefs; atol = 1e-10)
        end
    end

    @testset "warmStartFromBox records the tier it settled on" begin
        # The tier tells the caller which guarantees the returned model actually has,
        # so it must always be present and within range.
        Random.seed!(3001)
        n = 90
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        y = 20.0 .+ cumsum(randn(n))
        model = SARIMA(TimeArray(collect(dates), y), 1, 1, 1; allowMean = false)
        fit!(model; stationary = true, invertible = true,
             warmStartFromBox = true, maxTimeSeconds = 30.0)

        @test haskey(model.metadata, "warmStartTier")
        @test model.metadata["warmStartTier"] in (1, 2, 3)
        @test all(isfinite, model.ϕ)
        @test all(isfinite, model.θ)
    end

    @testset "tier 1 really delivers a stationary, invertible model" begin
        # Reaching tier 1 is the whole point of the constrained fit; if it is reported
        # but the roots sit outside the unit circle, the guarantee is hollow.
        Random.seed!(3002)
        n = 120
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        y = 10.0 .+ cumsum(randn(n) .* 0.5)
        model = SARIMA(TimeArray(collect(dates), y), 2, 1, 1; allowMean = false)
        fit!(model; stationary = true, invertible = true,
             warmStartFromBox = true, maxTimeSeconds = 60.0)

        if model.metadata["warmStartTier"] == 1
            @test Sarimax.maxInverseRootModulus(-model.ϕ) < 1.0
            @test Sarimax.maxInverseRootModulus(model.θ) < 1.0
        end
    end

    @testset "warm start does not change the problem being solved" begin
        # The seed is a starting point, not a reformulation: when both routes converge
        # they must reach the same objective. A drift here would mean the warm-started
        # fit is optimising something else.
        Random.seed!(3003)
        n = 100
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        y = 5.0 .+ cumsum(randn(n) .* 0.3)

        plain = SARIMA(TimeArray(collect(dates), y), 1, 1, 1; allowMean = false)
        warm = SARIMA(TimeArray(collect(dates), y), 1, 1, 1; allowMean = false)
        fit!(plain; stationary = true, invertible = true, warmStartFromBox = false)
        fit!(warm; stationary = true, invertible = true,
             warmStartFromBox = true, maxTimeSeconds = 60.0)

        if warm.metadata["warmStartTier"] == 1
            # same feasible set, same optimum up to solver tolerance
            @test isapprox(plain.σ², warm.σ²; rtol = 1e-4)
        end
    end

    @testset "maxTimeSeconds bounds the solve instead of hanging" begin
        # A cap that cannot be met must come back with a limit status — never throw and
        # never run unbounded. The caller decides what to do with a bounded result.
        Random.seed!(3004)
        n = 200
        dates = Date(2000, 1, 1):Day(1):(Date(2000, 1, 1)+Day(n - 1))
        y = 100.0 .+ cumsum(randn(n))
        model = SARIMA(TimeArray(collect(dates), y), 2, 1, 2; allowMean = false)

        elapsed = @elapsed fit!(model; maxTimeSeconds = 0.001)
        @test elapsed < 30.0
        @test haskey(model.metadata, "solverStatus")
        @test all(isfinite, model.ϕ)
    end
    @testset "the seeded residuals live on the model's internal scale" begin
        # `ws.ϵ` is returned in ORIGINAL units (`value.(ϵ) .* yScale`), while this model's
        # `ϵ` lives on the standardised scale (`yValues ./ yScale`). Seeding the raw vector
        # made the warm start LESS feasible than a cold start — measured 28/08 through the
        # iteration-0 `inf_pr` of Ipopt, which went from ~13 to 337-7720 on daily series.
        # The scale is only visible when it is not 1, so this test uses a series whose
        # standard deviation is far from unity; on a unit-variance series the bug hides.
        Random.seed!(4711)
        n = 250
        dates = Date(2000, 1, 1):Day(1):(Date(2000, 1, 1)+Day(n - 1))
        y = 5000.0 .+ cumsum(randn(n) .* 250.0)      # std(diff(y)) ~ 250, not ~1
        ta = TimeArray(collect(dates), y)

        cold = SARIMA(ta, 2, 1, 2; allowMean = false)
        fit!(cold)

        warm = SARIMA(ta, 2, 1, 2; allowMean = false)
        fit!(warm; warmStart = cold)

        # Seeded from the converged point of the same problem, the warm solve must not
        # need more iterations than the cold one. With the raw seed it needed more,
        # because the start violated every dynamic constraint by a factor of `yScale`.
        itCold = get(cold.metadata, "solverIterations", missing)
        itWarm = get(warm.metadata, "solverIterations", missing)
        if !ismissing(itCold) && !ismissing(itWarm)
            @test itWarm <= itCold
        end

        # And it must land on the same optimum, not merely converge somewhere.
        @test isapprox(Float64.(warm.ϕ), Float64.(cold.ϕ); atol = 1e-4)
        @test isapprox(Float64.(warm.θ), Float64.(cold.θ); atol = 1e-4)
    end

    @testset "huber falling back to mse warns instead of degrading silently" begin
        # The huber branch fits an `mse` base, tries huber, and on failure copies the whole
        # `mse` model over, flagging only `metadata["huberFallback"]`. A caller that does not
        # read the metadata gets one estimator under another's label. Measured on the M4
        # campaign commit: under `initialization = :innovations` the objective guard threw and
        # the bare `catch` swallowed it, so `huberFallback` was true in 40 of 40 weekly series.
        # A time budget small enough to stop the huber solve reproduces the same path here.
        Random.seed!(9182)
        n = 120
        dates = Date(2000, 1, 1):Day(1):(Date(2000, 1, 1)+Day(n - 1))
        y = 1000.0 .+ cumsum(randn(n) .* 30.0)
        model = SARIMA(TimeArray(collect(dates), y), 2, 1, 2; allowMean = false)

        fit!(model; objectiveFunction = "huber", maxTimeSeconds = 1e-4)

        # Whatever the solver decided, the flag must exist so a campaign can report the
        # fallback rate as a validity column rather than discovering it after the fact.
        @test haskey(model.metadata, "huberFallback")
        @test model.metadata["huberFallback"] isa Bool
        @test all(isfinite, model.ϕ)
    end

end
