# Quantile regression as the asymmetric extension of `mae`.
#
# The check loss is rho_a(e) = a*max(e,0) + (1-a)*max(-e,0) over eps = y - yhat. Two things
# need pinning and they are of different kinds:
#
#   1. THE REDUCTION. At a = 1/2 the loss is symmetric and the estimator must be `mae`. That
#      is an algebraic claim and is checked on coefficients and on the objective value.
#
#   2. THE ORIENTATION. Which tail gets the weight `a` is NOT visible in the objective value,
#      because a sign inversion maps rho_a to rho_(1-a) and the reported number stays a
#      plausible one. This package has already shipped an inverted `mae` residual for exactly
#      that reason: |e| is even, so the objective hid it while every forecast with q > 0 was
#      corrupted. Under an asymmetric loss the same inversion silently fits the wrong
#      quantile. Only EMPIRICAL COVERAGE pins it, so that is what is tested.

@testset "Quantile objective" begin
    "Fraction of residuals at or below zero: the empirical coverage of the fit."
    cobertura(m) = count(<=(1e-9), m.ϵ) / length(m.ϵ)

    function ajusta(y, p, q, s, P, Q, obj; nivel = 0.5, ini = :innovations)
        ta = TimeArray(
            collect(Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(length(y) - 1))),
            copy(y),
        )
        m = SARIMA(ta, p, 0, q; seasonality = s, P = P, D = 0, Q = Q, silent = true)
        Sarimax.fit!(
            m;
            objectiveFunction = obj,
            quantileLevel = nivel,
            initialization = ini,
            seasonalForm = :multiplicative,
            stationary = true,
            invertible = false,
            silent = true,
        )
        m
    end

    Random.seed!(20260904)
    T = 160
    y = 50.0 .+ cumsum(randn(T) .* 0.4)

    @testset "reduction: a = 1/2 IS mae" begin
        # Same argmin, and the objective exactly half — rho_(1/2) = |e|/2.
        for (p, q) in ((1, 0), (0, 1), (1, 1))
            mq = ajusta(y, p, q, 1, 0, 0, "quantile"; nivel = 0.5)
            mm = ajusta(y, p, q, 1, 0, 0, "mae")
            isnothing(mq.ϕ) || @test isapprox(mq.ϕ, mm.ϕ; atol = 1e-5)
            isnothing(mq.θ) || @test isapprox(mq.θ, mm.θ; atol = 1e-5)
            oq = get(mq.metadata, "objectiveValue", nothing)
            om = get(mm.metadata, "objectiveValue", nothing)
            @test !isnothing(oq) && !isnothing(om)
            @test isapprox(oq, om / 2; rtol = 1e-5)
        end
    end

    @testset "orientation: coverage tracks the level" begin
        # eps = y - yhat, and eps > 0 is an under-prediction weighted by `a`. A high `a`
        # therefore pushes the fit UP and leaves a fraction `a` of residuals at or below
        # zero. If the residual sign were inverted, coverage would come out at 1 - a and
        # this is the only test that would notice.
        cobs = Float64[]
        for nivel in (0.1, 0.25, 0.5, 0.75, 0.9)
            m = ajusta(y, 1, 0, 1, 0, 0, "quantile"; nivel = nivel)
            push!(cobs, cobertura(m))
        end
        # coverage is monotone in the level ...
        @test issorted(cobs)
        # ... and lands near it, not near its complement
        for (nivel, c) in zip((0.1, 0.25, 0.5, 0.75, 0.9), cobs)
            @test abs(c - nivel) < 0.12
            # The complement comparison is the one that catches a sign inversion, but it
            # carries no information at the median, where the level IS its own complement.
            nivel == 0.5 || @test abs(c - nivel) < abs(c - (1 - nivel))
        end
    end

    @testset "the two tails are genuinely different fits" begin
        m10 = ajusta(y, 1, 0, 1, 0, 0, "quantile"; nivel = 0.1)
        m90 = ajusta(y, 1, 0, 1, 0, 0, "quantile"; nivel = 0.9)
        # the level-0.9 fit sits above the level-0.1 fit, pointwise on the fitted values
        v10 = values(m10.fitInSample)
        v90 = values(m90.fitInSample)
        @test mean(v90 .- v10) > 0
    end

    @testset "the level reaches the seasonal and mixed paths" begin
        ys = 50.0 .+ cumsum(randn(T) .* 0.3) .+ 3 .* sin.(2π .* (1:T) ./ 12)
        for (p, q, P, Q) in ((0, 1, 0, 1), (1, 1, 1, 1))
            for nivel in (0.25, 0.75)
                m = ajusta(ys, p, q, 12, P, Q, "quantile"; nivel = nivel)
                @test get(m.metadata, "solverStatus", "") in
                      ("LOCALLY_SOLVED", "ALMOST_LOCALLY_SOLVED")
                @test abs(cobertura(m) - nivel) < 0.15
            end
        end
    end

    @testset "the level is recorded, and only for this objective" begin
        m = ajusta(y, 1, 0, 1, 0, 0, "quantile"; nivel = 0.3)
        @test get(m.metadata, "quantileLevel", nothing) == 0.3
        mm = ajusta(y, 1, 0, 1, 0, 0, "mae")
        @test isnothing(get(mm.metadata, "quantileLevel", nothing))
    end

    @testset "the pre-sample block carries the SAME asymmetry" begin
        # Under `:innovations` the pre-sample residuals join the same loss. Weighting them
        # symmetrically would target one quantile on the sample and another on the block,
        # which would show up as coverage drifting away from the level as the block grows.
        # `:zeroed` has no free block, so it is the control.
        for nivel in (0.2, 0.8)
            mi = ajusta(y, 0, 2, 1, 0, 0, "quantile"; nivel = nivel, ini = :innovations)
            mz = ajusta(y, 0, 2, 1, 0, 0, "quantile"; nivel = nivel, ini = :zeroed)
            @test abs(cobertura(mi) - nivel) < 0.12
            @test abs(cobertura(mz) - nivel) < 0.12
        end
    end

    @testset "the level is rejected outside (0, 1)" begin
        ta = TimeArray(
            collect(Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(T - 1))),
            copy(y),
        )
        for ruim in (0.0, 1.0, -0.1, 1.5)
            m = SARIMA(ta, 1, 0, 0; seasonality = 1, silent = true)
            @test_throws AssertionError Sarimax.fit!(
                m;
                objectiveFunction = "quantile",
                quantileLevel = ruim,
                silent = true,
            )
        end
    end

    @testset "auto accepts the objective and threads the level" begin
        ys = 50.0 .+ cumsum(randn(T) .* 0.3) .+ 3 .* sin.(2π .* (1:T) ./ 12)
        ta = TimeArray(
            collect(Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(T - 1))),
            ys,
        )
        m = auto(
            ta;
            seasonality = 12,
            objectiveFunction = "quantile",
            quantileLevel = 0.8,
            maxp = 1,
            maxq = 1,
            maxP = 1,
            maxQ = 1,
        )
        @test get(m.metadata, "quantileLevel", nothing) == 0.8
        @test cobertura(m) > 0.5
    end
end
