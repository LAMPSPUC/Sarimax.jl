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
        # The level travels ONLY with the objective that reads it. This helper is also used
        # to fit `mae`, and passing a quantile level there is refused — which is the point
        # of the guard: attaching a level to an objective that ignores it is how a sweep
        # ends up reporting one estimator under another's name.
        nivelKw = obj == "quantile" ? (; quantileLevel = nivel) : (;)
        Sarimax.fit!(
            m;
            objectiveFunction = obj,
            nivelKw...,
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
        # NaN is on the list on purpose: every comparison against it is false, so
        # `0 < NaN < 1` fails and the guard fires. A guard written as `!(x <= 0 || x >= 1)`
        # would let it through and the level would reach the objective as NaN.
        for ruim in (0.0, 1.0, -0.1, 1.5, NaN)
            m = SARIMA(ta, 1, 0, 0; seasonality = 1, silent = true)
            @test_throws AssertionError Sarimax.fit!(
                m;
                objectiveFunction = "quantile",
                quantileLevel = ruim,
                silent = true,
            )
        end
    end

    @testset "tau = 1/2 reproduces mae on EVERY observable" begin
        # The reduction is not a statement about the objective VALUE (which differs by the
        # factor 1/2) but about the ESTIMATOR. What a caller can observe must agree:
        # coefficients, fitted values, fitted residuals and forecasts.
        for (p, q) in ((1, 0), (0, 1), (1, 1))
            mq = ajusta(y, p, q, 1, 0, 0, "quantile"; nivel = 0.5, ini = :zeroed)
            mm = ajusta(y, p, q, 1, 0, 0, "mae"; ini = :zeroed)

            isnothing(mq.ϕ) || @test isapprox(mq.ϕ, mm.ϕ; atol = 1e-5)
            isnothing(mq.θ) || @test isapprox(mq.θ, mm.θ; atol = 1e-5)
            @test isapprox(values(mq.fitInSample), values(mm.fitInSample); atol = 1e-5)
            @test isapprox(mq.ϵ, mm.ϵ; atol = 1e-5)

            predict!(mq; stepsAhead = 8)
            predict!(mm; stepsAhead = 8)
            @test isapprox(values(mq.forecast), values(mm.forecast); atol = 1e-5)

            # ... and the objective value differs by exactly the documented factor, which
            # is why the equivalence had to be stated on the estimator and not on it.
            oq = get(mq.metadata, "objectiveValue", nothing)
            om = get(mm.metadata, "objectiveValue", nothing)
            @test !isnothing(oq) && !isnothing(om)
            @test isapprox(oq, om / 2; rtol = 1e-5)
        end
    end

    @testset "orientation: the fitted location is monotone in tau" begin
        # A deliberately SIMPLE specification on a deliberately ASYMMETRIC sample: with a
        # right-skewed innovation the three quantiles are far apart, and an AR(1) with a
        # mean leaves the fitted location readable rather than buried in the dynamics.
        Random.seed!(20260909)
        n = 220
        # exponential innovations: mean 1, median log(2), strongly right-skewed
        ruido = -log.(rand(n))
        z = 10.0 .+ ruido
        ta = TimeArray(
            collect(Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))),
            z,
        )
        ajustaNivel(nivel) = begin
            m = SARIMA(ta, 1, 0, 0; seasonality = 1, allowMean = true, silent = true)
            Sarimax.fit!(
                m;
                objectiveFunction = "quantile",
                quantileLevel = nivel,
                initialization = :zeroed,
                stationary = true,
                invertible = false,
                silent = true,
            )
            m
        end
        m10 = ajustaNivel(0.1)
        m50 = ajustaNivel(0.5)
        m90 = ajustaNivel(0.9)
        loc(m) = Statistics.mean(values(m.fitInSample))

        # THIS is the assertion that catches a reversed sign convention: with
        # eps = y - yhat, a larger tau prices under-prediction more and lifts the fit.
        @test loc(m10) <= loc(m50)
        @test loc(m50) <= loc(m90)
        # ... and strictly so, otherwise the three levels would be indistinguishable and
        # the ordering above would hold vacuously.
        @test loc(m90) - loc(m10) > 0.5

        # The fitted location also brackets the sample the way the quantiles do.
        @test loc(m10) < Statistics.median(z) < loc(m90)
    end

    @testset "the objective value IS the pinball loss of the fitted residuals" begin
        # Reconstructed by hand from the fitted innovations and compared with what the
        # solver reported. Under `:zeroed` there is no pre-sample block, so the objective is
        # the sample sum alone and `model.ϵ` covers exactly the summed range.
        #
        # `objectiveValue` lives in the model's INTERNAL units: the endogenous series is
        # divided by its standard deviation before the model is built, and `model.ϵ` is
        # mapped back. With d = D = 0 that factor is the standard deviation of the series
        # itself, so the comparison is exact rather than up to an unknown constant.
        rho(nivel, e) = nivel * max(e, 0.0) + (1 - nivel) * max(-e, 0.0)
        yScale = Statistics.std(y)
        for nivel in (0.2, 0.5, 0.75), (p, q) in ((1, 0), (1, 1))
            m = ajusta(y, p, q, 1, 0, 0, "quantile"; nivel = nivel, ini = :zeroed)
            reconstruido = sum(rho(nivel, e) for e in m.ϵ)
            reportado = get(m.metadata, "objectiveValue", nothing)
            @test !isnothing(reportado)
            @test isapprox(reconstruido, reportado * yScale; rtol = 1e-6)

            # The complement is what an inverted orientation would reconstruct to. Away
            # from the median the two differ, and that difference is the test.
            if nivel != 0.5
                invertido = sum(rho(1 - nivel, e) for e in m.ϵ)
                @test !isapprox(invertido, reportado * yScale; rtol = 1e-6)
            end
        end
    end

    @testset "the objective reaches models with exogenous regressors" begin
        Random.seed!(20260910)
        n = 180
        datas = collect(Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1)))
        X = hcat(randn(n), randn(n))
        b = [2.0, -1.0]
        eta = zeros(n)
        e = randn(n)
        for t = 2:n
            eta[t] = 0.5 * eta[t-1] + e[t]
        end
        yx = X * b .+ eta .+ 5.0
        ta = TimeArray(datas, yx)
        xa = TimeArray(datas, X, [:x1, :x2])

        # `allowMean = true` is REQUIRED for coverage to land on the level, and that is a
        # property of the estimator rather than of the test: placing the fit at the
        # tau-quantile means MOVING THE LOCATION, and without an intercept the model has no
        # free location parameter to move — `phi` and `beta` are pinned by the dynamics and
        # the regressors. Measured on this series, `allowMean = false` gives coverage
        # 0.16/0.35/0.54 at levels 0.25/0.5/0.75: still monotone in the level, still
        # correctly oriented, but shifted. The asymmetric loss reaches the exogenous path
        # either way; only the attainable location differs.
        cobs = Float64[]
        for nivel in (0.25, 0.5, 0.75)
            m = SARIMA(ta, xa, 1, 0, 0; seasonality = 1, allowMean = true, silent = true)
            Sarimax.fit!(
                m;
                objectiveFunction = "quantile",
                quantileLevel = nivel,
                initialization = :zeroed,
                silent = true,
            )
            @test Sarimax.isFitted(m)
            @test length(m.exogCoefficients) == 2
            @test all(isfinite, m.exogCoefficients)
            push!(cobs, count(<=(1e-9), m.ϵ) / length(m.ϵ))
        end
        @test issorted(cobs)
        for (nivel, c) in zip((0.25, 0.5, 0.75), cobs)
            @test abs(c - nivel) < 0.12
        end
    end

    @testset ":free keeps the sign convention with no pre-sample split" begin
        # Under `:free` the pre-sample block is open but NOT priced, so `mae` and
        # `quantile` no longer build the `eps_pre_plus`/`eps_pre_minus` pair: with neither
        # part in the objective it was an unbounded, unused block. Removing it is not
        # observable from outside — those variables entered no objective and constrained
        # nothing — so what this pins is that the mode still WORKS and still produces the
        # package's residual orientation.
        #
        # The orientation is the part worth guarding: the decomposition being edited here is
        # the same one whose inverted form once shipped as `eps = yhat - y`, corrupting
        # every forecast with q > 0 while leaving the objective value plausible.
        for (p, q, P, Q, s) in ((1, 1, 0, 0, 1), (0, 2, 0, 0, 1), (1, 1, 1, 1, 12))
            for obj in ("mae", "quantile")
                m = ajusta(y, p, q, s, P, Q, obj; nivel = 0.6, ini = :free)
                @test Sarimax.isFitted(m)
                @test all(isfinite, m.ϵ)
                # eps = y - yhat, positive on an under-prediction
                observado = values(m.y)[(end-length(m.ϵ)+1):end]
                ajustado = values(m.fitInSample)[(end-length(m.ϵ)+1):end]
                @test Statistics.cor(m.ϵ, observado .- ajustado) > 0.999
            end
        end
    end

    @testset "the level reaches every initialization mode" begin
        # The default (`:innovations`) plus the three other free/fixed conventions the
        # objective supports. The level must survive all of them: the pre-sample block joins
        # the SAME asymmetric loss, so opening it must not move the targeted quantile.
        for ini in (:innovations, :zeroed, :free, :penalized)
            for nivel in (0.3, 0.7)
                m = ajusta(y, 1, 0, 1, 0, 0, "quantile"; nivel = nivel, ini = ini)
                @test Sarimax.isFitted(m)
                @test abs(cobertura(m) - nivel) < 0.13
            end
        end
    end

    @testset "auto FITS with the pinball loss but RANKS with the criterion" begin
        # The two halves of the architecture, stated separately.
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
        # FIT: the selected candidate carries the objective and the level it was fitted
        # with, and its residuals show the asymmetry.
        @test get(m.metadata, "objectiveFunction", nothing) == "quantile"
        @test get(m.metadata, "quantileLevel", nothing) == 0.8
        @test cobertura(m) > 0.5

        # RANK: the criterion comes from the package's likelihood machinery, not from the
        # pinball sum. Recomputed here from its declared parts.
        ll, n, _ = Sarimax.criterionLoglikeAndN(m)
        K = get_hyperparameters_number(m)
        @test aicc(m) ≈ 2 * K - 2 * ll + ((2 * K * K + 2 * K) / (n - K - 1))
        @test aic(m) ≈ 2 * K - 2 * ll
        # The criterion machinery ran on this candidate (the key is written by
        # `criterionLoglikeAndN` itself).
        @test haskey(m.metadata, "criterionFallback")

        # The sharpest statement of the separation: `mae` and `quantile(0.5)` are the same
        # fit at two different objective VALUES, and the criterion must not notice the
        # difference. If ranking read `objectiveValue`, these two would differ by 2x.
        mq = ajusta(ys, 1, 0, 1, 0, 0, "quantile"; nivel = 0.5, ini = :zeroed)
        mm = ajusta(ys, 1, 0, 1, 0, 0, "mae"; ini = :zeroed)
        @test !isapprox(
            get(mq.metadata, "objectiveValue", NaN),
            get(mm.metadata, "objectiveValue", NaN);
            rtol = 1e-3,
        )
        @test aicc(mq) ≈ aicc(mm) atol = 1e-4
        @test bic(mq) ≈ bic(mm) atol = 1e-4
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
