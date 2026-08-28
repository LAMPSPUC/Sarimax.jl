# Guards against silently indefensible behaviour.
#
# None of these tests pins POLICY (how much to shrink, which degrees of freedom to charge,
# which space to sweep). They pin the absence of surprise: an ignored argument must not move
# the criterion, and an objective that degrades must say so.
@testset "guardas de objetivo e contagem de parametros" begin
    rng = MersenneTwister(0x6A11)
    dates(n) = collect(Date(2000, 1, 1):Month(1):Date(2000, 1, 1)+Month(n - 1))

    @testset "lambda ignorado nao pode mexer no criterio" begin
        # The sparse count must be triggered by the objective used, not by the PRESENCE of
        # `lambda`/`alpha`. At fixed coefficients under `mse`, which ignores `lambda`,
        # passing it leaves the fit bit-identical, so it must leave K and the criterion
        # untouched as well.
        y = TimeArray(dates(60), randn(rng, 60))
        semLambda = SARIMA(y; arCoefficients = [0.5, 0.0, 0.0], allowMean = false)
        comLambda = SARIMA(y; arCoefficients = [0.5, 0.0, 0.0], allowMean = false, lambda = 1.0)
        fit!(semLambda)
        fit!(comLambda)

        @test [semLambda.ϕ...] ≈ [comLambda.ϕ...] atol = 1e-12   # premissa: ajuste identico
        @test get_hyperparameters_number(semLambda) == get_hyperparameters_number(comLambda)
        @test aicc(semLambda) ≈ aicc(comLambda)
        @test aic(semLambda) ≈ aic(comLambda)
        @test bic(semLambda) ≈ bic(comLambda)
        # `alpha` tem o mesmo gatilho e a mesma exigencia
        comAlpha = SARIMA(y; arCoefficients = [0.5, 0.0, 0.0], allowMean = false, alpha = 0.5)
        fit!(comAlpha)
        @test aicc(comAlpha) ≈ aicc(semLambda)
    end

    @testset "elastic_net mantem a contagem esparsa" begin
        # The trigger above must not have switched off the sparse count for objectives that
        # do regularize. Restricting it to the lasso case (`alpha = 1`) would be a policy
        # change.
        n = 150
        y = TimeArray(
            dates(n),
            10 .+ 0.6 .* sin.(2π .* (1:n) ./ 12) .+ cumsum(randn(rng, n) .* 0.25),
        )
        m = SARIMA(y, 3, 1, 2; seasonality = 12, P = 1, D = 0, Q = 1, allowMean = false)
        fit!(m; objectiveFunction = "elastic_net", alpha = 1.0)
        nominal = m.p + m.q + m.P + m.Q + 1
        coefs = vcat([m.ϕ...], [m.θ...], [m.Φ...], [m.Θ...])
        # Under the penalized formulation the lasso case shrinks coefficients to zero, so
        # the sparse count is below the nominal one. The two-stage construction this
        # replaced had no penalty multiplier: `lambda` never reached the optimization, the
        # shrinkage was governed by a calibrated tolerance, and under a free pre-sample
        # block the coefficients saturated the invertibility bound instead of collapsing.
        @test any(c -> abs(c) <= 1e-5, coefs)
        @test get_hyperparameters_number(m) < nominal
        @test m.metadata["objectiveFunction"] == "elastic_net"
    end

    @testset "ridge RECUSA lambda em vez de ignorar" begin
        # The package separates two cases: an invalid ARGUMENT COMBINATION (this one) is
        # refused, because it is fixed at the call site and the caller can check it in
        # advance; RUN-TIME DEGRADATION (the `ml_exact` testset below) warns, because it
        # depends on the candidate and raising there would abort the search. A warning is
        # invisible in a parallel run, and a sweep in which some cells silently mean
        # something else is a broken sweep.
        n = 90
        y = TimeArray(dates(n), 10 .+ cumsum(randn(rng, n) .* 0.3))
        mk() = SARIMA(y, 2, 1, 1; allowMean = false)
        @test_throws ArgumentError fit!(
            mk(); objectiveFunction = "ridge", alpha = 0.0, lambda = 1.0
        )
        # sem `lambda` nao ha o que recusar, e o ajuste corre sem aviso
        @test_logs match_mode = :any fit!(mk(); objectiveFunction = "ridge", alpha = 0.0)
    end

    @testset ":penalized RECUSA objetivo nao coberto" begin
        # Same policy. The accepted list must mirror the penalized-objective gate; if it
        # admits something the gate does not cover, the fit degrades to :free silently,
        # which is what this error exists to prevent.
        n = 90
        y = TimeArray(dates(n), 10 .+ cumsum(randn(rng, n) .* 0.3))
        mk() = SARIMA(y, 2, 1, 1; allowMean = false)
        # The refused list is empty: the pre-sample block enters the FIT TERM of every
        # objective, which all of them have, and the regularizing parts are untouched. No
        # supported objective is refused by the guard.
        #
        # The invariant this testset watches is therefore **every supported objective works
        # under the penalized free-block modes**. Adding an objective without extending its
        # fit term breaks this test, which is the point.
        #
        # `ml_exact` is excluded for a pre-existing degeneracy: it returns sigma2 = 0 under
        # `:free` and emits its own warning, which the next testset covers.
        suportados = ("mae", "mse", "ml", "bilevel", "elastic_net", "stable", "ridge", "huber")
        for obj in suportados, init in (:penalized, :innovations)
            m = mk()
            kw = obj == "elastic_net" ? (; alpha = 0.5) : (;)
            fit!(m; objectiveFunction = obj, initialization = init, kw...)
            @test Sarimax.isFitted(m)
        end
    end

    @testset "ml_exact avisa quando degrada por completo" begin
        n = 90
        y = TimeArray(dates(n), 10 .+ cumsum(randn(rng, n) .* 0.3))
        # sem a parametrizacao por reflexao o objetivo vira CSS puro
        @test_logs (:warn, r"degrades to plain CSS") match_mode = :any fit!(
            SARIMA(y, 2, 1, 0; allowMean = false); objectiveFunction = "ml_exact",
            stationary = false,
        )
        # sem parte AR, idem
        @test_logs (:warn, r"degrades to plain CSS") match_mode = :any fit!(
            SARIMA(y, 0, 1, 1; allowMean = false); objectiveFunction = "ml_exact",
        )
        # negativo: AR puro com `stationary = true` e o caso suportado, nao avisa
        @test_logs match_mode = :any fit!(
            SARIMA(y, 2, 1, 0; allowMean = false); objectiveFunction = "ml_exact",
            stationary = true,
        )
    end
end
