# Composition of a LOSS with a coefficient PENALTY.
#
# Before this, `objectiveFunction` chose both at once: `"elastic_net"` meant "quadratic loss
# AND elastic net", and "quantile loss AND lasso" was unspellable. `penalty` is the second
# axis.
#
# The decisive test here is the OBJECTIVE RECONSTRUCTION under composition. Rebuilding
#
#     sum_t rho_tau(eps_t)  +  sum_j lambda_j [ alpha |psi_j| + (1-alpha)/2 psi_j^2 ]
#
# by hand and matching it against the reported objective proves the penalty landed on the
# QUANTILE loss and not on some quadratic one — a distinction no coefficient value would
# reveal on its own, which is why the squared-loss variant is asserted NOT to match.

@testset "composicao de perda e penalidade" begin
    Random.seed!(20260909)
    T = 200
    datas = collect(Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(T - 1)))
    e = randn(T)
    yv = zeros(T)
    for t = 3:T
        yv[t] = 0.5 * yv[t-1] - 0.3 * yv[t-2] + e[t] + 0.4 * e[t-1]
    end
    # CENTERED on purpose: with a level and `allowMean = false` the autoregressive
    # polynomial has to carry the level, phi runs to the stationarity boundary, and every
    # shrinkage reading below would be about that corner instead of about the penalty.
    serie = TimeArray(datas, yv)
    yScale = Statistics.std(yv)

    mk() = SARIMA(serie, 2, 0, 1; allowMean = false, silent = true)
    coefsOf(m) = vcat([m.ϕ...], [m.θ...])
    cobertura(m) = count(<=(1e-9), m.ϵ) / length(m.ϵ)
    rho(tau, x) = tau * max(x, 0.0) + (1 - tau) * max(-x, 0.0)

    @testset "elastic_net E mse+penalty: onde coincidem e onde NAO" begin
        par(ini) = begin
            a = mk()
            fit!(a; objectiveFunction = "elastic_net", alpha = 0.5, lambda = 60.0,
                 initialization = ini)
            b = mk()
            fit!(b; objectiveFunction = "mse", penalty = :elastic_net, alpha = 0.5,
                 lambda = 60.0, initialization = ini)
            (a, b)
        end

        # WHERE THEY COINCIDE: when the pre-sample block is not priced. The same expression
        # is emitted, so the two problems are one problem -- equality, not tolerance.
        for ini in (:zeroed, :free)
            a, b = par(ini)
            @test coefsOf(a) == coefsOf(b)
            @test get(a.metadata, "objectiveValue", NaN) ==
                  get(b.metadata, "objectiveValue", NaN)
        end

        # WHERE THEY DO NOT: under `:penalized` and `:innovations` the two fit terms differ,
        # and the difference is the GAUSSIAN DETERMINANT. `"mse"` prices the free pre-sample
        # block as a concentrated Gaussian likelihood, `S * prod(1-kappa^2)^(-j/T)`, while
        # the `"elastic_net"` objective's fit term is plain `sum(eps^2) + presampleSquares`
        # with no determinant. That asymmetry predates the composition axis -- it is how the
        # two branches have always been written -- and neither is changed here.
        #
        # It matters because `:innovations` is the DEFAULT: on the default path the two
        # spellings are NOT interchangeable, and the documentation says so.
        for ini in (:penalized, :innovations)
            a, b = par(ini)
            @test !isapprox(coefsOf(a), coefsOf(b); atol = 1e-8)
            @test !isapprox(
                get(a.metadata, "objectiveValue", NaN),
                get(b.metadata, "objectiveValue", NaN);
                rtol = 1e-8,
            )
        end

        # Whatever the mode, both declare the same penalty, so a result can be read without
        # knowing which spelling produced it.
        for ini in (:zeroed, :innovations)
            a, b = par(ini)
            @test get(a.metadata, "penalty", nothing) == "elastic_net"
            @test get(b.metadata, "penalty", nothing) == "elastic_net"
        end
    end

    @testset "penalty = :none nao muda nada" begin
        # The backward-compatibility guarantee: the new axis is inert at its default, on
        # every loss that admits it.
        for obj in ("mse", "mae", "huber", "quantile", "ml")
            a = mk()
            fit!(a; objectiveFunction = obj, initialization = :zeroed)
            b = mk()
            fit!(b; objectiveFunction = obj, penalty = :none, initialization = :zeroed)
            @test coefsOf(a) == coefsOf(b)
            @test get(b.metadata, "penalty", nothing) == "none"
        end
    end

    @testset "o objetivo reconstruido bate — e o da perda quadratica NAO" begin
        tau = 0.7
        casos = (
            (lambda = 30.0, alpha = 0.6, w = fill(30.0, 3)),
            (lambda = [40.0, 0.0, 10.0], alpha = 0.6, w = [40.0, 0.0, 10.0]),
            (lambda = (ar = 25.0, ma = 0.0), alpha = 1.0, w = [25.0, 25.0, 0.0]),
            (lambda = 15.0, alpha = 0.0, w = fill(15.0, 3)),
        )
        for caso in casos
            m = mk()
            fit!(m; objectiveFunction = "quantile", quantileLevel = tau,
                 penalty = :elastic_net, alpha = caso.alpha, lambda = caso.lambda,
                 initialization = :zeroed)
            psi = coefsOf(m)
            @test length(psi) == length(caso.w)          # premissa: indexacao alinhada
            # `objectiveValue` lives in the model's internal units; `model.ϵ` is mapped back,
            # and with d = D = 0 the factor is the series' own standard deviation.
            fitTerm = sum(rho(tau, x) for x in (m.ϵ ./ yScale))
            penTerm = sum(
                caso.w[j] * (caso.alpha * abs(psi[j]) + (1 - caso.alpha) / 2 * psi[j]^2)
                for j in eachindex(psi)
            )
            reportado = get(m.metadata, "objectiveValue", nothing)
            @test !isnothing(reportado)
            @test isapprox(fitTerm + penTerm, reportado; rtol = 1e-6)

            # THE HALF THAT MATTERS: had the penalty been attached to a squared loss — the
            # only loss the penalized objectives knew before — the total would be this, and
            # the coefficients alone would not say which of the two had been minimized.
            quadratico = sum((m.ϵ ./ yScale) .^ 2) + penTerm
            @test !isapprox(quadratico, reportado; rtol = 1e-3)
        end
    end

    @testset "a penalidade morde sob uma perda NAO quadratica" begin
        l1(m) = sum(abs.(coefsOf(m)))
        normas = Float64[]
        for lam in (0.0, 5.0, 25.0, 100.0)
            m = mk()
            fit!(m; objectiveFunction = "quantile", quantileLevel = 0.5,
                 penalty = :elastic_net, alpha = 1.0, lambda = lam,
                 initialization = :zeroed)
            push!(normas, l1(m))
        end
        # shrinkage is monotone in lambda ...
        @test issorted(normas; rev = true)
        # ... strictly, and at the top of the path everything is driven to zero, which is
        # the lasso behaviour and not merely "a smaller number".
        @test normas[1] > normas[end] + 1.0
        @test normas[end] < 1e-6
    end

    @testset "a orientacao do quantil sobrevive a penalidade" begin
        cobs = Float64[]
        locs = Float64[]
        for tau in (0.15, 0.5, 0.85)
            m = SARIMA(serie, 1, 0, 0; allowMean = true, silent = true)
            fit!(m; objectiveFunction = "quantile", quantileLevel = tau,
                 penalty = :elastic_net, alpha = 1.0, lambda = 10.0,
                 initialization = :zeroed)
            push!(cobs, cobertura(m))
            push!(locs, Statistics.mean(values(m.fitInSample)))
        end
        @test issorted(cobs)
        @test issorted(locs)
        for (tau, c) in zip((0.15, 0.5, 0.85), cobs)
            @test abs(c - tau) < 0.1
        end
    end

    @testset "pesos heterogeneos compoem com uma perda nao quadratica" begin
        # Same exact statement as in the unpenalized case: a zero weight is not a small
        # weight, it is the block being absent from the penalty.
        zerado = mk()
        fit!(zerado; objectiveFunction = "mae", penalty = :elastic_net, alpha = 1.0,
             lambda = (ar = [30.0, 30.0], ma = 0.0), penaltyTarget = :all,
             initialization = :zeroed)
        soAR = mk()
        fit!(soAR; objectiveFunction = "mae", penalty = :elastic_net, alpha = 1.0,
             lambda = 30.0, penaltyTarget = :dynamics, initialization = :zeroed)
        # `:dynamics` on this model reaches ar and ma alike, so the two differ by the ma
        # weight and must NOT coincide ...
        @test !isapprox(coefsOf(zerado), coefsOf(soAR); atol = 1e-6)
        # ... while weighting every block equally does reproduce the scalar.
        uniforme = mk()
        fit!(uniforme; objectiveFunction = "mae", penalty = :elastic_net, alpha = 1.0,
             lambda = (ar = [30.0, 30.0], ma = 30.0), penaltyTarget = :all,
             initialization = :zeroed)
        @test coefsOf(uniforme) == coefsOf(soAR)
    end

    @testset "a contagem esparsa segue a PENALIDADE, nao a perda" begin
        m = mk()
        fit!(m; objectiveFunction = "quantile", quantileLevel = 0.5,
             penalty = :elastic_net, alpha = 1.0, lambda = 100.0,
             initialization = :zeroed)
        nominal = m.p + m.q + m.P + m.Q + 1
        @test any(c -> abs(c) <= 1e-5, coefsOf(m))
        @test get_hyperparameters_number(m) < nominal

        # sem penalidade, a contagem nominal volta
        semPen = mk()
        fit!(semPen; objectiveFunction = "quantile", quantileLevel = 0.5,
             initialization = :zeroed)
        @test get_hyperparameters_number(semPen) == nominal
    end

    @testset "combinacoes recusadas" begin
        # DOUBLE SPECIFICATION: these objectives carry their own penalty.
        @test_throws ArgumentError fit!(
            mk(); objectiveFunction = "elastic_net", alpha = 0.5, penalty = :elastic_net,
        )
        @test_throws ArgumentError fit!(
            mk(); objectiveFunction = "ridge", penalty = :elastic_net, alpha = 0.5,
        )
        # SCALE: the fit term of these is not a sum on the scale `lambda` is calibrated for
        # (`ml_exact` is on a log scale, `stable` on a mean scale), and `bilevel` keeps the
        # moving-average coefficients outside the optimization altogether.
        for obj in ("ml_exact", "stable", "bilevel")
            @test_throws ArgumentError fit!(
                mk(); objectiveFunction = obj, penalty = :elastic_net, alpha = 0.5,
            )
        end
        # unknown penalty family
        @test_throws ArgumentError fit!(mk(); objectiveFunction = "mse", penalty = :nope)
        # alpha is required, exactly as under the `elastic_net` objective -- and with the
        # same exception type, so the same missing argument does not raise two different
        # things depending on which spelling selected the penalty.
        @test_throws ArgumentError fit!(
            mk(); objectiveFunction = "quantile", penalty = :elastic_net,
        )
        @test_throws ArgumentError fit!(mk(); objectiveFunction = "elastic_net")
        # and the weight validation is the same one, reached through the new axis
        @test_throws ArgumentError fit!(
            mk(); objectiveFunction = "mae", penalty = :elastic_net, alpha = 1.0,
            lambda = [1.0, -1.0, 1.0],
        )
        @test_throws ArgumentError fit!(
            mk(); objectiveFunction = "mae", penalty = :elastic_net, alpha = 1.0,
            lambda = (ar = 1.0,),
        )
    end

    @testset "a composicao sobrevive ao multistart" begin
        # `penalty` joins the argument bundles the re-fitting paths forward; without it a
        # multistart fit would drop the penalty and return an unpenalized model under the
        # caller's label.
        sem = mk()
        fit!(sem; objectiveFunction = "quantile", quantileLevel = 0.5,
             penalty = :elastic_net, alpha = 1.0, lambda = 100.0, initialization = :zeroed)
        com = mk()
        fit!(com; objectiveFunction = "quantile", quantileLevel = 0.5,
             penalty = :elastic_net, alpha = 1.0, lambda = 100.0, initialization = :zeroed,
             multistart = true)
        @test get(com.metadata, "penalty", nothing) == "elastic_net"
        @test sum(abs.(coefsOf(com))) < 1e-6      # a penalidade chegou ao ajuste
        @test sum(abs.(coefsOf(sem))) < 1e-6
    end

    @testset "auto encaminha a perda e a penalidade" begin
        m = auto(
            serie;
            seasonality = 1,
            objectiveFunction = "quantile",
            quantileLevel = 0.8,
            penalty = :elastic_net,
            alpha = 1.0,
            lambda = 20.0,
            maxp = 2,
            maxq = 1,
            maxP = 0,
            maxQ = 0,
        )
        @test get(m.metadata, "objectiveFunction", nothing) == "quantile"
        @test get(m.metadata, "penalty", nothing) == "elastic_net"
        @test get(m.metadata, "quantileLevel", nothing) == 0.8
        # ranking still goes through the criterion machinery, not through the penalized
        # objective value
        ll, n, _ = Sarimax.criterionLoglikeAndN(m)
        K = get_hyperparameters_number(m)
        @test aicc(m) ≈ 2 * K - 2 * ll + ((2 * K * K + 2 * K) / (n - K - 1))

        # `lambda`/`alpha` are admitted because a penalty is in play; without one they are
        # still refused, which is what keeps them from silently doing nothing.
        @test_throws ArgumentError auto(
            serie; seasonality = 1, objectiveFunction = "quantile", lambda = 20.0,
            maxp = 1, maxq = 0, maxP = 0, maxQ = 0,
        )
        @test_throws ArgumentError auto(
            serie; seasonality = 1, objectiveFunction = "quantile", alpha = 1.0,
            maxp = 1, maxq = 0, maxP = 0, maxQ = 0,
        )
        @test_throws ArgumentError auto(
            serie; seasonality = 1, objectiveFunction = "mse", lambda = 20.0, alpha = 1.0,
            maxp = 1, maxq = 0, maxP = 0, maxQ = 0,
        )
    end
end
