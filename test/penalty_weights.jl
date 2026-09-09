# Coefficient-specific regularization weights.
#
# The backbone of this file is an OBJECTIVE RECONSTRUCTION: after a penalized fit, the
# penalty is rebuilt by hand from the fitted coefficients and the weights the caller asked
# for, and the total is compared with the objective value the solver reported. That single
# identity pins, at once,
#
#   * WHICH coefficients the penalty reaches (the intercept and the drift are absent, and
#     their absence is observable rather than asserted);
#   * WHICH WEIGHT lands on each of them, i.e. the indexing;
#   * that a zero weight contributes NOTHING, rather than "not much";
#   * that `alpha` keeps mixing L1 against L2 and `lambda_j` is a strength, not a second
#     mixing parameter.
#
# Shrinkage magnitude is used only as a secondary, directional check: correlated AR lags
# make it an unreliable primary assertion, exactly as the qualification discussion noted.

@testset "pesos de penalizacao por coeficiente" begin
    rng = MersenneTwister(0xC0FFEE)
    datas(n) = collect(Date(2000, 1, 1):Month(1):Date(2000, 1, 1)+Month(n - 1))

    # --- fixtures -------------------------------------------------------------------------
    n = 160
    e = randn(rng, n)
    yv = zeros(n)
    for t = 3:n
        yv[t] = 0.55 * yv[t-1] - 0.25 * yv[t-2] + e[t] + 0.35 * e[t-1]
    end
    serie = TimeArray(datas(n), yv .+ 20.0)

    ns = 200
    es = randn(rng, ns)
    ysz = zeros(ns)
    for t = 14:ns
        ysz[t] = 0.5 * ysz[t-1] + 0.4 * ysz[t-12] + es[t] + 0.3 * es[t-1]
    end
    serieSaz = TimeArray(datas(ns), ysz .+ 30.0)

    # Orthogonal-by-construction regressors, so the two exogenous coefficients are
    # identified independently and a weight applied to one does not leak into the other.
    nx = 200
    X = zeros(nx, 2)
    X[:, 1] = [isodd(i) ? 1.0 : -1.0 for i = 1:nx]
    X[:, 2] = [(i % 4 in (0, 1)) ? 1.0 : -1.0 for i = 1:nx]
    @test abs(sum(X[:, 1] .* X[:, 2])) < 1e-12          # premissa: colunas ortogonais
    ex = randn(rng, nx)
    yx = X * [3.0, -2.0] .+ ex
    serieX = TimeArray(datas(nx), yx)
    regX = TimeArray(datas(nx), X, [:x1, :x2])

    coefsOf(m) = begin
        v = Float64[]
        for f in (:ϕ, :θ, :Φ, :Θ)
            w = getfield(m, f)
            isnothing(w) || append!(v, Float64.([w...]))
        end
        isnothing(m.exogCoefficients) || append!(v, Float64.([m.exogCoefficients...]))
        v
    end

    """Penalized coefficients in the canonical order, in the model's INTERNAL units.

    `ϕ`, `θ`, `Φ` and `Θ` are dimensionless and pass through; the exogenous coefficients are
    returned to the user multiplied by `yScale`, so the penalty saw them divided by it."""
    internalCoefs(m, yScale; penaltyTarget = :all) = begin
        v = Float64[]
        blocos = Dict(:ar => :ϕ, :ma => :θ, :sar => :Φ, :sma => :Θ)
        for b in Sarimax.penaltyBlocks(m, penaltyTarget)
            if b === :exog
                append!(v, Float64.([m.exogCoefficients...]) ./ yScale)
            else
                append!(v, Float64.([getfield(m, blocos[b])...]))
            end
        end
        v
    end

    """Objective value the elastic net should have reported, rebuilt by hand."""
    reconstruct(m, yScale, weights, alpha; penaltyTarget = :all) = begin
        psi = internalCoefs(m, yScale; penaltyTarget = penaltyTarget)
        @test length(psi) == length(weights)          # premissa: indexacao alinhada
        fitLoss = sum((m.ϵ ./ yScale) .^ 2)
        pen = sum(
            weights[j] * (alpha * abs(psi[j]) + (1 - alpha) / 2 * psi[j]^2) for
            j in eachindex(psi);
            init = 0.0,
        )
        fitLoss + pen
    end

    # =====================================================================================
    # TEST 1 / TEST 2 — o escalar historico e o caso uniforme da forma geral
    # =====================================================================================
    @testset "escalar, vetor igual e NamedTuple igual dao o MESMO ajuste" begin
        L = 45.0
        mk() = SARIMA(serie, 2, 0, 1; allowMean = false, silent = true)
        ajusta(lam) = begin
            m = mk()
            fit!(m; objectiveFunction = "elastic_net", alpha = 0.5, lambda = lam,
                 initialization = :zeroed)
            m
        end
        base = ajusta(L)
        @test Sarimax.penaltyCoefficientNames(base) == ["ar1", "ar2", "ma1"]

        for lam in (fill(L, 3), (ar = L, ma = L), (ar = [L, L], ma = [L]),
                    Dict(:ar => L, :ma => L))
            outro = ajusta(lam)
            # Not merely "within tolerance": the uniform branch emits the historical scalar
            # expression verbatim, so the two problems are the same problem.
            @test coefsOf(outro) == coefsOf(base)
            @test get(outro.metadata, "objectiveValue", NaN) ==
                  get(base.metadata, "objectiveValue", NaN)
        end
    end

    @testset "os apelidos gregos nomeiam os mesmos blocos" begin
        mk() = SARIMA(serie, 2, 0, 1; allowMean = false, silent = true)
        a = mk(); fit!(a; objectiveFunction = "elastic_net", alpha = 1.0,
                       lambda = (ar = 30.0, ma = 5.0), initialization = :zeroed)
        b = mk(); fit!(b; objectiveFunction = "elastic_net", alpha = 1.0,
                       lambda = (ϕ = 30.0, θ = 5.0), initialization = :zeroed)
        @test coefsOf(a) == coefsOf(b)
    end

    # =====================================================================================
    # TEST 3 / 5 / 6 / 8 — reconstrucao do objetivo: quem entra, com que peso, com que alpha
    # =====================================================================================
    @testset "o objetivo reconstruido bate com o reportado" begin
        yScale = Statistics.std(values(serie))
        casos = (
            (lambda = 40.0, alpha = 0.5, w = fill(40.0, 3)),
            (lambda = (ar = [50.0, 0.0], ma = 12.0), alpha = 1.0, w = [50.0, 0.0, 12.0]),
            (lambda = (ar = [0.0, 0.0], ma = 0.0), alpha = 0.5, w = [0.0, 0.0, 0.0]),
            (lambda = [5.0, 60.0, 25.0], alpha = 0.0, w = [5.0, 60.0, 25.0]),
            (lambda = [5.0, 60.0, 25.0], alpha = 0.25, w = [5.0, 60.0, 25.0]),
        )
        for caso in casos
            m = SARIMA(serie, 2, 0, 1; allowMean = false, silent = true)
            fit!(m; objectiveFunction = "elastic_net", alpha = caso.alpha,
                 lambda = caso.lambda, initialization = :zeroed)
            reportado = get(m.metadata, "objectiveValue", nothing)
            @test !isnothing(reportado)
            @test isapprox(reconstruct(m, yScale, caso.w, caso.alpha), reportado;
                           rtol = 1e-6)
        end
    end

    @testset "peso zero nao penaliza: o ajuste e o do bloco ausente do alvo" begin
        # An EXACT statement rather than an argument from magnitude: zeroing the weights of
        # the exogenous block must give the same fit as excluding that block through
        # `penaltyTarget`, with the dynamics weighted identically in both.
        mk() = SARIMA(serieX, regX, 1, 0, 0; seasonality = 1, allowMean = false,
                      silent = true)
        zerado = mk()
        fit!(zerado; objectiveFunction = "elastic_net", alpha = 1.0,
             lambda = (ar = 20.0, exog = [0.0, 0.0]), penaltyTarget = :all,
             initialization = :zeroed)
        excluido = mk()
        fit!(excluido; objectiveFunction = "elastic_net", alpha = 1.0,
             lambda = 20.0, penaltyTarget = :dynamics, initialization = :zeroed)
        @test isapprox(coefsOf(zerado), coefsOf(excluido); atol = 1e-6)

        # ... and the same in the other direction: zeroing the dynamics reproduces
        # `penaltyTarget = :exogenous`.
        zeradoDin = mk()
        fit!(zeradoDin; objectiveFunction = "elastic_net", alpha = 1.0,
             lambda = (ar = 0.0, exog = [15.0, 15.0]), penaltyTarget = :all,
             initialization = :zeroed)
        soExog = mk()
        fit!(soExog; objectiveFunction = "elastic_net", alpha = 1.0, lambda = 15.0,
             penaltyTarget = :exogenous, initialization = :zeroed)
        @test isapprox(coefsOf(zeradoDin), coefsOf(soExog); atol = 1e-6)
    end

    @testset "TEST 8: intercepto e drift ficam fora, e nao podem ser nomeados" begin
        comMedia = SARIMA(serie, 1, 0, 0; allowMean = true, silent = true)
        @test Sarimax.penaltyCoefficientNames(comMedia) == ["ar1"]
        comDrift = SARIMA(serie, 1, 1, 0; allowMean = false, allowDrift = true,
                          silent = true)
        @test Sarimax.penaltyCoefficientNames(comDrift) == ["ar1"]

        # A weight vector is sized by the penalized coefficients only, so a caller who
        # counted the intercept in gets an error rather than a silent misalignment.
        m = SARIMA(serie, 1, 0, 0; allowMean = true, silent = true)
        @test_throws ArgumentError fit!(
            m; objectiveFunction = "elastic_net", alpha = 0.5, lambda = [1.0, 1.0],
        )
        # ... and naming the intercept as a block is not a spelling the API knows.
        m2 = SARIMA(serie, 1, 0, 0; allowMean = true, silent = true)
        @test_throws ArgumentError fit!(
            m2; objectiveFunction = "elastic_net", alpha = 0.5,
            lambda = (ar = 1.0, c = 1.0),
        )

        # The intercept is estimated, not shrunk: hammering the dynamics leaves it alone.
        forte = SARIMA(serie, 1, 0, 0; allowMean = true, silent = true)
        fit!(forte; objectiveFunction = "elastic_net", alpha = 1.0, lambda = 5000.0,
             initialization = :zeroed)
        @test abs(forte.ϕ[1]) < 1e-4          # a dinamica colapsa
        @test abs(forte.c) > 1.0              # o nivel nao
    end

    # =====================================================================================
    # TEST 4 / 5 / 6 — ridge, lasso e elastic net heterogeneos mordem na direcao certa
    # =====================================================================================
    @testset "TEST 4: ridge heterogeneo em regressores ortogonais" begin
        # `alpha = 0` is the ridge-type penalty of this objective. With orthogonal columns
        # the two exogenous coefficients are identified independently, so the comparison
        # below isolates the weight instead of measuring collinearity.
        m = SARIMA(serieX, regX, 1, 0, 0; seasonality = 1, allowMean = false, silent = true)
        fit!(m; objectiveFunction = "elastic_net", alpha = 0.0,
             lambda = (ar = 0.0, exog = [400.0, 1.0]), penaltyTarget = :all,
             initialization = :zeroed)
        b = Float64.([m.exogCoefficients...])
        # true beta = [3.0, -2.0]; x1 is penalized 400x harder, so it is pulled towards 0
        # in RELATIVE terms while x2 stays near its unpenalized value.
        @test abs(b[1]) / 3.0 < abs(b[2]) / 2.0
        @test abs(b[2]) > 1.5

        # ... and swapping the weights swaps which one is pulled in.
        m2 = SARIMA(serieX, regX, 1, 0, 0; seasonality = 1, allowMean = false, silent = true)
        fit!(m2; objectiveFunction = "elastic_net", alpha = 0.0,
             lambda = (ar = 0.0, exog = [1.0, 400.0]), penaltyTarget = :all,
             initialization = :zeroed)
        b2 = Float64.([m2.exogCoefficients...])
        @test abs(b2[2]) / 2.0 < abs(b2[1]) / 3.0
        @test abs(b2[1]) > 2.5
    end

    @testset "TEST 5: lasso heterogeneo zera o coeficiente pesado, nao o leve" begin
        m = SARIMA(serieX, regX, 1, 0, 0; seasonality = 1, allowMean = false, silent = true)
        fit!(m; objectiveFunction = "elastic_net", alpha = 1.0,
             lambda = (ar = 0.0, exog = [5000.0, 0.0]), penaltyTarget = :all,
             initialization = :zeroed)
        b = Float64.([m.exogCoefficients...])
        @test abs(b[1]) < 1e-4        # x1: peso esmagador, colapsa
        @test abs(b[2]) > 1.5         # x2: peso zero, sobrevive
    end

    @testset "TEST 6: elastic net usa o peso nas DUAS componentes" begin
        # If a weight reached only the L1 term (or only the L2 term), the reconstruction
        # would fail for some alpha and pass for others. Sweeping alpha is what separates
        # the two components.
        yScale = Statistics.std(values(serie))
        w = [80.0, 0.0, 20.0]
        for a in (0.0, 0.3, 0.7, 1.0)
            m = SARIMA(serie, 2, 0, 1; allowMean = false, silent = true)
            fit!(m; objectiveFunction = "elastic_net", alpha = a, lambda = w,
                 initialization = :zeroed)
            @test isapprox(
                reconstruct(m, yScale, w, a),
                get(m.metadata, "objectiveValue", NaN);
                rtol = 1e-6,
            )
        end
    end

    # =====================================================================================
    # TEST 9 — familias de modelo: AR/MA, bloco sazonal, bloco exogeno
    # =====================================================================================
    @testset "TEST 9: os cinco blocos sao alcancaveis e ordenados" begin
        m = SARIMA(serieSaz, 1, 0, 1; seasonality = 12, P = 1, D = 0, Q = 1,
                   allowMean = false, silent = true)
        @test Sarimax.penaltyCoefficientNames(m) == ["ar1", "ma1", "sar1", "sma1"]
        fit!(m; objectiveFunction = "elastic_net", alpha = 1.0,
             lambda = (ar = 0.0, ma = 0.0, sar = 4000.0, sma = 0.0),
             initialization = :zeroed)
        # only the seasonal AR block was hammered
        @test abs(m.Φ[1]) < 1e-3
        @test abs(m.ϕ[1]) > 1e-2

        mx = SARIMA(serieX, regX, 1, 0, 1; seasonality = 1, allowMean = false, silent = true)
        @test Sarimax.penaltyCoefficientNames(mx) ==
              ["ar1", "ma1", "exog:x1", "exog:x2"]
        @test Sarimax.penaltyCoefficientNames(mx; penaltyTarget = :exogenous) ==
              ["exog:x1", "exog:x2"]
        @test Sarimax.penaltyCoefficientNames(mx; penaltyTarget = :dynamics) ==
              ["ar1", "ma1"]
    end

    @testset "um lambda estruturado convive com penaltyTarget" begin
        # Under `:exogenous` only the exogenous block is penalized, so it is the only one a
        # structured lambda may (and must) name.
        m = SARIMA(serieX, regX, 1, 0, 0; seasonality = 1, allowMean = false, silent = true)
        fit!(m; objectiveFunction = "elastic_net", alpha = 1.0, penaltyTarget = :exogenous,
             lambda = (exog = [5000.0, 0.0],), initialization = :zeroed)
        b = Float64.([m.exogCoefficients...])
        @test abs(b[1]) < 1e-4
        @test abs(b[2]) > 1.5

        # naming a block the target excluded is a spec error, not a silent no-op
        m2 = SARIMA(serieX, regX, 1, 0, 0; seasonality = 1, allowMean = false, silent = true)
        @test_throws ArgumentError fit!(
            m2; objectiveFunction = "elastic_net", alpha = 1.0, penaltyTarget = :exogenous,
            lambda = (ar = 1.0, exog = [1.0, 1.0]),
        )
    end

    # =====================================================================================
    # TEST 7 — pesos invalidos
    # =====================================================================================
    @testset "TEST 7: pesos invalidos sao recusados com mensagem util" begin
        mk() = SARIMA(serie, 2, 0, 1; allowMean = false, silent = true)
        ruins = (
            [1.0, -1.0, 1.0],                      # negativo
            [1.0, 1.0],                            # dimensao errada (curta)
            [1.0, 1.0, 1.0, 1.0],                  # dimensao errada (longa)
            (ar = 1.0, nope = 1.0),                # chave desconhecida
            (ar = 1.0,),                           # bloco penalizado nao nomeado
            (ar = 1.0, ma = 1.0, sar = 1.0),       # bloco inexistente no modelo
            (ar = [1.0], ma = 1.0),                # comprimento de bloco inconsistente
            [1.0, NaN, 1.0],                       # NaN
            [1.0, Inf, 1.0],                       # Inf
            (ar = [1.0, -0.5], ma = 1.0),          # negativo dentro de um bloco
        )
        for ruim in ruins
            m = mk()
            @test_throws ArgumentError fit!(
                m; objectiveFunction = "elastic_net", alpha = 0.5, lambda = ruim,
            )
        end

        # O construtor recusa valores invalidos onde ja pode ve-los, mantendo o tipo de
        # excecao que sempre lancou.
        @test_throws Sarimax.InvalidParametersCombination SARIMA(
            serie; arCoefficients = [0.5], lambda = -1.0,
        )
        @test_throws Sarimax.InvalidParametersCombination SARIMA(
            serie; arCoefficients = [0.5], lambda = [1.0, -1.0],
        )
        @test_throws Sarimax.InvalidParametersCombination SARIMA(
            serie; arCoefficients = [0.5], lambda = (ar = NaN,),
        )
    end

    # =====================================================================================
    # B5 — os pesos que um lasso adaptativo precisa
    # =====================================================================================
    @testset "pesos no estilo lasso adaptativo sao aceitos como estao" begin
        # The package does NOT run the two-stage procedure; it accepts the weights the
        # caller computed. This is the whole point of the generalization, so it is exercised
        # end to end here.
        #
        # The specification is a PURE REGRESSION (p = q = 0) on orthogonal columns, so the
        # soft-thresholding prediction below is exact. With an autoregressive term the ARX
        # form makes the effective regression non-orthogonal — the AR coefficient absorbs
        # part of whatever the penalty takes out of beta — and the shrinkage ordering stops
        # being a property of the weights alone. Measured on this same data with an AR(1)
        # added, the ordering reverses, which is a fact about the design matrix and not
        # about the penalty.
        primeiro = SARIMA(serieX, regX, 0, 0, 0; seasonality = 1, allowMean = false,
                          silent = true)
        fit!(primeiro; objectiveFunction = "mse", initialization = :zeroed)

        nomes = Sarimax.penaltyCoefficientNames(primeiro; penaltyTarget = :exogenous)
        @test nomes == ["exog:x1", "exog:x2"]
        btil = Float64.([primeiro.exogCoefficients...])
        # w_j = 1 / |b_j|^gamma, gamma = 1
        w = 1.0 ./ abs.(btil)
        @test all(isfinite, w)
        @test abs(btil[1]) > abs(btil[2])                      # premissa do desenho
        @test w[1] < w[2]                                      # ... logo o peso se inverte

        adaptativo = SARIMA(serieX, regX, 0, 0, 0; seasonality = 1, allowMean = false,
                            silent = true)
        fit!(adaptativo; objectiveFunction = "elastic_net", alpha = 1.0,
             penaltyTarget = :exogenous, lambda = 50.0 .* w, initialization = :zeroed)
        @test Sarimax.isFitted(adaptativo)
        badap = Float64.([adaptativo.exogCoefficients...])

        # Orthogonal lasso is soft-thresholding: the absolute shrinkage of each coefficient
        # is proportional to ITS OWN weight. The coefficient that was already large gets the
        # small weight and is moved less — the adaptive-Lasso behaviour the weights encode.
        @test (abs(btil[1]) - abs(badap[1])) < (abs(btil[2]) - abs(badap[2]))
        @test (abs(btil[1]) - abs(badap[1])) > 0               # ambos encolhem
        # ... and the same in relative terms.
        @test (abs(btil[1]) - abs(badap[1])) / abs(btil[1]) <
              (abs(btil[2]) - abs(badap[2])) / abs(btil[2])

        # A uniform lambda of the same average strength is a DIFFERENT fit: the weights are
        # not decoration.
        uniforme = SARIMA(serieX, regX, 0, 0, 0; seasonality = 1, allowMean = false,
                          silent = true)
        fit!(uniforme; objectiveFunction = "elastic_net", alpha = 1.0,
             penaltyTarget = :exogenous, lambda = 50.0 * Statistics.mean(w),
             initialization = :zeroed)
        @test !isapprox(
            Float64.([uniforme.exogCoefficients...]), badap; atol = 1e-6,
        )
    end

    # =====================================================================================
    # multistart / warm start nao podem perder a especificacao da penalidade
    # =====================================================================================
    @testset "a especificacao da penalidade sobrevive ao multistart" begin
        # `penaltyTarget` used to be dropped from the argument bundle the multistart path
        # forwards, so a fit asked to shrink only the regressors shrank the dynamics too and
        # reported the result under the caller's label.
        mk() = SARIMA(serieX, regX, 1, 0, 0; seasonality = 1, allowMean = false,
                      silent = true)
        semMulti = mk()
        fit!(semMulti; objectiveFunction = "elastic_net", alpha = 1.0,
             penaltyTarget = :exogenous, lambda = 5000.0, initialization = :zeroed)
        comMulti = mk()
        fit!(comMulti; objectiveFunction = "elastic_net", alpha = 1.0,
             penaltyTarget = :exogenous, lambda = 5000.0, initialization = :zeroed,
             multistart = true)
        # Under `:exogenous` the AR coefficient is never penalized: with the target dropped
        # it would have been, and it would have collapsed.
        @test abs(comMulti.ϕ[1]) > 1e-3
        @test abs(semMulti.ϕ[1]) > 1e-3
        @test all(abs.(Float64.([comMulti.exogCoefficients...])) .< 1e-3)
    end

    # =====================================================================================
    # ridge: contrato inalterado
    # =====================================================================================
    @testset "ridge continua recusando lambda, escalar ou heterogeneo" begin
        mk() = SARIMA(serie, 2, 0, 1; allowMean = false, silent = true)
        @test_throws ArgumentError fit!(mk(); objectiveFunction = "ridge", lambda = 1.0)
        @test_throws ArgumentError fit!(
            mk(); objectiveFunction = "ridge", lambda = [1.0, 1.0, 1.0],
        )
        # e o ajuste sem lambda corre normalmente
        r = mk()
        fit!(r; objectiveFunction = "ridge", initialization = :zeroed)
        @test Sarimax.isFitted(r)
    end
end
