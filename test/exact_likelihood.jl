# Tests of the exact Gaussian likelihood (src/exact_likelihood.jl).
#
# The independent reference is the full MVN: for z ~ N(0, sigma^2 * Gamma) with
# Gamma_{ij} = gamma(|i-j|) and sigma^2 concentrated out,
#
#     l = -(n/2) (log(2 pi sigma2) + 1) - (1/2) logdet(Gamma),
#     sigma2 = z' Gamma^{-1} z / n.
#
# That is exactly the quantity Durbin-Levinson computes in O(n^2), obtained here by dense
# linear algebra, sharing no line of code with the implementation. (LinearAlgebra is
# qualified through Sarimax so that it need not be declared in the test target.)
referenceExactLoglike(z::Vector{Float64}, γ::Vector{Float64}) = begin
    LA = Sarimax.LinearAlgebra
    n = length(z)
    Γ = [γ[abs(i - j)+1] for i = 1:n, j = 1:n]
    σ² = LA.dot(z, Γ \ z) / n
    -(n / 2) * (log(2π * σ²) + 1) - LA.logdet(Γ) / 2
end

@testset "exact likelihood" begin
    rng = MersenneTwister(0x5A71)

    @testset "expandMultiplicativePolynomial" begin
        # (1 - 0.5B)(1 - 0.3B^12) = 1 - 0.5B - 0.3B^12 + 0.15B^13  (convencao AR)
        ar = Sarimax.expandMultiplicativePolynomial([0.5], [0.3], 12; negate = true)
        @test length(ar) == 13
        @test ar[1] ≈ 0.5
        @test ar[12] ≈ 0.3
        @test ar[13] ≈ -0.15
        @test all(iszero, ar[2:11])
        # (1 + 0.5B)(1 + 0.3B^12): lado MA, produto cruzado com sinal positivo
        ma = Sarimax.expandMultiplicativePolynomial([0.5], [0.3], 12; negate = false)
        @test ma[13] ≈ 0.15
        # casos degenerados
        @test isempty(Sarimax.expandMultiplicativePolynomial(Float64[], Float64[], 12))
        @test Sarimax.expandMultiplicativePolynomial([0.7], Float64[], 12) ≈ [0.7]
    end

    @testset "psi weights: sem colisao e base correta" begin
        # ruido branco: psi_k = 0 para k >= 1, psi_0 = 1
        @test Sarimax.psiWeights(Float64[], Float64[], 5) == zeros(5)
        @test Sarimax.psiWeightsFromZero(Float64[], Float64[], 5) == [1.0; zeros(5)]
        # AR(1): psi_k = phi^k
        @test Sarimax.psiWeights([0.5], Float64[], 4) ≈ [0.5, 0.25, 0.125, 0.0625]
        @test Sarimax.psiWeightsFromZero([0.5], Float64[], 4) ≈ [1.0, 0.5, 0.25, 0.125, 0.0625]
        # a variancia de previsao de um ARMA(0,0) ajustado e constante = sigma^2
        # (o bug historico de sobrescrita do psiWeights dava [1,2,2,2] * sigma^2)
        yWN = TimeArray(
            collect(Date(2000, 1, 1):Month(1):Date(2004, 12, 1)),
            randn(rng, 60),
        )
        mWN = SARIMA(yWN, 0, 0, 0; allowMean = false)
        fit!(mWN)
        fe = Sarimax.forecastErrors(mWN, 4)
        @test all(v -> isapprox(v, fe[1]; rtol = 1e-8), fe)
    end

    @testset "ruido branco: forma fechada" begin
        z = randn(rng, 40)
        n = length(z)
        σ² = sum(abs2, z) / n
        expected = -(n / 2) * (log(2π * σ²) + 1)
        @test Sarimax.exactGaussianLogLikelihood(z, Float64[], Float64[]) ≈ expected
    end

    @testset "AR(1): forma fechada" begin
        φ = 0.6
        z = randn(rng, 50)
        n = length(z)
        # v_1 = 1/(1-phi^2), v_t = 1 (t >= 2); e_1 = z_1, e_t = z_t - phi z_{t-1}
        σ² = ((1 - φ^2) * z[1]^2 + sum((z[t] - φ * z[t-1])^2 for t = 2:n)) / n
        expected = -(n / 2) * (log(2π * σ²) + 1) - log(1 / (1 - φ^2)) / 2
        @test Sarimax.exactGaussianLogLikelihood(z, [φ], Float64[]) ≈ expected
    end

    @testset "MA(1), ARMA(1,1) e sazonal: contra a MVN densa" begin
        for (ar, ma) in (
            (Float64[], [0.4]),                                  # MA(1)
            ([0.5], [-0.3]),                                     # ARMA(1,1)
            (
                Sarimax.expandMultiplicativePolynomial([0.4], [0.3], 4; negate = true),
                Sarimax.expandMultiplicativePolynomial([-0.2], [0.25], 4; negate = false),
            ),                                                   # SARMA(1,1)(1,1)_4 expandido
        )
            z = randn(rng, 30)
            γfull = Sarimax.theoreticalACF(ar, ma, length(z))
            @test !isnothing(γfull)
            @test Sarimax.exactGaussianLogLikelihood(z, ar, ma) ≈
                  referenceExactLoglike(z, γfull) rtol = 1e-10
        end
    end

    @testset "recusa principiada perto da fronteira" begin
        # AR(1) with a near-unit root: the psi tail does not decay within the truncation,
        # and the right answer is `nothing`, never a silently wrong number.
        @test isnothing(Sarimax.theoreticalACF([0.99999], Float64[], 50))
        @test isnothing(Sarimax.exactGaussianLogLikelihood(randn(rng, 50), [0.99999], Float64[]))
        # ponto explosivo
        @test isnothing(Sarimax.exactGaussianLogLikelihood(randn(rng, 50), [1.5], Float64[]))
        # serie vazia
        @test isnothing(Sarimax.exactGaussianLogLikelihood(Float64[], [0.5], Float64[]))
        # modelo nao ajustado
        yShort = TimeArray(collect(Date(2020, 1, 1):Month(1):Date(2021, 12, 1)), randn(rng, 24))
        @test isnothing(Sarimax.exactLoglike(SARIMA(yShort, 1, 0, 0)))
    end

    @testset "exactLoglike do modelo ajustado bate com a avaliacao direta" begin
        y = TimeArray(collect(Date(2000, 1, 1):Month(1):Date(2006, 12, 1)), randn(rng, 84))
        m = SARIMA(y, 1, 0, 1; allowMean = false)
        fit!(m)
        ll = Sarimax.exactLoglike(m)
        if !isnothing(ll)
            z = Float64.(values(m.y))
            @test ll ≈ Sarimax.exactGaussianLogLikelihood(z, [m.ϕ...], [m.θ...])
        end
    end

    @testset "n do AICc casa com a amostra da verossimilhanca" begin
        # The exact likelihood entering the criterion is evaluated over the T points of the
        # differenced series; the small-sample correction must use that same n, not the
        # `length(observedResiduals) = T - lb + 1` of the CSS conditioning.
        y = TimeArray(collect(Date(2000, 1, 1):Month(1):Date(2006, 12, 1)), randn(rng, 84))
        m = SARIMA(y, 2, 0, 0; allowMean = false)
        # `:zeroed` EXPLICITLY: this testset assumes the conditioning truncation EXISTS
        # (`nRes < T`). Under the `:innovations` default the sum starts at t = 1, `lb = 1`
        # and `nRes == T`, so the premise disappears and the test would stop measuring what
        # it was written to measure.
        fit!(m; initialization = :zeroed)  # residualLags = p = 2 => lb = 3
        if !isnothing(Sarimax.exactLoglike(m))   # criterio esta no caminho exato
            T = length(values(m.y))
            nRes = length(Sarimax.observedResiduals(m))
            @test nRes < T   # premissa do teste: a truncagem existe de fato
            K = Sarimax.get_hyperparameters_number(m)
            correctionOn(n) = (2K^2 + 2K) / (n - K - 1)
            @test aicc(m) ≈ aic(m) + correctionOn(T)
            @test aicc(m) ≉ aic(m) + correctionOn(nRes)   # o defeito antigo, nao regredir
            llAndN = Sarimax.criterionLoglikeAndN(m)
            @test llAndN[2] == T
            @test llAndN[3] === true
            @test m.metadata["criterionFallback"] === false
        end
    end

    @testset "telemetria de custo soma todos os ajustes do candidato" begin
        # `warmStartFromBox` ajusta o mesmo candidato ate 3 vezes (solve da caixa + dois
        # tiers restritos) e retorna cedo; `ensureAdmissible!` pode refitar por cima. Se a
        # telemetria sobrescrevesse, ela reportaria um subconjunto do custo — e no tier 3,
        # onde `merge!` copia o metadata do seed, justamente o solve MAIS BARATO dos tres.
        yW = TimeArray(
            collect(Date(2000, 1, 1):Month(1):Date(2007, 6, 1)),
            10 .+ 0.5 .* sin.(2π .* (1:90) ./ 12) .+ cumsum(randn(rng, 90) .* 0.2),
        )

        mPlain = SARIMA(yW, 1, 1, 1; seasonality = 12, P = 1, D = 0, Q = 1)
        fit!(mPlain)
        @test mPlain.metadata["fitCount"] == 1
        @test mPlain.metadata["buildTimeSecTotal"] ≈ mPlain.metadata["buildTimeSec"]
        @test mPlain.metadata["solveTimeSecTotal"] ≈ mPlain.metadata["solveTimeSec"]

        mWarm = SARIMA(yW, 1, 1, 1; seasonality = 12, P = 1, D = 0, Q = 1)
        fit!(mWarm; stationary = true, invertible = true, warmStartFromBox = true)
        # pelo menos o solve da caixa + uma tentativa restrita
        @test mWarm.metadata["fitCount"] >= 2
        @test mWarm.metadata["solveTimeSecTotal"] > mWarm.metadata["solveTimeSec"]
        @test mWarm.metadata["buildTimeSecTotal"] >= mWarm.metadata["buildTimeSec"]

        # refit sobre o mesmo objeto continua acumulando
        before = mPlain.metadata["fitCount"]
        fit!(mPlain)
        @test mPlain.metadata["fitCount"] == before + 1
    end

    @testset "contagem de hiperparametros sobrevive a nomes-string desligados" begin
        # `fit!` disables the string names of the JuMP variables (a build cost). With them
        # off, `variable_by_name` returns `nothing` WITHOUT error, so any consumer relying on
        # it counts wrongly and silently. `get_hyperparameters_number(::JuMP.Model)` reads
        # the object dictionary instead; this test pins that difference.
        jm = Sarimax.JuMP.Model()
        Sarimax.JuMP.set_string_names_on_creation(jm, false)
        Sarimax.JuMP.@variable(jm, c)
        Sarimax.JuMP.@variable(jm, trend)
        @test Sarimax.JuMP.variable_by_name(jm, "c") === nothing
        @test haskey(jm, :c)
        @test haskey(jm, :trend)
        @test !haskey(jm, :naoexiste)

        # e o caminho que de fato consome isso (elastic_net) continua ajustando
        yEN = TimeArray(
            collect(Date(2000, 1, 1):Month(1):Date(2007, 6, 1)),
            10 .+ 0.5 .* sin.(2π .* (1:90) ./ 12) .+ cumsum(randn(rng, 90) .* 0.2),
        )
        mEN = SARIMA(yEN, 1, 1, 1; seasonality = 12, P = 1, D = 0, Q = 1)
        fit!(mEN; objectiveFunction = "elastic_net", alpha = 0.5)
        @test Sarimax.isFitted(mEN)
    end

    @testset "criterionSampleSize: aritmetica == differentiate" begin
        # `criterionSampleSize` uses `n - d - D*s` instead of allocating the differenced
        # series. The equivalence must hold for every combination of differencing orders.
        yLong = TimeArray(collect(Date(2000, 1, 1):Month(1):Date(2010, 12, 1)), randn(rng, 132))
        for (d, D, s) in ((0, 0, 1), (1, 0, 1), (2, 0, 1), (0, 1, 12), (1, 1, 12), (2, 1, 12), (1, 2, 4))
            m = SARIMA(yLong, 1, d, 0; seasonality = s, D = D)
            @test Sarimax.criterionSampleSize(m) ==
                  length(values(differentiate(yLong, d, D, s)))
        end
    end

    @testset "selecao: recuo CSS nunca vence candidato com exata" begin
        # The SEARCH criterion (getInformationCriteriaFunction) penalizes candidates whose
        # criterion came from the fallback; the public accessors are unchanged.
        y = TimeArray(collect(Date(2000, 1, 1):Month(1):Date(2006, 12, 1)), randn(rng, 84))
        m = SARIMA(y, 1, 0, 0; allowMean = false)
        fit!(m)
        searchAicc = Sarimax.getInformationCriteriaFunction("aicc")
        publicValue = aicc(m)
        if !isnothing(Sarimax.exactLoglike(m))
            @test searchAicc(m) ≈ publicValue          # caminho exato: sem penalidade
        end
        # forca o recuo: coeficiente AR na fronteira mantido fixo
        mBoundary = SARIMA(y; arCoefficients = [0.99999], allowMean = false)
        fit!(mBoundary)
        @test isnothing(Sarimax.exactLoglike(mBoundary))
        publicBoundary = aicc(mBoundary)          # avalia o criterio e grava o metadata
        @test publicBoundary isa AbstractFloat
        @test mBoundary.metadata["criterionFallback"] === true
        @test searchAicc(mBoundary) ≈ publicBoundary + Sarimax.FALLBACK_CRITERION_PENALTY
        @test searchAicc(mBoundary) > searchAicc(m)
    end

    @testset "caminhos de busca antes mortos: grid e stepwiseNaive" begin
        yAuto = TimeArray(
            collect(Date(2000, 1, 1):Month(1):Date(2004, 12, 1)),
            0.7 .* sin.(2π .* (1:60) ./ 12) .+ randn(rng, 60) .* 0.5,
        )
        for method in ("grid", "stepwiseNaive")
            m = auto(
                yAuto;
                seasonality = 1,
                d = 0,
                D = 0,
                maxp = 1,
                maxq = 1,
                maxP = 0,
                maxQ = 0,
                searchMethod = method,
            )
            @test Sarimax.isFitted(m)
        end
    end
end
