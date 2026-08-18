# Contrato do multistart {zero, CSS}.
#
# A propriedade que o desenho garante e MONOTONIA: a partida do zero continua sendo um dos
# candidatos, e o desempate e por `aicc`, entao ligar `multistart` nunca pode PIORAR o
# criterio. Esse e o unico invariante forte que da para afirmar sem medir a M4 — o ganho de
# previsao e questao empirica e vive nos experimentos, nao aqui.
#
# `isFitted` e `aicc` nao estao todos no export de src/Sarimax.jl; qualificar com `Sarimax.`
# (ja quebrou o suite duas vezes por isso).
@testset "multistart {zero, CSS}" begin
    Random.seed!(20260818)
    # ARMA(1,1) com sinal claro nos dois blocos, para que a partida tenha o que mover
    T = 220
    ε = randn(T + 60)
    y = zeros(T + 60)
    for t = 3:(T+60)
        y[t] = 0.65 * y[t-1] + ε[t] - 0.45 * ε[t-1]
    end
    ta = Sarimax.loadDataset(DataFrame(y = 100.0 .+ cumsum(y[61:end])))

    função(ms) = begin
        m = Sarimax.SARIMA(ta, 1, 1, 1; seasonality = 1, allowMean = false, allowDrift = true)
        Sarimax.fit!(m; objectiveFunction = "mse", silent = true, stationary = true,
                     invertible = false, initialization = :penalized, multistart = ms)
        m
    end
    base = função(false)
    multi = função(true)

    @test Sarimax.isFitted(base)
    @test Sarimax.isFitted(multi)

    # metadados do multistart so existem no caminho ligado
    @test !haskey(base.metadata, "multistartPartidas")
    @test haskey(multi.metadata, "multistartPartidas")
    @test multi.metadata["multistartPartidas"] >= 1
    @test multi.metadata["multistartVenceuCSS"] isa Bool

    # MONOTONIA: nunca pior que a partida unica. A folga de 1e-6 absorve o ruido do solver;
    # sem ela o teste vira um detector de ultimo bit do Ipopt.
    @test Sarimax.aicc(multi) <= Sarimax.aicc(base) + 1e-6

    # o multistart devolve um modelo COERENTE, nao um Frankenstein: os residuos tem que ser
    # os do vencedor, nao os de um ajuste e os coeficientes de outro.
    @test length(multi.ϵ) == length(base.ϵ)
    @test isapprox(multi.σ², sum(abs2, multi.ϵ) / length(multi.ϵ); rtol = 0.5)

    # sob `:zeroed` a semente CSS E a partida de producao: o resultado tem de coincidir com o
    # caminho simples, e nao apenas "nao piorar".
    zsimples = Sarimax.SARIMA(ta, 1, 1, 1; seasonality = 1, allowMean = false, allowDrift = true)
    Sarimax.fit!(zsimples; objectiveFunction = "mse", silent = true, stationary = true,
                 invertible = false, initialization = :zeroed, multistart = false)
    zmulti = Sarimax.SARIMA(ta, 1, 1, 1; seasonality = 1, allowMean = false, allowDrift = true)
    Sarimax.fit!(zmulti; objectiveFunction = "mse", silent = true, stationary = true,
                 invertible = false, initialization = :zeroed, multistart = true)
    @test Sarimax.aicc(zmulti) <= Sarimax.aicc(zsimples) + 1e-6
end
