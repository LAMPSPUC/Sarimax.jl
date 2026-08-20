# A penalidade do elastic_net multiplica o peso adaptativo pela posicao do parametro no
# bloco. Para phi/theta/Phi/Theta a posicao E a ordem da defasagem e o escalonamento e
# deliberado. Para :beta a posicao e so a ordem em que o usuario passou as colunas, que
# nao carrega informacao -- e a consequencia observavel e que PERMUTAR AS COLUNAS DE X
# MUDA O MODELO. E essa a propriedade que o teste afirma, e nao a formula interna.
@testset "penalidade exogena e invariante a permutacao das colunas" begin
    Random.seed!(20250820)
    T, S, K = 120, 12, 6
    β = [2.0, 0.0, -1.5, 0.0, 0.0, 0.8]
    X = randn(T, K)
    ϵ = randn(T)
    η = zeros(T)
    for t = 2:T
        η[t] = 0.5 * η[t-1] + ϵ[t]
    end
    y = X * β .+ η
    datas = collect(Date(2000, 1, 1):Month(1):Date(2000, 1, 1)+Month(T-1))

    ajusta(cols) = begin
        m = SARIMA(TimeArray(datas, y), TimeArray(datas, X[:, cols]), 1, 0, 0;
                   P = 0, D = 0, Q = 0, seasonality = S,
                   allowMean = false, allowDrift = false)
        fit!(m; objectiveFunction = "elastic_net", alpha = 1.0,
             initialization = :free, stationary = true, invertible = false)
        Float64.([m.exogCoefficients...])
    end

    direta = collect(1:K)
    trocada = reverse(direta)
    b1 = ajusta(direta)
    b2 = ajusta(trocada)[sortperm(trocada)]   # desfaz a permutacao para comparar
    @test b1 ≈ b2 atol = 1e-4
end
