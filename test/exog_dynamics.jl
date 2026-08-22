# Semantica do bloco exogeno: :armax vs :regression_errors.
#
# As duas equacoes (Hyndman, "The ARIMAX model muddle", 2010):
#   :armax              phi(B) Phi(B^s) y_t = X_t'b + theta(B) Theta(B^s) e_t
#   :regression_errors  y_t = X_t'b + eta_t ,  phi(B) Phi(B^s) eta_t = theta(B) Theta(B^s) e_t
#
# Os testes afirmam PROPRIEDADES DE SAIDA, nao internos: (i) o default nao mudou nada,
# (ii) a identidade algebrica vale exatamente onde a algebra diz que vale e so ali, e
# (iii) sob a forma de erros ARIMA o coeficiente recupera o efeito marginal do DGP.

@testset "exogDynamics" begin
    mkTA(v) = TimeArray(
        collect(Date(2000, 1, 1):Month(1):Date(2000, 1, 1)+Month(length(v)-1)), v)
    mkTX(X) = TimeArray(
        collect(Date(2000, 1, 1):Month(1):Date(2000, 1, 1)+Month(size(X, 1)-1)), X)

    # `y` tem T pontos; `X` tem T + HMAX, porque o `predict!` exige o regressor futuro
    # para o horizonte pedido -- e foi exatamente esse descasamento que fez a primeira
    # versao deste arquivo lancar em vez de testar.
    HMAX = 12

    "y = X*b + eta, com eta ~ ARMA(1,1)x(P,0,Q)[12]. P = 0 => polinomio AR unitario."
    function geraErrosArima(seed, P; T = 120, S = 12, β = [2.0, -1.5])
        rng = MersenneTwister(seed)
        burn, K = 300, length(β)
        n = burn + T + HMAX
        X, e = randn(rng, n, K), randn(rng, n)
        ϕ, θ, Φ, Θ = 0.5, 0.4, (P > 0 ? 0.6 : 0.0), (P > 0 ? 0.0 : 0.7)
        η = zeros(n)
        for t = 14:n
            η[t] = ϕ * η[t-1] + Φ * η[t-12] - ϕ * Φ * η[t-13] +
                   e[t] + θ * e[t-1] + Θ * e[t-12] + θ * Θ * e[t-13]
        end
        y = X * β .+ η
        rg = (n-T-HMAX+1):n
        (mkTA(y[rg][1:T]), mkTX(X[rg, :]))
    end

    ajusta(y, X, p, P, q, Q; kw...) = begin
        m = SARIMA(y, X, p, 0, q; P = P, D = 0, Q = Q, seasonality = 12,
                   allowMean = false, allowDrift = false)
        fit!(m; objectiveFunction = "mse", initialization = :free,
             stationary = true, invertible = false, kw...)
        m
    end

    @testset "argumento invalido e recusado" begin
        y, X = geraErrosArima(1, 1)
        m = SARIMA(y, X, 1, 0, 1; P = 1, D = 0, Q = 0, seasonality = 12)
        @test_throws ArgumentError fit!(m; exogDynamics = :naosei)
    end

    # O DEFAULT E `:regression_errors` (era `:armax` ate a virada deliberada). Este teste
    # afirma que nomear o default explicitamente da o mesmo resultado que omiti-lo — se
    # divergirem, o kwarg nao esta sendo lido no caminho que o default toma.
    @testset "o default e :regression_errors, e nomea-lo e bit-identico a omitir" begin
        y, X = geraErrosArima(2, 1)
        a = ajusta(y, X, 1, 1, 1, 0)
        b = ajusta(y, X, 1, 1, 1, 0; exogDynamics = :regression_errors)
        @test a.exogCoefficients == b.exogCoefficients
        @test a.ϕ == b.ϕ
        @test a.Φ == b.Φ
        @test a.θ == b.θ
        predict!(a; stepsAhead = 6)
        predict!(b; stepsAhead = 6)
        @test values(a.forecast) == values(b.forecast)
    end

    # A IDENTIDADE. As duas formas coincidem sse o polinomio AR (regular e sazonal) e
    # unitario e nao ha diferenciacao -- filtrar X por 1 nao faz nada. Fixar isto impede
    # que uma refatoracao futura quebre a equivalencia em silencio.
    @testset "p = P = 0: as duas semanticas sao a MESMA equacao" begin
        y, X = geraErrosArima(3, 0)
        a = ajusta(y, X, 0, 0, 1, 1)
        b = ajusta(y, X, 0, 0, 1, 1; exogDynamics = :regression_errors)
        @test a.exogCoefficients ≈ b.exogCoefficients atol = 1e-8
        @test a.θ ≈ b.θ atol = 1e-8
        @test a.Θ ≈ b.Θ atol = 1e-8
        predict!(a; stepsAhead = 12)
        predict!(b; stepsAhead = 12)
        @test values(a.forecast) ≈ values(b.forecast) atol = 1e-6
    end

    # ... e so ali. Com p > 0 as duas TEM que divergir: se nao divergirem, o flag nao
    # esta vinculando, que e a classe de defeito que ja custou caro neste pacote.
    # Os DOIS lados nomeados de proposito. Antes este teste usava o default de um lado, e
    # quando o default virou ele passou a comparar `:regression_errors` consigo mesmo — a
    # diferenca deu 0,0 e o teste falhou por motivo que nao era o defeito que ele vigia.
    # Teste de divergencia entre dois modos tem de nomear os dois.
    @testset "p > 0: as duas semanticas divergem (o flag vincula)" begin
        y, X = geraErrosArima(4, 1)
        a = ajusta(y, X, 1, 1, 1, 0; exogDynamics = :armax)
        b = ajusta(y, X, 1, 1, 1, 0; exogDynamics = :regression_errors)
        predict!(a; stepsAhead = 12)
        predict!(b; stepsAhead = 12)
        @test maximum(abs.(values(a.forecast) .- values(b.forecast))) > 1e-3
    end

    # Sob a forma de erros ARIMA, b e o efeito marginal do DGP e tem de ser recuperado.
    # Sob :armax, b e multiplicador de impacto e NAO tem essa interpretacao -- por isso
    # o teste afirma so o lado que a teoria garante, e nao uma comparacao de erro.
    @testset ":regression_errors recupera o efeito marginal" begin
        β = [2.0, -1.5]
        erros = Float64[]
        for seed = 10:15
            y, X = geraErrosArima(seed, 1; β = β)
            m = ajusta(y, X, 1, 1, 1, 0; exogDynamics = :regression_errors)
            push!(erros, maximum(abs.(Float64.([m.exogCoefficients...]) .- β)))
        end
        @test median(erros) < 0.25
    end
end
