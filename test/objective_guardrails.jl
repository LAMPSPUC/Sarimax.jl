# Guardas contra comportamento indefensavel em silencio.
#
# Nenhum destes testes fixa POLITICA (quanto encolher, quais graus de liberdade cobrar, que
# espaco varrer) — todos fixam a ausencia de surpresa: parametro ignorado nao pode mexer no
# criterio, e objetivo que degrada tem que avisar.
@testset "guardas de objetivo e contagem de parametros" begin
    rng = MersenneTwister(0x6A11)
    dates(n) = collect(Date(2000, 1, 1):Month(1):Date(2000, 1, 1)+Month(n - 1))

    @testset "lambda ignorado nao pode mexer no criterio" begin
        # Reproducao do defeito: o gatilho da contagem esparsa era a PRESENCA de
        # `lambda`/`alpha`, nao o objetivo usado. Com coeficientes fixos e objetivo `mse`
        # (que ignora `lambda`), passar `lambda` mantinha o ajuste bit-a-bit identico mas
        # levava K de 4 para 2 e o AICc de 190.9058 para 186.3890.
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
        # A correcao acima nao pode ter desligado a contagem esparsa de quem realmente
        # regulariza. Restringi-la ao lasso (`alpha = 1`) e mudanca de politica, nao de defeito.
        n = 150
        y = TimeArray(
            dates(n),
            10 .+ 0.6 .* sin.(2π .* (1:n) ./ 12) .+ cumsum(randn(rng, n) .* 0.25),
        )
        m = SARIMA(y, 3, 1, 2; seasonality = 12, P = 1, D = 0, Q = 1, allowMean = false)
        fit!(m; objectiveFunction = "elastic_net", alpha = 1.0)
        nominal = m.p + m.q + m.P + m.Q + 1
        coefs = vcat([m.ϕ...], [m.θ...], [m.Φ...], [m.Θ...])
        @test any(c -> abs(c) <= 1e-5, coefs)              # premissa: houve zeragem
        @test get_hyperparameters_number(m) < nominal
        @test m.metadata["objectiveFunction"] == "elastic_net"
    end

    @testset "ridge RECUSA lambda em vez de ignorar" begin
        # Era aviso e passou a ser erro. A politica do pacote separa dois casos que antes
        # estavam juntos: COMBINACAO DE ARGUMENTOS invalida (esta) recusa, porque e fixa na
        # chamada e o chamador pode checar antes; DEGRADACAO EM TEMPO DE EXECUCAO (o
        # `ml_exact` do testset seguinte) avisa, porque depende do candidato e erro ali
        # abortaria a busca.
        #
        # O que motivou a troca: num run paralelo o `@warn` e invisivel, e uma varredura em
        # que parte das celulas silenciosamente significa outra coisa e uma varredura
        # quebrada.
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
        # Mesma politica. A lista aceita tem de espelhar o portao do objetivo penalizado;
        # se ela permitir algo que o portao nao cobre, o ajuste degrada para :free em
        # silencio — o defeito que este erro existe para impedir.
        n = 90
        y = TimeArray(dates(n), 10 .+ cumsum(randn(rng, n) .* 0.3))
        mk() = SARIMA(y, 2, 1, 1; allowMean = false)
        @test_throws ArgumentError fit!(
            mk(); objectiveFunction = "mae", initialization = :penalized
        )
        # `:innovations` tem de recusar pelo MESMO motivo: o portao do objetivo e
        # `penalizado = initialization in (:penalized, :innovations)`, entao os dois nomes
        # precisam estar na guarda. Este teste e o que impede a lista e o portao de se
        # separarem quando alguem acrescentar um modo novo.
        @test_throws ArgumentError fit!(
            mk(); objectiveFunction = "stable", initialization = :innovations
        )
        # `mse` e `ridge` sao os casos cobertos: nao recusam.
        #
        # O `ridge` ESTAVA nesta lista de recusados e saiu: o termo L2 nao muda a metrica da
        # perda — o ajuste segue sendo soma de quadrados — entao o prior pre-amostral, que
        # tambem e forma quadratica, e somavel a ela. A penalidade entra dentro de `S` antes
        # do fator do determinante; ver `test/ridge_innovations.jl` para a verificacao de
        # que as duas formas tem o mesmo argmin.
        @test_logs match_mode = :any fit!(
            mk(); objectiveFunction = "mse", initialization = :penalized
        )
        @test_logs match_mode = :any fit!(
            mk(); objectiveFunction = "ridge", initialization = :innovations
        )
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
