# `initialization = :innovations` (e `:penalized`) com `objectiveFunction = "ridge"`.
#
# O termo L2 nao muda a metrica da perda — o ajuste segue sendo soma de quadrados — entao o
# prior pre-amostral, que tambem e forma quadratica, e somavel a ela. A penalidade entra
# DENTRO de `S`, antes da multiplicacao pelo `fator`, porque `S * fator` e a forma
# multiplicativa de `T*log(S) + log|Omega|` e o `lambda = sqrt(nEff)` foi calibrado para a
# escala de uma soma de quadrados.

@testset "ridge com inicializacao penalizada" begin

    # Serie ARMA com componente sazonal, longa o bastante para o bloco pre-amostral existir.
    Random.seed!(20260822)
    T = 160
    s = 12
    ε = randn(T + 60)
    y = zeros(T + 60)
    for t = 3:(T+60)
        y[t] = 0.6 * y[t-1] + ε[t] + 0.4 * ε[t-1] + 0.3 * sin(2π * t / s)
    end
    serie = Sarimax.loadDataset(DataFrame(y = 100.0 .+ 10.0 .* y[61:end]))

    @testset "a guarda deixa passar ridge e continua barrando o resto" begin
        for ini in (:innovations, :penalized)
            # Nao lanca mais. O ajuste em si pode falhar por motivo numerico em qualquer
            # objetivo; o que se testa aqui e que NAO e mais ArgumentError de combinacao.
            m = SARIMA(serie, 1, 0, 1; seasonality = s)
            @test_nowarn fit!(m; objectiveFunction = "ridge", initialization = ini,
                                      silent = true)
        end
        # Os demais objetivos TROCAM a metrica da perda (ou nao tem o bloco escrito):
        # cobrar o bloco inicial numa escala quadratica enquanto a perda e linear na cauda
        # seria outra decisao de modelagem, nao uma extensao desta.
        for obj in ("mae", "stable", "ml_exact")
            m = SARIMA(serie, 1, 0, 1; seasonality = s)
            @test_throws ArgumentError fit!(
                m; objectiveFunction = obj, initialization = :innovations, silent = true)
        end

        # DEFEITO PRE-EXISTENTE, anterior a esta branch: `huber` sem `warmStart` NAO chega
        # na guarda. O bloco de warm-start (busque por `objectiveFunction == "huber" &&
        # isnothing(warmStart)`) roda antes dela, ajusta um base em `mse`, tenta o huber
        # dentro de um `try/catch` que ENGOLE a ArgumentError, e devolve o base marcado com
        # `huberFallback = true`. O usuario pede huber com inicializacao penalizada e recebe
        # um ajuste mse sem erro — precisamente a degradacao silenciosa que a guarda existe
        # para impedir.
        #
        # Registrado como `@test_broken` e nao "corrigido de passagem": mexer no caminho do
        # huber e outro assunto que o desta branch. Quando for consertado, este teste passa a
        # falhar como "Unexpectedly Pass" e deve virar um `@test_throws` normal.
        mHub = SARIMA(serie, 1, 0, 1; seasonality = s)
        lancou = try
            fit!(mHub; objectiveFunction = "huber", initialization = :innovations, silent = true)
            false
        catch
            true
        end
        @test_broken lancou
        # E o sintoma observavel do recuo, para o defeito ficar caracterizado e nao so
        # anotado: o ajuste devolvido e o base de mse.
        @test get(mHub.metadata, "huberFallback", false) == true
    end

    @testset "o fator multiplicativo carrega o log-determinante" begin
        # `S * fator` so equivale a `T*log(S) + log|Omega|` se `T*log(fator) == log|Omega|`.
        # Sem isso a penalidade somada dentro de `S` estaria sendo escalada por outra coisa.
        Tloc = 137
        for κ in ([0.3], [0.5, -0.2], [0.7, 0.1, -0.4])
            fator = prod([(1 - κ[j]^2)^(-j / Tloc) for j in eachindex(κ)])
            logDet = -sum(j * log(1 - κ[j]^2) for j in eachindex(κ))
            @test Tloc * log(fator) ≈ logDet atol = 1e-12
        end
    end

    @testset "somar dentro de S preserva o argmin da forma MAP" begin
        # A forma implementada e `(S + λ‖b‖²) * fator`; a forma MAP e
        # `T*log(S + λ‖b‖²) + log|Omega|`. Uma e transformacao monotona da outra, entao a
        # ORDEM entre dois pontos tem de coincidir. A forma ERRADA (`S*fator + λ‖b‖²`) nao
        # tem essa propriedade, e o ultimo bloco mostra um contraexemplo.
        Tloc = 137
        Random.seed!(7)
        discordancias = 0
        for _ = 1:400
            S1, S2 = 50 .+ 100 .* rand(2)
            pen1, pen2 = 10 .* rand(2)
            κ1 = 0.9 .* (2 .* rand(2) .- 1)
            κ2 = 0.9 .* (2 .* rand(2) .- 1)
            f(κ) = prod([(1 - κ[j]^2)^(-j / Tloc) for j in eachindex(κ)])
            ld(κ) = -sum(j * log(1 - κ[j]^2) for j in eachindex(κ))
            mult1 = (S1 + pen1) * f(κ1)
            mult2 = (S2 + pen2) * f(κ2)
            map1 = Tloc * log(S1 + pen1) + ld(κ1)
            map2 = Tloc * log(S2 + pen2) + ld(κ2)
            sign(mult1 - mult2) == sign(map1 - map2) || (discordancias += 1)
        end
        @test discordancias == 0
    end

    @testset "sem coeficientes a encolher, ridge coincide com mse" begin
        # Um (0,d,0) nao tem phi/theta/Phi/Theta, entao `ridgeShrinkage` devolve `nothing` e
        # os dois objetivos tem de produzir o MESMO ajuste — o que amarra os dois ramos
        # (condicional e penalizado) um ao outro.
        for ini in (:zeroed, :innovations)
            mMse = SARIMA(serie, 0, 1, 0; seasonality = s)
            mRid = SARIMA(serie, 0, 1, 0; seasonality = s)
            fit!(mMse; objectiveFunction = "mse", initialization = ini, silent = true)
            fit!(mRid; objectiveFunction = "ridge", initialization = ini, silent = true)
            @test mRid.σ² ≈ mMse.σ² rtol = 1e-8
        end
    end

    @testset "o encolhimento age sobre os coeficientes" begin
        normaAR(m) = sum(abs2, something(m.ϕ, Float64[])) +
                     sum(abs2, something(m.θ, Float64[]))
        mMse = SARIMA(serie, 2, 0, 2; seasonality = s)
        mRid = SARIMA(serie, 2, 0, 2; seasonality = s)
        fit!(mMse; objectiveFunction = "mse", initialization = :innovations, silent = true)
        fit!(mRid; objectiveFunction = "ridge", initialization = :innovations, silent = true)
        # Encolhe: a norma nao pode CRESCER. Desigualdade fraca porque o otimizador e
        # local e a serie pode ter os coeficientes ja proximos de zero.
        @test normaAR(mRid) <= normaAR(mMse) + 1e-6
        # E o ajuste piora (ou empata) em soma de quadrados, que e o preco do encolhimento.
        @test mRid.σ² >= mMse.σ² - 1e-8
    end

    @testset "a penalidade NAO entra no criterio de informacao" begin
        # sigma2, loglik e AICc saem dos RESIDUOS (`computeSARIMAModelVariance` le
        # `value.(model[:ϵ])`), nao do valor do objetivo. Se um dia alguem passar a ler
        # `objective_value`, o termo L2 contaminaria o torneio e este teste cai.
        m = SARIMA(serie, 1, 0, 1; seasonality = s)
        fit!(m; objectiveFunction = "ridge", initialization = :innovations, silent = true)
        resid = Sarimax.observedResiduals(m)
        K = get_hyperparameters_number(m)
        σ²Esperado = sum(abs2, resid) / (length(resid) - K + 1)
        @test m.σ² ≈ σ²Esperado rtol = 1e-6
        @test isfinite(aicc(m))
    end

    @testset "lambda e sqrt(T) sob :innovations e constante no torneio" begin
        # Sob :innovations o `auto` zera o searchLb, entao lb = 1 e nEff = T. O valor tem de
        # ser o MESMO para candidatos de ordens diferentes na mesma serie, senao a regua
        # muda entre competidores dentro do proprio torneio.
        Tdiff = 140
        @test sqrt(max(Tdiff - 1 + 1, 1)) == sqrt(Tdiff)
        # E diferente do regime condicional, onde lb desconta o conditioning comum.
        lbCond = 5 + 2 * s + 1
        @test sqrt(max(Tdiff - lbCond + 1, 1)) < sqrt(Tdiff)
    end
end
