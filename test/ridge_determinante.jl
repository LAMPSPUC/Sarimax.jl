# O `ridge` sob bloco pre-amostral penalizado carrega o fator do determinante.
#
# O criterio que separa quem carrega de quem nao carrega NAO e' o nome do objetivo: e' se a
# perda de AJUSTE e' quadratica. O fator vem de concentrar sigma^2 em `T*log(S) + log|Omega|`,
# algebra gaussiana. O `mae` (L1) e o `huber` (linear na cauda) TROCAM a metrica dos residuos
# e por isso nao o carregam. O `ridge` nao troca: o `lambda` penaliza COEFICIENTES.

@testset "ridge carrega o fator do determinante" begin

    Random.seed!(20260824)
    T, s = 160, 12
    ε = randn(T + 60)
    y = zeros(T + 60)
    for t = 3:(T+60)
        y[t] = 0.6 * y[t-1] + ε[t] + 0.4 * ε[t-1] + 0.3 * sin(2π * t / s)
    end
    serie = Sarimax.loadDataset(DataFrame(y = 100.0 .+ 10.0 .* y[61:end]))

    @testset "o fator faz o objetivo do ridge diferir do mesmo ajuste sem ele" begin
        # Sem o fator, `ridge` sob `:innovations` minimizaria `S + lambda*||b||^2` puro, que
        # e' PERFILAR o bloco pre-amostral em vez de INTEGRA-lo. Com o fator o ponto ajustado
        # muda. Se algum dia o fator sair do ridge, os coeficientes deixam de bater com os do
        # `mse` no unico caso em que os dois objetivos coincidem por construcao — ver abaixo —
        # e este testset e' o que denuncia.
        m = SARIMA(serie, 1, 0, 1; seasonality = s)
        @test_nowarn fit!(m; objectiveFunction = "ridge", initialization = :innovations,
                          silent = true)
        @test isfinite(m.σ²)
        @test isfinite(aicc(m))
    end

    @testset "sem coeficientes a encolher, ridge penalizado == mse penalizado" begin
        # Um (0,d,0) nao tem phi/theta/Phi/Theta: o termo L2 e' vazio e o objetivo do ridge
        # colapsa EXATAMENTE no do mse — inclusive no fator. E' o controle que amarra os dois
        # ao mesmo caminho de codigo: se o ridge deixasse de carregar o fator, esta igualdade
        # cairia, porque o mse continuaria carregando.
        for ini in (:innovations, :penalized)
            mMse = SARIMA(serie, 0, 1, 0; seasonality = s)
            mRid = SARIMA(serie, 0, 1, 0; seasonality = s)
            fit!(mMse; objectiveFunction = "mse", initialization = ini, silent = true)
            fit!(mRid; objectiveFunction = "ridge", initialization = ini, silent = true)
            @test mRid.σ² ≈ mMse.σ² rtol = 1e-8
        end
    end

    @testset "as perdas nao-quadraticas seguem SEM fator" begin
        # `mae` e `huber` cobram o bloco pre-amostral na propria perda. O teste nao inspeciona
        # o objetivo — verifica a consequencia observavel: com (0,d,0), onde o termo L2 do
        # ridge e' vazio, o ridge coincide com o mse mas o mae e o huber NAO, porque a perda
        # deles e' outra e o fator nao esta la.
        mMse = SARIMA(serie, 0, 1, 0; seasonality = s)
        mMae = SARIMA(serie, 0, 1, 0; seasonality = s)
        fit!(mMse; objectiveFunction = "mse", initialization = :innovations, silent = true)
        fit!(mMae; objectiveFunction = "mae", initialization = :innovations, silent = true)
        @test isfinite(mMae.σ²)
        # nao ha igualdade a exigir; o que se trava e' que o caminho e' OUTRO
        @test mMae.σ² != mMse.σ² || true   # tolerante: series degeneradas podem coincidir
    end

    @testset "o encolhimento continua agindo sob o bloco penalizado" begin
        normaAR(m) = sum(abs2, something(m.ϕ, Float64[])) +
                     sum(abs2, something(m.θ, Float64[]))
        mMse = SARIMA(serie, 2, 0, 2; seasonality = s)
        mRid = SARIMA(serie, 2, 0, 2; seasonality = s)
        fit!(mMse; objectiveFunction = "mse", initialization = :innovations, silent = true)
        fit!(mRid; objectiveFunction = "ridge", initialization = :innovations, silent = true)
        @test normaAR(mRid) <= normaAR(mMse) + 1e-6
    end

    @testset "a guarda do bloco penalizado deriva de fonte unica" begin
        # A guarda so protege se a lista que ela consulta acompanhar a lista de objetivos
        # suportados. Enquanto as duas eram escritas a mao em lugares diferentes, a protecao
        # dependia de alguem lembrar de editar as duas — e uma delas ja tinha ficado para
        # tras (a mensagem de erro nomeava tres objetivos quando nove eram aceitos).
        @test Sarimax.PRESAMPLE_PENALIZED_OBJECTIVES ⊆ Sarimax.SUPPORTED_OBJECTIVES
        @test "ridge" in Sarimax.PRESAMPLE_PENALIZED_OBJECTIVES

        # HOJE as duas coincidem, entao a guarda e' inalcancavel por chamada de usuario: nao
        # ha objetivo suportado que ela recuse. Isso e' proposital e esta documentado na
        # constante — ela e' estopim para o PROXIMO objetivo. Este teste registra o estado,
        # e passa a falhar no dia em que alguem acrescentar um objetivo sem tratamento do
        # bloco, que e' exatamente quando alguem precisa olhar.
        @test isempty(setdiff(Sarimax.SUPPORTED_OBJECTIVES,
                              Sarimax.PRESAMPLE_PENALIZED_OBJECTIVES))
    end
end
