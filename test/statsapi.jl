@testset "StatsAPI interface" begin
    Random.seed!(123)
    n = 300
    phi = 0.5
    v = zeros(n)
    for t = 2:n
        v[t] = phi * v[t-1] + randn()
    end
    ar1Dates = Date(1990, 1, 1):Month(1):(Date(1990, 1, 1)+Month(n - 1))
    model = SARIMA(TimeArray(collect(ar1Dates), v), 1, 0, 0; allowMean = true)

    @test_throws ModelNotFitted coef(model)
    @test_throws ModelNotFitted residuals(model)
    @test_throws ModelNotFitted nobs(model)
    @test_throws ModelNotFitted vcov(model)

    # `:zeroed` explicito: os testes abaixo comparam `cssResiduals` com `residuals`, e
    # `cssResiduals` implementa a recursao ZERADA (statsapi.jl: `resid = zeros(T)`, termo MA
    # pulado para `t - j <= 0`). Ela nao reproduz nenhum modo de bloco pre-amostral livre.
    fit!(model; initialization = :zeroed)

    @testset "accessors" begin
        @test coefnames(model) == ["c", "ar_1"]
        @test length(coef(model)) == 2
        @test nobs(model) == length(residuals(model))
        @test length(fitted(model)) == nobs(model)
    end

    @testset "cssResiduals replicates the JuMP fit" begin
        cr = Sarimax.cssResiduals(model, coef(model))
        @test maximum(abs.(cr .- residuals(model))) < 1e-8

        # DEFEITO PRE-EXISTENTE, nao regressao desta branch. `cssResiduals` promete no
        # docstring `cssResiduals(model, coef(model)) ~= residuals(model)`, e essa promessa
        # NAO vale para os modos de bloco livre. Medido na base, AR(1) n=300:
        #
        #   :zeroed 4,4e-16 | :warmup 4,4e-16 | :free 5,1e-2 | :penalized 1,2e-2 | :innovations 2,4e-2
        #
        # Importa porque `vcov` diferencia numericamente `sum(abs2, cssResiduals(...))`:
        # o `stderror` sai de um objetivo DIFERENTE do que foi otimizado. No airline o
        # `se[2]` muda 51% entre modos.
        #
        # Conserto certo nao e mecanico — o estimador minimiza S(coefs, eps_pre)
        # CONJUNTAMENTE, entao a curvatura correta e a do objetivo PERFILADO, nao a do
        # plug-in. Perfilado vs plug-in e decisao de metodo do mantenedor.
        for modo in (:free, :penalized, :innovations)
            mLivre = SARIMA(TimeArray(collect(ar1Dates), v), 1, 0, 0; allowMean = true)
            fit!(mLivre; initialization = modo)
            crLivre = Sarimax.cssResiduals(mLivre, coef(mLivre))
            @test_broken maximum(abs.(crLivre .- residuals(mLivre))) < 1e-8
        end
    end

    @testset "standard errors (CSS asymptotics)" begin
        se = stderror(model)
        @test length(se) == 2
        @test all(isfinite, se)
        # theory: se(ϕ̂) ≈ sqrt((1-ϕ²)/n) ≈ 0.05
        @test 0.01 < se[2] < 0.2
        @test abs(coef(model)[2] - phi) < 3 * se[2]
        V = vcov(model)
        @test size(V) == (2, 2)
        @test V ≈ V' atol = 1e-10
    end

    @testset "multiplicative seasonal model consistency" begin
        airPassengersLog = log.(load_dataset(AIR_PASSENGERS))
        airline = SARIMA(airPassengersLog, 0, 1, 1; seasonality = 12, P = 0, D = 1, Q = 1)
        fit!(airline; initialization = :zeroed)  # ver nota sobre `cssResiduals` acima
        cr = Sarimax.cssResiduals(airline, coef(airline))
        @test maximum(abs.(cr .- residuals(airline))) < 1e-8
        se = stderror(airline)
        @test length(se) == length(coef(airline))
    end

    @testset "deprecated camelCase names still work" begin
        @test length(loadDataset(AIR_PASSENGERS)) == 204
        @test hasFitMethods(SARIMAModel)
    end
end
