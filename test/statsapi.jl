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

    # `:zeroed` explicitly: the tests below compare `cssResiduals` with `residuals`, and
    # `cssResiduals` implements the ZEROED recursion (statsapi.jl: `resid = zeros(T)`, MA
    # term skipped for `t - j <= 0`). It reproduces no free pre-sample block mode.
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

        # Known limitation. `cssResiduals` documents
        # `cssResiduals(model, coef(model)) ~= residuals(model)`, and that does NOT hold for
        # the free-block modes. Measured on an AR(1) with n = 300:
        #
        #   :zeroed 4.4e-16 | :warmup 4.4e-16 | :free 5.1e-2 | :penalized 1.2e-2 | :innovations 2.4e-2
        #
        # It matters because `vcov` numerically differentiates
        # `sum(abs2, cssResiduals(...))`, so `stderror` comes from a DIFFERENT objective from
        # the one optimized; on the airline model `se[2]` moves by 51% between modes.
        #
        # The fix is not mechanical: the estimator minimizes S(coefs, eps_pre) JOINTLY, so
        # the correct curvature is that of the PROFILED objective, not of the plug-in.
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
        # `initialization` e `seasonalForm` NOMEADOS: o teste nao pode depender do valor
        # dos defaults. Ver nota sobre `cssResiduals` acima.
        fit!(airline; initialization = :zeroed, seasonalForm = :multiplicative)
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
