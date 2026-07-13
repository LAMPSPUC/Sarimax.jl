@testset "Residual diagnostics" begin
    @testset "Ljung-Box" begin
        Random.seed!(21)
        wn = randn(300)
        resWN = Sarimax.ljung_box_test(wn; lags = 10)
        @test resWN["p_value"] > 0.05          # white noise: no autocorrelation
        @test resWN["dof"] == 10

        ar = zeros(300)
        for t = 2:300
            ar[t] = 0.8 * ar[t-1] + randn()
        end
        resAR = Sarimax.ljung_box_test(ar; lags = 10)
        @test resAR["p_value"] < 1e-6          # strongly autocorrelated
        @test resAR["test_statistic"] > resWN["test_statistic"]

        # fitdf reduces the degrees of freedom
        resDF = Sarimax.ljung_box_test(wn; lags = 10, fitdf = 3)
        @test resDF["dof"] == 7

        @test_throws ArgumentError Sarimax.ljung_box_test(wn; lags = 0)
        @test_throws ArgumentError Sarimax.ljung_box_test(wn; lags = 300)

        # model method: residuals of a well-specified fit should pass
        Random.seed!(22)
        n = 200
        diagDates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        yAR = zeros(n)
        for t = 2:n
            yAR[t] = 0.5 * yAR[t-1] + randn()
        end
        model = SARIMA(TimeArray(collect(diagDates), yAR), 1, 0, 0; allowMean = false)
        fit!(model)
        resModel = Sarimax.ljung_box_test(model)
        @test resModel["p_value"] > 0.01
    end

    @testset "Jarque-Bera" begin
        Random.seed!(23)
        normal = randn(500)
        resN = Sarimax.jarque_bera_test(normal)
        @test resN["p_value"] > 0.05

        skewed = exp.(randn(500))               # log-normal: heavily skewed
        resS = Sarimax.jarque_bera_test(skewed)
        @test resS["p_value"] < 1e-6
        @test resS["skewness"] > 1.0

        @test_throws ArgumentError Sarimax.jarque_bera_test([1.0, 2.0])
    end
end

@testset "Box-Cox" begin
    Random.seed!(24)
    y = exp.(randn(120) .* 0.2 .+ 2.0)

    # round-trip identity for several λ
    for λ in (-0.5, 0.0, 0.33, 1.0)
        z = Sarimax.boxcox_transform(y, λ)
        @test Sarimax.inverse_boxcox(z, λ) ≈ y rtol = 1e-10
    end
    # λ = 0 is the log
    @test Sarimax.boxcox_transform(y, 0.0) ≈ log.(y) rtol = 1e-12

    # positivity requirement
    @test_throws ArgumentError Sarimax.boxcox_transform([1.0, -2.0], 0.5)

    # Guerrero λ lies in the search interval and stabilizes variance
    airPassengers = load_dataset(AIR_PASSENGERS)
    λ = Sarimax.boxcox_lambda(airPassengers; seasonality = 12)
    @test -1.0 <= λ <= 2.0
    # the series has variance increasing with level → λ decidedly below 1
    @test λ < 0.9

    # TimeArray methods
    ta = Sarimax.boxcox_transform(airPassengers, λ)
    @test ta isa TimeArray
    @test values(Sarimax.inverse_boxcox(ta, λ)) ≈ values(airPassengers) rtol = 1e-8
end

@testset "Temporal cross-validation" begin
    # Deterministic linear series: a drift-only model forecasts it exactly,
    # so every rolling-origin error must be ~0.
    n = 60
    cvDates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
    yLinear = TimeArray(collect(cvDates), 2.0 .+ 0.5 .* collect(1.0:n))

    cv = Sarimax.cross_validation(
        yLinear;
        initialTrainSize = 40,
        stepsAhead = 3,
        step = 5,
        fitFunction = train -> begin
            m = SARIMA(train, 0, 1, 0; allowMean = false, allowDrift = true)
            fit!(m)
            m
        end,
    )
    @test size(cv.errors) == (length(cv.origins), 3)
    @test cv.origins == [40, 45, 50, 55]
    @test all(abs.(cv.errors[.!isnan.(cv.errors)]) .< 1e-3)
    @test all(cv.rmse .< 1e-3)
    @test length(cv.mae) == 3

    @test_throws ArgumentError Sarimax.cross_validation(yLinear; initialTrainSize = 59, stepsAhead = 3)
end
