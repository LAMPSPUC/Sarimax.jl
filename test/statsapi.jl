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

    fit!(model)

    @testset "accessors" begin
        @test coefnames(model) == ["c", "ar_1"]
        @test length(coef(model)) == 2
        @test nobs(model) == length(residuals(model))
        @test length(fitted(model)) == nobs(model)
    end

    @testset "cssResiduals replicates the JuMP fit" begin
        cr = Sarimax.cssResiduals(model, coef(model))
        @test maximum(abs.(cr .- residuals(model))) < 1e-8
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
        fit!(airline)
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
