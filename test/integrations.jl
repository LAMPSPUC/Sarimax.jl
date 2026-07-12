@testset "Ecosystem integrations" begin
    @testset "Tables.jl input" begin
        n = 24
        tblDates = Date(2020, 1, 1):Month(1):(Date(2020, 1, 1)+Month(n - 1))
        namedTupleTable = (date = collect(tblDates), value = collect(1.0:n))
        ta = load_dataset(namedTupleTable)
        @test ta isa TimeArray
        @test length(ta) == n
        @test values(ta)[end] == Float64(n)

        # custom timestamp column name
        renamed = (when = collect(tblDates), value = collect(1.0:n))
        ta2 = load_dataset(renamed; timestampColumn = :when)
        @test timestamp(ta2) == collect(tblDates)

        @test_throws ArgumentError load_dataset(42)
    end

    @testset "Plots recipe" begin
        Random.seed!(41)
        n = 60
        recDates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        model = SARIMA(TimeArray(collect(recDates), cumsum(randn(n))), 0, 1, 0; allowMean = false)
        fit!(model)
        predict!(model; stepsAhead = 6, displayConfidenceIntervals = true)
        recipes = Sarimax.RecipesBase.apply_recipe(Dict{Symbol,Any}(), model)
        @test length(recipes) == 3   # observed + fitted + forecast(with ribbon)
        # forecast series carries the confidence ribbon
        @test haskey(recipes[3].plotattributes, :ribbon)
    end

    @testset "MLJ interface" begin
        Random.seed!(42)
        n = 120
        ar = zeros(n)
        for t = 2:n
            ar[t] = 0.6 * ar[t-1] + randn()
        end
        spec = Sarimax.SARIMAForecaster(p = 1, allowMean = false)
        fitresult, _, report = Sarimax.MLJModelInterface.fit(spec, 0, nothing, ar)
        @test fitresult isa SARIMAModel
        @test report.coefficient_names == ["ar_1"]
        @test abs(report.coefficients[1] - 0.6) < 0.15

        horizon = (dummy = zeros(5),)   # 5 rows -> 5-step forecast
        forecast = Sarimax.MLJModelInterface.predict(spec, fitresult, horizon)
        @test length(forecast) == 5
        @test all(isfinite, forecast)
    end

    @testset "type stability of pure helpers" begin
        @inferred Sarimax.polynomialMultiplication([1.0, -1.0], [1.0, -1.0])
        @inferred Sarimax.psiWeights([0.5], [0.3], 5)
        @inferred differentiated_coefficients(1, 1, 12)
        @inferred Sarimax.reflectionToMA([0.5, 0.3])
        @inferred Sarimax.ljung_box_test(randn(100); lags = 5)
        @inferred Sarimax.boxcox_transform([1.0, 2.0, 3.0], 0.5)
    end
end
