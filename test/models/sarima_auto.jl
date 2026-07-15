@testset "auto sarima" begin
    @testset "Constant series" begin
        series = TimeArray(Dates.Date(2019, 1, 1):Dates.Day(1):Dates.Date(2019, 1, 31), ones(31))
        model = Sarimax.auto(series)
        @test model.c == 1
        @test model.p == 0
        @test model.q == 0
        @test model.P == 0
        @test model.Q == 0
        @test model.d == 0
        @test model.D == 0
        @test model.seasonality == 1
    end

    @testset "constantDiffSeriesModelSpecification Tests" begin
        y = TimeArray(Dates.Date(2019, 1, 1):Dates.Day(1):Dates.Date(2019, 1, 31), ones(31))
        exog = nothing

        # Test case where D > 0 and d == 0
        model1 = Sarimax.constantDiffSeriesModelSpecification(y, exog, 0, 1, 12, false, true)
        @test model1 isa Sarimax.SarimaxModel
        @test model1.allowDrift == true

        # Test case where D > 0 and d > 0
        model2 = Sarimax.constantDiffSeriesModelSpecification(y, exog, 1, 1, 12, false, false)
        @test model2 isa Sarimax.SarimaxModel
        @test model2.allowDrift == false

        # Test case where d == 2
        model3 = Sarimax.constantDiffSeriesModelSpecification(y, exog, 2, 0, 12, false, false)
        @test model3 isa Sarimax.SarimaxModel

        # Test case where d < 2
        model4 = Sarimax.constantDiffSeriesModelSpecification(y, exog, 1, 0, 12, true, false)
        @test model4 isa Sarimax.SarimaxModel
        @test model4.allowMean == true

        # Test case where data follow a simple polynomial
        @test_throws ArgumentError Sarimax.constantDiffSeriesModelSpecification(y, exog, 3, 0, 12, true, false)

        # Test case with exog
        exog1 = TimeArray(Dates.Date(2019, 1, 1):Dates.Day(1):Dates.Date(2019, 1, 31), ones(31))
        model5 = Sarimax.constantDiffSeriesModelSpecification(y, exog1, 1, 0, 12, false, false)
        @test model5 isa Sarimax.SarimaxModel
        @test model5.allowDrift == false
        @test model5.allowMean == false

        # Test case with exog and D > 0
        model6 = Sarimax.constantDiffSeriesModelSpecification(y, exog1, 1, 1, 12, false, false)
        @test model6 isa Sarimax.SarimaxModel
        @test model6.p == 0
        @test model6.q == 0
        @test model6.P == 0
        @test model6.Q == 0
    end

    @testset "getInformationCriteriaFunction" begin
        @test_throws ArgumentError Sarimax.getInformationCriteriaFunction("mse")

        # Test AIC
        func1 = Sarimax.getInformationCriteriaFunction("aic")
        @test func1 isa Function
        @test aic(2, 3.0) ≈ -2.0
        @test aic(3, 4.0) ≈ -2.0

        # Test AICC
        func2 = Sarimax.getInformationCriteriaFunction("aicc")
        @test func2 isa Function
        @test func2(10, 2, 3.0) ≈ -0.2857142857142858
        @test func2(10, 3, 4.0) ≈ 2.0

        # Test BIC
        func3 = Sarimax.getInformationCriteriaFunction("bic")
        @test func3 isa Function
        @test func3(10, 2, 3.0) ≈ -1.3948298140119082
        @test func3(10, 3, 4.0) ≈ -1.0922447210178623
    end

    @testset "detectOutliers" begin
        series = TimeArray(Dates.Date(2019, 1, 1):Dates.Day(1):Dates.Date(2019, 1, 31), ones(31))
        outliers = Sarimax.detectOutliers(series, nothing, 0, 0, 1, false)
        @test isnothing(outliers)

        values(series)[5] = 100
        outliers = Sarimax.detectOutliers(series, nothing, 0, 0, 1, false)
        @test isa(outliers, TimeSeries.TimeArray)
        @test length(colnames(outliers)) == 1
        @test colnames(outliers)[1] == Symbol("outlier_5")

        values(series)[5] = 1
        values(series)[10] = 100
        outliers = Sarimax.detectOutliers(series, nothing, 0, 0, 1, false)
        @test isa(outliers, TimeSeries.TimeArray)
        @test length(colnames(outliers)) == 1
        @test colnames(outliers)[1] == Symbol("outlier_10")

        values(series)[10] = 1
        values(series)[15] = 100
        outliers = Sarimax.detectOutliers(series, nothing, 0, 0, 1, false)
        @test isa(outliers, TimeSeries.TimeArray)
        @test length(colnames(outliers)) == 1
        @test colnames(outliers)[1] == Symbol("outlier_15")

        values(series)[20] = 100
        outliers = Sarimax.detectOutliers(series, nothing, 0, 0, 1, false)
        @test isa(outliers, TimeSeries.TimeArray)
        @test length(colnames(outliers)) == 2
        @test colnames(outliers)[1] == Symbol("outlier_15")
        @test colnames(outliers)[2] == Symbol("outlier_20")

        # test with exog
        exog = TimeArray(Dates.Date(2019, 1, 1):Dates.Day(1):Dates.Date(2019, 1, 31), 2 .* ones(31))
        outliers = Sarimax.detectOutliers(series, exog, 0, 0, 1, false)
        @test !isnothing(outliers)
        @test length(colnames(outliers)) == 3
        @test colnames(outliers)[1] == Symbol("A")
        @test colnames(outliers)[2] == Symbol("outlier_15")
        @test colnames(outliers)[3] == Symbol("outlier_20")

    end

    @testset "auto with default stepwise" begin
        airpassengers = load_dataset(AIR_PASSENGERS)
        model = auto(airpassengers; seasonality = 12)
        # Pinned selection. The tryCandidate! rewrite of the neighbourhood scan was
        # verified to reproduce the unrolled implementation exactly (same model,
        # same AICc to the last digit) before this pin was added.
        @test model.p == 4
        @test model.q == 1
        @test model.P == 0
        @test model.Q == 1
        @test model.d == 1
        @test model.D == 1
        @test aicc(model) ≈ 465.45706926625013 atol = 1e-6
    end

    @testset "parallel candidate fitting (smoke)" begin
        # With a single thread @threads degrades to serial; the selection must be
        # identical either way (min-IC selection is order-independent).
        Random.seed!(31)
        n = 90
        parDates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        yPar = TimeArray(collect(parDates), cumsum(randn(n)) .+ 0.3 .* collect(1.0:n))
        mSerial = auto(yPar; searchMethod = "grid", maxp = 1, maxq = 1, maxP = 0, maxQ = 0)
        mParallel = auto(yPar; searchMethod = "grid", maxp = 1, maxq = 1, maxP = 0, maxQ = 0, parallel = true)
        @test (mSerial.p, mSerial.q, mSerial.d) == (mParallel.p, mParallel.q, mParallel.d)
        @test aicc(mSerial) ≈ aicc(mParallel) atol = 1e-8
    end

    @testset "auto with stepwise naive" begin
        airpassengers = load_dataset(AIR_PASSENGERS)
        log_airpassengers = log.(airpassengers)
        model = auto(airpassengers; searchMethod="stepwiseNaive", seasonality=12)
        # Selected orders under CSS information criteria and the multiplicative
        # seasonal form (v0.3 default). Identical to the exhaustive grid optimum.
        @test model.p == 1
        @test model.q == 1
        @test model.P == 2
        @test model.Q == 0
        @test model.d == 1
        @test model.D == 1
    end

    @testset "auto with grid search" begin
        airpassengers = load_dataset(AIR_PASSENGERS)
        log_airpassengers = log.(airpassengers)
        model = auto(airpassengers; searchMethod="grid", seasonality=12)
        # Selected orders under CSS information criteria and the multiplicative
        # seasonal form (see above).
        @test model.p == 1
        @test model.q == 1
        @test model.P == 2
        @test model.Q == 0
        @test model.d == 1
        @test model.D == 1
        # Exhaustive search must not do worse than the stepwise heuristic.
        stepwiseModel = auto(airpassengers; searchMethod="stepwiseNaive", seasonality=12)
        @test aicc(model) <= aicc(stepwiseModel) + 1e-6
    end
end
