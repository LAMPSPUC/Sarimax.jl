function generateARseries(
    p,
    s,
    ARcoeff,
    seasCoeff,
    trend,
    seed::Int = 1234,
    error::Bool = true,
)
    dates = Date(1991, 7, 1):Month(1):Date(2008, 2, 1)
    Random.seed!(seed)
    #Error terms:
    if error
        whiteNoise = randn(200) # Normal distribution mean = 0 and std error = 1
    else
        whiteNoise = zeros(200)
    end
    #trend
    x = 1:200
    numInitialValues = max(s, p)
    seriesValues::Vector{Float64} = Vector{Float64}()
    for i = 1:numInitialValues
        value = randn() + trend * x[i] + whiteNoise[i]
        push!(seriesValues, value)
    end
    for i = numInitialValues+1:200
        value =
            seriesValues[i-s] * seasCoeff +
            sum(ARcoeff[j] * seriesValues[i-j] for j = 1:p) +
            trend * x[i] +
            whiteNoise[i]
        push!(seriesValues, value)
    end
    return TimeArray(dates, seriesValues)
end

function generateSeries(p, s, coeff, trend, seed::Int = 1234, error::Bool = true)
    dates = Date(1991, 7, 1):Month(1):Date(2008, 2, 1)
    Random.seed!(seed)
    #Error terms:
    if error
        whiteNoise = randn(200) # Normal distribution mean = 0 and std error = 1
    else
        whiteNoise = zeros(200)
    end
    seriesValues = randn(p)
    #Seasonality
    x = 1:200
    if s > 1
        seas = 5 * sin.(x * 2 * pi / s)
        # adding seasonality to the initial terms
        for i = 1:p
            seriesValues[i] += seas[i]
        end
    else
        seas = zeros(200)
    end

    #adding trend to the initial terms
    for i = 1:p
        seriesValues[i] += trend * x[i]
    end

    # generating AR series
    for i = p+1:200
        value = whiteNoise[i] + seas[i] + x[i] * trend
        for j = 1:p
            value += coeff[j] * seriesValues[i-j]
        end
        push!(seriesValues, value)
    end
    return TimeArray(dates, seriesValues)
end

function MAPE(actual, forecast)
    mape = values(mean(abs.((actual .- forecast) ./ actual)))[1] * 100
    return mape
end

function MAE(actual, forecast)
    return values(mean(abs.(actual .- forecast)))[1]
end
@testset "Sarima predict" begin
    @testset "predict sarima without white noise" begin
        #p=2 P=1 trend =0.1
        ARcoeff = [-0.3, -0.2]
        seasCoeff = 0.4
        trend = 0.1
        ARseries = generateARseries(2, 12, ARcoeff, seasCoeff, trend, 1234, false)
        trainingSet, testingSet = splitTrainTest(ARseries)
        modelMSE = SARIMA(trainingSet, 2, 1, 0; seasonality = 12, P = 1, D = 0, Q = 0)
        Sarimax.fit!(modelMSE)
        print(modelMSE)
        forecastMSE = Sarimax.predict!(modelMSE; stepsAhead = length(testingSet))
        maeMSE = MAE(testingSet, forecastMSE)
        mapeMSE = MAPE(testingSet, forecastMSE)
        @test maeMSE ≈ 0 atol = 1e-3
        @test mapeMSE ≈ 0 atol = 1e-3

        #sin
        seriesSin = generateSeries(0, 12, 0, 0, 1234, false)
        trainingSin, testingSin = splitTrainTest(seriesSin)
        modelSin = SARIMA(trainingSin, 0, 0, 0; seasonality = 12, P = 1, D = 0, Q = 0)
        Sarimax.fit!(modelSin)
        print(modelSin)
        forecastSin = Sarimax.predict!(modelSin; stepsAhead = length(testingSet))
        maeSin = MAE(testingSin, forecastSin)
        @test maeSin ≈ 0 atol = 1e-3
    end

    @testset "auto predict without white noise" begin
        #p=2 P=1 trend =1
        ARcoeff = [0.3, 0.3]
        seasCoeff = 0.5
        trend = 0.1
        ARseries = generateARseries(2, 12, ARcoeff, seasCoeff, trend, 1234, false)
        trainingSet, testingSet = splitTrainTest(ARseries)
        modelAuto = Sarimax.auto(
            trainingSet;
            seasonality = 12,
            objectiveFunction = "mse",
            allowMean = false,
            allowDrift = true,
        )
        forecastAuto = Sarimax.predict!(modelAuto; stepsAhead = length(testingSet))
        mapeAuto = MAPE(testingSet, forecastAuto)
        maeAuto = MAE(testingSet, forecastAuto)
        # @test mapeAuto ≈ 0 atol = 1e-3
        # @test maeAuto ≈ 0 atol = 1e-3

        #p=2 sin seasonality trend=0.1
        seriesARSeas = generateSeries(2, 12, [0.3, 0.2], 0.1, 1234, false)
        trainingARSeas, testingARSeas = splitTrainTest(seriesARSeas)
        modelARSeasAuto =
            Sarimax.auto(trainingARSeas; seasonality = 12, objectiveFunction = "mse")
        forecastARSeasAuto = Sarimax.predict!(modelARSeasAuto; stepsAhead = 40)
        mapeARSeasAuto = MAPE(testingARSeas, forecastARSeasAuto)
        maeARSeasAuto = MAE(testingARSeas, forecastARSeasAuto)
        @test mapeARSeasAuto ≈ 0 atol = 1e-3
        @test maeARSeasAuto ≈ 0 atol = 1e-3
    end

    @testset "Sarima predict with exog" begin
        # Create a time series that is a linear Function and one exog that is also linear
        # use the auto function to fit the model split the train and test sets and compare
        # the forecast with the test set
        x = 1:200
        y::Vector{Float64} = [0.3 * i for i in x]
        exog::Vector{Float64} = [0.15 * i for i in x]
        series = TimeArray(Date(1991, 7, 1):Month(1):Date(2008, 2, 1), y)
        exogSeries = TimeArray(Date(1991, 7, 1):Month(1):Date(2008, 2, 1), exog)
        trainingSet, testingSet = splitTrainTest(series)
        modelExog = Sarimax.auto(
            trainingSet;
            exog = exogSeries,
            seasonality = 12,
            objectiveFunction = "elastic_net",
            alpha = 1.0,
            seasonalIntegrationTest = "ocsb"
        )
        forecastExog = Sarimax.predict!(modelExog; stepsAhead = length(testingSet))
        mapeExog = MAPE(testingSet, forecastExog)
        maeExog = MAE(testingSet, forecastExog)
        @test mapeExog ≈ 0 atol = 1e-1
        @test maeExog ≈ 0 atol = 1e-1
    end

    @testset "exog forecast uses matching horizon row" begin
        # Regression test for a bug where the exog forecast loop always used
        # the LAST horizon's exog row for every forecasted step instead of
        # the row matching that step.
        nTrain = 200
        nTotal = 205
        dates = Date(1991, 7, 1):Month(1):(Date(1991, 7, 1)+Month(nTotal - 1))
        xValues::Vector{Float64} = collect(1.0:nTotal)
        yValues::Vector{Float64} = 2.0 .* xValues[1:nTrain]
        trainSeries = TimeArray(collect(dates)[1:nTrain], yValues)
        exogSeries = TimeArray(collect(dates), xValues)

        model = SARIMA(trainSeries, exogSeries, 0, 0, 0; allowMean = false)
        fit!(model)
        forecast = Sarimax.predict!(model; stepsAhead = 5)

        forecastedValues = values(model.forecast)
        for i = 1:5
            @test forecastedValues[i] ≈ 2.0 * (nTrain + i) rtol = 1e-2
        end
    end

    @testset "forecast variances propagate through integration" begin
        # ARIMA(0,1,0): Var[h] = σ²·h on the original scale — the ψ-weights must
        # include the differencing operator (1-B)^d (1-B^s)^D.
        Random.seed!(42)
        varDates = Date(2000, 1, 1):Month(1):Date(2004, 12, 1)
        rw = TimeArray(collect(varDates), cumsum(randn(60)))
        mRW = SARIMA(rw, 0, 1, 0; allowMean = false)
        fit!(mRW)
        feRW = Sarimax.forecastErrors(mRW, 5)
        @test feRW ./ mRW.σ² ≈ [1.0, 2.0, 3.0, 4.0, 5.0] rtol = 1e-8

        # White noise: constant variance σ².
        wn = TimeArray(collect(varDates), randn(60))
        mWN = SARIMA(wn, 0, 0, 0; allowMean = true)
        fit!(mWN)
        feWN = Sarimax.forecastErrors(mWN, 4)
        @test feWN ./ mWN.σ² ≈ ones(4) rtol = 1e-8

        # Polynomial helper: (1-B)² = 1 - 2B + B²
        @test Sarimax.polynomialMultiplication([1.0, -1.0], [1.0, -1.0]) == [1.0, -2.0, 1.0]
    end

    @testset "short series seasonal predict" begin
        # Regression test: seasonal AR term in `predict` had no bounds guard,
        # so a short series (fewer than P*seasonality observations) threw a
        # BoundsError instead of simply skipping unavailable lags.
        Random.seed!(1234)
        dates = Date(1991, 7, 1):Month(1):Date(1991, 7, 1)+Month(14)
        y = TimeArray(collect(dates), randn(15))
        model = SARIMA(y, 0, 0, 0; seasonality = 12, P = 1)
        fit!(model)
        Sarimax.predict!(model; stepsAhead = 3)
        @test true
    end
end
