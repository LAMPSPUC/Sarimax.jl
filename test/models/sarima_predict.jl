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
            objectiveFunction = "lasso",
            seasonalIntegrationTest = "ch"
        )
        forecastExog = Sarimax.predict!(modelExog; stepsAhead = length(testingSet))
        mapeExog = MAPE(testingSet, forecastExog)
        maeExog = MAE(testingSet, forecastExog)
        @test mapeExog ≈ 0 atol = 1e-1
        @test maeExog ≈ 0 atol = 1e-1
    end
end
