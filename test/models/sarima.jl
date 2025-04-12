@testset "SARIMA" begin

    @test hasFitMethods(SARIMAModel)
    @test hasHyperparametersMethods(SARIMAModel)

    airPassengers = loadDataset(AIR_PASSENGERS)
    airPassengersLog = log.(airPassengers)

    modeloLog = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)
    @test Sarimax.typeofModelElements(modeloLog) == Float64
    @test Sarimax.isFitted(modeloLog) == false
    @test getHyperparametersNumber(modeloLog) == 8
    fit!(modeloLog)
    predict!(modeloLog; stepsAhead = 10, displayConfidenceIntervals = true)
    @test size(modeloLog.forecast, 1) == 10
    @test size(modeloLog.forecast, 2) == 3

    modeloLogMAE = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)
    @test Sarimax.typeofModelElements(modeloLogMAE) == Float64
    @test Sarimax.isFitted(modeloLogMAE) == false
    @test getHyperparametersNumber(modeloLogMAE) == 8
    fit!(modeloLogMAE; objectiveFunction = "mae")
    predict!(modeloLogMAE; stepsAhead = 10, displayConfidenceIntervals = true)
    @test size(modeloLogMAE.forecast, 1) == 10
    @test size(modeloLogMAE.forecast, 2) == 3


    @test Sarimax.isFitted(modeloLog) == true
    @test mean(modeloLog.ϵ) ≈ 0 atol = 1e-1

    autoModelML = auto(airPassengersLog; seasonality = 12, objectiveFunction = "ml")
    @test autoModelML.d == 0 # Output of forecast package in R
    @test autoModelML.D == 1 # Output of forecast package in R

    simulation = simulate(modeloLog, 10, 301)
    @test length(simulation) == 301
    @test length(simulation[1]) == 10

    gdpc = loadDataset(GDPC1)
    modelGDPC1 = auto(
        gdpc;
        seasonality = 4,
        objectiveFunction = "mse",
        assertStationarity = true,
        assertInvertibility = true,
    )
    @test modelGDPC1.d == 2 # Output of forecast package in R
    @test modelGDPC1.D == 0 # Output of forecast package in R

    nrou = loadDataset(NROU)
    modelNROU = auto(
        nrou;
        seasonality = 4,
        objectiveFunction = "mse",
        assertStationarity = true,
        assertInvertibility = true,
    )
    @test modelNROU.d == 2 # Output of forecast package in R
    @test modelNROU.D == 0 # Output of forecast package in R
end

@testset "ARMA to MA" begin
    airPassengers = loadDataset(AIR_PASSENGERS)
    airPassengersLog = log.(airPassengers)

    firstModel = SARIMA(airPassengersLog; arCoefficients = [-0.2, 0.0, 0.5])
    firstCoefficients = toMA(firstModel, 5)
    @test firstCoefficients ≈ [-0.2, 0.04, 0.492, -0.1984, 0.05968] atol = 1e-3

    secondModel =
        SARIMA(airPassengersLog; arCoefficients = [0.2, -0.1], maCoefficients = [0.5])
    secondCoefficients = toMA(secondModel, 5)
    @test secondCoefficients ≈ [0.7, 0.0399999, -0.062, -0.0164, 0.00292] atol = 1e-3

    thirdModel = SARIMA(
        airPassengersLog;
        arCoefficients = [0.2, -0.1],
        maCoefficients = [0.5],
        seasonalARCoefficients = [0.1],
        seasonality = 12,
    )
    thirdCoefficients = toMA(thirdModel, 15)
    data = [
        7.000000e-01,
        4.000000e-02,
        -6.200000e-02,
        -1.640000e-02,
        2.920000e-03,
        2.224000e-03,
        1.528000e-04,
        -1.918400e-04,
        -5.364800e-05,
        8.454400e-06,
        7.055680e-06,
        1.000006e-01,
        8.999941e-02,
        1.199982e-02,
        -1.279998e-02,
    ]

    @test thirdCoefficients ≈ data atol = 1e-3
end
