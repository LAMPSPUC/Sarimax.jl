@testset "Base functions of Sarima model" begin
    airPassengers = loadDataset(AIR_PASSENGERS)
    airPassengersLog = log.(airPassengers)

    modeloLog = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)
    io = IOBuffer()
    show(io, modeloLog)
    output = String(take!(io))
    @test "SARIMA (3, 0 ,1)(1, 1 ,1 s=12) with zero mean and non zero drift" == output

    @test_throws  Sarimax.InvalidParametersCombination SARIMA(airPassengersLog)
    @test_throws  Sarimax.InvalidParametersCombination SARIMA(airPassengersLog; seasonalMACoefficients=[0.9])
    @test_throws  Sarimax.InvalidParametersCombination SARIMA(airPassengersLog; exogCoefficients=[0.9])
    @test_throws  Sarimax.InvalidParametersCombination SARIMA(airPassengersLog; exog=airPassengersLog, exogCoefficients=[0.9,0.1,0.3])

    initModel = SARIMA(airPassengersLog; exog=airPassengersLog, seasonalARCoefficients=[0.5], seasonality=12 ,exogCoefficients=[0.5])
    fit!(initModel;automaticExogDifferentiation=true)
    @test initModel.exogCoefficients[1] == 0.5
    @test initModel.Φ[1] == 0.5
end
