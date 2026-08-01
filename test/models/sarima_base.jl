@testset "Base functions of Sarima model" begin
    airPassengers = load_dataset(AIR_PASSENGERS)
    airPassengersLog = log.(airPassengers)

    modeloLog = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)
    io = IOBuffer()
    show(io, modeloLog)
    output = String(take!(io))
    @test output == "SARIMA(3,0,1)(1,1,1)[12] | not fitted"

    @test_throws  Sarimax.InvalidParametersCombination SARIMA(airPassengersLog)
    @test_throws  Sarimax.InvalidParametersCombination SARIMA(airPassengersLog; seasonalMACoefficients=[0.9])
    @test_throws  Sarimax.InvalidParametersCombination SARIMA(airPassengersLog; exogCoefficients=[0.9])
    @test_throws  Sarimax.InvalidParametersCombination SARIMA(airPassengersLog; exog=airPassengersLog, exogCoefficients=[0.9,0.1,0.3])
    @test_throws  Sarimax.InvalidParametersCombination SARIMA(airPassengersLog; alpha=-1.0)
    @test_throws  Sarimax.InvalidParametersCombination SARIMA(airPassengersLog; alpha=10.0)
    @test_throws  Sarimax.InvalidParametersCombination SARIMA(airPassengersLog; lambda=-1.0)


    initModel = SARIMA(airPassengersLog; exog=airPassengersLog, seasonalARCoefficients=[0.5], seasonality=12 ,exogCoefficients=[0.5])
    fit!(initModel;automaticExogDifferentiation=true)
    # Scale-dependent provided values round-trip through the internal rescaling
    # (value/yScale on entry, *yScale on exit), so they are preserved to 1 ulp,
    # not bit-for-bit; dimensionless Φ passes through untouched and stays exact.
    @test initModel.exogCoefficients[1] ≈ 0.5 atol = 1e-12
    @test initModel.Φ[1] == 0.5
end
