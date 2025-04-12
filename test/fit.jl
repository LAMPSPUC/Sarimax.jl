@testset "Fit" begin
    mutable struct ARIMA_TEST <: Sarimax.SarimaxModel end
    @testset "hasFitMethods" begin
        @test hasFitMethods(SARIMAModel)
        @test !hasFitMethods(ARIMA_TEST)
    end

    @testset "hasHyperparametersMethods" begin
        @test hasHyperparametersMethods(SARIMAModel)
        @test !hasHyperparametersMethods(ARIMA_TEST)
    end

    @testset "aic_function" begin
        @test aic(2, 3.0) ≈ -2.0
        @test aic(3, 4.0) ≈ -2.0

        # Test with Float16
        @test aic(2, Float16(3.0)) ≈ Float16(-2.0)
        @test aic(3, Float16(4.0)) ≈ Float16(-2.0)
    end

    @testset "aicc_function" begin
        @test aicc(10, 2, 3.0) ≈ -0.2857142857142858
        @test aicc(10, 3, 4.0) ≈ 2.0

        # Test with Float16
        @test aicc(10, 2, Float16(3.0)) ≈ Float16(-0.2857142857142858)
        @test aicc(10, 3, Float16(4.0)) ≈ Float16(2.0)
    end

    @testset "bic_function" begin
        @test bic(10, 2, 3.0) ≈ -1.3948298140119082
        @test bic(10, 3, 4.0) ≈ -1.0922447210178623

        # Test with Float16
        @test bic(10, 2, Float16(3.0)) ≈ Float16(-1.3948298140119082)
        @test bic(10, 3, Float16(4.0)) ≈ Float16(-1.0922447210178623)
    end

    @testset "informationCriteriaModel" begin
        @test_throws MissingMethodImplementation begin
            aic(ARIMA_TEST())
        end

        @test_throws MissingMethodImplementation begin
            aicc(ARIMA_TEST())
        end

        @test_throws MissingMethodImplementation begin
            bic(ARIMA_TEST())
        end

        airPassengers = loadDataset(AIR_PASSENGERS)
        airPassengersLog = log.(airPassengers)
        testModel = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)
        fit!(testModel)
        @test aic(testModel) ≈ -1063.1519532534248
        @test aicc(testModel) ≈ -1062.3650680075232
        @test bic(testModel) ≈ -1037.0919902772025
    end


end
