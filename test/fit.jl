@testset "Fit" begin
    mutable struct ARIMA_TEST <: Sarimax.SarimaxModel end
    @testset "has_fit_methods" begin
        @test has_fit_methods(SARIMAModel)
        @test !has_fit_methods(ARIMA_TEST)
    end

    @testset "has_hyperparameters_methods" begin
        @test has_hyperparameters_methods(SARIMAModel)
        @test !has_hyperparameters_methods(ARIMA_TEST)
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

        airPassengers = load_dataset(AIR_PASSENGERS)
        airPassengersLog = log.(airPassengers)
        testModel = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)
        fit!(testModel)
        # Criteria are scored with `criterionLoglike`: the EXACT Gaussian log-likelihood
        # when computable (the case here), CSS fallback otherwise. `K = ncoef + 1` counts
        # all declared parameters (+σ²), and the small-sample correction / BIC factor use
        # the sample size of the likelihood actually used (`T` on the exact path — NOT
        # `length(observedResiduals)`, which discounts the CSS conditioning).
        K = get_hyperparameters_number(testModel)
        @test K == 8
        ll, n, usedExact = Sarimax.criterionLoglikeAndN(testModel)
        @test usedExact
        @test n == length(values(differentiate(airPassengersLog, 0, 1, 12)))
        @test aic(testModel) ≈ 2 * K - 2 * ll atol = 1e-10
        @test aicc(testModel) ≈ aic(testModel) + (2 * K * K + 2 * K) / (n - K - 1) atol = 1e-10
        @test bic(testModel) ≈ K * log(n) - 2 * ll atol = 1e-10
        # the exact likelihood is never below the CSS one evaluated at the same point on
        # fewer observations by more than the determinant term; what matters here is that
        # the public accessors and the criterion core agree with each other
        @test loglike(testModel) ≉ ll  # CSS accessor and exact criterion are distinct quantities
    end


end
