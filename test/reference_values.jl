@testset "Cross-implementation reference values (R CSS)" begin
    # Reference values generated 2026-07-12 with R stats::arima(method = "CSS")
    # on log(datasets/airpassengers.csv), frequency 12 (R 4.x, macOS arm64):
    #
    #   arima(y, order=c(0,1,1), seasonal=list(order=c(0,1,1), period=12), method="CSS")
    #     theta = -0.7872984190   Theta = -0.7140764458   sigma2 = 0.0041414547
    #   arima(y, order=c(1,1,1), method="CSS")
    #     ar    =  0.3227790019   ma    = -0.8242122152   sigma2 = 0.0261163050
    #
    # R's CSS convention conditions only on the AR-side lags and warm-starts the
    # MA recursion from the beginning of the differenced sample — this is the
    # `initialization = :warmup` mode. (The default `:zeroed` mode drops the
    # first q + s·Q differenced observations instead; both are legitimate CSS
    # variants and agree asymptotically.)
    airPassengersLog = log.(load_dataset(AIR_PASSENGERS))

    @testset "airline (0,1,1)(0,1,1)12 vs R" begin
        model = SARIMA(airPassengersLog, 0, 1, 1;
            seasonality = 12, P = 0, D = 1, Q = 1, allowMean = false)
        fit!(model; initialization = :warmup)
        @test model.θ[1] ≈ -0.7872984190 atol = 1e-4
        @test model.Θ[1] ≈ -0.7140764458 atol = 1e-4
        # σ² conventions differ by the denominator: R uses RSS/n, the package
        # uses the df-adjusted RSS/(n-K+1). Compare on R's convention:
        rss = sum(abs2, residuals(model))
        @test rss / nobs(model) ≈ 0.0041414547 atol = 1e-5
        @test nobs(model) == 191   # R n.used for this spec
    end

    @testset "ARIMA(1,1,1) vs R" begin
        model = SARIMA(airPassengersLog, 1, 1, 1; allowMean = false)
        fit!(model; initialization = :warmup)
        @test model.ϕ[1] ≈ 0.3227790019 atol = 1e-4
        @test model.θ[1] ≈ -0.8242122152 atol = 1e-4
        rss = sum(abs2, residuals(model))
        @test rss / nobs(model) ≈ 0.0261163050 atol = 1e-5
    end

    @testset "initialization modes are distinct but same family" begin
        zeroed = SARIMA(airPassengersLog, 0, 1, 1;
            seasonality = 12, P = 0, D = 1, Q = 1, allowMean = false)
        # Explicit by design: this testset exists to contrast `:zeroed` with `:warmup`, and
        # the assertion `nobs(warmup) - nobs(zeroed) == 13` is about the conditioning
        # `:zeroed` applies. Under the `:innovations` default (lb = 1, no observation
        # discarded) relying on the default here would measure something else.
        fit!(zeroed; initialization = :zeroed)
        warmup = SARIMA(airPassengersLog, 0, 1, 1;
            seasonality = 12, P = 0, D = 1, Q = 1, allowMean = false)
        fit!(warmup; initialization = :warmup)
        # different conditioning ⇒ different (but neighboring) estimates
        @test zeroed.θ[1] != warmup.θ[1]
        @test abs(zeroed.θ[1] - warmup.θ[1]) < 0.1
        # warmup uses the full differenced sample; zeroed drops q + s·Q obs
        @test nobs(warmup) - nobs(zeroed) == 13
        # invalid mode
        bad = SARIMA(airPassengersLog, 0, 1, 1; seasonality = 12, P = 0, D = 1, Q = 1)
        @test_throws ArgumentError fit!(bad; initialization = :exact)
    end
end
