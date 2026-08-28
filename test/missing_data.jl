@testset "Missing data (stationary models)" begin
    @testset "AR(1) coefficient recovery and two-sided smoother" begin
        Random.seed!(20)
        n = 400
        ϕtrue = 0.6
        y = zeros(n)
        for t = 2:n
            y[t] = ϕtrue * y[t-1] + randn()
        end
        dts = Date(1990, 1, 1):Month(1):(Date(1990, 1, 1)+Month(n - 1))

        mFull = SARIMA(TimeArray(collect(dts), copy(y)), 1, 0, 0; allowMean = false)
        fit!(mFull)

        ym = copy(y)
        holes = [37, 50, 51, 120, 200, 201, 202, 260, 300, 333, 360, 390]
        for h in holes
            ym[h] = NaN
        end
        mMiss = SARIMA(TimeArray(collect(dts), ym), 1, 0, 0; allowMean = false)
        fit!(mMiss)

        # coefficient stays close to the complete-data fit
        @test abs(mMiss.ϕ[1] - mFull.ϕ[1]) < 0.05
        # Effective sample: it excludes the gaps AND depends on the mode's CONDITIONING.
        # Under the `:innovations` default the error sum starts at t = 1 and no observation
        # is conditioned out; under `:zeroed` the first one is (p = 1). Both regimes are
        # pinned deliberately, rather than a single magic number, because the conditioning
        # is the axis that moves.
        @test nobs(mMiss) == n - length(holes)
        mMissZeroed = SARIMA(TimeArray(collect(dts), ym), 1, 0, 0; allowMean = false)
        fit!(mMissZeroed; initialization = :zeroed)
        @test nobs(mMissZeroed) == (n - 1) - length(holes)
        @test mMiss.metadata["nMissing"] == length(holes)
        @test length(residuals(mMiss)) == nobs(mMiss)

        # an isolated gap (index 120) is imputed by the exact two-sided AR(1)
        # smoother:  ỹ_t = ϕ (y_{t-1} + y_{t+1}) / (1 + ϕ²)
        imputed = values(mMiss.y)[120]
        smoother = mMiss.ϕ[1] * (y[119] + y[121]) / (1 + mMiss.ϕ[1]^2)
        @test isapprox(imputed, smoother; atol = 1e-3)

        # downstream quantities are finite and forecasting still works
        @test isfinite(loglike(mMiss))
        @test isfinite(aic(mMiss))
        predict!(mMiss; stepsAhead = 3)
        @test all(isfinite, values(mMiss.forecast))
    end

    @testset "MA(1) with gaps" begin
        Random.seed!(5)
        n = 300
        e = randn(n + 1)
        θtrue = 0.5
        y = [e[t+1] + θtrue * e[t] for t = 1:n]
        dts = Date(1990, 1, 1):Month(1):(Date(1990, 1, 1)+Month(n - 1))
        ym = copy(y)
        for h in [40, 41, 150, 220, 221, 270]
            ym[h] = NaN
        end
        m = SARIMA(TimeArray(collect(dts), ym), 0, 0, 1; allowMean = false)
        fit!(m)
        @test abs(m.θ[1] - θtrue) < 0.2
        # ver nota sobre condicionamento no testset AR(1) acima
        @test nobs(m) == n - 6
        mZeroed = SARIMA(TimeArray(collect(dts), ym), 0, 0, 1; allowMean = false)
        fit!(mZeroed; initialization = :zeroed)
        @test nobs(mZeroed) == (n - 1) - 6
    end

    @testset "unsupported combinations throw" begin
        Random.seed!(1)
        n = 60
        dts = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        ym = cumsum(randn(n))
        ym[10] = NaN

        # differencing not yet supported with gaps
        md = SARIMA(TimeArray(collect(dts), copy(ym)), 1, 1, 0; allowMean = false)
        @test_throws ArgumentError fit!(md)

        # auto rejects missing data
        @test_throws ArgumentError auto(TimeArray(collect(dts), copy(ym)); seasonality = 1)

        # exogenous + missing not yet supported
        exog = TimeArray(collect(dts), collect(1.0:n))
        mx = SARIMA(TimeArray(collect(dts), copy(ym)), exog, 1, 0, 0; allowMean = false)
        @test_throws ArgumentError fit!(mx)
    end
end
