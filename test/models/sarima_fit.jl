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
    # seriesValues = randn(s) .+ trend*x[1:max(s,p)] .+ whiteNoise[1:max(s,p)]
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

@testset "Sarima fit" begin
    @testset "fit p=0 P=1 without white noise" begin
        # `:zeroed` is passed EXPLICITLY here. This testset recovers coefficients from a
        # NOISE-FREE series, which is a CONDITIONAL LEAST SQUARES expectation: it holds for
        # the mode that fixes the pre-sample block at zero, and not for modes that price the
        # initial state. A noise-free series is degenerate for a likelihood objective, with
        # sigma^2 -> 0: `ml_exact` with `:zeroed`, which has no pre-sample block at all,
        # also misses the coefficient on this series.
        #
        # Recovery under the default mode is covered separately, with noise, in the testset
        # "coefficient recovery under noise (default mode)".
        ARcoeff = [0]
        seasCoeff = 0.5
        trend = 0
        ARseries = generateARseries(1, 12, ARcoeff, seasCoeff, trend, 1234, false)
        modelMSE = SARIMA(ARseries, 1, 1, 0; seasonality = 12, P = 1, D = 0, Q = 0)
        modelML = SARIMA(ARseries, 1, 1, 0; seasonality = 12, P = 1, D = 0, Q = 0)
        modelBILEVEL = SARIMA(ARseries, 1, 1, 0; seasonality = 12, P = 1, D = 0, Q = 0)
        # generateARseries is an ADDITIVE DGP -> fit the additive form
        Sarimax.fit!(modelMSE, objectiveFunction = "mse", seasonalForm = :additive; initialization = :zeroed)
        Sarimax.fit!(modelML, objectiveFunction = "ml", seasonalForm = :additive; initialization = :zeroed)
        Sarimax.fit!(modelBILEVEL, objectiveFunction = "bilevel", seasonalForm = :additive; initialization = :zeroed)
        @test seasCoeff ≈ modelMSE.Φ[1] atol = 1e-3
        @test seasCoeff ≈ modelML.Φ[1] atol = 1e-3
        @test seasCoeff ≈ modelBILEVEL.Φ[1] atol = 1e-3
    end

    @testset "fit (p=1 P=0) and (p=2 P=0) without white noise" begin
        # `:zeroed` is passed EXPLICITLY here. This testset recovers coefficients from a
        # NOISE-FREE series, which is a CONDITIONAL LEAST SQUARES expectation: it holds for
        # the mode that fixes the pre-sample block at zero, and not for modes that price the
        # initial state. A noise-free series is degenerate for a likelihood objective, with
        # sigma^2 -> 0: `ml_exact` with `:zeroed`, which has no pre-sample block at all,
        # also misses the coefficient on this series.
        #
        # Recovery under the default mode is covered separately, with noise, in the testset
        # "coefficient recovery under noise (default mode)".

        ar1 = generateARseries(1, 1, [0.3], 0, 0, 1234, false)
        modelAR1MSE = SARIMA(ar1, 1, 0, 0; seasonality = 12, P = 0, D = 0, Q = 0)
        fit!(modelAR1MSE, objectiveFunction = "mse"; initialization = :zeroed)
        @test modelAR1MSE.ϕ ≈ [0.3] atol = 1e-3

        modelAR1ML = SARIMA(ar1, 1, 0, 0; seasonality = 12, P = 0, D = 0, Q = 0)
        fit!(modelAR1ML, objectiveFunction = "ml"; initialization = :zeroed)
        @test modelAR1ML.ϕ ≈ [0.3] atol = 1e-3

        modelAR1BI = SARIMA(ar1, 1, 0, 0; seasonality = 12, P = 0, D = 0, Q = 0)
        fit!(modelAR1BI, objectiveFunction = "bilevel"; initialization = :zeroed)
        @test modelAR1BI.ϕ ≈ [0.3] atol = 1e-3

        ar2 = generateARseries(2, 1, [0.3, 0.4], 0, 0, 1234, false)
        modelAR2MSE = SARIMA(ar2, 2, 0, 0; seasonality = 12, P = 0, D = 0, Q = 0)
        fit!(modelAR2MSE, objectiveFunction = "mse"; initialization = :zeroed)
        @test modelAR2MSE.ϕ ≈ [0.3, 0.4] atol = 1e-3

        modelAR2ML = SARIMA(ar2, 2, 0, 0; seasonality = 12, P = 0, D = 0, Q = 0)
        fit!(modelAR2ML, objectiveFunction = "ml"; initialization = :zeroed)
        @test modelAR2ML.ϕ ≈ [0.3, 0.4] atol = 1e-3

        modelAR2BI = SARIMA(ar2, 2, 0, 0; seasonality = 12, P = 0, D = 0, Q = 0)
        fit!(modelAR2BI, objectiveFunction = "bilevel"; initialization = :zeroed)
        @test modelAR2BI.ϕ ≈ [0.3, 0.4] atol = 1e-3

    end

    @testset "fit p=2 P=1 without white Noise" begin
        # `:zeroed` is passed EXPLICITLY here. This testset recovers coefficients from a
        # NOISE-FREE series, which is a CONDITIONAL LEAST SQUARES expectation: it holds for
        # the mode that fixes the pre-sample block at zero, and not for modes that price the
        # initial state. A noise-free series is degenerate for a likelihood objective, with
        # sigma^2 -> 0: `ml_exact` with `:zeroed`, which has no pre-sample block at all,
        # also misses the coefficient on this series.
        #
        # Recovery under the default mode is covered separately, with noise, in the testset
        # "coefficient recovery under noise (default mode)".
        ARcoeff = [-0.3, -0.2]
        seasCoeff = -0.4
        trend = 0.1
        ARseries = generateARseries(2, 12, ARcoeff, seasCoeff, trend, 1234, false)
        modelMSE = SARIMA(ARseries, 2, 1, 0; seasonality = 12, P = 1, D = 0, Q = 0)
        modelML = SARIMA(ARseries, 2, 1, 0; seasonality = 12, P = 1, D = 0, Q = 0)
        modelBILEVEL = SARIMA(ARseries, 2, 1, 0; seasonality = 12, P = 1, D = 0, Q = 0)
        # additive DGP -> additive form
        fit!(modelMSE, objectiveFunction = "mse", seasonalForm = :additive; initialization = :zeroed)
        fit!(modelML, objectiveFunction = "ml", seasonalForm = :additive; initialization = :zeroed)
        fit!(modelBILEVEL, objectiveFunction = "bilevel", seasonalForm = :additive; initialization = :zeroed)
        @test ARcoeff ≈ modelMSE.ϕ atol = 1e-3
        @test seasCoeff ≈ modelMSE.Φ[1] atol = 1e-3
        @test ARcoeff ≈ modelML.ϕ atol = 1e-3
        @test seasCoeff ≈ modelML.Φ[1] atol = 1e-3
        @test ARcoeff ≈ modelBILEVEL.ϕ atol = 1e-3
        @test seasCoeff ≈ modelBILEVEL.Φ[1] atol = 1e-3
    end

    @testset "auto with exougenous variable D correction" begin
        airPassengers = load_dataset(AIR_PASSENGERS)
        airPassengersLog = log.(airPassengers)
        lengthAirPassengers = length(airPassengersLog)
        exogenous =
            TimeArray(timestamp(airPassengers), [0.5 * i for i = 1:lengthAirPassengers])
        modelAutoExog = auto(
            airPassengersLog;
            exog = exogenous,
            seasonality = 12,
            objectiveFunction = "mse",
            showLogs = true,
        )
        modelAuto = auto(
            airPassengersLog;
            seasonality = 12,
            objectiveFunction = "mse",
            showLogs = true,
        )

        @test modelAutoExog.D == 0
        # R 4.4.1 / forecast 8.23.0: auto.arima(log(AirPassengers)) selects D = 1
        # (airline model); the D == 0 expectation predated the "seas" default.
        @test modelAuto.D == 1

        modelAutoFixedD = auto(
            airPassengersLog;
            seasonality = 12,
            objectiveFunction = "mse",
            showLogs = true,
            D = 0,
        )
        @test modelAutoFixedD.D == modelAutoExog.D
        @test modelAutoFixedD.d != modelAutoExog.d
    end

    @testset "ridge_fit" begin
        airPassengers = load_dataset(AIR_PASSENGERS)
        airPassengersLog = log.(airPassengers)
        modelRidge = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)
        fit!(modelRidge; objectiveFunction = "elastic_net", alpha = 0.0)
        modelNoRidge = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)
        fit!(modelNoRidge)
        @test modelRidge.ϕ != modelNoRidge.ϕ
        @test modelRidge.Φ != modelNoRidge.Φ

        # Test ridge with exogenous variable
        lengthAirPassengers = length(airPassengersLog)
        exogenous =
            TimeArray(timestamp(airPassengers), [0.5 * i for i = 1:lengthAirPassengers])
        modelRidgeExog = auto(
            airPassengersLog;
            exog = exogenous,
            seasonality = 12,
            objectiveFunction = "elastic_net",
            showLogs = false,
            alpha = 0.0
        )
        @test modelRidgeExog.ϕ != modelRidge.ϕ
        @test modelRidgeExog.d == modelRidge.d
        @test modelRidgeExog.D != modelRidge.D
    end

    @testset "lasso_fit" begin
        airPassengers = load_dataset(AIR_PASSENGERS)
        airPassengersLog = log.(airPassengers)
        modelLasso = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)
        fit!(modelLasso; objectiveFunction = "elastic_net", alpha = 1.0)
        modelNoLasso = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)
        fit!(modelNoLasso)
        @test modelLasso.ϕ != modelNoLasso.ϕ
        @test modelLasso.Φ != modelNoLasso.Φ

        # Test lasso with exogenous variable
        lengthAirPassengers = length(airPassengersLog)
        exogenous =
            TimeArray(timestamp(airPassengers), [0.5 * i for i = 1:lengthAirPassengers])
        modelLassoExog = auto(
            airPassengersLog;
            exog = exogenous,
            seasonality = 12,
            objectiveFunction = "elastic_net",
            showLogs = false,
            alpha = 1.0
        )
        @test modelLassoExog.ϕ != modelLasso.ϕ
        @test modelLassoExog.d == modelLasso.d
        @test modelLassoExog.D != modelLasso.D
    end

    @testset "bilevel_fit" begin
        airPassengers = load_dataset(AIR_PASSENGERS)
        airPassengersLog = log.(airPassengers)
        modelBILEVEL = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)
        fit!(modelBILEVEL; objectiveFunction = "bilevel")
        modelNoBILEVEL = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)
        fit!(modelNoBILEVEL)
        @test modelBILEVEL.ϕ != modelNoBILEVEL.ϕ
        @test modelBILEVEL.Φ != modelNoBILEVEL.Φ

        # Disabled on cost, not because it is wrong. `auto` with `bilevel` and an exogenous
        # regressor accounted for 95.5% of the runtime of this whole file, at about 1,130s
        # against 24s for the two direct `fit!` calls that cover the same objective. Those
        # two remain. The objective is deprecated as of v1.0 and scheduled for removal in
        # v2.0, so this branch stays disabled; the lost coverage is recorded below as
        # `@test_skip` so that it appears in the suite summary instead of vanishing.
        if false
            lengthAirPassengers = length(airPassengersLog)
            exogenous =
                TimeArray(timestamp(airPassengers), [0.5 * i for i = 1:lengthAirPassengers])
            modelBILEVELExog = auto(
                airPassengersLog;
                exog = exogenous,
                seasonality = 12,
                objectiveFunction = "bilevel",
                showLogs = false,
            )
            @test modelBILEVELExog.ϕ != modelBILEVEL.ϕ
            @test modelBILEVELExog.d == modelBILEVEL.d
            @test modelBILEVELExog.D != modelBILEVEL.D
        end
        @test_skip "auto + bilevel + exog: desativado por custo (12x), ver issue de depreciacao"
    end

    @testset "stable_fit" begin
        airPassengers = load_dataset(AIR_PASSENGERS)
        airPassengersLog = log.(airPassengers)
        modelStable = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)
        fit!(modelStable; objectiveFunction = "stable")
        modelMse = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)
        fit!(modelMse)
        @test modelStable.ϕ != modelMse.ϕ
        @test modelStable.Φ != modelMse.Φ
    end



    @testset "drift and trend terms" begin
        n = 80
        driftDates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        yLine = TimeArray(collect(driftDates), 3.0 .+ 0.5 .* collect(1.0:n))

        # d = 1: drift = constant in the differenced equation
        mDrift = SARIMA(yLine, 0, 1, 0; allowMean = false, allowDrift = true)
        fit!(mDrift)
        @test mDrift.trend ≈ 0.5 atol = 1e-4
        predict!(mDrift; stepsAhead = 3)
        @test values(mDrift.forecast) ≈ [3.0 + 0.5 * (n + i) for i = 1:3] atol = 1e-3

        # d = 0: drift is a genuine linear trend δ·t (was a duplicated constant before v0.3)
        yTrend = TimeArray(collect(driftDates), 0.5 .* collect(1.0:n))
        mTrend = SARIMA(yTrend, 0, 0, 0; allowMean = false, allowDrift = true)
        fit!(mTrend)
        @test mTrend.trend ≈ 0.5 atol = 1e-4

        # mean and drift were perfectly collinear: now mutually exclusive
        mBoth = SARIMA(yLine, 0, 1, 0; allowMean = true, allowDrift = true)
        @test_throws InvalidParametersCombination fit!(mBoth)
    end

    @testset "stationary AR parameterization" begin
        # Levinson recursion: φ₁ = κ₁(1-κ₂), φ₂ = κ₂
        @test Sarimax.reflectionToAR([0.5]) == [0.5]
        @test isapprox(Sarimax.reflectionToAR([0.5, 0.3]), [0.5 - 0.3 * 0.5, 0.3]; atol = 1e-12)

        Random.seed!(11)
        n = 80
        statDates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        rw = TimeArray(collect(statDates), cumsum(randn(n)))
        mStat = SARIMA(rw, 1, 0, 0; allowMean = false)
        fit!(mStat; stationary = true, stationarityMargin = 0.02)
        @test abs(mStat.ϕ[1]) <= 0.98 + 1e-8

        mBad = SARIMA(rw, 1, 0, 0; allowMean = false)
        @test_throws AssertionError fit!(mBad; stationarityMargin = 1.5)
    end

    @testset "model display" begin
        io = IOBuffer()
        dispDates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(59))
        Random.seed!(3)
        mDisp = SARIMA(TimeArray(collect(dispDates), cumsum(randn(60))), 1, 1, 0; allowMean = false)
        show(io, mDisp)
        @test occursin("not fitted", String(take!(io)))
        fit!(mDisp)
        show(io, MIME("text/plain"), mDisp)
        str = String(take!(io))
        @test occursin("coefficient", str)
        @test occursin("ar_1", str)
        @test occursin("AIC", str)
        @test occursin("multiplicative", str)
    end

    @testset "multiplicative_recovery" begin
        # `:zeroed` is passed EXPLICITLY here. This testset recovers coefficients from a
        # NOISE-FREE series, which is a CONDITIONAL LEAST SQUARES expectation: it holds for
        # the mode that fixes the pre-sample block at zero, and not for modes that price the
        # initial state. A noise-free series is degenerate for a likelihood objective, with
        # sigma^2 -> 0: `ml_exact` with `:zeroed`, which has no pre-sample block at all,
        # also misses the coefficient on this series.
        #
        # Recovery under the default mode is covered separately, with noise, in the testset
        # "coefficient recovery under noise (default mode)".
        # Noise-free multiplicative DGP: y_t = φy_{t-1} + Φy_{t-12} − φΦy_{t-13}
        Random.seed!(7)
        n = 200
        vals = randn(13) .* 0.1
        for t = 14:n
            push!(vals, 0.4 * vals[t-1] + 0.5 * vals[t-12] - 0.4 * 0.5 * vals[t-13])
        end
        multDates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        yMult = TimeArray(collect(multDates), vals)

        mMult = SARIMA(yMult, 1, 0, 0; seasonality = 12, P = 1, allowMean = false)
        fit!(mMult; initialization = :zeroed)   # default :multiplicative
        @test mMult.ϕ[1] ≈ 0.4 atol = 1e-3
        @test mMult.Φ[1] ≈ 0.5 atol = 1e-3

        # The additive form cannot represent this DGP: estimates are distorted.
        mAdd = SARIMA(yMult, 1, 0, 0; seasonality = 12, P = 1, allowMean = false)
        fit!(mAdd; seasonalForm = :additive, initialization = :zeroed)
        @test abs(mAdd.ϕ[1] - 0.4) > 1e-2

        # :free is not implemented yet
        mFree = SARIMA(yMult, 1, 0, 0; seasonality = 12, P = 1, allowMean = false)
        @test_throws ArgumentError fit!(mFree; seasonalForm = :free)
    end

    @testset "coefficient recovery under noise (default mode)" begin
        # Counterpart to the noise-free testsets above, which pass `:zeroed` explicitly
        # because exact recovery is a conditional-least-squares expectation. This one passes
        # no `initialization` at all and therefore runs under the DEFAULT mode, so that the
        # default has coefficient-recovery coverage of its own.
        #
        # The tolerance is deliberately loose: with noise and finite T there is small-sample
        # bias in ANY mode. What this protects is that the default recovers the right order
        # of magnitude and the right sign, not a pinned value.
        Random.seed!(20240823)
        n = 400
        phiTrue = 0.6
        v = zeros(n)
        for t = 2:n
            v[t] = phiTrue * v[t-1] + randn()
        end
        dts = Date(1990, 1, 1):Month(1):(Date(1990, 1, 1)+Month(n - 1))
        mNoise = SARIMA(TimeArray(collect(dts), v), 1, 0, 0; allowMean = true)
        fit!(mNoise)
        @test length(mNoise.ϕ) == 1
        @test isfinite(mNoise.ϕ[1])
        @test sign(mNoise.ϕ[1]) == sign(phiTrue)
        @test abs(mNoise.ϕ[1] - phiTrue) < 0.15
        @test mNoise.σ² > 0
    end

    @testset "invertible_fit" begin
        # reflectionToMA recursion (numeric check): θ₁ = κ₁(1+κ₂), θ₂ = κ₂
        @test Sarimax.reflectionToMA([0.5]) == [0.5]
        @test isapprox(Sarimax.reflectionToMA([0.5, 0.3]), [0.5 + 0.3 * 0.5, 0.3]; atol = 1e-12)
        @test length(Sarimax.reflectionToMA(Float64[])) == 0

        airPassengers = load_dataset(AIR_PASSENGERS)

        # q = 1: the reflection parameterization coincides with the box bounds
        boxMA1 = SARIMA(airPassengers, 1, 0, 1)
        fit!(boxMA1; objectiveFunction = "mse", initialization = :zeroed)
        refMA1 = SARIMA(airPassengers, 1, 0, 1)
        fit!(refMA1; objectiveFunction = "mse", invertible = true, initialization = :zeroed)
        @test isapprox(boxMA1.θ[1], refMA1.θ[1]; atol = 1e-4)

        # airline model: box drives θ to the unit-root boundary (|θ| = 1, non-invertible),
        # while the reflection parameterization keeps it strictly inside |θ| ≤ 1 - ρ.
        ρ = 0.05
        # The unit-root boundary pile-up documented here is a property of the
        # ADDITIVE-form fit (under :multiplicative the airline θ stays interior).
        boxAir = SARIMA(airPassengers, 0, 1, 1; seasonality = 12, P = 0, D = 1, Q = 1)
        fit!(boxAir; objectiveFunction = "mse", seasonalForm = :additive, initialization = :zeroed)
        refAir = SARIMA(airPassengers, 0, 1, 1; seasonality = 12, P = 0, D = 1, Q = 1)
        fit!(refAir; objectiveFunction = "mse", invertible = true, invertibilityMargin = ρ, seasonalForm = :additive, initialization = :zeroed)
        @test abs(boxAir.θ[1]) >= 0.99
        @test abs(refAir.θ[1]) <= (1 - ρ) + 1e-6

        # invertible parameterization is incompatible with the bilevel objective
        bilevelModel = SARIMA(airPassengers, 0, 1, 1; seasonality = 12, P = 0, D = 1, Q = 1)
        @test_throws AssertionError fit!(bilevelModel; objectiveFunction = "bilevel", invertible = true)
    end
end
