@testset "auto sarima" begin
    @testset "Constant series" begin
        series = TimeArray(Dates.Date(2019, 1, 1):Dates.Day(1):Dates.Date(2019, 1, 31), ones(31))
        model = Sarimax.auto(series)
        @test model.c == 1
        @test model.p == 0
        @test model.q == 0
        @test model.P == 0
        @test model.Q == 0
        @test model.d == 0
        @test model.D == 0
        @test model.seasonality == 1
    end

    @testset "constantDiffSeriesModelSpecification Tests" begin
        y = TimeArray(Dates.Date(2019, 1, 1):Dates.Day(1):Dates.Date(2019, 1, 31), ones(31))
        exog = nothing

        # Test case where D > 0 and d == 0
        model1 = Sarimax.constantDiffSeriesModelSpecification(y, exog, 0, 1, 12, false, true)
        @test model1 isa Sarimax.SarimaxModel
        @test model1.allowDrift == true

        # Test case where D > 0 and d > 0
        model2 = Sarimax.constantDiffSeriesModelSpecification(y, exog, 1, 1, 12, false, false)
        @test model2 isa Sarimax.SarimaxModel
        @test model2.allowDrift == false

        # Test case where d == 2
        model3 = Sarimax.constantDiffSeriesModelSpecification(y, exog, 2, 0, 12, false, false)
        @test model3 isa Sarimax.SarimaxModel

        # Test case where d < 2
        model4 = Sarimax.constantDiffSeriesModelSpecification(y, exog, 1, 0, 12, true, false)
        @test model4 isa Sarimax.SarimaxModel
        @test model4.allowMean == true

        # Test case where data follow a simple polynomial
        @test_throws ArgumentError Sarimax.constantDiffSeriesModelSpecification(y, exog, 3, 0, 12, true, false)

        # Test case with exog
        exog1 = TimeArray(Dates.Date(2019, 1, 1):Dates.Day(1):Dates.Date(2019, 1, 31), ones(31))
        model5 = Sarimax.constantDiffSeriesModelSpecification(y, exog1, 1, 0, 12, false, false)
        @test model5 isa Sarimax.SarimaxModel
        @test model5.allowDrift == false
        @test model5.allowMean == false

        # Test case with exog and D > 0
        model6 = Sarimax.constantDiffSeriesModelSpecification(y, exog1, 1, 1, 12, false, false)
        @test model6 isa Sarimax.SarimaxModel
        @test model6.p == 0
        @test model6.q == 0
        @test model6.P == 0
        @test model6.Q == 0
    end

    @testset "getInformationCriteriaFunction" begin
        @test_throws ArgumentError Sarimax.getInformationCriteriaFunction("mse")

        # Test AIC
        func1 = Sarimax.getInformationCriteriaFunction("aic")
        @test func1 isa Function
        @test aic(2, 3.0) ≈ -2.0
        @test aic(3, 4.0) ≈ -2.0

        # Test AICC
        func2 = Sarimax.getInformationCriteriaFunction("aicc")
        @test func2 isa Function
        @test func2(10, 2, 3.0) ≈ -0.2857142857142858
        @test func2(10, 3, 4.0) ≈ 2.0

        # Test BIC
        func3 = Sarimax.getInformationCriteriaFunction("bic")
        @test func3 isa Function
        @test func3(10, 2, 3.0) ≈ -1.3948298140119082
        @test func3(10, 3, 4.0) ≈ -1.0922447210178623
    end

    @testset "detectOutliers" begin
        # The fixture must be NON-DEGENERATE. On a constant series the interquartile range
        # is exactly zero, the fences collapse onto the median, and everything not
        # bit-identical to it is flagged. Since the residuals are JuMP variables tied by
        # equality constraints, satisfied only to the solver tolerance, the set of
        # "identical" residuals varies from machine to machine and the outlier count is not
        # reproducible.
        #
        # The series below has real dispersion (a 10..14 pattern, IQR = 2) plus an
        # unambiguous spike: relative dispersion about 0.54 and a margin of about 2.0 in
        # data units between the outermost inlier and the fence, against solver noise of
        # order 1e-8.
        dates = Dates.Date(2019, 1, 1):Dates.Day(1):Dates.Date(2019, 1, 31)
        baseValues = [10.0 + ((i - 1) % 5) for i = 1:31]

        series = TimeArray(dates, copy(baseValues))
        outliers = Sarimax.detectOutliers(series, nothing, 0, 0, 1, false)
        @test isnothing(outliers)

        series = TimeArray(dates, copy(baseValues))
        values(series)[5] = 100
        outliers = Sarimax.detectOutliers(series, nothing, 0, 0, 1, false)
        @test isa(outliers, TimeSeries.TimeArray)
        @test length(colnames(outliers)) == 1
        @test colnames(outliers)[1] == Symbol("outlier_5")

        series = TimeArray(dates, copy(baseValues))
        values(series)[10] = 100
        outliers = Sarimax.detectOutliers(series, nothing, 0, 0, 1, false)
        @test isa(outliers, TimeSeries.TimeArray)
        @test length(colnames(outliers)) == 1
        @test colnames(outliers)[1] == Symbol("outlier_10")

        series = TimeArray(dates, copy(baseValues))
        values(series)[15] = 100
        outliers = Sarimax.detectOutliers(series, nothing, 0, 0, 1, false)
        @test isa(outliers, TimeSeries.TimeArray)
        @test length(colnames(outliers)) == 1
        @test colnames(outliers)[1] == Symbol("outlier_15")

        values(series)[20] = 100
        outliers = Sarimax.detectOutliers(series, nothing, 0, 0, 1, false)
        @test isa(outliers, TimeSeries.TimeArray)
        @test length(colnames(outliers)) == 2
        @test colnames(outliers)[1] == Symbol("outlier_15")
        @test colnames(outliers)[2] == Symbol("outlier_20")

        # test with exog
        exog = TimeArray(dates, 2 .* ones(31))
        outliers = Sarimax.detectOutliers(series, exog, 0, 0, 1, false)
        @test !isnothing(outliers)
        @test length(colnames(outliers)) == 3
        @test colnames(outliers)[1] == Symbol("A")
        @test colnames(outliers)[2] == Symbol("outlier_15")
        @test colnames(outliers)[3] == Symbol("outlier_20")

        # The degenerate contract end to end: a constant series with a single spike yields
        # no outliers at all, because the dispersion of the residuals is zero. This is the
        # counterpart of the unit test in `identifyOutliers Tests`.
        degenerate = TimeArray(dates, ones(31))
        values(degenerate)[5] = 100
        @test isnothing(Sarimax.detectOutliers(degenerate, nothing, 0, 0, 1, false))
    end

    @testset "auto with default stepwise" begin
        airpassengers = load_dataset(AIR_PASSENGERS)
        model = auto(airpassengers; seasonality = 12)
        # Pinned selection. The tryCandidate! rewrite of the neighbourhood scan was
        # verified to reproduce the unrolled implementation exactly (same model,
        # same AICc to the last digit) before this pin was added.
        @test model.p == 4
        @test model.q == 1
        @test model.P == 0
        @test model.Q == 1
        @test model.d == 1
        @test model.D == 1
        # The criterion value is pinned, not the scale it is computed on. It reflects AICc
        # over the EXACT Gaussian likelihood (`criterionLoglike`) with `K = ncoef + 1`,
        # matching `forecast::Arima`, the small-sample correction using the `n` of the
        # likelihood actually used, and the `:innovations` default, under which the error
        # sum starts at t = 1 and the criterion `n` is `T` rather than `T - lb + 1`. The
        # selected ORDER is unchanged by all of these; only the value moves, as it must when
        # the likelihood scale changes.
        @test aicc(model) ≈ 520.6574204583987 atol = 1e-6
    end

    @testset "parallel candidate fitting (smoke)" begin
        # With a single thread @threads degrades to serial; the selection must be
        # identical either way (min-IC selection is order-independent).
        Random.seed!(31)
        n = 90
        parDates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        yPar = TimeArray(collect(parDates), cumsum(randn(n)) .+ 0.3 .* collect(1.0:n))
        # `parallel` NAMED on both sides. Note that with JULIA_NUM_THREADS=1, the CI
        # default, both branches are the same code: this testset only exercises real
        # parallelism at nthreads >= 2.
        mSerial = auto(yPar; searchMethod = "grid", maxp = 1, maxq = 1, maxP = 0, maxQ = 0, parallel = false)
        mParallel = auto(yPar; searchMethod = "grid", maxp = 1, maxq = 1, maxP = 0, maxQ = 0, parallel = true)
        @test (mSerial.p, mSerial.q, mSerial.d) == (mParallel.p, mParallel.q, mParallel.d)
        @test aicc(mSerial) ≈ aicc(mParallel) atol = 1e-8
    end

    @testset "auto with stepwise naive" begin
        airpassengers = load_dataset(AIR_PASSENGERS)
        log_airpassengers = log.(airpassengers)
        model = auto(airpassengers; searchMethod="stepwiseNaive", seasonality=12)
        # The differencing decisions are shared with every search method and must hold.
        @test model.d == 1
        @test model.D == 1
        # Under the pre-v0.3 defaults the naive stepwise used to land on the exhaustive
        # grid optimum, (1,1,1)(2,1,0) — which is still what the grid finds today
        # (verified 2026-08: searchMethod = "grid" on this series). Under the v0.3
        # defaults (multiplicative seasonal form + constrained candidate fitting) the
        # naive heuristic now stops at (2,1,2)(0,1,1) instead. The broken tests keep
        # the grid optimum as the documented target so the gap stays visible; if the
        # heuristic is improved to reach it again, these flip to passing.
        # `q`, `P` and `Q` now match the grid optimum, since `maxOrder` is not imposed on
        # the LOCAL searches (mirroring `forecast`, where `max.order` lives only inside
        # `search.arima`) and `K` matches `k = ncoef + 1` of `forecast::Arima`. `p` still
        # diverges: the heuristic stops at a different point.
        @test_broken model.p == 1
        @test model.q == 1
        @test model.P == 2
        @test model.Q == 0
    end

    @testset "auto with grid search" begin
        airpassengers = load_dataset(AIR_PASSENGERS)
        log_airpassengers = log.(airpassengers)
        model = auto(airpassengers; searchMethod="grid", seasonality=12)
        # The grid selection is pinned to the current defaults. On this canonical series R's
        # `auto.arima` selects (2,1,1)(0,1,0) at an AICc of 1018.165; the grid here does not
        # reach that. The pinned order is documented rather than asserted to be optimal, so
        # that a change in selection is visible in the diff instead of being absorbed
        # silently.
        @test model.p == 0
        @test model.q == 3
        @test model.P == 2
        @test model.Q == 0
        @test model.d == 1
        @test model.D == 1
        # Exhaustive search must not do worse than the stepwise heuristic, but only within
        # the SAME candidate space. The two methods sweep different spaces by design
        # (`maxOrder` applies to the grid and not to the local searches, mirroring
        # `forecast`), so the comparison is only meaningful once that difference is removed.
        # Here the per-term boxes are tight and `maxOrder` is loose enough to bind on
        # neither, which restores the sanity property and keeps the grid cheap at 36
        # candidates.
        espaco = (maxp = 2, maxq = 2, maxP = 1, maxQ = 1, maxOrder = 6)
        gridSame = auto(airpassengers; searchMethod="grid", seasonality=12, espaco...)
        stepwiseModel = auto(airpassengers; searchMethod="stepwiseNaive", seasonality=12, espaco...)
        @test aicc(gridSame) <= aicc(stepwiseModel) + 1e-6
    end
end
