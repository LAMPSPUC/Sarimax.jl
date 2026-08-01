using Test
using Random
using Sarimax
using JSON

@testset "KPSS Test" begin
    @testset "Basic Functionality" begin
        # Test with white noise (should be stationary)
        Random.seed!(123)
        stationary_series = randn(100)
        result = Sarimax.kpss_test(stationary_series)
        @test haskey(result, "test_statistic")
        @test haskey(result, "p_value")
        @test haskey(result, "critical_values")
        @test haskey(result, "lags")
        @test result["test_statistic"] < result["critical_values"][0.05]  # Should be stationary

        # Test with random walk (should be non-stationary)
        random_walk = cumsum(randn(100))
        result = Sarimax.kpss_test(random_walk)
        # @test result["test_statistic"] < result["critical_values"][0.05]  # Should be non-stationary

        # Test with trend stationary series
        t = 1:100
        trend_stationary = 0.1 .* t .+ randn(100)
        result = Sarimax.kpss_test(trend_stationary, regression=:ct)
        @test result["test_statistic"] < result["critical_values"][0.05]  # Should be trend stationary
    end

    @testset "Different Float Types" begin
        Random.seed!(123)
        data = randn(100)

        # Test Float64
        result64 = Sarimax.kpss_test(Float64.(data))
        @test eltype(Float64.(data)) == Float64

        # Test Float32
        result32 = Sarimax.kpss_test(Float32.(data))
        @test eltype(Float32.(data)) == Float32

        # Test BigFloat
        resultbig = Sarimax.kpss_test(BigFloat.(data))
        @test eltype(BigFloat.(data)) == BigFloat

        # Results should be approximately equal across types
        @test isapprox(result64["test_statistic"], result32["test_statistic"], atol=1e-5)
        @test isapprox(result64["test_statistic"], resultbig["test_statistic"], atol=1e-5)
    end

    @testset "Regression Types" begin
        Random.seed!(123)
        data = randn(100)

        # Test constant (:c) regression
        result_c = Sarimax.kpss_test(data, regression=:c)
        @test haskey(result_c["critical_values"], 0.10)
        @test result_c["critical_values"][0.05] ≈ 0.463 atol=5e-3

        # Test trend (:ct) regression
        result_ct = Sarimax.kpss_test(data, regression=:ct)
        @test haskey(result_ct["critical_values"], 0.10)
        @test result_ct["critical_values"][0.05] ≈ 0.146 atol=5e-3

        # Test invalid regression type
        @test_throws ArgumentError Sarimax.kpss_test(data, regression=:invalid)
    end

    @testset "Test in airpassengers" begin
        kpss_datasets = JSON.parsefile(joinpath(@__DIR__, "datasets", "kpss_results_datasets.json"))
        airpassengers = load_dataset(AIR_PASSENGERS)
        kpss_result = Sarimax.kpss_test(values(airpassengers);regression=:c)
        @test isapprox(kpss_result["test_statistic"], kpss_datasets["airpassengers.csv"]["test_stat"], atol=5e-3)
        @test kpss_result["p_value"] == kpss_datasets["airpassengers.csv"]["p_value"]

        airpassengersLog = log.(values(airpassengers))
        kpss_result = Sarimax.kpss_test(airpassengersLog;regression=:c)
        @test isapprox(kpss_result["test_statistic"], kpss_datasets["log_airpassengers.csv"]["test_stat"], atol=5e-3)
        @test kpss_result["p_value"] == kpss_datasets["log_airpassengers.csv"]["p_value"]

        gdpc1 = load_dataset(GDPC1)
        kpss_result = Sarimax.kpss_test(values(gdpc1);regression=:c)
        @test isapprox(kpss_result["test_statistic"], kpss_datasets["GDPC1.csv"]["test_stat"], atol=5e-3)
        @test kpss_result["p_value"] == kpss_datasets["GDPC1.csv"]["p_value"]

        nrou = load_dataset(NROU)
        kpss_result = Sarimax.kpss_test(values(nrou);regression=:c)
        @test isapprox(kpss_result["test_statistic"], kpss_datasets["NROU.csv"]["test_stat"], atol=5e-3)
        @test kpss_result["p_value"] == kpss_datasets["NROU.csv"]["p_value"]
    end


    @testset "Lag Selection" begin
        Random.seed!(123)
        data = randn(100)

        # Test legacy lag selection
        result_legacy = Sarimax.kpss_test(data, nlags=:legacy)
        @test result_legacy["lags"] == min(Int(ceil(12.0 * (100/100.0)^0.25)), 99)

        # Test custom lag
        result_custom = Sarimax.kpss_test(data, nlags=5)
        @test result_custom["lags"] == 5

        # Test invalid lag values
        @test_throws ArgumentError Sarimax.kpss_test(data, nlags=:invalid)
        @test_throws ArgumentError Sarimax.kpss_test(data, nlags=100)  # >= n
    end

    @testset "Comparison with Python statsmodels" begin
        # Load test data
        kpss_time_series = JSON.parsefile(joinpath(@__DIR__, "datasets", "kpss_time_series.json"))
        kpss_results = JSON.parsefile(joinpath(@__DIR__, "datasets", "kpss_results.json"))

        # Separate stationary and non-stationary series
        stationary_series = Dict(name => data for (name, data) in kpss_time_series if occursin("stationary_series", name))
        nonstationary_series = Dict(name => data for (name, data) in kpss_time_series if occursin("nonstationary_series", name))

        @testset "Test Statistics" begin
            @testset "Stationary Series" begin
                for (series_name, series_data) in stationary_series
                    python_results = kpss_results[series_name]
                    julia_results = Sarimax.kpss_test(series_data, regression=:c, nlags=:legacy)

                    @test isapprox(julia_results["test_statistic"],
                                  python_results["KPSS Statistic"],
                                  atol=5e-3)
                    @test julia_results["lags"] == python_results["Lags Used"]
                end
            end

            @testset "Non-stationary Series" begin
                for (series_name, series_data) in nonstationary_series
                    python_results = kpss_results[series_name]
                    julia_results = Sarimax.kpss_test(series_data, regression=:c, nlags=:legacy)

                    @test isapprox(julia_results["test_statistic"],
                                  python_results["KPSS Statistic"],
                                  atol=5e-3)
                    @test julia_results["lags"] == python_results["Lags Used"]
                end
            end
        end

        @testset "P-values" begin
            @testset "Stationary Series" begin
                for (series_name, series_data) in stationary_series
                    python_results = kpss_results[series_name]
                    julia_results = Sarimax.kpss_test(series_data, regression=:c, nlags=:legacy)

                    @test isapprox(julia_results["p_value"],
                                  python_results["p-value"],
                                  atol=5e-3)
                end
            end

            @testset "Non-stationary Series" begin
                for (series_name, series_data) in nonstationary_series
                    python_results = kpss_results[series_name]
                    julia_results = Sarimax.kpss_test(series_data, regression=:c, nlags=:legacy)

                    @test isapprox(julia_results["p_value"],
                                  python_results["p-value"],
                                  atol=5e-3)
                end
            end
        end

        @testset "Critical Values" begin
            # Critical values should be the same for all series when regression=:c
            series_name, series_data = first(kpss_time_series)
            python_results = kpss_results[series_name]
            julia_results = Sarimax.kpss_test(series_data, regression=:c)

            # Convert Python's percentage strings to our decimal format
            python_crit = Dict(
                0.10 => python_results["Critical Values"]["10%"],
                0.05 => python_results["Critical Values"]["5%"],
                0.025 => python_results["Critical Values"]["2.5%"],
                0.01 => python_results["Critical Values"]["1%"]
            )

            for (level, value) in julia_results["critical_values"]
                @test isapprox(value, python_crit[level], atol=5e-3)
            end
        end
    end
end

@testset "OCSB Test" begin
    @testset "Basic Functionality" begin
        Random.seed!(123)
        series::Vector{Float32} = randn(100)
        result = Sarimax.ocsb_test(series;max_lag=0)
        @test haskey(result, "test_statistic")
        @test haskey(result, "critical_value")
        @test haskey(result, "seasonal_difference")
        @test result["seasonal_difference"] == 0
    end

    @testset "Test in airpassengers" begin
        ocsb_datasets = JSON.parsefile(joinpath(@__DIR__, "datasets", "ocsb_results_datasets.json"))
        airpassengers = load_dataset(AIR_PASSENGERS)
        ocsb_result = Sarimax.ocsb_test(values(airpassengers);max_lag=3)
        @test ocsb_result["seasonal_difference"] == ocsb_datasets["airpassengers.csv"]["D"]
        @test isapprox(ocsb_result["test_statistic"], ocsb_datasets["airpassengers.csv"]["test_stat"], atol=5e-3)

        airpassengersLog = log.(values(airpassengers))
        ocsb_result = Sarimax.ocsb_test(airpassengersLog;max_lag=3)
        @test ocsb_result["seasonal_difference"] == ocsb_datasets["log_airpassengers.csv"]["D"]
        @test isapprox(ocsb_result["test_statistic"], ocsb_datasets["log_airpassengers.csv"]["test_stat"], atol=5e-3)

        gdpc1 = load_dataset(GDPC1)
        ocsb_result = Sarimax.ocsb_test(values(gdpc1);max_lag=3)
        @test ocsb_result["seasonal_difference"] == ocsb_datasets["GDPC1.csv"]["D"]
        @test isapprox(ocsb_result["test_statistic"], ocsb_datasets["GDPC1.csv"]["test_stat"], atol=5e-3)

        nrou = load_dataset(NROU)
        ocsb_result = Sarimax.ocsb_test(values(nrou);max_lag=3)
        @test ocsb_result["seasonal_difference"] == ocsb_datasets["NROU.csv"]["D"]
        # @test isapprox(ocsb_result["test_statistic"], ocsb_datasets["NROU.csv"]["test_stat"], atol=5e-3)
    end

    @testset "Comparison with Python pmdarima" begin
        ocsb_time_series = JSON.parsefile(joinpath(@__DIR__, "datasets", "ocsb_time_series.json"))
        ocsb_results = JSON.parsefile(joinpath(@__DIR__, "datasets", "ocsb_results.json"))

        for (series_name, series_data) in ocsb_time_series
            python_results = ocsb_results[series_name]
            series::Vector{Float32} = series_data
            julia_results = Sarimax.ocsb_test(series;max_lag=3)

            @test isapprox(julia_results["test_statistic"],
                          python_results["OCSB test statistic"],
                          atol=5e-3)
            @test julia_results["seasonal_difference"] == python_results["D"]
        end
    end
end

# ------------------------------------------------------------------------------------------
# R-compatibility guarantees.
# Reference values generated with R 4.4.1, forecast 8.23.0, urca 1.3-4:
#   urca::ur.kpss(y, type = "mu", lags = "short")@teststat
#   forecast::ndiffs(y, alpha = 0.05, test = "kpss")
#   forecast::nsdiffs(ts(y, frequency = 12), test = "seas")
# on the deterministic series defined below (bit-reproducible across languages) and on the
# package's AIR_PASSENGERS dataset (datasets/airpassengers.csv — note: this is NOT R's
# classic AirPassengers; the references were generated on this exact CSV).
# ------------------------------------------------------------------------------------------
@testset "R compatibility (urca/forecast)" begin
    t1 = collect(1.0:120.0)
    yTrend = 0.5 .* t1 .+ 10 .* sin.(2 .* pi .* t1 ./ 12)   # trend + seasonal
    t2 = collect(1.0:150.0)
    yDeter = sin.(t2) .+ cos.(3 .* t2)                      # deterministic stationary
    yRw = cumsum(sin.(t2) .+ cos.(2 .* t2) .+ 0.3)          # accumulated drift

    airPassengers = Float64.(values(loadDataset(AIR_PASSENGERS)))

    @testset "kpss_test nlags=:short matches urca::ur.kpss(lags=\"short\")" begin
        atol = 1e-6
        @test Sarimax.kpss_test(yTrend; nlags=:short)["test_statistic"] ≈ 2.2655841674 atol = atol
        @test Sarimax.kpss_test(diff(yTrend); nlags=:short)["test_statistic"] ≈ 0.0132526946 atol = atol
        @test Sarimax.kpss_test(yDeter; nlags=:short)["test_statistic"] ≈ 0.0401231596 atol = atol
        @test Sarimax.kpss_test(diff(yDeter); nlags=:short)["test_statistic"] ≈ 0.0238223296 atol = atol
        @test Sarimax.kpss_test(yRw; nlags=:short)["test_statistic"] ≈ 3.0956486894 atol = atol
        @test Sarimax.kpss_test(diff(yRw); nlags=:short)["test_statistic"] ≈ 0.0291375337 atol = atol
        @test Sarimax.kpss_test(airPassengers; nlags=:short)["test_statistic"] ≈ 3.8269740597 atol = atol
        @test Sarimax.kpss_test(diff(airPassengers); nlags=:short)["test_statistic"] ≈ 0.0193280027 atol = atol
    end

    @testset "selectIntegrationOrder kpssShort matches forecast::ndiffs" begin
        @test Sarimax.selectIntegrationOrder(yTrend, 2, 0, 1, "kpssShort") == 1
        @test Sarimax.selectIntegrationOrder(yDeter, 2, 0, 1, "kpssShort") == 0
        @test Sarimax.selectIntegrationOrder(yRw, 2, 0, 1, "kpssShort") == 1
        @test Sarimax.selectIntegrationOrder(airPassengers, 2, 0, 1, "kpssShort") == 1
    end

    @testset "seasonalStrengthTest matches forecast::nsdiffs(test=\"seas\")" begin
        @test Sarimax.seasonalStrengthTest(yTrend, 12)["seasonal_difference"] == 1
        @test Sarimax.seasonalStrengthTest(yDeter, 12)["seasonal_difference"] == 0
        @test Sarimax.seasonalStrengthTest(airPassengers, 12)["seasonal_difference"] == 1
        # selectSeasonalIntegrationOrder("seas") delegates to the internal test
        @test Sarimax.selectSeasonalIntegrationOrder(yTrend, 12, "seas") == 1
        @test Sarimax.selectSeasonalIntegrationOrder(yDeter, 12, "seas") == 0
    end

    @testset "root admissibility margin (auto.arima's 1.001 rule)" begin
        threshold = 1 / (1 + 1e-3)
        # AR(1): inverse root modulus = |phi| -> AR poly coeffs a = -phi
        @test Sarimax.maxInverseRootModulus([-0.9995]) ≈ 0.9995 atol = 1e-12
        @test Sarimax.maxInverseRootModulus([-0.9995]) >= threshold   # rejected (near unit root)
        @test Sarimax.maxInverseRootModulus([-0.95]) < threshold      # accepted
        # MA(1): 1 + theta z -> inverse root modulus = |theta|
        @test Sarimax.maxInverseRootModulus([0.9995]) >= threshold
        @test Sarimax.maxInverseRootModulus([0.9]) < threshold
        # AR(2) with known roots: (1 - 0.5z)(1 - 0.4z) = 1 - 0.9z + 0.2z^2
        @test Sarimax.maxInverseRootModulus([-0.9, 0.2]) ≈ 0.5 atol = 1e-10
        # empty / all-zero -> trivially admissible
        @test Sarimax.maxInverseRootModulus(Float64[]) == 0.0
        @test Sarimax.maxInverseRootModulus([0.0, 0.0]) == 0.0
    end
end

@testset "kpssShort reproduces forecast::ndiffs (bandwidth regression)" begin
    # forecast::ndiffs does NOT use urca's lags = "short": its kpss_wrap fixes
    # use.lag = trunc(3*sqrt(n)/13) (verified in the forecast 8.23.0 source). All
    # reference numbers below are pinned against R 4.4.1 / forecast 8.23.0 / urca
    # 1.3.4 run on the exact same inputs.

    # (1) Numerical exactness against urca, on the package's own dataset (n = 203).
    # Both bandwidths must reproduce ur.kpss to ~1e-6; on this series they happen to
    # agree on the decision (p ≈ 0.10 -> d = 0, matching auto.arima's d = 0 here).
    airp = Float64.(values(loadDataset(AIR_PASSENGERS)))
    logAirp = log.(airp)
    dsLog = logAirp[13:end] .- logAirp[1:end-12]

    resNdiffs = Sarimax.kpss_test(dsLog; nlags = :ndiffs)
    @test resNdiffs["test_statistic"] ≈ 0.2316541 atol = 1e-5   # ur.kpss, use.lag = 3
    resShort = Sarimax.kpss_test(dsLog; nlags = :short)
    @test resShort["test_statistic"] ≈ 0.2088131 atol = 1e-5    # ur.kpss, use.lag = 4
    @test Sarimax.selectIntegrationOrder(logAirp, 2, 1, 12, "kpssShort") == 0

    # (2) Decision divergence: the two bandwidths disagree on Box-Jenkins' classic
    # AirPassengers (1949-1960, n = 144; embedded literally — it is not the packaged
    # dataset of the same name). After the seasonal difference of the log (n = 132)
    # the ndiffs bandwidth gives 2 lags, statistic 0.5367, rejection at 5% -> d = 1
    # (agreeing with auto.arima), while urca's "short" gives 4 lags, 0.3682, no
    # rejection -> d = 0. This is the case that motivated the bandwidth fix.
    classicAirp = Float64[
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
        196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
        242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
        284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
        315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
        340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
        417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432]
    logClassic = log.(classicAirp)
    dsClassic = logClassic[13:end] .- logClassic[1:end-12]

    cNdiffs = Sarimax.kpss_test(dsClassic; nlags = :ndiffs)
    @test cNdiffs["test_statistic"] ≈ 0.5366879 atol = 1e-5
    @test cNdiffs["p_value"] < 0.05                    # rejects -> one more difference

    cShort = Sarimax.kpss_test(dsClassic; nlags = :short)
    @test cShort["test_statistic"] ≈ 0.3681637 atol = 1e-5
    @test cShort["p_value"] > 0.05                     # urca-short would say d = 0

    # End-to-end on the classic series: corrected bandwidth reproduces auto.arima's
    # d = 1 (the airline model), which the urca-short bandwidth failed to select.
    @test Sarimax.selectIntegrationOrder(logClassic, 2, 1, 12, "kpssShort") == 1
end

@testset "seasonalStrengthTest bounded-robust STL (no upstream hang)" begin
    # Regression guard for the SeasonalTrendLoess robust-loop hang: these short M4
    # Monthly series (train windows) make stl(...; robust = true) spin forever; the
    # bounded-robust call must terminate and return a well-formed result.
    # A degenerate near-constant series with a single spike: the kind of profile that
    # makes the old robust = true STL loop fail to converge and spin forever.
    y = vcat(fill(100.0, 60), [1.0e6], fill(100.0, 8))
    local res
    @test (res = Sarimax.seasonalStrengthTest(y, 12)) isa Dict   # must terminate
    @test 0.0 <= res["seasonal_strength"] <= 1.0
    @test res["seasonal_difference"] in (0, 1)
end
