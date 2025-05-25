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
        airpassengers = loadDataset(AIR_PASSENGERS)
        ocsb_result = Sarimax.ocsb_test(values(airpassengers);max_lag=3)
        @test ocsb_result["seasonal_difference"] == ocsb_datasets["airpassengers.csv"]["D"]
        @test isapprox(ocsb_result["test_statistic"], ocsb_datasets["airpassengers.csv"]["test_stat"], atol=5e-3)

        gdpc1 = loadDataset(GDPC1)
        ocsb_result = Sarimax.ocsb_test(values(gdpc1);max_lag=3)
        @test ocsb_result["seasonal_difference"] == ocsb_datasets["GDPC1.csv"]["D"]
        @test isapprox(ocsb_result["test_statistic"], ocsb_datasets["GDPC1.csv"]["test_stat"], atol=5e-3)

        nrou = loadDataset(NROU)
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
