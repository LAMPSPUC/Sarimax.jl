@testset "utils" begin

    @testset "Differentiate" begin
        airPassengers = loadDataset(AIR_PASSENGERS)
        diff_0_0 = differentiate(airPassengers, 0, 0, 12)
        @test size(diff_0_0) == (204,)
        @test values(diff_0_0) == values(airPassengers)

        diff_1_0 = differentiate(airPassengers, 1, 0, 12)
        @test size(diff_1_0) == (203,)
        @test values(diff_1_0) ==
              [values(airPassengers)[i] - values(airPassengers)[i-1] for i = 2:204]

        diff_0_1 = differentiate(airPassengers, 0, 1, 12)
        @test size(diff_0_1) == (192,)
        @test values(diff_0_1) ==
              [values(airPassengers)[i] - values(airPassengers)[i-12] for i = 13:204]

        diff_1_1 = differentiate(airPassengers, 1, 1, 12)
        @test size(diff_1_1) == (191,)
        @test isapprox(
            values(diff_1_1),
            [
                values(airPassengers)[i] - values(airPassengers)[i-1] -
                values(airPassengers)[i-12] + values(airPassengers)[i-13] for i = 14:204
            ],
            atol = 1e-6,
        )
    end
    @testset "Test Differentiated Coefficients Function" begin
        # Test case 1
        d = 1
        D = 0
        s = 1
        expected_output = [1.0, -1.0]
        @test differentiatedCoefficients(d, D, s) == expected_output

        # Test case 2
        d = 2
        D = 1
        s = 4
        expected_output = [1.0, -2.0, 1.0, 0.0, -1.0, 2.0, -1.0]
        @test differentiatedCoefficients(d, D, s) == expected_output

        # Test case 3
        d = 1
        D = 1
        s = 12
        expected_output =
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]
        @test differentiatedCoefficients(d, D, s) == expected_output
    end

    @testset "Testing integrate function" begin
        # Load dataset and differentiate series
        y = loadDataset(AIR_PASSENGERS)
        diff_1_1 = differentiate(y, 1, 1, 12)
        diff_0_1 = differentiate(y, 0, 1, 12)
        diff_1_0 = differentiate(y, 1, 0, 12)
        diff_2_0 = differentiate(y, 2, 0, 12)
        diff_2_1 = differentiate(y, 2, 1, 12)

        # Extract values from differentiated series
        values_diff_1_0::Vector{Float64} = values(diff_1_0)
        values_diff_0_1::Vector{Float64} = values(diff_0_1)
        values_diff_1_1::Vector{Float64} = values(diff_1_1)
        values_diff_2_0::Vector{Float64} = values(diff_2_0)
        values_diff_2_1::Vector{Float64} = values(diff_2_1)

        @test isapprox(
            integrate(values(y[1:1]), values_diff_1_0, 1, 0, 12),
            values(y);
            atol = 1e-5,
        )
        @test isapprox(
            integrate(values(y[1:12]), values_diff_0_1, 0, 1, 12),
            values(y);
            atol = 1e-5,
        )
        @test isapprox(
            integrate(values(y[1:13]), values_diff_1_1, 1, 1, 12),
            values(y);
            atol = 1e-5,
        )
        @test isapprox(
            integrate(values(y[1:2]), values_diff_2_0, 2, 0, 12),
            values(y);
            atol = 1e-5,
        )
        @test isapprox(
            integrate(values(y[1:14]), values_diff_2_1, 2, 1, 12),
            values(y);
            atol = 1e-5,
        )
    end

    @testset "selectSeasonalIntegrationOrder" begin
        airPassengers = loadDataset(AIR_PASSENGERS)
        @test Sarimax.selectSeasonalIntegrationOrder(values(airPassengers), 12, "seas") == 1
        @test Sarimax.selectSeasonalIntegrationOrder(values(airPassengers), 12, "ch") == 0
        @test_throws ArgumentError Sarimax.selectSeasonalIntegrationOrder(
            values(airPassengers),
            12,
            "hegy",
        )
        # TODO: Fix test of PyCall and RCall
        # @testset "ocsb test without PyCall" begin
        #     @test Sarimax.selectSeasonalIntegrationOrder(values(airPassengers), 12, "ocsb") == StateSpaceModels.seasonal_strength_test(y, seasonality)
        # end

        # @testset "ocsbR test without RCall" begin
        #     @test Sarimax.selectSeasonalIntegrationOrder(values(airPassengers), 12, "ocsbR") == StateSpaceModels.seasonal_strength_test(y, seasonality)
        # end

        # @testset "ocsb test with PyCall" begin
        #     # Mock the Pkg.project().dependencies to simulate PyCall being installed
        #     @test Sarimax.selectSeasonalIntegrationOrder(values(airPassengers), 12, "ocsb") == 1
        # end

        # @testset "ocsbR test with RCall" begin
        #     # Mock the Pkg.project().dependencies to simulate RCall being installed

        #     @test Sarimax.selectSeasonalIntegrationOrder(values(airPassengers), 12, "ocsbR") == 1
        # end
    end

    @testset "selectIntegrationOrder" begin
        airPassengers = loadDataset(AIR_PASSENGERS)
        @test Sarimax.selectIntegrationOrder(values(airPassengers), 2, 0, 12, "kpss") == 1
        @test_throws ArgumentError Sarimax.selectIntegrationOrder(
            values(airPassengers),
            2,
            0,
            12,
            "hegy",
        )
    end

    # @testset "selectIntegrationOrderR" begin
    #     airPassengers = loadDataset(AIR_PASSENGERS)
    #     @test Sarimax.selectIntegrationOrder(values(airPassengers), 2, 0, 12, "kpssR") == 1
    # end

    @testset "automaticDifferentiation" begin
        gdpc1Data = loadDataset(GDPC1)
        nrouData = loadDataset(NROU)
        seriesVector::Vector{TimeArray} = [gdpc1Data, nrouData]
        mergedTimeArray = Sarimax.merge(seriesVector)

        @test_throws AssertionError automaticDifferentiation(
            mergedTimeArray;
            integrationTest = "test",
        )
        @test_throws AssertionError automaticDifferentiation(
            mergedTimeArray;
            seasonalIntegrationTest = "test",
        )
        @test_throws AssertionError automaticDifferentiation(
            mergedTimeArray;
            seasonalPeriod = -1,
        )

        mergedDiffSeries, diffMetadata = automaticDifferentiation(
            mergedTimeArray;
            integrationTest = "kpss",
            seasonalIntegrationTest = "ch",
            seasonalPeriod = 12,
        )

        @test size(mergedDiffSeries, 2) == size(mergedTimeArray, 2)
        @test colnames(mergedDiffSeries) == colnames(mergedTimeArray)

        for col in colnames(mergedTimeArray)
            @test diffMetadata[col][:d] == 2
            @test diffMetadata[col][:D] == 0
            @test size(mergedDiffSeries[col], 1) ==
                  size(mergedTimeArray[col], 1) - diffMetadata[col][:d]
        end
    end

    @testset "automaticDifferentiation Outlier Case" begin
        timestamps = Date(2020, 1, 1):Day(1):Date(2020, 1, 5)
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        outlier_values = [0, 0, 1, 0, 0]
        data = (datetime = timestamps, data = values, outlier_3 = outlier_values)
        series = TimeArray(data; timestamp = :datetime)

        diffSeries, metadata = automaticDifferentiation(series)
        @test haskey(metadata, :outlier_3)
        @test metadata[:outlier_3][:d] == 0
        @test metadata[:outlier_3][:D] == 0
        @test TimeSeries.values(diffSeries[:outlier_3]) == TimeSeries.values(series[:outlier_3])  # Outlier column should remain unchanged
    end

    @testset "isConstant" begin
        # Create Dataframe with constant values and one date column
        df = DataFrame(date = Date(2020, 1, 1):Day(1):Date(2020, 1, 10), value = ones(10))
        dataset = loadDataset(df)
        @test Sarimax.isConstant(dataset) == true

        # Add a new column with different values
        df[!, "newCol"] = [ones(5); 2 * ones(5)]
        dataset = loadDataset(df)

        @test Sarimax.isConstant(dataset) == true

        df.value = [ones(5); 2 * ones(5)]
        dataset = loadDataset(df)

        @test Sarimax.isConstant(dataset) == false
    end

    @testset "logLikelihood and loglike" begin
        mutable struct TestModelUtil <: Sarimax.SarimaxModel end

        @test_throws MissingMethodImplementation loglikelihood(TestModelUtil())
        @test_throws MissingMethodImplementation loglike(TestModelUtil())

        airPassengers = loadDataset(AIR_PASSENGERS)
        airPassengersLog = log.(airPassengers)
        testModel = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)

        @test_throws ModelNotFitted loglikelihood(testModel)
        @test_throws ModelNotFitted loglike(testModel)

        fit!(testModel)

        @test loglikelihood(testModel) ≈ 254.01202403694745 atol = 1e-1
        @test loglike(testModel) ≈ 254.01202403694745 atol = 1e-1
    end

    @testset "identifyOutliers Tests" begin
        # Basic test with no outliers
        data1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        @test Sarimax.identifyOutliers(data1) == [false, false, false, false, false]

        # Test with a single outlier
        data2 = [1.0, 2.0, 3.0, 100.0]
        @test Sarimax.identifyOutliers(data2) == [false, false, false, true]

        # Test with multiple outliers
        data3 = [1.0, 2.0, 3.0, 100.0, -50.0]
        @test Sarimax.identifyOutliers(data3) == [false, false, false, true, true]

        # Test with a different threshold
        data4 = [1.0, 2.0, 3.0, 20, -10.0]
        @test Sarimax.identifyOutliers(data4, "iqr", 10.0) == [false, false, false, false, false]  # Higher threshold, no outliers

        # Test with an empty vector
        data5 = Float64[]
        @test Sarimax.identifyOutliers(data5) == Bool[]

        # Test with identical values (no outliers expected)
        data6 = fill(5.0, 10)
        @test Sarimax.identifyOutliers(data6) == fill(false, 10)

        # Test invalid method
        @test_throws ArgumentError Sarimax.identifyOutliers([1.0, 2.0, 3.0], "unknown")
    end

    @testset "createOutliersDummies Tests" begin
        # Test with no outliers
        outliers1 = falses(5)
        df1 = Sarimax.createOutliersDummies(outliers1)
        @test size(df1, 2) == 0  # No columns should be created

        # Test with a single outlier
        outliers2 = falses(5)
        outliers2[3] = true
        df2 = Sarimax.createOutliersDummies(outliers2)
        @test size(df2, 2) == 1  # One column should be created
        @test df2[!, "outlier_3"] == [0, 0, 1, 0, 0]

        # Test with multiple outliers
        outliers3::BitVector = [true, false, true, false, true]
        df3 = Sarimax.createOutliersDummies(outliers3)
        @test size(df3, 2) == 3  # Three columns should be created
        @test df3[!, "outlier_1"] == [1, 0, 0, 0, 0]
        @test df3[!, "outlier_3"] == [0, 0, 1, 0, 0]
        @test df3[!, "outlier_5"] == [0, 0, 0, 0, 1]

        # Test with initial offset
        df4 = Sarimax.createOutliersDummies(outliers2, 1)
        @test size(df4, 1) == 6  # One extra row due to offset
        @test df4[!, "outlier_3"] == [0, 0, 0, 1, 0, 0]

        # Test with end offset
        df5 = Sarimax.createOutliersDummies(outliers2, 0, 1)
        @test size(df5, 1) == 6  # One extra row due to offset
        @test df5[!, "outlier_3"] == [0, 0, 1, 0, 0, 0]

        # Test with both initial and end offsets
        df6 = Sarimax.createOutliersDummies(outliers2, 2, 2)
        @test size(df6, 1) == 9  # Two extra rows at start and end
        @test df6[!, "outlier_3"] == [0, 0, 0, 0, 1, 0, 0, 0, 0]
    end
end
