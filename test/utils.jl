@testset "utils" begin

    @testset "Differentiate" begin
        airPassengers = load_dataset(AIR_PASSENGERS)
        diff_0_0 = differentiate(values(airPassengers), 0, 0, 12)
        @test size(diff_0_0) == (204,)
        @test values(diff_0_0) == values(airPassengers)

        diff_1_0 = differentiate(values(airPassengers), 1, 0, 12)
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
        @test differentiated_coefficients(d, D, s) == expected_output

        # Test case 2
        d = 2
        D = 1
        s = 4
        expected_output = [1.0, -2.0, 1.0, 0.0, -1.0, 2.0, -1.0]
        @test differentiated_coefficients(d, D, s) == expected_output

        # Test case 3
        d = 1
        D = 1
        s = 12
        expected_output =
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]
        @test differentiated_coefficients(d, D, s) == expected_output
    end

    @testset "Testing integrate function" begin
        # Load dataset and differentiate series
        y = load_dataset(AIR_PASSENGERS)
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
        airPassengers = load_dataset(AIR_PASSENGERS)
        @test Sarimax.selectSeasonalIntegrationOrder(values(airPassengers), 12, "seas") == 1
        @test Sarimax.selectSeasonalIntegrationOrder(values(airPassengers), 12, "ch") == 0
        @test_throws ArgumentError Sarimax.selectSeasonalIntegrationOrder(
            values(airPassengers),
            12,
            "hegy",
        )
    end

    @testset "selectIntegrationOrder" begin
        airPassengers = load_dataset(AIR_PASSENGERS)
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
    #     airPassengers = load_dataset(AIR_PASSENGERS)
    #     @test Sarimax.selectIntegrationOrder(values(airPassengers), 2, 0, 12, "kpssR") == 1
    # end

    @testset "automatic_differentiation" begin
        gdpc1Data = load_dataset(GDPC1)
        nrouData = load_dataset(NROU)
        seriesVector::Vector{TimeArray} = [gdpc1Data, nrouData]
        mergedTimeArray = Sarimax.merge(seriesVector)

        @test_throws AssertionError automatic_differentiation(
            mergedTimeArray;
            integrationTest = "test",
        )
        @test_throws AssertionError automatic_differentiation(
            mergedTimeArray;
            seasonalIntegrationTest = "test",
        )
        @test_throws AssertionError automatic_differentiation(
            mergedTimeArray;
            seasonalPeriod = -1,
        )

        mergedDiffSeries, diffMetadata = automatic_differentiation(
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

    @testset "automatic_differentiation Outlier Case" begin
        timestamps = Date(2020, 1, 1):Day(1):Date(2020, 1, 5)
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        outlier_values = [0, 0, 1, 0, 0]
        data = (datetime = timestamps, data = values, outlier_3 = outlier_values)
        series = TimeArray(data; timestamp = :datetime)

        diffSeries, metadata = automatic_differentiation(series)
        @test haskey(metadata, :outlier_3)
        @test metadata[:outlier_3][:d] == 0
        @test metadata[:outlier_3][:D] == 0
        @test TimeSeries.values(diffSeries[:outlier_3]) == TimeSeries.values(series[:outlier_3])  # Outlier column should remain unchanged
    end

    @testset "isConstant" begin
        # Create Dataframe with constant values and one date column
        df = DataFrame(date = Date(2020, 1, 1):Day(1):Date(2020, 1, 10), value = ones(10))
        dataset = load_dataset(df)
        @test Sarimax.isConstant(dataset) == true

        # Add a new column with different values
        df[!, "newCol"] = [ones(5); 2 * ones(5)]
        dataset = load_dataset(df)

        @test Sarimax.isConstant(dataset) == true

        df.value = [ones(5); 2 * ones(5)]
        dataset = load_dataset(df)

        @test Sarimax.isConstant(dataset) == false
    end

    @testset "logLikelihood and loglike" begin
        mutable struct TestModelUtil <: Sarimax.SarimaxModel end

        @test_throws MissingMethodImplementation loglikelihood(TestModelUtil())
        @test_throws MissingMethodImplementation loglike(TestModelUtil())

        airPassengers = load_dataset(AIR_PASSENGERS)
        airPassengersLog = log.(airPassengers)
        testModel = SARIMA(airPassengersLog, 3, 0, 1; seasonality = 12, P = 1, D = 1, Q = 1)

        @test_throws ModelNotFitted loglikelihood(testModel)
        @test_throws ModelNotFitted loglike(testModel)

        fit!(testModel)

        # CSS loglik of the (3,0,1)(1,1,1)12 fit under the multiplicative form.
        #
        # The value reflects the `:innovations` default: the error sum starts at t = 1, so
        # the effective sample is `T` rather than `T - lb + 1` and the loglik is evaluated
        # over more points. That is a different scale, neither better nor worse.
        @test loglikelihood(testModel) ≈ 277.6017213855341 atol = 1e-1
        @test loglike(testModel) ≈ 277.6017213855341 atol = 1e-1
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

    @testset "identifyOutliers dispersao degenerada" begin
        # Contract: a zero interquartile range — or one negligible on the scale of the data
        # — implies no outliers. Without dispersion the IQR fences collapse onto the
        # quartiles and the rule flags everything not bit-identical to them. Zero dispersion
        # is zero evidence of atypicality. Tested as a UNIT, over constructed vectors: a
        # fixture that runs the solver would inherit the solver's tolerance.

        # Maioria identica mais um pico: sem dispersao central nao ha do que o pico destoar.
        degenerateWithSpike = vcat(fill(5.0, 30), 100.0)
        @test Sarimax.identifyOutliers(degenerateWithSpike) == falses(31)

        # Residuals of a SARIMA(0,0,0) fit on a constant series, with three values shifted
        # by 1-2 ULP, which is the difference between one machine and another. Without the
        # guard, positions 7, 24 and 30 appear as outliers alongside 5; with it, none does,
        # on any machine.
        base = -3.1935483870967762
        solverNoise = fill(base, 31)
        solverNoise[5] = 95.80645161290323
        solverNoise[7] = prevfloat(base)
        solverNoise[24] = nextfloat(base)
        solverNoise[30] = nextfloat(base, 2)
        @test Sarimax.identifyOutliers(solverNoise) == falses(31)

        # Escala nula: os dois quartis em zero. A guarda usa `<=`, entao cobre este caso.
        allZeros = zeros(20)
        allZeros[3] = 1e-9
        @test Sarimax.identifyOutliers(allZeros) == falses(20)

        # Binding control: the guard must stay SILENT when there is real dispersion. Same
        # shape as the cases above but with a real IQR, and the spike is still flagged.
        nonDegenerate = vcat(repeat([10.0, 11.0, 12.0, 13.0, 14.0], 6), 100.0)
        @test findall(Sarimax.identifyOutliers(nonDegenerate)) == [31]

        # E a dispersao minima que ainda NAO e degenerada segue tratada como dispersao:
        # IQR relativo de 1e-6, cem vezes acima de `DEGENERATE_IQR_RTOL`, preserva a regra.
        justAboveTolerance = fill(1.0, 30)
        justAboveTolerance[1:15] .= 1.0 + 1e-6
        push!(justAboveTolerance, 100.0)
        @test findall(Sarimax.identifyOutliers(justAboveTolerance)) == [31]
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
