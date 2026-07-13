@testset "Datasets" begin

    @testset "Datasets Enum" begin
        @test AIR_PASSENGERS == Datasets(1)
        @test GDPC1 == Datasets(2)
        @test NROU == Datasets(3)
    end

    @testset "load_dataset" begin
        @testset "load_dataset(Datasets)" begin
            @testset "AIR_PASSENGERS" begin
                airPassengersData = load_dataset(AIR_PASSENGERS)
                @test size(airPassengersData, 1) == 204
                @test values(airPassengersData)[1] == 3.526591
                @test values(airPassengersData)[end] == 19.43174
            end

            @testset "GDPC1" begin
                GDPC1Data = load_dataset(GDPC1)
                @test size(GDPC1Data, 1) == 344
                @test values(GDPC1Data)[1] == 2260.807
                @test values(GDPC1Data)[end] == 27944.500
            end

            @testset "NROU" begin
                NROUData = load_dataset(NROU)
                @test size(NROUData, 1) == 344
                @test values(NROUData)[1] == 5.2550525665283200
                @test values(NROUData)[end] == 4.2031234672198900
            end
        end

        @testset "load_dataset(DataFrame)" begin
            @testset "Air Passengers" begin
                airPassengersDf = CSV.File("../datasets/airpassengers.csv") |> DataFrame
                airPassengersData = load_dataset(airPassengersDf)
                @test size(airPassengersData, 1) == 204
                @test values(airPassengersData)[1] == 3.526591
                @test values(airPassengersData)[end] == 19.43174
            end

            @testset "GDPC1" begin
                GDPC1Df = CSV.File("../datasets/GDPC1.csv") |> DataFrame
                GDPC1Data = load_dataset(GDPC1Df)
                @test size(GDPC1Data, 1) == 344
                @test values(GDPC1Data)[1] == 2260.807
                @test values(GDPC1Data)[end] == 27944.500
            end

            @testset "NROU" begin
                NROUDf = CSV.File("../datasets/NROU.csv") |> DataFrame
                NROUData = load_dataset(NROUDf)
                @test size(NROUData, 1) == 344
                @test values(NROUData)[1] == 5.2550525665283200
                @test values(NROUData)[end] == 4.2031234672198900
            end

            @testset "Date in not the first column" begin
                df = DataFrame(Datas = ["2020-01-01", "2020-01-02", "2020-01-03"], Values = [1, 2, 3])
                data = load_dataset(df, true)
                println(data)
                println(values(data))
                @test size(data, 1) == 3
                @test values(data)[1,2] == 1
                @test values(data)[end,2] == 3
            end
        end
    end

    @testset "split_train_test" begin
        @testset "split_train_test(Datasets)" begin
            @testset "AIR_PASSENGERS" begin
                airPassengers = load_dataset(AIR_PASSENGERS)
                train, test = split_train_test(airPassengers; trainPercentage = 0.8)
                @test size(train, 1) == 163
                @test size(test, 1) == 41
            end

            @testset "GDPC1" begin
                GDPC1Data = load_dataset(GDPC1)
                train, test = split_train_test(GDPC1Data; trainPercentage = 0.8)
                @test size(train, 1) == 275
                @test size(test, 1) == 69
            end

            @testset "NROU" begin
                NROUData = load_dataset(NROU)
                train, test = split_train_test(NROUData; trainPercentage = 0.8)
                @test size(train, 1) == 275
                @test size(test, 1) == 69
            end
        end
    end
end
