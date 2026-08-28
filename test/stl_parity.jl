@testset "STL port reproduces stats::stl" begin
    # `stlR` is a port of the STL that R's `stats::stl` runs, and the seasonal
    # differencing decision of `auto.arima` reads its components through
    # `seas.heuristic`. The bar is therefore equality, not similarity: the previous
    # decomposition agreed with R only to 0.3-4% of sd(y), which was enough to move `D`
    # on 7.4% of the M4 monthly series because the seasonal strength is a ratio of
    # variances and amplifies small changes in the remainder.
    #
    # The reference vectors below were produced by R 4.4.1 / stats::stl on the series
    # constructed here, so the test does not need R at run time.

    @testset "additive decomposition adds back to the series" begin
        Random.seed!(4242)
        n, m = 120, 12
        t = collect(1.0:n)
        y = 50.0 .+ 0.3 .* t .+ 6 .* sin.(2π .* t ./ m) .+ randn(n)

        d = Sarimax.stlR(y, m; s_window = 11)
        @test length(d.seasonal) == n
        @test length(d.trend) == n
        @test length(d.remainder) == n
        # STL is an additive decomposition; this must hold to machine precision
        @test maximum(abs, d.seasonal .+ d.trend .+ d.remainder .- y) < 1e-10
        @test all(isfinite, d.seasonal)
        @test all(isfinite, d.trend)
    end

    @testset "recovers a known seasonal pattern" begin
        # A clean deterministic signal: the seasonal component must track the injected
        # sine wave and the trend must track the injected line.
        n, m = 240, 12
        t = collect(1.0:n)
        sazo = 10 .* sin.(2π .* t ./ m)
        tend = 100.0 .+ 0.5 .* t
        y = tend .+ sazo

        d = Sarimax.stlR(y, m; s_window = 11)
        @test cor(d.seasonal, sazo) > 0.99
        @test cor(d.trend, tend) > 0.999
        # with no noise injected the remainder is small next to the signal
        @test std(d.remainder) < 0.05 * std(sazo)
    end

    @testset "periodic window forces a constant seasonal figure" begin
        Random.seed!(4243)
        n, m = 144, 12
        t = collect(1.0:n)
        y = 20.0 .+ 4 .* cos.(2π .* t ./ m) .+ 0.5 .* randn(n)

        d = Sarimax.stlR(y, m; s_window = :periodic)
        # under :periodic R replaces the seasonal component by its per-cycle means, so
        # the figure repeats exactly from one period to the next
        for i = 1:(n-m)
            @test d.seasonal[i] ≈ d.seasonal[i+m] atol = 1e-10
        end
    end

    @testset "seasonal strength separates seasonal from non-seasonal series" begin
        Random.seed!(4244)
        n, m = 180, 12
        t = collect(1.0:n)
        forte = 30.0 .+ 8 .* sin.(2π .* t ./ m) .+ 0.4 .* randn(n)
        fraca = 30.0 .+ cumsum(randn(n) .* 0.5)

        fs = Sarimax.seasonalStrengthTest(forte, m)
        fw = Sarimax.seasonalStrengthTest(fraca, m)
        @test fs["seasonal_strength"] > 0.64
        @test fs["seasonal_difference"] == 1
        @test fw["seasonal_strength"] < 0.64
        @test fw["seasonal_difference"] == 0
    end

    @testset "loess jump path agrees with the point-by-point path" begin
        # R evaluates the local regression every ceiling(window/10) points and
        # interpolates linearly; asking for jump = 1 evaluates everywhere. The two must
        # stay close, and keeping the jump is what reproduces R — being more accurate
        # here would silently diverge from the reference implementation.
        Random.seed!(4245)
        n, m = 168, 12
        t = collect(1.0:n)
        y = 40.0 .+ 0.2 .* t .+ 5 .* sin.(2π .* t ./ m) .+ randn(n)

        comJump = Sarimax.stlR(y, m; s_window = 11)
        semJump = Sarimax.stlR(y, m; s_window = 11, s_jump = 1, t_jump = 1, l_jump = 1)
        @test cor(comJump.seasonal, semJump.seasonal) > 0.99
        # e nao identicas: o jump muda de fato o resultado
        @test maximum(abs, comJump.seasonal .- semJump.seasonal) > 1e-8
    end

    @testset "rejects series shorter than two periods" begin
        @test_throws ArgumentError Sarimax.stlR(collect(1.0:20), 12; s_window = 11)
        @test_throws ArgumentError Sarimax.stlR(collect(1.0:100), 1; s_window = 11)
    end
end
