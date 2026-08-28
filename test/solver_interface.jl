@testset "Solver interface and reproducibility" begin
    @testset "optimizer accepts attributes, not just a constructor" begin
        # `Model(optimizer)` takes either form, but a `::DataType` annotation used to
        # reject `optimizer_with_attributes`. That left the caller stuck with solver
        # defaults — for a global solver such as SCIP the tolerances and limits are
        # attributes, so without this there is no way to ask for anything but an
        # indefinite search for a zero duality gap.
        Random.seed!(4001)
        n = 60
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        y = 10.0 .+ cumsum(randn(n) .* 0.4)

        configured = Sarimax.JuMP.optimizer_with_attributes(
            Sarimax.Ipopt.Optimizer, "max_iter" => 50, "print_level" => 0)
        model = SARIMA(TimeArray(collect(dates), y), 1, 1, 1; allowMean = false)
        @test_nowarn fit!(model; optimizer = configured)
        @test all(isfinite, model.ϕ)
        @test haskey(model.metadata, "solverStatus")
    end

    @testset "fitting is reproducible" begin
        # Same input, same result: a package used for benchmarking has to be
        # deterministic, otherwise published numbers cannot be reproduced.
        Random.seed!(4002)
        n = 80
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        y = 30.0 .+ cumsum(randn(n) .* 0.6)
        ta = TimeArray(collect(dates), y)

        m1 = SARIMA(ta, 1, 1, 1; allowMean = false)
        m2 = SARIMA(ta, 1, 1, 1; allowMean = false)
        fit!(m1)
        fit!(m2)
        @test m1.ϕ == m2.ϕ
        @test m1.θ == m2.θ
        @test m1.σ² == m2.σ²

        predict!(m1; stepsAhead = 5)
        predict!(m2; stepsAhead = 5)
        @test values(m1.forecast) == values(m2.forecast)
    end

    @testset "auto is reproducible and respects a time budget" begin
        # The budget is per candidate fit, so `auto` is not bounded by it exactly;
        # what must hold is that passing one does not break the search and that the
        # selection stays deterministic.
        Random.seed!(4003)
        n = 70
        dates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(n - 1))
        y = 15.0 .+ cumsum(randn(n) .* 0.5)
        ta = TimeArray(collect(dates), y)

        a1 = auto(ta; seasonality = 1, maxTimeSeconds = 30.0)
        a2 = auto(ta; seasonality = 1, maxTimeSeconds = 30.0)
        @test (a1.p, a1.d, a1.q) == (a2.p, a2.d, a2.q)
        @test isapprox(a1.σ², a2.σ²; rtol = 1e-10)
    end
end
