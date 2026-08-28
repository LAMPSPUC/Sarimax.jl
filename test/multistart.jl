# Contract of the {zero, CSS} multistart.
#
# The property the design guarantees is MONOTONICITY: the zero start remains one of the
# candidates and the tie-break is by `aicc`, so enabling `multistart` can never WORSEN the
# criterion. That is the strong invariant assertable without an M4-scale measurement; any
# forecasting gain is an empirical question and lives in the experiments.
#
# `isFitted` and `aicc` are not all exported from src/Sarimax.jl: qualify them with
# `Sarimax.`
@testset "multistart {zero, CSS}" begin
    Random.seed!(20260818)
    # ARMA(1,1) com sinal claro nos dois blocos, para que a partida tenha o que mover
    T = 220
    ε = randn(T + 60)
    y = zeros(T + 60)
    for t = 3:(T+60)
        y[t] = 0.65 * y[t-1] + ε[t] - 0.45 * ε[t-1]
    end
    ta = Sarimax.loadDataset(DataFrame(y = 100.0 .+ cumsum(y[61:end])))

    função(ms) = begin
        m = Sarimax.SARIMA(ta, 1, 1, 1; seasonality = 1, allowMean = false, allowDrift = true)
        Sarimax.fit!(m; objectiveFunction = "mse", silent = true, stationary = true,
                     invertible = false, initialization = :penalized, multistart = ms)
        m
    end
    base = função(false)
    multi = função(true)

    @test Sarimax.isFitted(base)
    @test Sarimax.isFitted(multi)

    # metadados do multistart so existem no caminho ligado
    @test !haskey(base.metadata, "multistartPartidas")
    @test haskey(multi.metadata, "multistartPartidas")
    @test multi.metadata["multistartPartidas"] >= 1
    @test multi.metadata["multistartVenceuCSS"] isa Bool

    # MONOTONIA: nunca pior que a partida unica. A folga de 1e-6 absorve o ruido do solver;
    # sem ela o teste vira um detector de ultimo bit do Ipopt.
    @test Sarimax.aicc(multi) <= Sarimax.aicc(base) + 1e-6

    # The multistart returns a COHERENT model: the residuals must be the winner's, not
    # those of one fit alongside the coefficients of another.
    @test length(multi.ϵ) == length(base.ϵ)
    @test isapprox(multi.σ², sum(abs2, multi.ϵ) / length(multi.ϵ); rtol = 0.5)

    # Under `:zeroed` the CSS seed IS the production start, so the result must coincide with
    # the simple path, not merely fail to worsen.
    zsimples = Sarimax.SARIMA(ta, 1, 1, 1; seasonality = 1, allowMean = false, allowDrift = true)
    Sarimax.fit!(zsimples; objectiveFunction = "mse", silent = true, stationary = true,
                 invertible = false, initialization = :zeroed, multistart = false)
    zmulti = Sarimax.SARIMA(ta, 1, 1, 1; seasonality = 1, allowMean = false, allowDrift = true)
    Sarimax.fit!(zmulti; objectiveFunction = "mse", silent = true, stationary = true,
                 invertible = false, initialization = :zeroed, multistart = true)
    @test Sarimax.aicc(zmulti) <= Sarimax.aicc(zsimples) + 1e-6
end
