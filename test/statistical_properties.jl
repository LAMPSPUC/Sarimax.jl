@testset "Statistical properties (simulation)" begin
    @testset "prediction-interval nominal coverage: random walk" begin
        # ARIMA(0,1,0): the 95% interval uses Var[h] = σ̂²·h on the original scale.
        # Empirical coverage over replications must be close to nominal — this is
        # the end-to-end check that uncertainty propagates through re-integration.
        Random.seed!(101)
        nReplicas = 100
        nTrain = 80
        horizon = 4
        covDates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(nTrain + horizon - 1))
        hits = zeros(Int, horizon)
        for _ = 1:nReplicas
            path = cumsum(randn(nTrain + horizon))
            train = TimeArray(collect(covDates)[1:nTrain], path[1:nTrain])
            model = SARIMA(train, 0, 1, 0; allowMean = false)
            fit!(model)
            predict!(model; stepsAhead = horizon, displayConfidenceIntervals = true,
                     confidenceLevel = 0.95)
            fc = values(model.forecast)   # columns: forecast, lower, upper
            for h = 1:horizon
                (fc[h, 2] <= path[nTrain+h] <= fc[h, 3]) && (hits[h] += 1)
            end
        end
        coverage = hits ./ nReplicas
        # binomial se ≈ 2.2%; allow a generous band around the 95% nominal level
        for h = 1:horizon
            @test 0.85 <= coverage[h] <= 1.0
        end
    end

    @testset "prediction-interval nominal coverage: estimated AR(1)" begin
        # Stationary case: intervals rely on the ψ-weights of the ESTIMATED model.
        Random.seed!(102)
        nReplicas = 50
        nTrain = 100
        horizon = 3
        ϕtrue = 0.6
        covDates = Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(nTrain + horizon - 1))
        hits = zeros(Int, horizon)
        for _ = 1:nReplicas
            path = zeros(nTrain + horizon)
            for t = 2:(nTrain+horizon)
                path[t] = ϕtrue * path[t-1] + randn()
            end
            train = TimeArray(collect(covDates)[1:nTrain], path[1:nTrain])
            model = SARIMA(train, 1, 0, 0; allowMean = false)
            fit!(model)
            predict!(model; stepsAhead = horizon, displayConfidenceIntervals = true,
                     confidenceLevel = 0.95)
            fc = values(model.forecast)
            for h = 1:horizon
                (fc[h, 2] <= path[nTrain+h] <= fc[h, 3]) && (hits[h] += 1)
            end
        end
        coverage = hits ./ nReplicas
        for h = 1:horizon
            @test 0.82 <= coverage[h] <= 1.0   # wider band: ϕ and σ² are estimated
        end
    end

    @testset "Monte Carlo bias: AR(1) with noise" begin
        Random.seed!(103)
        nReplicas = 20
        nObs = 300
        ϕtrue = 0.5
        mcDates = Date(1990, 1, 1):Month(1):(Date(1990, 1, 1)+Month(nObs - 1))
        estimates = zeros(nReplicas)
        for r = 1:nReplicas
            path = zeros(nObs)
            for t = 2:nObs
                path[t] = ϕtrue * path[t-1] + randn()
            end
            model = SARIMA(TimeArray(collect(mcDates), path), 1, 0, 0; allowMean = false)
            fit!(model)
            estimates[r] = model.ϕ[1]
        end
        # se(mean) ≈ sqrt((1-ϕ²)/n)/√R ≈ 0.011; allow small-sample (Kendall) bias too
        @test abs(mean(estimates) - ϕtrue) < 0.04
        @test all(abs.(estimates .- ϕtrue) .< 0.2)
        @test 0.01 < std(estimates) < 0.12
    end
end

@testset "SCIP optimizer path (certified global)" begin
    # The 'certified global optimum through the same fit! call' claim: on a small
    # MA(1), SCIP must terminate OPTIMAL and agree with Ipopt's local solution.
    Random.seed!(104)
    n = 40
    θtrue = 0.5
    noise = randn(n + 1)
    maPath = [noise[t+1] + θtrue * noise[t] for t = 1:n]
    scipDates = Date(2010, 1, 1):Month(1):(Date(2010, 1, 1)+Month(n - 1))
    series = TimeArray(collect(scipDates), maPath)

    ipoptModel = SARIMA(series, 0, 0, 1; allowMean = false)
    fit!(ipoptModel)

    scipModel = SARIMA(series, 0, 0, 1; allowMean = false)
    fit!(scipModel; optimizer = Sarimax.SCIP.Optimizer)
    @test Sarimax.isFitted(scipModel)
    @test scipModel.metadata["solverStatus"] == "OPTIMAL"   # global certificate
    @test scipModel.θ[1] ≈ ipoptModel.θ[1] atol = 1e-3
    # the certified optimum can be no worse than the local one
    rssSCIP = sum(abs2, residuals(scipModel))
    rssIpopt = sum(abs2, residuals(ipoptModel))
    @test rssSCIP <= rssIpopt + 1e-6
end
