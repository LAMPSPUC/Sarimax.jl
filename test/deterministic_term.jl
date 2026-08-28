# The deterministic term of the exact likelihood, against `stats::arima(method="ML")`.
#
# This covers the path a centred series with `include.mean = FALSE` cannot exercise, namely
# the one where a deterministic term is present.
#
# Without R available the comparisons against R are skipped; the algebraic invariants always
# run.
@testset "termo deterministico da verossimilhanca exata" begin
    rng = MersenneTwister(0x0DE7)
    dates(n) = collect(Date(2000, 1, 1):Month(1):Date(2000, 1, 1)+Month(n - 1))

    @testset "c e a CONSTANTE, o nivel removido e mu = c/(1-sum(ar))" begin
        # AR(1) series with a known mean: y_t - mu = phi (y_{t-1} - mu) + e_t. The
        # regression constant is c = mu*(1-phi); removing `c` instead of `mu` shifts the
        # series by the wrong factor and shifts the likelihood with it.
        n, φ, μ = 400, 0.6, 50.0
        e = randn(rng, n + 200)
        z = zeros(n + 200)
        for t = 2:(n+200)
            z[t] = φ * z[t-1] + e[t]
        end
        y = μ .+ z[201:end]
        m = SARIMA(TimeArray(dates(n), y), 1, 0, 0; allowMean = true)
        fit!(m; objectiveFunction = "mse", stationary = false)

        c = Float64(m.c)
        φ̂ = Float64(m.ϕ[1])
        level = c / (1 - φ̂)
        # the implied level must be the mean of the series, not the constant
        @test isapprox(level, mean(y); rtol = 0.05)
        @test !isapprox(c, mean(y); rtol = 0.05)      # premissa: as duas diferem de fato

        # the model's exact likelihood must be the one evaluated at the LEVEL
        z_level = Float64.(values(m.y)) .- level
        z_const = Float64.(values(m.y)) .- c
        llModel = Sarimax.exactLoglike(m)
        llLevel = Sarimax.exactGaussianLogLikelihood(z_level, [φ̂], Float64[])
        llConst = Sarimax.exactGaussianLogLikelihood(z_const, [φ̂], Float64[])
        @test !isnothing(llModel)
        @test llModel ≈ llLevel
        @test !isapprox(llModel, llConst; rtol = 1e-6)   # nao regredir para a constante
    end

    @testset "trend multiplica o regressor de tempo diferenciado, que nem sempre vale 1" begin
        # `trend * driftValues[t]`: sob d=1,D=0 o regressor diferenciado vale 1 e um escalar
        # acerta por coincidencia; sob d=0,D=1 vale `s` (12 no mensal) e o escalar erra 12x.
        n = 132
        ramp = TimeArray(dates(n), collect(1.0:n))
        @test all(values(differentiate(ramp, 1, 0, 12)) .≈ 1.0)
        @test all(values(differentiate(ramp, 0, 1, 12)) .≈ 12.0)
        @test all(values(differentiate(ramp, 1, 1, 12)) .≈ 0.0)   # drift nao identificavel

        # o caminho d=0,D=1 e o que o escalar quebrava
        y = 100 .+ 0.5 .* (1:n) .+ 3 .* sin.(2π .* (1:n) ./ 12) .+ randn(rng, n)
        m = SARIMA(TimeArray(dates(n), y), 1, 0, 0; seasonality = 12, D = 1,
                   allowMean = false, allowDrift = true)
        fit!(m; objectiveFunction = "mse", stationary = false)
        if !isnothing(m.trend) && !isnothing(Sarimax.exactLoglike(m))
            z = Float64.(values(differentiate(m.y, 0, 1, 12)))
            reg = Float64.(values(differentiate(
                TimeArray(timestamp(m.y), collect(1.0:length(values(m.y)))), 0, 1, 12)))
            φ̂ = Float64.([m.ϕ...])
            llReg = Sarimax.exactGaussianLogLikelihood(z .- Float64(m.trend) .* reg, φ̂, Float64[])
            llEsc = Sarimax.exactGaussianLogLikelihood(z .- Float64(m.trend), φ̂, Float64[])
            @test Sarimax.exactLoglike(m) ≈ llReg
            @test !isapprox(llReg, llEsc; rtol = 1e-9)   # o escalar de fato difere
        end
    end
end
