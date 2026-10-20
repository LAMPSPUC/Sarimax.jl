# The exact likelihood that scores models with regressors (aic, aicc, bic and every `auto`
# search) must be the likelihood of the model `fit!` estimated.
#
#   :armax              phi*(B) z_t = c + X_t'b + theta*(B) e_t
#                       => z_t - c/phi*(1) - mu_t ~ ARMA,  phi*(B) mu_t = X_t'b
#   :regression_errors  z_t - X_t'b - c/phi*(1) ~ ARMA
#
# with z the differenced series and X the regressors as `fit!` used them (not differenced).
# The reference below is the dense multivariate normal density of the ARMA part, with its
# autocovariances from psi weights computed here, sharing no code with
# src/exact_likelihood.jl. The regressor path under :armax starts at its steady state,
# mu_t = X_1'b / phi*(1) for t <= 0 (the convention documented in `armaxRegressionPath`).
#
# Up to 1.0.0 the package scored both semantics as regression with ARMA errors on regressors
# differenced with the series' d, and a single one-dimensional regressor made the likelihood
# unavailable (silent CSS fallback). On that code 15 of these 21 checks fail or error; the 6
# that pass are cases it already scored correctly (regression with errors at d = 0, no AR terms).

@testset "exact likelihood with regressors" begin
    LA = Sarimax.LinearAlgebra

    function psiRef(ar, ma, n)
        ψ = zeros(n + 1)
        ψ[1] = 1.0
        for k = 1:n
            acc = k <= length(ma) ? ma[k] : 0.0
            for i = 1:min(k, length(ar))
                acc += ar[i] * ψ[k-i+1]
            end
            ψ[k+1] = acc
        end
        ψ
    end
    function mvnRef(w, ar, ma)
        n, L = length(w), 4000
        ψ = psiRef(ar, ma, L)
        γ = [sum(ψ[j] * ψ[j+h] for j = 1:(L+1-h)) for h = 0:(n-1)]
        Γ = [γ[abs(i - j)+1] for i = 1:n, j = 1:n]
        σ² = LA.dot(w, Γ \ w) / n
        -(n / 2) * (log(2π * σ²) + 1) - LA.logdet(Γ) / 2
    end
    function pathRef(xb, φ)
        μ, prev = zeros(length(xb)), xb[1] / (1 - φ)
        for t in eachindex(xb)
            μ[t] = xb[t] + φ * prev
            prev = μ[t]
        end
        μ
    end

    # Two regressors with non-zero means; z follows the `kind` equation, y = z (d = 0) or
    # its cumulative sum (d = 1). `extra` future rows of X are appended for forecasting.
    function simulate(kind; d = 0, nreg = 2, T = 240, burn = 200, extra = 0, seed = 20261021)
        rng = MersenneTwister(seed)
        N = T + burn + extra
        X = zeros(N, 2)
        u = randn(rng, N, 2)
        for t = 2:N
            X[t, 1] = 5 + 0.7 * (X[t-1, 1] - 5) + u[t, 1]
            X[t, 2] = -2 + 0.4 * (X[t-1, 2] + 2) + u[t, 2]
        end
        b = [2.0, -1.5][1:nreg]
        e, z, η = randn(rng, N), zeros(N), zeros(N)
        for t = 2:N
            xb = sum(X[t, j] * b[j] for j = 1:nreg)
            if kind === :armax
                z[t] = 0.6 * z[t-1] + xb + e[t] + 0.3 * e[t-1]
            else
                η[t] = 0.6 * η[t-1] + e[t] + 0.3 * e[t-1]
                z[t] = xb + η[t]
            end
        end
        y = d == 0 ? z : cumsum(z)
        ts = collect(Date(2000, 1, 1):Day(1):(Date(2000, 1, 1)+Day(T + extra - 1)))
        # a single regressor as the one-dimensional TimeArray that `load_dataset` returns
        Xta = nreg == 1 ? TimeArray(ts, X[(burn+1):end, 1], [:x1]) :
                          TimeArray(ts, X[(burn+1):end, 1:nreg], [:x1, :x2][1:nreg])
        TimeArray(ts[1:T], y[(burn+1):(burn+T)], [:y]), Xta
    end

    fitWith(y, X, p, d, q, sem) = begin
        m = SARIMA(y, X, p, d, q)
        fit!(m; exogDynamics = sem)
        m
    end
    function reference(m, y, X, d, sem)
        φ = m.p > 0 ? m.ϕ[1] : 0.0
        θ = m.q > 0 ? [m.θ[1]] : Float64[]
        z = Float64.(values(Sarimax.differentiate(y, d, 0, 1)))
        Xv = Float64.(values(X))
        Xv = ndims(Xv) == 1 ? reshape(Xv, :, 1) : Xv
        xb = Xv[(d+1):length(values(y)), :] * Float64.(m.exogCoefficients)
        level = something(m.c, 0.0) / (1 - φ)
        w = z .- level .- (sem === :armax ? pathRef(xb, φ) : xb)
        mvnRef(w, m.p > 0 ? [φ] : Float64[], θ)
    end

    @testset "the likelihood of the fitted semantics (d = $d)" for d in (0, 1)
        for kind in (:armax, :regression_errors), sem in (:armax, :regression_errors)
            y, X = simulate(kind; d = d)
            m = fitWith(y, X, 1, d, 1, sem)
            @test Sarimax.exactLoglike(m) ≈ reference(m, y, X, d, sem) rtol = 1e-8
        end
    end

    @testset "the criterion prefers the semantics that generated the data" begin
        for kind in (:armax, :regression_errors)
            y, X = simulate(kind)
            la = Sarimax.exactLoglike(fitWith(y, X, 1, 0, 1, :armax))
            lr = Sarimax.exactLoglike(fitWith(y, X, 1, 0, 1, :regression_errors))
            @test kind === :armax ? la > lr + 10 : lr > la + 10
        end
    end

    @testset "a single regressor is scored, not sent to the CSS fallback" begin
        y, X = simulate(:armax; nreg = 1)
        m = fitWith(y, X, 1, 0, 1, :armax)
        @test !isnothing(Sarimax.exactLoglike(m))
        @test Sarimax.exactLoglike(m) ≈ reference(m, y, X, 0, :armax) rtol = 1e-8
        aic(m)
        @test m.metadata["criterionFallback"] == false
    end

    @testset "without AR terms the two semantics are the same likelihood" begin
        y, X = simulate(:regression_errors)
        a = fitWith(y, X, 0, 0, 1, :armax)
        r = fitWith(y, X, 0, 0, 1, :regression_errors)
        @test Sarimax.exactLoglike(a) ≈ Sarimax.exactLoglike(r) rtol = 1e-8
        @test Sarimax.exactLoglike(a) ≈ reference(a, y, X, 0, :armax) rtol = 1e-8
    end

    @testset "future regressor rows do not enter the likelihood" begin
        y, X = simulate(:armax; extra = 12)
        @test length(values(X)[:, 1]) == length(values(y)) + 12
        m = fitWith(y, X, 1, 0, 1, :armax)
        @test Sarimax.exactLoglike(m) ≈ reference(m, y, X, 0, :armax) rtol = 1e-8
    end

    @testset "armaxRegressionPath: a constant input is the constant's level" begin
        ar = [0.5, 0.2]
        @test Sarimax.armaxRegressionPath(fill(3.0, 20), ar) ≈ fill(3.0 / 0.3, 20)
        @test Sarimax.armaxRegressionPath([1.0, 2.0, 3.0], Float64[]) == [1.0, 2.0, 3.0]
        @test isnothing(Sarimax.armaxRegressionPath(ones(5), [0.4, 0.6]))   # unit root
    end

    @testset "auto with a regressor ranks by the exact likelihood" begin
        y, X = simulate(:armax; nreg = 1)
        m = auto(y; exog = X, seasonality = 1)
        @test m.metadata["criterionFallback"] == false
    end
end
