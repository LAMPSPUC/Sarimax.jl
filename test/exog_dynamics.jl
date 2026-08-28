# Semantics of the exogenous block: :armax against :regression_errors.
#
# The two equations (Hyndman, "The ARIMAX model muddle", 2010):
#   :armax              phi(B) Phi(B^s) y_t = X_t'b + theta(B) Theta(B^s) e_t
#   :regression_errors  y_t = X_t'b + eta_t ,  phi(B) Phi(B^s) eta_t = theta(B) Theta(B^s) e_t
#
# These tests assert OUTPUT PROPERTIES rather than internals: (i) naming the default changes
# nothing, (ii) the algebraic identity holds exactly where the algebra says it does and
# nowhere else, and (iii) under the ARIMA-errors form the coefficient recovers the marginal
# effect of the data generating process.

@testset "exogDynamics" begin
    mkTA(v) = TimeArray(
        collect(Date(2000, 1, 1):Month(1):Date(2000, 1, 1)+Month(length(v)-1)), v)
    mkTX(X) = TimeArray(
        collect(Date(2000, 1, 1):Month(1):Date(2000, 1, 1)+Month(size(X, 1)-1)), X)

    # `y` has T points; `X` has T + HMAX, because `predict!` requires the future regressor
    # over the requested horizon.
    HMAX = 12

    "y = X*b + eta, com eta ~ ARMA(1,1)x(P,0,Q)[12]. P = 0 => polinomio AR unitario."
    function geraErrosArima(seed, P; T = 120, S = 12, β = [2.0, -1.5])
        rng = MersenneTwister(seed)
        burn, K = 300, length(β)
        n = burn + T + HMAX
        X, e = randn(rng, n, K), randn(rng, n)
        ϕ, θ, Φ, Θ = 0.5, 0.4, (P > 0 ? 0.6 : 0.0), (P > 0 ? 0.0 : 0.7)
        η = zeros(n)
        for t = 14:n
            η[t] = ϕ * η[t-1] + Φ * η[t-12] - ϕ * Φ * η[t-13] +
                   e[t] + θ * e[t-1] + Θ * e[t-12] + θ * Θ * e[t-13]
        end
        y = X * β .+ η
        rg = (n-T-HMAX+1):n
        (mkTA(y[rg][1:T]), mkTX(X[rg, :]))
    end

    ajusta(y, X, p, P, q, Q; kw...) = begin
        m = SARIMA(y, X, p, 0, q; P = P, D = 0, Q = Q, seasonality = 12,
                   allowMean = false, allowDrift = false)
        fit!(m; objectiveFunction = "mse", initialization = :free,
             stationary = true, invertible = false, kw...)
        m
    end

    @testset "argumento invalido e recusado" begin
        y, X = geraErrosArima(1, 1)
        m = SARIMA(y, X, 1, 0, 1; P = 1, D = 0, Q = 0, seasonality = 12)
        @test_throws ArgumentError fit!(m; exogDynamics = :naosei)
    end

    # The default is `:armax`, the ARX/impact-multiplier form. This asserts that naming the
    # default explicitly gives the same result as omitting it: if they diverge, the keyword
    # is not being read on the path the default takes.
    @testset "o default e :armax, e nomea-lo e bit-identico a omitir" begin
        y, X = geraErrosArima(2, 1)
        a = ajusta(y, X, 1, 1, 1, 0)
        b = ajusta(y, X, 1, 1, 1, 0; exogDynamics = :armax)
        @test a.exogCoefficients == b.exogCoefficients
        @test a.ϕ == b.ϕ
        @test a.Φ == b.Φ
        @test a.θ == b.θ
        predict!(a; stepsAhead = 6)
        predict!(b; stepsAhead = 6)
        @test values(a.forecast) == values(b.forecast)
    end

    # THE IDENTITY. The two forms coincide iff the AR polynomial (regular and seasonal) is
    # unitary and there is no differencing, since filtering X by 1 does nothing. Pinning it
    # stops a future refactor from breaking the equivalence silently.
    @testset "p = P = 0: as duas semanticas sao a MESMA equacao" begin
        y, X = geraErrosArima(3, 0)
        a = ajusta(y, X, 0, 0, 1, 1)
        b = ajusta(y, X, 0, 0, 1, 1; exogDynamics = :regression_errors)
        @test a.exogCoefficients ≈ b.exogCoefficients atol = 1e-8
        @test a.θ ≈ b.θ atol = 1e-8
        @test a.Θ ≈ b.Θ atol = 1e-8
        predict!(a; stepsAhead = 12)
        predict!(b; stepsAhead = 12)
        @test values(a.forecast) ≈ values(b.forecast) atol = 1e-6
    end

    # ...and nowhere else. With p > 0 the two MUST diverge: if they do not, the flag is not
    # binding. BOTH sides are named deliberately — a divergence test that leaves one side on
    # the default silently compares a mode with itself whenever the default moves.
    @testset "p > 0: as duas semanticas divergem (o flag vincula)" begin
        y, X = geraErrosArima(4, 1)
        a = ajusta(y, X, 1, 1, 1, 0; exogDynamics = :armax)
        b = ajusta(y, X, 1, 1, 1, 0; exogDynamics = :regression_errors)
        predict!(a; stepsAhead = 12)
        predict!(b; stepsAhead = 12)
        @test maximum(abs.(values(a.forecast) .- values(b.forecast))) > 1e-3
    end

    # `auto` must expose the same keyword as `fit!`: without it the alternative semantics
    # is unreachable from the package's main entry point. With p = P = 0 the two forms
    # coincide by the identity above, so the flag is exercised where it can bind.
    @testset "auto expoe exogDynamics" begin
        y, X = geraErrosArima(7, 0)
        common = (exog = X, seasonality = 12, maxp = 2, maxq = 0, maxP = 0, maxQ = 0,
                  d = 0, D = 0)
        a = auto(y; common..., exogDynamics = :armax)
        b = auto(y; common..., exogDynamics = :regression_errors)
        @test a.metadata["exogDynamics"] == "armax"
        @test b.metadata["exogDynamics"] == "regression_errors"
        @test_throws Exception auto(y; common..., exogDynamics = :naosei)
    end

    # `auto` differences the exogenous block whenever d > 0, which requires `differentiate`
    # to accept more than one column. Anything else makes multi-regressor SARIMAX
    # unreachable from `auto`.
    @testset "auto aceita mais de uma exogena" begin
        y, X = geraErrosArima(8, 0)
        m = auto(y; exog = X, seasonality = 12, maxp = 1, maxq = 0, maxP = 0, maxQ = 0)
        @test Sarimax.isFitted(m)
        @test length([m.exogCoefficients...]) == size(values(X), 2)
    end

    # Under the ARIMA-errors form, b is the marginal effect of the data generating process
    # and must be recovered. Under :armax, b is an impact multiplier and has no such
    # interpretation, so this asserts only the side the theory guarantees.
    @testset ":regression_errors recupera o efeito marginal" begin
        β = [2.0, -1.5]
        erros = Float64[]
        for seed = 10:15
            y, X = geraErrosArima(seed, 1; β = β)
            m = ajusta(y, X, 1, 1, 1, 0; exogDynamics = :regression_errors)
            push!(erros, maximum(abs.(Float64.([m.exogCoefficients...]) .- β)))
        end
        @test median(erros) < 0.25
    end
end
