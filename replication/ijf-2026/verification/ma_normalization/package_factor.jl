# Which moving-average normalization the package applies, measured on the package itself.
#
#     julia +1.10.12 package_factor.jl PATH_TO_A_CHECKOUT_OF_THE_REVISION
#
# A seasonal MA model y_t = Psi(B) e_t with Psi = theta(B) Theta(B^s) (multiplicative) or
# 1 + theta B + Theta B^s (additive) is fitted with its coefficients held fixed, under
# `objectiveFunction = "mse"` and `initialization = :innovations`. The solver then minimises over
# the pre-sample innovations only, and every residual is affine in them, so the inner minimum
# S_innov is a linear least-squares problem solved here without the package. The recorded
# objective divided by S_innov is the normalization factor the package applies, which is
# compared with
#   * D_reg(theta) D_seas(Theta)                         (the separate factors), and
#   * the complete factor of Supplementary Section S2: D_reg D_seas [R^2]^(-1/T) for the
#     multiplicative polynomial, and the factor of the complete polynomial for the additive one.
# Each case pins the separate factors with a @test and asserts that the complete factor is NOT
# what the package applies, so a change either way shows.
using Pkg
Pkg.activate(; temp = true)
Pkg.develop(path = ARGS[1])
Pkg.add("TimeSeries")
using Dates, LinearAlgebra, Random, Sarimax, Statistics, Test, TimeSeries
println("Sarimax at ", pkgdir(Sarimax))

# reflection coordinates of 1 + c_1 x + ... + c_m x^m (inverse of Sarimax.reflectionToMA)
function reflection(coef)
    c, η = collect(Float64, coef), zeros(length(coef))
    for k = length(coef):-1:1
        e = c[k]
        η[k] = e
        c = [(c[i] - e * c[k-i]) / (1 - e^2) for i = 1:(k-1)]
    end
    return η
end
logdet_limit(coef, mult = 1) = -sum(mult * j * log(1 - η^2) for (j, η) in enumerate(reflection(coef)))

function profile(y, θ, Θ, s, form)          # min over e_pre of |e_pre|^2 + sum_t e_t^2
    T, m = length(y), 1 + s                   # the package's window: q + s*Q
    function resid(x)
        e = Dict(t => x[k] for (k, t) in enumerate((1-m):0))
        out = zeros(T)
        for t = 1:T
            v = y[t] - θ * e[t-1] - Θ * e[t-s] - (form === :multiplicative ? θ * Θ * e[t-s-1] : 0.0)
            e[t] = v
            out[t] = v
        end
        return out
    end
    r0 = resid(zeros(m))
    W = reduce(hcat, [resid(Matrix(1.0I, m, m)[:, k]) .- r0 for k = 1:m])
    x = -((I + W' * W) \ (W' * r0))
    return sum(abs2, x) + sum(abs2, r0 .+ W * x)
end

@testset "MA normalization applied by the package" begin
    s, T = 4, 120
    rng = MersenneTwister(20261024)
    e = randn(rng, T + s + 1)
    θ0, Θ0 = 0.7, 0.6
    y = [e[t+s+1] + θ0 * e[t+s] + Θ0 * e[t+1] + θ0 * Θ0 * e[t] for t = 1:T]
    ta = TimeArray(collect(Date(2000, 1, 1):Month(1):(Date(2000, 1, 1)+Month(T - 1))), y)
    ys = y ./ std(y)                          # the solver works on y / std(differenced series)
    # additive pairs are chosen so that the complete polynomial 1 + theta B + Theta B^s is
    # invertible, where its factor is defined
    cases = [(:multiplicative, (0.7, 0.6)), (:multiplicative, (-0.8, 0.7)), (:multiplicative, (0.9, -0.5)),
             (:additive, (0.5, 0.3)), (:additive, (-0.4, 0.5)), (:additive, (0.6, -0.3))]
    for (form, (θ, Θ)) in cases
        m = SARIMA(ta, 0, 0, 1; seasonality = s, Q = 1, allowMean = false)
        m.θ, m.Θ = [θ], [Θ]
        m.keepProvidedCoefficients = true
        fit!(m; objectiveFunction = "mse", initialization = :innovations, invertible = false,
             seasonalForm = form)
        applied = m.metadata["objectiveValue"] / profile(ys, θ, Θ, s, form)
        separate = exp((logdet_limit([θ]) + logdet_limit([Θ], s)) / T)
        if form === :multiplicative
            R = 1 + Θ * (-θ)^s                # Res(P_theta, Q_Theta) for q = Q = 1
            complete = separate * (R^2)^(-1 / T)
        else
            complete = exp(logdet_limit([θ; zeros(s - 2); Θ]) / T)
        end
        println(rpad("$form theta=$θ Theta=$Θ", 40), " applied ", round(applied; digits = 8),
                "  separate ", round(separate; digits = 8), "  complete ", round(complete; digits = 8))
        @test isapprox(applied, separate; rtol = 1e-6)
        @test !isapprox(applied, complete; rtol = 1e-5)
    end
end
