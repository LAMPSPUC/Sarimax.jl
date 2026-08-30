# The Gaussian determinant normalization of the quadratic pre-sample objective belongs to the
# COMPLETE moving-average polynomial the model represents, not to the regular and seasonal
# factors taken separately.
#
# The separate form is exact when one of the blocks is empty, which is why it survived: every
# pure case reduces to it. It is wrong whenever q > 0 AND Q > 0, because the product
# theta(B)*Theta(B^s) carries cross coefficients theta_j*Theta_w at lags j + s*w and the
# reflection coordinates of a product are not a function of the coordinates of its factors.
#
# The tests below are mathematical, not end-to-end: the identity is checked against a
# determinant built from the inverse-MA recursion, which shares no code with the reflection
# path being tested.

"""
Determinant of the infinite-history Gram matrix of the homogeneous inverse-MA recursion,
computed WITHOUT any reflection coefficient.

For `Psi(B) = 1 + a_1 B + ... + a_L B^L` the recursion `x_t = -sum_l a_l x_{t-l}` has
companion matrix `A`, and

    H_inf = I + sum_{t>=1} (A^t)' e_1 e_1' A^t.

`det(H_inf)` is the quantity the reflection formula `prod_l (1-kappa_l^2)^(-l)` claims to
equal. Converges whenever `Psi` is invertible, since the spectral radius of `A` is then < 1.
"""
function inverseMAGramDeterminant(a::AbstractVector; tol::Float64 = 1e-15, maxIter::Int = 500_000)
    L = length(a)
    L == 0 && return 1.0
    af = collect(float.(a))
    A = zeros(Float64, L, L)
    A[1, :] = -af
    for i = 2:L
        A[i, i-1] = 1.0
    end
    H = Matrix{Float64}(LinearAlgebra.I, L, L)
    P = Matrix{Float64}(LinearAlgebra.I, L, L)
    for _ = 1:maxIter
        P = A * P                      # P = A^t
        v = P[1, :]                    # e_1' A^t
        update = v * transpose(v)
        H .+= update
        maximum(abs, update) < tol && break
    end
    return LinearAlgebra.det(H)
end

"The determinant factor the OLD code computed: the two blocks normalized independently."
function separateBlockDeterminantFactor(θ::Vector{Float64}, Θ::Vector{Float64}, s::Int, T::Real)
    f = 1.0
    if !isempty(θ)
        κ = Sarimax.maToReflection(θ)
        f *= prod((1 - κ[j]^2)^(-j / T) for j = 1:length(κ))
    end
    if !isempty(Θ)
        κs = Sarimax.maToReflection(Θ)
        f *= prod((1 - κs[w]^2)^(-s * w / T) for w = 1:length(κs))
    end
    return f
end

@testset "Complete MA polynomial: construction" begin
    @testset "multiplicative expansion carries the cross terms" begin
        # (1 + 0.7B)(1 + 0.4B^3) = 1 + 0.7B + 0.4B^3 + 0.28B^4
        a = Sarimax.fullMACoefficients([0.7], [0.4], 3, :multiplicative)
        @test length(a) == 1 + 3 * 1
        @test isapprox(Float64[x for x in a], [0.7, 0.0, 0.4, 0.28]; atol = 1e-12)
    end

    @testset "a seasonal lag colliding with a regular one ADDS" begin
        # (1 + 0.5B + 0.3B^2)(1 + 0.4B^2) = 1 + 0.5B + 0.7B^2 + 0.2B^3 + 0.12B^4
        a = Sarimax.fullMACoefficients([0.5, 0.3], [0.4], 2, :multiplicative)
        @test isapprox(Float64[x for x in a], [0.5, 0.7, 0.2, 0.12]; atol = 1e-12)
    end

    @testset "additive form has no cross terms" begin
        a = Sarimax.fullMACoefficients([0.7], [0.4], 3, :additive)
        @test isapprox(Float64[x for x in a], [0.7, 0.0, 0.4]; atol = 1e-12)
    end

    @testset "degenerate orders" begin
        @test isempty(Sarimax.fullMACoefficients(Float64[], Float64[], 12, :multiplicative))
        @test isapprox(
            Float64[x for x in Sarimax.fullMACoefficients([0.6], Float64[], 12, :multiplicative)],
            [0.6];
            atol = 1e-12,
        )
        pure = Float64[x for x in Sarimax.fullMACoefficients(Float64[], [0.5], 4, :multiplicative)]
        @test pure == [0.0, 0.0, 0.0, 0.5]
    end

    @testset "single source of truth: agrees with the existing MA expansion" begin
        # `expandMultiplicativePolynomial(...; negate = false)` is the numeric-only MA
        # expansion the exact-likelihood path used. Pinned equal so the two cannot drift.
        for (θ, Θ, s) in (
            ([0.5], [0.3], 12),
            ([0.5, -0.2], [0.3], 4),
            ([0.4], [0.2, 0.1], 3),
            (Float64[], [0.3], 12),
            ([0.7], Float64[], 12),
        )
            @test isapprox(
                Float64[x for x in Sarimax.fullMACoefficients(θ, Θ, s, :multiplicative)],
                Sarimax.expandMultiplicativePolynomial(θ, Θ, s; negate = false);
                atol = 1e-14,
            )
        end
    end
end

@testset "Determinant normalization of the complete MA polynomial" begin
    @testset "independent validation against the inverse-MA Gram determinant" begin
        # This is the test that matters: it validates the statistical identity WITHOUT the
        # reflection code path, so a bug shared by the formula and its implementation cannot
        # hide here.
        for a in (
            [0.5],
            [0.4, 0.2],
            [0.6, -0.3],
            [0.5, 0.3, 0.2],
            [0.7, 0.0, 0.4, 0.28],          # (1+0.7B)(1+0.4B^3)
            [0.5, 0.7, 0.2, 0.12],          # (1+0.5B+0.3B^2)(1+0.4B^2)
        )
            @test isapprox(
                Sarimax.fullMADeterminantFactor(a, 1),
                inverseMAGramDeterminant(a);
                rtol = 1e-10,
            )
        end
    end

    @testset "reduction: q = Q = 0 gives exactly 1" begin
        @test Sarimax.fullMADeterminantFactor(Float64[], 100) == 1.0
    end

    @testset "reduction: Q = 0 reproduces the regular-block formula exactly" begin
        T = 137
        for θ in ([0.5], [0.4, 0.2], [0.6, -0.3])
            a = Float64[x for x in Sarimax.fullMACoefficients(θ, Float64[], 12, :multiplicative)]
            @test isapprox(
                Sarimax.fullMADeterminantFactor(a, T),
                separateBlockDeterminantFactor(θ, Float64[], 12, T);
                rtol = 1e-12,
            )
        end
    end

    @testset "reduction: q = 0 reproduces the pure-seasonal formula exactly" begin
        # With q = 0 the polynomial is a polynomial in B^s: the s phase chains decouple and
        # the seasonal block really does behave as s independent copies. That reading is what
        # fails once q > 0.
        T = 211
        for s in (4, 12), Θ in ([0.5], [0.4, 0.2], [-0.35])
            a = Float64[x for x in Sarimax.fullMACoefficients(Float64[], Θ, s, :multiplicative)]
            @test isapprox(
                Sarimax.fullMADeterminantFactor(a, T),
                separateBlockDeterminantFactor(Float64[], Θ, s, T);
                rtol = 1e-10,
            )
        end
    end

    @testset "COUNTEREXAMPLE: with q > 0 and Q > 0 the separate form is wrong" begin
        for (θ, Θ, s) in (([0.7], [0.4], 3), ([0.7], [0.4], 2), ([0.5, 0.2], [0.3], 4))
            a = Float64[x for x in Sarimax.fullMACoefficients(θ, Θ, s, :multiplicative)]
            truth = inverseMAGramDeterminant(a)
            novo = Sarimax.fullMADeterminantFactor(a, 1)
            antigo = separateBlockDeterminantFactor(θ, Θ, s, 1)
            # the new implementation is the true determinant ...
            @test isapprox(novo, truth; rtol = 1e-10)
            # ... and the old one is measurably not
            @test abs(antigo - truth) / truth > 0.01
        end
        # The headline case, stated numerically so a regression is legible:
        a = Float64[x for x in Sarimax.fullMACoefficients([0.7], [0.4], 3, :multiplicative)]
        @test isapprox(Sarimax.fullMADeterminantFactor(a, 1), 4.443975877; rtol = 1e-8)
        @test isapprox(separateBlockDeterminantFactor([0.7], [0.4], 3, 1), 3.308201588; rtol = 1e-8)
    end
end

@testset "Staged JuMP construction equals the numeric recursion" begin
    # `maDeterminantDenominators!` stages the step-down recursion through auxiliary variables
    # instead of nesting rational expressions. It must produce the same `1 - kappa_l^2`.
    for a in ([0.5], [0.4, 0.2], [0.7, 0.0, 0.4, 0.28], [0.5, 0.7, 0.2, 0.12])
        mod = Sarimax.JuMP.Model(Sarimax.Ipopt.Optimizer)
        Sarimax.JuMP.set_silent(mod)
        L = length(a)
        Sarimax.JuMP.@variable(mod, x[1:L])
        Sarimax.JuMP.fix.(x, a; force = true)
        d = Sarimax.maDeterminantDenominators!(mod, [x[i] for i = 1:L])
        Sarimax.JuMP.@objective(mod, Min, 0.0)
        Sarimax.JuMP.optimize!(mod)
        κ = Sarimax.maToReflection(a)
        esperado = [1 - κ[l]^2 for l = 1:L]
        @test isapprox(Sarimax.JuMP.value.(d), esperado; atol = 1e-8)
    end

    @testset "agrees with the small-order expression reference" begin
        for a in ([0.5], [0.4, 0.2], [0.6, -0.3])
            expr = Sarimax.maToReflectionExpr(a, length(a))
            @test isapprox(
                [1 - Float64(expr[l])^2 for l in eachindex(expr)],
                [1 - Sarimax.maToReflection(a)[l]^2 for l in eachindex(expr)];
                atol = 1e-12,
            )
        end
    end
end

@testset "Invertibility domain of the normalization" begin
    # The normalization is real only where every 1 - kappa_l^2 > 0, i.e. on the invertible
    # region of the COMPLETE polynomial. What that costs depends on the seasonal form.
    invertivel(a) = isempty(a) || all(abs.(Sarimax.maToReflection(collect(float.(a)))) .< 1)

    @testset "multiplicative: the region is the SAME as before" begin
        # Psi(B) = theta(B)*Theta(B^s), so the roots of Psi are the roots of the two factors
        # together: Psi is invertible exactly when both blocks are. Requiring the complete
        # polynomial to be invertible therefore asks for nothing the separate normalization
        # did not already require, and the explicit floor replaces an implicit domain error
        # rather than shrinking the feasible set.
        for (θ, Θ, s) in (
            ([0.5], [0.4], 4),
            ([0.9], [0.8], 3),
            ([0.4, 0.3], [0.5], 12),
            ([-0.7], [0.6, -0.2], 4),
        )
            blocos = invertivel(θ) && invertivel(Θ)
            completo = invertivel(Float64[x for x in
                                          Sarimax.fullMACoefficients(θ, Θ, s, :multiplicative)])
            @test blocos == completo
        end
        # and a non-invertible factor makes the complete polynomial non-invertible too
        naoInv = Float64[x for x in Sarimax.fullMACoefficients([1.5], [0.4], 3, :multiplicative)]
        @test !invertivel(naoInv)
    end

    @testset "additive: the region is NOT the per-block one" begin
        # Under the additive form Psi(B) is a SUM, not a product, so its invertibility does
        # not follow from the blocks'. Here the complete-polynomial requirement is a genuinely
        # different (and correct) condition, and the per-block check was answering another
        # question. Recorded so the two forms are not treated alike.
        # Psi(B) = 1 - 0.9B - 0.9B^2. Each block is an invertible MA(1) on its own, but the
        # MA(2) they sum to violates a_1 + a_2 > -1 and is not invertible.
        θ, Θ, s = [-0.9], [-0.9], 2
        @test invertivel(θ) && invertivel(Θ)
        a = Float64[x for x in Sarimax.fullMACoefficients(θ, Θ, s, :additive)]
        @test a == [-0.9, -0.9]
        @test !invertivel(a)
    end
end

@testset "Objective regression: the objective carries the full-polynomial factor" begin
    # Validates the OBJECTIVE VALUE, not merely fitted coefficients.
    #
    # The argument avoids having to reconstruct the sum of squares, which is itself the
    # result of an optimization over the free pre-sample block. Instead it uses the fact
    # that ONE polynomial can be written two ways:
    #
    #   A: q = 1, Q = 1, s = 3, theta = 0.7, Theta = 0.4   ->  Psi(B) = (1+0.7B)(1+0.4B^3)
    #   B: q = 4, Q = 0, s = 1, theta = [0.7, 0, 0.4, 0.28] ->  the same Psi(B), expanded
    #
    # Both produce the same one-step recursion, so the same feasible set and the same
    # optimal S. Both have residualLags = 4, hence the same conditioning and the same
    # pre-sample block. They must therefore return the SAME objective.
    #
    # Case B is a PURE REGULAR MA: there the separate and the complete normalizations
    # coincide, and the regular formula is the one already validated against a directly
    # constructed MA covariance. So B's objective is known to be S * D_full. Equality of the
    # two objectives then pins A's normalization to D_full as well.
    #
    # Under the OLD code the two disagreed: A was normalized by the separate-block product
    # and B by the complete one, for the same polynomial and the same data.
    Random.seed!(20260829)
    s = 3
    T = 120
    y = 10.0 .+ cumsum(randn(T) .* 0.1)
    dates = Date(2020, 1, 1):Day(1):(Date(2020, 1, 1)+Day(T - 1))
    ta = TimeArray(collect(dates), y)

    θfix, Θfix = [0.7], [0.4]
    expandido = Float64[x for x in Sarimax.fullMACoefficients(θfix, Θfix, s, :multiplicative)]

    function ajusta(; p, q, P, Q, seasonality, θ, Θ)
        m = SARIMA(ta, p, 0, q; seasonality = seasonality, P = P, D = 0, Q = Q, silent = true)
        m.θ = θ
        isnothing(Θ) || (m.Θ = Θ)
        m.keepProvidedCoefficients = true
        Sarimax.fit!(
            m;
            objectiveFunction = "mse",
            initialization = :innovations,
            seasonalForm = :multiplicative,
            stationary = true,
            invertible = false,
            silent = true,
        )
        return m
    end

    mA = ajusta(p = 0, q = 1, P = 0, Q = 1, seasonality = s, θ = θfix, Θ = Θfix)
    mB = ajusta(p = 0, q = 4, P = 0, Q = 0, seasonality = 1, θ = expandido, Θ = nothing)

    objA = get(mA.metadata, "objectiveValue", nothing)
    objB = get(mB.metadata, "objectiveValue", nothing)
    @test !isnothing(objA)
    @test !isnothing(objB)
    @test isapprox(objA, objB; rtol = 1e-6)

    # And the two candidate normalizations are far apart at this point, so the equality
    # above discriminates rather than passing under either.
    dFull = Sarimax.fullMADeterminantFactor(expandido, T)
    dSeparate = separateBlockDeterminantFactor(θfix, Θfix, s, T)
    @test abs(dFull - dSeparate) / dFull > 1e-3
    # the ratio the old code would have introduced between A and B
    @test isapprox(objA / objB * (dFull / dSeparate), dFull / dSeparate; rtol = 1e-5)

    # The domain restriction the normalization imposes must be announced, not implicit.
    @test get(mA.metadata, "innovationsFullMAInvertible", "") == "true"
    @test get(mA.metadata, "innovationsFullMAOrder", "") == string(1 + s * 1)
    @test get(mB.metadata, "innovationsFullMAOrder", "") == "4"
end
