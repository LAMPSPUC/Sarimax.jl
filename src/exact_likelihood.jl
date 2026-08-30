"""
EXACT Gaussian likelihood of a SARIMA model, evaluated at GIVEN coefficients.

The package distinguishes three likelihood-like objects that do not coincide: the objective
function being minimized, the exact ML, and the `loglike()` that feeds the information
criteria. This file provides the exact one: given a series and coefficients, it returns the
exact Gaussian log-likelihood, the same quantity `stats::arima(method = "ML")` reports.

It is what lets the criteria account for pre-sample uncertainty by construction, instead of
charging free pre-sample values as parameters. It also carries the determinant term whose
absence biases selection towards high `q`.

METHOD: Durbin-Levinson on the theoretical autocovariance. Neither Kalman nor Ansley.

    1. expande os polinomios multiplicativos:  phi(B)Phi(B^s)  e  theta(B)Theta(B^s)
    2. pesos psi da representacao MA(inf) e dai a autocovariancia gamma(k)
    3. Durbin-Levinson devolve os erros de previsao um-passo v_t e os residuos e_t
    4. concentrating sigma^2:

           -2l = n*log(sigma2) + sum_t log(v_t) + n,     sigma2 = (1/n) sum_t e_t^2 / v_t

Exact for any stationary ARMA, including multiplicative seasonal and mixed cases, which are
precisely the ones where the closed form obtained by profiling the pre-sample values does not
close (measured: a spread of about 35 units of -2logLik in the mixed ARMA, against 1e-13 in
the pure AR and pure MA cases).

COST: O(n^2). At n = 300 that is about 90 thousand operations per evaluation, negligible next
to an Ipopt solve. Since it only needs to EVALUATE and not optimize, it runs in plain Julia:
it need not be expressible in JuMP, differentiable, or interact with the solver, so the
optimizer remains free to use any objective.
"""

"""
    expandMultiplicativePolynomial(nonSeasonal, seasonal, s; negate) -> Vector

Expande `(1 - sum a_i B^i)(1 - sum b_k B^{s k})` na forma plana `1 - sum c_j B^j`, devolvendo
`c`. Com `negate = false` expande a forma `(1 + sum a_i B^i)(1 + sum b_k B^{s k})`, que e a
convencao do lado MA deste pacote (`y_t = ... + theta_j eps_{t-j} + eps_t`).
"""
function expandMultiplicativePolynomial(
    nonSeasonal::Vector{Fl},
    seasonal::Vector{Fl},
    s::Int;
    negate::Bool = true,
) where {Fl<:AbstractFloat}
    na, nb = length(nonSeasonal), length(seasonal)
    (na == 0 && nb == 0) && return Fl[]
    sgn = negate ? -one(Fl) : one(Fl)
    c = zeros(Fl, na + s * nb)
    for i = 1:na
        c[i] += nonSeasonal[i]
    end
    for k = 1:nb
        c[s*k] += seasonal[k]
        for i = 1:na
            # o produto cruzado troca de sinal na convencao AR (1 - aB)(1 - bB^s)
            c[s*k+i] += sgn * nonSeasonal[i] * seasonal[k]
        end
    end
    return c
end

"""
    psiWeightsFromZero(ar, ma, n) -> Vector

`psi` weights of the MA(infinity) representation INCLUDING `psi_0 = 1` in the first position,
so that `psi[k+1] == psi_k`. The computation is delegated to [`psiWeights`](@ref), which
already exists in the package and returns `psi_1..psi_n` with `psi_0` implicit.

It is a separate function rather than a redefinition on purpose: `exact_likelihood.jl` is
included after `models/sarima.jl`, so a same-signature `psiWeights` with zero-based indexing
would override the existing one, and `forecastErrors`, which expects one-based indexing, would
read `psi_0` where it means `psi_1`. Where a convention differs, adapt at the boundary.
"""
psiWeightsFromZero(ar::Vector{Fl}, ma::Vector{Fl}, n::Int) where {Fl<:AbstractFloat} =
    vcat(one(Fl), psiWeights(ar, ma, n))

"""
    theoreticalACF(ar, ma, n; psiLength) -> Vector

Autocovariancias `gamma(0..n-1)` do ARMA estacionario com variancia de inovacao unitaria,
obtidas da representacao MA(infinito): `gamma(k) = sum_j psi_j * psi_{j+k}`.

A truncagem em `psiLength` e legitima porque os `psi` de um processo ESTACIONARIO decaem
geometricamente. Se a truncagem morder (processo perto da fronteira), a funcao devolve
`nothing` em vez de um numero silenciosamente errado — a checagem e a cauda dos `psi`.
"""
function theoreticalACF(
    ar::Vector{Fl},
    ma::Vector{Fl},
    n::Int;
    psiLength::Int = max(2 * n, 1000),
) where {Fl<:AbstractFloat}
    ψ = psiWeightsFromZero(ar, ma, psiLength)
    all(isfinite, ψ) || return nothing
    # cauda ainda relevante => truncagem morde => nao devolver numero errado
    tailMax = maximum(abs, @view ψ[max(1, psiLength - 9):end])
    tailMax > 1e-6 && return nothing
    γ = zeros(Fl, n)
    for k = 0:(n-1)
        acc = zero(Fl)
        @inbounds for j = 1:(psiLength+1-k)
            acc += ψ[j] * ψ[j+k]
        end
        γ[k+1] = acc
    end
    (γ[1] > 0 && all(isfinite, γ)) || return nothing
    return γ
end

"""
    exactGaussianLogLikelihood(z, ar, ma) -> Union{Float64,Nothing}

Exact Gaussian log-likelihood of the series `z` (already differenced and already cleared of
the deterministic terms) under the ARMA with coefficients `ar`/`ma`, with `sigma^2`
concentrated out.

Returns `nothing` when the point is inadmissible or numerically infeasible (non-stationary,
autocovariance not positive definite) — never a silently wrong number.

Durbin-Levinson: `v_t` is the one-step forecast error variance in units of `sigma^2`, and the
forecast coefficients come from the recursion. `sum log v_t` is the determinant term the CSS
plug-in lacks, whose absence biases selection towards high orders.
"""
function exactGaussianLogLikelihood(
    z::Vector{Fl},
    ar::Vector{Fl},
    ma::Vector{Fl},
) where {Fl<:AbstractFloat}
    n = length(z)
    n == 0 && return nothing
    γ = theoreticalACF(ar, ma, n)
    isnothing(γ) && return nothing

    v = zeros(Fl, n)          # variancia do erro de previsao (unidades de sigma^2)
    e = zeros(Fl, n)          # erro de previsao um-passo
    ϕprev = zeros(Fl, n)
    ϕcur = zeros(Fl, n)

    v[1] = γ[1]
    e[1] = z[1]
    for t = 1:(n-1)
        # coeficiente de reflexao
        acc = γ[t+1]
        for j = 1:(t-1)
            acc -= ϕprev[j] * γ[t-j+1]
        end
        v[t] <= 0 && return nothing
        κ = acc / v[t]
        abs(κ) >= 1 && return nothing            # perdeu positividade definida
        ϕcur[t] = κ
        for j = 1:(t-1)
            ϕcur[j] = ϕprev[j] - κ * ϕprev[t-j]
        end
        v[t+1] = v[t] * (1 - κ^2)
        pred = zero(Fl)
        for j = 1:t
            pred += ϕcur[j] * z[t-j+1]
        end
        e[t+1] = z[t+1] - pred
        ϕprev, ϕcur = ϕcur, ϕprev
    end
    (all(x -> x > 0 && isfinite(x), v) && all(isfinite, e)) || return nothing

    σ² = sum(e[t]^2 / v[t] for t = 1:n) / n
    (σ² > 0 && isfinite(σ²)) || return nothing
    sumLogV = sum(log, v)
    # -2l = n log(2 pi sigma2) + sum log v + n   =>   l = -(n/2)(log(2 pi sigma2) + 1) - (1/2) sum log v
    return -(n / 2) * (log(2π * σ²) + 1) - sumLogV / 2
end

"""
    exactLoglike(model::SARIMAModel) -> Union{Float64,Nothing}

Exact Gaussian log-likelihood of the FITTED model, evaluated at the estimated coefficients.

Differencing and deterministic terms are removed first: the likelihood is that of the
differenced series minus the deterministic part, which is the `stats::arima` convention.

NOTE on evaluating away from the MLE: the coefficients come from the objective the user chose
(which may be `mae`, `ridge`, `huber`, ...), not from the maximum of this function. As a
statistic for COMPARING candidates that is legitimate as long as the same function scores all
of them, but it is not the maximized likelihood AIC theory assumes. For a closer match, refine
the finalists before the final decision — one Newton step on this function from the fitted
point is already asymptotically equivalent to the MLE — which is what `auto.arima` does with
`approximation = TRUE`.
"""
function exactLoglike(model::SARIMAModel)
    isFitted(model) || return nothing
    s = model.seasonality
    ϕ = isnothing(model.ϕ) ? Float64[] : Float64.([model.ϕ...])
    θ = isnothing(model.θ) ? Float64[] : Float64.([model.θ...])
    Φ = isnothing(model.Φ) ? Float64[] : Float64.([model.Φ...])
    Θ = isnothing(model.Θ) ? Float64[] : Float64.([model.Θ...])
    ar = expandMultiplicativePolynomial(ϕ, Φ, s; negate = true)
    # MA side through the shared builder, so this and the determinant normalization of the
    # quadratic pre-sample objective cannot disagree about which polynomial the model's MA
    # part is. Equivalent to `expandMultiplicativePolynomial(θ, Θ, s; negate = false)` for
    # numeric input — pinned by a test — but it also accepts JuMP expressions and knows the
    # additive form, neither of which that function does. The AR side keeps its own
    # expansion because its sign convention differs.
    ma = Float64[x for x in fullMACoefficients(θ, Θ, s, :multiplicative)]

    diffY = differentiate(model.y, model.d, model.D, s)
    z = Float64.(values(diffY))
    isempty(z) && return nothing

    # Remove the deterministic part on the same scale as the differenced series. Two
    # conversions are required, both validated against `stats::arima(method="ML")`:
    #
    # 1. `model.c` is the regression CONSTANT, not the mean. The level to remove is
    #    mu = c / (1 - sum(ar)). On M4 series 44895, subtracting `c` gives -2305.891 and
    #    subtracting `mu` gives -2304.253, which is exactly R's value. A 1.64 error in the
    #    log-likelihood is 3.3 AICc units, enough to change a selection.
    #
    # 2. `model.trend` MULTIPLIES the differenced time regressor (`trend * driftValues[t]`,
    #    see `sarima.jl`), and that regressor is not 1 in general: it is 1 at d=1, D=0, but
    #    `s` at d=0, D=1 (12 for monthly data). Subtracting the scalar is then off by a
    #    factor of 12.
    #
    # On M4 monthly the two together reach 15.7% of the 48k series (10.7% + 5.0%).
    if !isnothing(model.c) && model.allowMean
        arSum = isempty(ar) ? zero(eltype(z)) : sum(ar)
        denom = 1 - arSum
        # `denom = phi(1)*Phi(1)`, with the polynomial already expanded. When it goes to
        # zero the process has a unit root and THE MEAN DOES NOT EXIST, so no value here is
        # right. Falling back to `model.c` is a finite choice, not a correct one; such a
        # candidate should have been rejected by the admissibility check before reaching
        # this point.
        if abs(denom) <= 1e-8
            @warn "exactGaussianLogLikelihood: phi(1)*Phi(1) = $(denom) ~ 0 (raiz unitaria); " *
                  "a media do processo nao existe e o nivel removido e apenas finito, nao correto."
        end
        level = abs(denom) > 1e-8 ? model.c / denom : model.c
        z = z .- level
    end
    if !isnothing(model.trend) && model.allowDrift
        # The `fill(1.0, ...)` fallback subtracts the scalar instead of the differenced
        # regressor, which is wrong whenever that regressor is not 1. It warns rather than
        # failing silently, so the case cannot reappear untraced.
        driftReg = try
            r = values(differentiate(
                    TimeArray(timestamp(model.y), collect(1.0:length(values(model.y)))),
                    model.d, model.D, s))
            if length(r) == length(z)
                Float64.(r)
            else
                @warn "exactGaussianLogLikelihood: regressor de drift diferenciado tem " *
                      "comprimento $(length(r)), esperado $(length(z)); usando 1.0. " *
                      "Com d=$(model.d), D=$(model.D), s=$s o termo deterministico fica errado."
                fill(1.0, length(z))
            end
        catch e
            @warn "exactGaussianLogLikelihood: falha ao diferenciar o regressor de drift " *
                  "($(typeof(e))); usando 1.0, o que reintroduz o erro de escala."
            fill(1.0, length(z))
        end
        z = z .- model.trend .* driftReg
    end
    if !isnothing(model.exog) && !isnothing(model.exogCoefficients)
        try
            X = Float64.(values(differentiate(model.exog, model.d, model.D, s)))
            β = Float64.([model.exogCoefficients...])
            size(X, 1) == length(z) && length(β) == size(X, 2) && (z = z .- X * β)
        catch
            return nothing
        end
    end
    return exactGaussianLogLikelihood(z, ar, ma)
end
