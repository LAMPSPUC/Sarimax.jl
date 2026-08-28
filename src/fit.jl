"""
    has_fit_methods(modelType::Type{<:SarimaxModel}) -> Bool

Check if a given `SarimaxModel` type has the `fit!` method implemented.

# Arguments
- `modelType::Type{<:SarimaxModel}`: Type of the Sarimax model to check.

# Returns
A boolean indicating whether the `fit!` method is implemented for the specified model type.

"""
function has_fit_methods(modelType::Type{<:SarimaxModel})
    tupleModelType = Tuple{modelType}
    return hasmethod(fit!, tupleModelType)
end

"""
    has_hyperparameters_methods(modelType::Type{<:SarimaxModel}) -> Bool

Checks if a given `SarimaxModel` type has methods related to hyperparameters.

# Arguments
- `modelType::Type{<:SarimaxModel}`: Type of the Sarimax model to check.

# Returns
A boolean indicating whether the hyperparameter-related methods are implemented for the specified model type.

"""
function has_hyperparameters_methods(modelType::Type{<:SarimaxModel})
    tupleModelType = Tuple{modelType}
    return hasmethod(get_hyperparameters_number, tupleModelType)
end

"""
    aic(K::Int, loglikeVal::Fl) where Fl<:AbstractFloat -> Fl

Calculate the Akaike Information Criterion (AIC) for a given number of parameters and log-likelihood value.

# Arguments
- `K::Int`: Number of parameters in the model.
- `loglikeVal::Fl`: Log-likelihood value of the model.

# Returns
The AIC value calculated using the formula: AIC = 2*K - 2*loglikeVal.

"""
function aic(K::Int, loglikeVal::Fl) where {Fl<:AbstractFloat}
    return 2 * K - 2 * loglikeVal
end

"""
    aicc(T::Int, K::Int, loglikeVal::Fl) where Fl<:AbstractFloat -> Fl

Calculate the corrected Akaike Information Criterion (AICc) for a given number of observations, number of parameters, and log-likelihood value.

# Arguments
- `T::Int`: Number of observations in the data.
- `K::Int`: Number of parameters in the model.
- `loglikeVal::Fl`: Log-likelihood value of the model.

# Returns
The AICc value calculated using the formula: AICc = AIC(K, loglikeVal) + ((2*K*K + 2*K) / (T - K - 1)).

"""
function aicc(T::Int, K::Int, loglikeVal::Fl) where {Fl<:AbstractFloat}
    return aic(K, loglikeVal) + ((2 * K * K + 2 * K) / (T - K - 1))
end

"""
    bic(T::Int, K::Int, loglikeVal::Fl) -> Fl

Calculate the Bayesian Information Criterion (BIC) for a given number of observations, number of parameters, and log-likelihood value.

# Arguments
- `T::Int`: Number of observations in the data.
- `K::Int`: Number of parameters in the model.
- `loglikeVal::Fl`: Log-likelihood value of the model.

# Returns
The BIC value calculated using the formula: BIC = log(T) * K - 2 * loglikeVal.

"""
function bic(T::Int, K::Int, loglikeVal::Fl) where {Fl<:AbstractFloat}
    return log(T) * K - 2 * loglikeVal
end

"""
    criterionLoglike(model) -> Fl

Log-likelihood used by the INFORMATION CRITERIA: the EXACT Gaussian one
([`exactLoglike`](@ref)) when it is computable, falling back to the CSS plug-in
([`loglike`](@ref)) when it is not.

Why the exact likelihood and not CSS. `forecast::auto.arima` selects by AICc computed on the
exact likelihood, so a CSS plug-in criterion would make AICc values and search trajectories
incomparable with it. CSS also ignores pre-sample uncertainty, which the exact likelihood
accounts for by construction.

The fallback exists because `exactLoglike` deliberately returns `nothing` rather than a wrong
number (a non-stationary point, an autocovariance that is not positive definite, or psi
truncation biting). In those cases the criterion reverts to the CSS plug-in: degraded
behaviour, never silently incorrect.

No `try/catch`: the contract of `exactLoglike` is already nothing-on-refusal, so any exception
here is a bug and must propagate. A broad `catch` would mask `MethodError`/`UndefVarError` and
make the fallback rate uninterpretable. Model types without `exactLoglike` fall back through
`applicable`.

QUASI-AIC, NOT AIC. This likelihood is evaluated at the coefficients THE USER'S OBJECTIVE
produced (`mae`, `huber`, CVaR, ridge, ...), not at the Gaussian maximum. As a statistic for
comparing candidates it is legitimate, since the same function scores all of them, but it is
not the maximized likelihood that AIC theory assumes. On simulated ARMA(1,1) the deficit
`2*(l(MLE) - l(fitted point))` has median 0.01 under `mse` and grows with the MA root radius:
median 0.98 and p90 7.03 at a root of 0.98, already on the scale at which AICc decides. Under
`ridge` the median reaches 1.79, because the shrinkage moves the point away from the maximum
by design. Refining by one Newton step does not fix it: in the regime that matters the optimum
sits on the invertibility boundary, where Le Cam's asymptotic equivalence does not hold — one
step closes 6% of the deficit and worsens the value in 40% of cases.
"""
function criterionLoglike(model::SarimaxModel)
    return first(criterionLoglikeAndN(model))
end

"""
    criterionLoglikeAndN(model) -> (loglike, n, usedExact)

Core of [`criterionLoglike`](@ref): it also returns the SAMPLE SIZE over which the
log-likelihood was evaluated, and which path was used.

The `n` matters because the two likelihoods do not live on the same sample: the exact one is
evaluated over the whole differenced series (`T`), while the CSS one lives on the conditioned
residuals (`length(observedResiduals) = T - lb + 1`, with `lb` up to 30 under the monthly
defaults of `auto`). Finite-sample corrections (AICc) and the `log(n)` factor of BIC must use
the `n` of the likelihood actually used; otherwise the criterion applies an extra size penalty,
growing in K, that `forecast::Arima` does not have.

The fallback is recorded in `model.metadata["criterionFallback"]` so that it is measurable.
"""
function criterionLoglikeAndN(model::SarimaxModel)
    exact = applicable(exactLoglike, model) ? exactLoglike(model) : nothing
    usedExact = !isnothing(exact)
    if hasproperty(model, :metadata) && isa(model.metadata, AbstractDict)
        model.metadata["criterionFallback"] = !usedExact
    end
    usedExact && return exact, criterionSampleSize(model), true
    return loglike(model), length(observedResiduals(model)), false
end

"""
    criterionSampleSize(model) -> Int

Number of observations of the EXACT likelihood: the length of the differenced series, which is
the `n` that `stats::arima` uses. Not to be confused with `length(observedResiduals)`, which
discounts the CSS conditioning.

Computed arithmetically (`n - d - D*s`) rather than as `length(values(differentiate(...)))`:
the latter allocates the whole differenced series only to measure its length, costing about
7.5us per criterion evaluation. The equivalence of the two forms is pinned in
`test/exact_likelihood.jl`.

Without a type annotation because `fit.jl` is included before `models/sarima.jl` defines
`SARIMAModel`; it is only called on the exact path, which requires
`applicable(exactLoglike, model)`.
"""
criterionSampleSize(model) =
    length(values(model.y)) - model.d - model.D * model.seasonality

"""
Penalty added to the criterion of a SEARCH candidate whose criterion came from the CSS
fallback.

Without it the fallback rewards exactly the wrong candidates: CSS is evaluated over fewer
observations than the exact likelihood (`T - lb + 1` against `T`), so it is less negative and
gives a smaller AICc, while what triggers the fallback is a root near the boundary. A
near-non-stationary candidate would gain tens of AICc units of advantage purely from having no
computable exact likelihood.

The penalty imposes a two-level ordering, analogous to `myarima` (which returns `Inf` when the
likelihood is not finite): a candidate with an exact likelihood always beats one without, and
among those without, the CSS comparison remains valid because they all condition on the same
sample (`searchLb`). It is additive rather than `Inf` so that the search is never left without
a selection when the exact likelihood fails for every candidate (short or pathological
series).

So se aplica a SELECAO ([`searchCriterionFunction`](@ref)); os acessores publicos `aic`,
`aicc` e `bic` continuam devolvendo o valor com recuo documentado.
"""
const FALLBACK_CRITERION_PENALTY = 1e10

"""
    searchCriterionFunction(baseCriterion) -> Function

Wraps `aic`/`aicc`/`bic` with SELECTION semantics: it adds
[`FALLBACK_CRITERION_PENALTY`](@ref) when the candidate's criterion came from the CSS
fallback (read from `model.metadata["criterionFallback"]`, written by
[`criterionLoglikeAndN`](@ref) during the criterion evaluation itself).
"""
function searchCriterionFunction(baseCriterion::Function)
    # Argument-transparent: besides the model form, `aic`/`aicc`/`bic` have scalar forms
    # ((K, ll), (T, K, ll)) that remain reachable through the returned function; the
    # penalty applies only when the first argument is a model.
    return function (args...; kwargs...)
        value = baseCriterion(args...; kwargs...)
        model = first(args)
        fallback =
            model isa SarimaxModel &&
            hasproperty(model, :metadata) &&
            isa(model.metadata, AbstractDict) &&
            get(model.metadata, "criterionFallback", false)
        return fallback ? value + FALLBACK_CRITERION_PENALTY : value
    end
end

"""
    aic(model::SarimaxModel; offset::Fl) -> Fl where Fl<:AbstractFloat

Calculate the Akaike Information Criterion (AIC) for a Sarimax model:
`AIC = 2K - 2ℓ`, where `ℓ` is [`criterionLoglike`](@ref) — the exact Gaussian
log-likelihood when computable, falling back to the conditional (CSS) one — and
`K = ncoef + 1` the number of estimated parameters, matching `forecast::Arima`.

# Arguments
- `model::SarimaxModel`: The Sarimax model for which AIC is calculated.
- `offset::Fl`: Optional value added to the AIC (kept for call compatibility).
- `K::Int`: Optional parameter-count override.

# Returns
The AIC value calculated using the number of parameters and log-likelihood value of the model.

# Errors
- Throws a `MissingMethodImplementation` if the `get_hyperparameters_number` method is not implemented for the given model type.

"""
function aic(model::SarimaxModel; offset::Union{AbstractFloat,Nothing} = nothing, K::Union{Int,Nothing} = nothing)
    !has_hyperparameters_methods(typeof(model)) &&
        throw(MissingMethodImplementation("get_hyperparameters_number"))
    K = isnothing(K) ? get_hyperparameters_number(model) : K
    offsetValue = isnothing(offset) ? 0.0 : offset
    return 2 * K - 2 * criterionLoglike(model) + offsetValue
end

"""
    aicc(model::SarimaxModel; offset::Fl) -> Fl where Fl<:AbstractFloat

Calculate the Corrected Akaike Information Criterion (AICc) for a Sarimax model.

# Arguments
- `model::SarimaxModel`: The Sarimax model for which AICc is calculated.
- `offset::Fl=0.0`: Offset value to be added to the AICc value.

# Returns
The AICc value calculated using the number of parameters, sample size, and log-likelihood value of the model.

# Errors
- Throws a `MissingMethodImplementation` if the `get_hyperparameters_number` method is not implemented for the given model type.

"""
function aicc(model::SarimaxModel; offset::Union{AbstractFloat,Nothing} = nothing, K::Union{Int,Nothing} = nothing)
    !has_hyperparameters_methods(typeof(model)) &&
        throw(MissingMethodImplementation("get_hyperparameters_number"))
    K = isnothing(K) ? get_hyperparameters_number(model) : K
    offsetValue = isnothing(offset) ? 0.0 : offset
    # The `n` of the correction is that of the sample the likelihood actually used: `T`
    # (the whole differenced series) on the exact path, `length(observedResiduals)` on the
    # CSS fallback. Mixing them — correcting a likelihood over T with a smaller conditioned
    # n — charges an extra penalty growing in K that `forecast::Arima` does not have.
    ll, n, _ = criterionLoglikeAndN(model)
    return 2 * K - 2 * ll + offsetValue + ((2 * K * K + 2 * K) / (n - K - 1))
end

"""
    bic(model::SarimaxModel; offset::Fl) -> Fl where Fl<:AbstractFloat

Calculate the Bayesian Information Criterion (BIC) for a Sarimax model.

# Arguments
- `model::SarimaxModel`: The Sarimax model for which BIC is calculated.
- `offset::Fl=0.0`: Offset value to be added to the BIC value.

# Returns
The BIC value calculated using the number of parameters, sample size, and log-likelihood value of the model.

# Errors
- Throws a `MissingMethodImplementation` if the `get_hyperparameters_number` method is not implemented for the given model type.

"""
function bic(model::SarimaxModel; offset::Union{AbstractFloat,Nothing} = nothing, K::Union{Int,Nothing} = nothing)
    !has_hyperparameters_methods(typeof(model)) &&
        throw(MissingMethodImplementation("get_hyperparameters_number"))
    K = isnothing(K) ? get_hyperparameters_number(model) : K
    offsetValue = isnothing(offset) ? 0.0 : offset
    # Mesmo `n` da verossimilhanca usada — ver `aicc`.
    ll, n, _ = criterionLoglikeAndN(model)
    return K * log(n) - 2 * ll + offsetValue
end


