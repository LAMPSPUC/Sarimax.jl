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
    aic(model::SarimaxModel; offset::Fl) -> Fl where Fl<:AbstractFloat

Calculate the Akaike Information Criterion (AIC) for a Sarimax model:
`AIC = 2K - 2ℓ`, where `ℓ` is the conditional (CSS) Gaussian log-likelihood
([`loglike`](@ref)) and `K` the number of estimated parameters. Comparable to
R's `arima(..., method = "CSS")` convention, not to exact-likelihood AICs.

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
    return 2 * K - 2 * loglike(model) + offsetValue
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
    n = length(model.ϵ)
    return aic(model; offset = offset, K = K) + ((2 * K * K + 2 * K) / (n - K - 1))
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
    n = length(model.ϵ)
    offsetValue = isnothing(offset) ? 0.0 : offset
    return K * log(n) - 2 * loglike(model) + offsetValue
end


