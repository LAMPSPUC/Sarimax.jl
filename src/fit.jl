"""
    hasFitMethods(modelType::Type{<:SarimaxModel}) -> Bool

Check if a given `SarimaxModel` type has the `fit!` method implemented.

# Arguments
- `modelType::Type{<:SarimaxModel}`: Type of the SARIMAX model to check.

# Returns
A boolean indicating whether the `fit!` method is implemented for the specified model type.

"""
function hasFitMethods(modelType::Type{<:SarimaxModel})
    tupleModelType = Tuple{modelType}
    return hasmethod(fit!, tupleModelType)
end

"""
    hasHyperparametersMethods(modelType::Type{<:SarimaxModel}) -> Bool

Checks if a given `SarimaxModel` type has methods related to hyperparameters.

# Arguments
- `modelType::Type{<:SarimaxModel}`: Type of the SARIMAX model to check.

# Returns
A boolean indicating whether the hyperparameter-related methods are implemented for the specified model type.

"""
function hasHyperparametersMethods(modelType::Type{<:SarimaxModel})
    tupleModelType = Tuple{modelType}
    return hasmethod(getHyperparametersNumber, tupleModelType)
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

Calculate the Akaike Information Criterion (AIC) for a SARIMAX model.

# Arguments
- `model::SarimaxModel`: The SARIMAX model for which AIC is calculated.
- `offset::Fl=0.0`: Offset value to be added to the AIC value.

# Returns
The AIC value calculated using the number of parameters and log-likelihood value of the model.

# Errors
- Throws a `MissingMethodImplementation` if the `getHyperparametersNumber` method is not implemented for the given model type.

"""
function aic(model::SarimaxModel; offset::Fl = 0.0) where {Fl<:AbstractFloat}
    !hasHyperparametersMethods(typeof(model)) &&
        throw(MissingMethodImplementation("getHyperparametersNumber"))
    K = SARIMAX.getHyperparametersNumber(model)
    # T = length(model.ϵ)
    # return aic(K, loglike(model))
    # offset = -2 * loglike(model) - length(model.y) * log(model.σ²)
    # return offset + T * log(model.σ²) + 2*K
    T = length(model.y) - model.d - model.D * model.seasonality
    return 2 * K + T * log(model.σ²) + offset
end

"""
    aicc(model::SarimaxModel; offset::Fl) -> Fl where Fl<:AbstractFloat

Calculate the Corrected Akaike Information Criterion (AICc) for a SARIMAX model.

# Arguments
- `model::SarimaxModel`: The SARIMAX model for which AICc is calculated.
- `offset::Fl=0.0`: Offset value to be added to the AICc value.

# Returns
The AICc value calculated using the number of parameters, sample size, and log-likelihood value of the model.

# Errors
- Throws a `MissingMethodImplementation` if the `getHyperparametersNumber` method is not implemented for the given model type.

"""
function aicc(model::SarimaxModel; offset::Fl = 0.0) where {Fl<:AbstractFloat}
    !hasHyperparametersMethods(typeof(model)) &&
        throw(MissingMethodImplementation("getHyperparametersNumber"))
    K = getHyperparametersNumber(model)
    # T = length(model.ϵ)
    # return aicc(T, K, loglike(model))
    T = length(model.y) - model.d - model.D * model.seasonality
    return aic(model; offset = offset) + ((2 * K * K + 2 * K) / (T - K - 1))
end

"""
    bic(model::SarimaxModel; offset::Fl) -> Fl where Fl<:AbstractFloat

Calculate the Bayesian Information Criterion (BIC) for a SARIMAX model.

# Arguments
- `model::SarimaxModel`: The SARIMAX model for which BIC is calculated.
- `offset::Fl=0.0`: Offset value to be added to the BIC value.

# Returns
The BIC value calculated using the number of parameters, sample size, and log-likelihood value of the model.

# Errors
- Throws a `MissingMethodImplementation` if the `getHyperparametersNumber` method is not implemented for the given model type.

"""
function bic(model::SarimaxModel; offset::Fl = 0.0) where {Fl<:AbstractFloat}
    !hasHyperparametersMethods(typeof(model)) &&
        throw(MissingMethodImplementation("getHyperparametersNumber"))
    K = getHyperparametersNumber(model)
    # T = length(model.ϵ)
    # return bic(T, K, loglike(model))
    T = length(model.y) - model.d - model.D * model.seasonality
    return aic(model; offset = offset) + K * (log(T) - 2)
end
