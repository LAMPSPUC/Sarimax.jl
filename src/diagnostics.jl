# Residual diagnostics: Ljung-Box and Jarque-Bera tests.

"""
    ljung_box_test(residuals::Vector; lags, fitdf = 0)
    ljung_box_test(model::SARIMAModel; lags, fitdf = p + q + P + Q)

Ljung-Box portmanteau test for residual autocorrelation.

`Q = n(n+2) Σₖ ρ̂ₖ²/(n−k)` over `k = 1:lags`; under the null of no
autocorrelation `Q ~ χ²(lags − fitdf)`. For a fitted model, `fitdf` defaults to
the number of estimated ARMA coefficients (R's `checkresiduals` convention) and
`lags` to `min(2s, n÷5)` for seasonal models (`min(10, n÷5)` otherwise).

# Returns
`Dict` with `"test_statistic"`, `"p_value"`, `"lags"`, `"dof"`.

# References
- Ljung, G. M. & Box, G. E. P. (1978). On a measure of lack of fit in time
  series models. Biometrika 65(2), 297-303.
"""
function ljung_box_test(
    residuals::Vector{Fl};
    lags::Int = min(10, length(residuals) ÷ 5),
    fitdf::Int = 0,
) where {Fl<:AbstractFloat}
    n = length(residuals)
    lags >= 1 || throw(ArgumentError("lags must be ≥ 1"))
    lags < n || throw(ArgumentError("lags must be smaller than the number of residuals"))
    fitdf >= 0 || throw(ArgumentError("fitdf must be non-negative"))
    centered = residuals .- mean(residuals)
    denom = sum(abs2, centered)
    Q = 0.0
    for k = 1:lags
        ρk = sum(centered[k+1:end] .* centered[1:end-k]) / denom
        Q += ρk^2 / (n - k)
    end
    Q *= n * (n + 2)
    dof = max(lags - fitdf, 1)
    pValue = 1 - cdf(Chisq(dof), Q)
    return Dict{String,Any}(
        "test_statistic" => Q,
        "p_value" => pValue,
        "lags" => lags,
        "dof" => dof,
    )
end

function ljung_box_test(
    model::SARIMAModel;
    lags::Int = model.seasonality > 1 ?
                min(2 * model.seasonality, length(observedResiduals(model)) ÷ 5) :
                min(10, length(observedResiduals(model)) ÷ 5),
    fitdf::Int = model.p + model.q + model.P + model.Q,
)
    !isFitted(model) && throw(ModelNotFitted())
    return ljung_box_test(observedResiduals(model); lags = lags, fitdf = min(fitdf, lags - 1))
end

"""
    jarque_bera_test(residuals::Vector)
    jarque_bera_test(model::SARIMAModel)

Jarque-Bera test for normality: `JB = n/6 · (S² + (K−3)²/4)`, where `S` and `K`
are the sample skewness and kurtosis; under the null of normality
`JB ~ χ²(2)`.

# Returns
`Dict` with `"test_statistic"`, `"p_value"`, `"skewness"`, `"kurtosis"`.
"""
function jarque_bera_test(residuals::Vector{Fl}) where {Fl<:AbstractFloat}
    n = length(residuals)
    n >= 4 || throw(ArgumentError("At least 4 observations are required"))
    μ = mean(residuals)
    centered = residuals .- μ
    m2 = sum(abs2, centered) / n
    m3 = sum(x -> x^3, centered) / n
    m4 = sum(x -> x^4, centered) / n
    S = m3 / m2^1.5
    K = m4 / m2^2
    JB = n / 6 * (S^2 + (K - 3)^2 / 4)
    pValue = 1 - cdf(Chisq(2), JB)
    return Dict(
        "test_statistic" => JB,
        "p_value" => pValue,
        "skewness" => S,
        "kurtosis" => K,
    )
end

function jarque_bera_test(model::SARIMAModel)
    !isFitted(model) && throw(ModelNotFitted())
    return jarque_bera_test(observedResiduals(model))
end
