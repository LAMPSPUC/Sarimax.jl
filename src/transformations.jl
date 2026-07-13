# Box-Cox transformation utilities.

"""
    boxcox_transform(y, λ)

Box-Cox transform: `(y^λ − 1)/λ` for `λ ≠ 0`, `log(y)` for `λ = 0`.
Requires strictly positive data. Accepts a vector or a `TimeArray`.
"""
function boxcox_transform(y::Vector{Fl}, λ::Real) where {Fl<:AbstractFloat}
    all(>(0), y) || throw(ArgumentError("Box-Cox requires strictly positive data"))
    return abs(λ) < 1e-10 ? log.(y) : (y .^ λ .- 1) ./ λ
end

boxcox_transform(y::TimeArray, λ::Real) =
    TimeArray(timestamp(y), boxcox_transform(collect(values(y)), λ), colnames(y))

"""
    inverse_boxcox(z, λ)

Inverse of [`boxcox_transform`](@ref): `(λz + 1)^(1/λ)` for `λ ≠ 0`, `exp(z)`
for `λ = 0`. Accepts a vector or a `TimeArray`.
"""
function inverse_boxcox(z::Vector{Fl}, λ::Real) where {Fl<:AbstractFloat}
    return abs(λ) < 1e-10 ? exp.(z) : (λ .* z .+ 1) .^ (1 / λ)
end

inverse_boxcox(z::TimeArray, λ::Real) =
    TimeArray(timestamp(z), inverse_boxcox(collect(values(z)), λ), colnames(z))

"""
    boxcox_lambda(y; seasonality = 2, lower = -1.0, upper = 2.0)

Select the Box-Cox λ by Guerrero's (1993) method: split the series into
non-overlapping groups of `seasonality` observations (at least 2) and choose
the λ that minimizes the coefficient of variation of `σᵢ / μᵢ^(1−λ)` across
groups — the λ that best stabilizes the variance. This is the method used by
`forecast::BoxCox.lambda` in R.

# Returns
The selected `λ` (grid search on `lower:0.01:upper`).

# References
- Guerrero, V. M. (1993). Time-series analysis supported by power
  transformations. Journal of Forecasting 12, 37-48.
"""
function boxcox_lambda(
    y::Vector{Fl};
    seasonality::Int = 2,
    lower::Real = -1.0,
    upper::Real = 2.0,
) where {Fl<:AbstractFloat}
    all(>(0), y) || throw(ArgumentError("Box-Cox requires strictly positive data"))
    groupSize = max(seasonality, 2)
    nGroups = length(y) ÷ groupSize
    nGroups >= 2 || throw(ArgumentError("Series too short for Guerrero's method"))

    μs = zeros(Fl, nGroups)
    σs = zeros(Fl, nGroups)
    for g = 1:nGroups
        chunk = y[(g-1)*groupSize+1:g*groupSize]
        μs[g] = mean(chunk)
        σs[g] = std(chunk)
    end

    bestλ = lower
    bestCV = Inf
    for λ in lower:0.01:upper
        r = σs ./ μs .^ (1 - λ)
        cv = std(r) / mean(r)
        if isfinite(cv) && cv < bestCV
            bestCV = cv
            bestλ = λ
        end
    end
    return bestλ
end

boxcox_lambda(y::TimeArray; kwargs...) = boxcox_lambda(collect(values(y)); kwargs...)
