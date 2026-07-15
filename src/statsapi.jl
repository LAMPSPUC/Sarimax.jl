# StatsAPI interface for SARIMAModel: coef, coefnames, residuals, nobs, fitted,
# vcov and stderror (CSS asymptotics via a numerical Hessian of the RSS).

"""
    coef(model::SARIMAModel)

Estimated coefficients in the order: mean `c` (if `allowMean`), drift (if
`allowDrift`), ϕ (AR), θ (MA), Φ (seasonal AR), Θ (seasonal MA), exogenous β.
σ² is not included.
"""
function StatsAPI.coef(model::SARIMAModel)
    !isFitted(model) && throw(ModelNotFitted())
    Fl = typeofModelElements(model)
    out = Fl[]
    model.allowMean && push!(out, model.c)
    model.allowDrift && push!(out, model.trend)
    model.p > 0 && append!(out, model.ϕ)
    model.q > 0 && append!(out, model.θ)
    model.P > 0 && append!(out, model.Φ)
    model.Q > 0 && append!(out, model.Θ)
    isnothing(model.exogCoefficients) || append!(out, model.exogCoefficients)
    return out
end

"""
    coefnames(model::SARIMAModel)

Names matching [`coef`](@ref): `"c"`, `"drift"`, `"ar_i"`, `"ma_j"`, `"sar_k"`,
`"sma_w"`, and one name per exogenous column.
"""
function StatsAPI.coefnames(model::SARIMAModel)
    !isFitted(model) && throw(ModelNotFitted())
    out = String[]
    model.allowMean && push!(out, "c")
    model.allowDrift && push!(out, "drift")
    append!(out, ["ar_$i" for i = 1:model.p])
    append!(out, ["ma_$j" for j = 1:model.q])
    append!(out, ["sar_$k" for k = 1:model.P])
    append!(out, ["sma_$w" for w = 1:model.Q])
    isnothing(model.exog) ||
        isnothing(model.exogCoefficients) ||
        append!(out, ["exog_" * String(col) for col in colnames(model.exog)])
    return out
end

"""
    residuals(model::SARIMAModel)

The CSS residuals over the effective sample (`t = lb:T`).
"""
function StatsAPI.residuals(model::SARIMAModel)
    !isFitted(model) && throw(ModelNotFitted())
    return observedResiduals(model)
end

"""
    nobs(model::SARIMAModel)

Number of effective observations (residuals) used by the CSS fit.
"""
function StatsAPI.nobs(model::SARIMAModel)
    !isFitted(model) && throw(ModelNotFitted())
    return length(observedResiduals(model))
end

"""
    fitted(model::SARIMAModel)

In-sample one-step-ahead fitted values on the original (integrated) scale.
"""
function StatsAPI.fitted(model::SARIMAModel)
    !isFitted(model) && throw(ModelNotFitted())
    return values(model.fitInSample)
end

"""
    cssResiduals(model::SARIMAModel, coefficients::Vector)

Pure-Julia replica of the CSS recursion used by `fit!`: given a coefficient
vector in the [`coef`](@ref) order, returns the residuals over the effective
sample. Used to differentiate the RSS numerically for [`vcov`](@ref);
`cssResiduals(model, coef(model)) ≈ residuals(model)`.

Assumes the exogenous variables enter as provided (the default
`automaticExogDifferentiation = false` path of `fit!`).
"""
function cssResiduals(model::SARIMAModel, coefficients::Vector{Fl}) where {Fl<:AbstractFloat}
    !isFitted(model) && throw(ModelNotFitted())
    diffY = differentiate(model.y, model.d, model.D, model.seasonality)
    if !isnothing(model.exog)
        diffY = TimeSeries.merge(diffY, model.exog)
    end
    yv = values(diffY)[:, 1]
    Xv = isnothing(model.exog) ? zeros(Fl, length(yv), 0) : values(diffY)[:, 2:end]
    T = length(yv)
    lb = T - length(model.ϵ) + 1

    idx = 1
    c = model.allowMean ? coefficients[idx] : zero(Fl)
    model.allowMean && (idx += 1)
    trend = model.allowDrift ? coefficients[idx] : zero(Fl)
    model.allowDrift && (idx += 1)
    arC = coefficients[idx:idx+model.p-1]
    idx += model.p
    maC = coefficients[idx:idx+model.q-1]
    idx += model.q
    sarC = coefficients[idx:idx+model.P-1]
    idx += model.P
    smaC = coefficients[idx:idx+model.Q-1]
    idx += model.Q
    nX = size(Xv, 2)
    exogC = coefficients[idx:idx+nX-1]

    driftV = if model.allowDrift
        diffT = differentiate(
            collect(Fl, 1:length(values(model.y))), model.d, model.D, model.seasonality)
        diffT[end-T+1:end]
    else
        Fl[]
    end

    mult = modelSeasonalForm(model) === :multiplicative
    s = model.seasonality
    resid = zeros(Fl, T)
    for t = lb:T
        fittedValue = c + (model.allowDrift ? trend * driftV[t] : trend)
        for i = 1:nX
            fittedValue += exogC[i] * Xv[t, i]
        end
        for i = 1:model.p
            (t - i > 0) && (fittedValue += arC[i] * yv[t-i])
        end
        for j = 1:model.q
            (t - j > 0) && (fittedValue += maC[j] * resid[t-j])
        end
        for k = 1:model.P
            (t - s * k > 0) && (fittedValue += sarC[k] * yv[t-s*k])
        end
        for w = 1:model.Q
            (t - s * w > 0) && (fittedValue += smaC[w] * resid[t-s*w])
        end
        if mult
            for i = 1:model.p, k = 1:model.P
                (t - i - s * k > 0) &&
                    (fittedValue -= arC[i] * sarC[k] * yv[t-i-s*k])
            end
            for j = 1:model.q, w = 1:model.Q
                (t - j - s * w > 0) &&
                    (fittedValue += maC[j] * smaC[w] * resid[t-j-s*w])
            end
        end
        resid[t] = yv[t] - fittedValue
    end
    return resid[lb:T]
end

"""
    vcov(model::SARIMAModel)

Asymptotic covariance of the CSS coefficient estimates,
`Var(θ̂) ≈ 2σ̂² H⁻¹`, where `H` is the numerical Hessian of the residual sum of
squares at the estimate (for least squares `H ≈ 2J'J`, so this equals the
classical `σ²(J'J)⁻¹`).
"""
function StatsAPI.vcov(model::SARIMAModel)
    !isFitted(model) && throw(ModelNotFitted())
    x0 = StatsAPI.coef(model)
    np = length(x0)
    np == 0 && return zeros(typeofModelElements(model), 0, 0)
    mask = get(model.metadata, "missingResidualMask", nothing)
    rss(x) = isnothing(mask) ? sum(abs2, cssResiduals(model, x)) :
             sum(abs2, cssResiduals(model, x)[.!mask])

    h = [eps(Float64)^0.25 * max(1.0, abs(x0[i])) for i = 1:np]
    f0 = rss(x0)
    H = zeros(Float64, np, np)
    for i = 1:np
        ei = zeros(np)
        ei[i] = h[i]
        H[i, i] = (rss(x0 .+ ei) - 2 * f0 + rss(x0 .- ei)) / h[i]^2
        for j = (i+1):np
            ej = zeros(np)
            ej[j] = h[j]
            H[i, j] =
                (
                    rss(x0 .+ ei .+ ej) - rss(x0 .+ ei .- ej) - rss(x0 .- ei .+ ej) +
                    rss(x0 .- ei .- ej)
                ) / (4 * h[i] * h[j])
            H[j, i] = H[i, j]
        end
    end
    return 2 * model.σ² * LinearAlgebra.pinv(H)
end

"""
    stderror(model::SARIMAModel)

Asymptotic standard errors of the CSS coefficient estimates (square root of the
diagonal of [`vcov`](@ref)). Entries whose variance estimate is not positive
(e.g. estimates on a bound, or flat RSS directions) are returned as `NaN` with
a warning.
"""
function StatsAPI.stderror(model::SARIMAModel)
    V = StatsAPI.vcov(model)
    d = LinearAlgebra.diag(V)
    if any(d .<= 0)
        @warn "Non-positive variance estimates encountered; returning NaN for those coefficients."
    end
    return [di > 0 ? sqrt(di) : NaN for di in d]
end
