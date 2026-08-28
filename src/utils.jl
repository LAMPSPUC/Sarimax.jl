"""
    differentiate(
        series::TimeArray,
        d::Int=0,
        D::Int=0,
        s::Int=1
    )

Differentiates a `TimeArray` `series` `d` times and `D` times with a seasonal difference of `s` periods.

# Arguments
- `series::TimeArray`: The time series data to differentiate.
- `d::Int=0`: The number of non-seasonal differences to take.
- `D::Int=0`: The number of seasonal differences to take.
- `s::Int=1`: The seasonal period for the differences.

# Returns
A differentiated `TimeArray`.

# Example
```jldoctest
julia> airPassengers = load_dataset(AIR_PASSENGERS)

julia> stationaryAirPassengers = differentiate(airPassengers, d=1, D=1, s=12)
```
"""
function differentiate(series::TimeArray, d::Int = 0, D::Int = 0, s::Int = 1)
    Fl = eltype(values(series))
    copiedValues::Vector{Fl} = values(series)
    coeffs = differentiated_coefficients(d, D, s, Fl)
    lenCoeffs = length(coeffs)
    diffValues::Vector{Fl} = Vector{Fl}()
    for i = lenCoeffs:length(copiedValues)
        y_diff = coeffs'copiedValues[i:-1:i-lenCoeffs+1]
        push!(diffValues, y_diff)
    end
    series = TimeArray(timestamp(series)[lenCoeffs:end], diffValues, colnames(series))
    return series
end

"""
    differentiate(series::Vector{Fl}, d::Int = 0, D::Int = 0, s::Int = 1) where Fl <: AbstractFloat

Differentiates a vector of values with `d` non-seasonal differences and `D` seasonal differences of period `s`.

# Arguments
- `series::Vector{Fl}`: The time series data to differentiate.
- `d::Int=0`: The number of non-seasonal differences to take.
- `D::Int=0`: The number of seasonal differences to take.
- `s::Int=1`: The seasonal period for the differences.

# Returns
A differentiated vector of values.

# Example
```julia
# Differentiate a time series with first-order difference and seasonal difference
diff_values = differentiate(values, 1, 1, 12)
```
"""
function differentiate(series::Vector{Fl}, d::Int = 0, D::Int = 0, s::Int = 1) where Fl <: AbstractFloat
    coeffs = differentiated_coefficients(d, D, s, Fl)
    lenCoeffs = length(coeffs)
    diffValues::Vector{Fl} = Vector{Fl}()
    for i = lenCoeffs:length(series)
        y_diff = coeffs'series[i:-1:i-lenCoeffs+1]
        push!(diffValues, y_diff)
    end
    return diffValues
end

"""
    differentiated_coefficients(d::Int, D::Int, s::Int, Fl::DataType=Float64)

Compute the coefficients for differentiating a time series.

# Arguments
- `d::Int`: Order of non-seasonal differencing.
- `D::Int`: Order of seasonal differencing.
- `s::Int`: Seasonal period.
- `Fl`: The type of the coefficients. Default is `Float64`.

# Returns
- `coeffs::Vector{AbstractFloat}`: Coefficients for differentiation.
"""
function differentiated_coefficients(d::Int, D::Int, s::Int, Fl::DataType = Float64)
    # Calculate the length of the resulting coefficients array
    lenCoeffs = d + D * s + 1
    # Initialize an array to store the coefficients
    coeffs = zeros(Fl, lenCoeffs)
    # Calculate the binomial coefficients
    binomialCoeffsd = [binomial(d, i) for i = 0:d]
    binomialCoeffsD = [binomial(D, j) for j = 0:D]

    # Calculate the coefficients
    for i = 0:d
        for j = 0:D
            coeffs[i+j*s+1] = (-1)^i * binomialCoeffsd[i+1] * (-1)^j * binomialCoeffsD[j+1]
        end
    end

    return coeffs
end


"""
    integrate(initialValues::Vector{Fl}, diffSeries::Vector{Fl}, d::Int, D::Int, s::Int) where Fl<:AbstractFloat

Converts a differentiated time series back to its original scale.

# Arguments
- `initialValues::Vector{Fl}`: Initial values of the original time series.
- `diffSeries::Vector{Fl}`: Differentiated time series.
- `d::Int`: Order of non-seasonal differencing.
- `D::Int`: Order of seasonal differencing.
- `s::Int`: Seasonal period.

# Returns
- `origSeries::Vector{Fl}`: Time series in the original scale.
"""
function integrate(
    initialValues::Vector{Fl},
    diffSeries::Vector{Fl},
    d::Int,
    D::Int,
    s::Int,
) where {Fl<:AbstractFloat}
    # Get the coefficients for differentiation
    # initialValues = b
    # diffSeries = a
    # d = 0
    # D = 1
    # s = 12
    # Fl = Float64
    coeffs = differentiated_coefficients(d, D, s, Fl)
    lenCoeffs = length(coeffs)

    # Calculate the length of the original series
    lenSeries = length(diffSeries) + d + D * s

    # Initialize an array to store the original series
    origSeries = zeros(Fl, lenSeries)

    # Copy the initial values to the original series
    origSeries[1:length(initialValues)] .= initialValues
    initialOffset = length(initialValues)

    # Iterate through the differentiated series and compute the original series
    for i = 1:length(diffSeries)
        # Calculate the value at the current index
        y_t::Fl = 0.0
        y_t += diffSeries[i]
        # y_t += (-1) * coeffs[2:end]'origSeries[initialOffset+i-1:-1:initialOffset+i-lenCoeffs+1]
        for j = 2:lenCoeffs
            y_t += (-1) * coeffs[j] * origSeries[initialOffset+i-(j-1)]
        end

        # Add contributions from past observations
        # origSeries[initialOffset+i] -= coeffs[2:end]'origSeries[initialOffset+i-1:-1:initialOffset+i-lenCoeffs+1]
        origSeries[initialOffset+i] = y_t
    end

    return origSeries
end


"""
    selectSeasonalIntegrationOrder{Fl}(y, seasonality, test) where Fl<:AbstractFloat

Selects the seasonal integration order for a time series based on the specified test.

# Arguments
- `y::Vector{Fl}`: The time series data.
- `seasonality::Int`: The seasonal period of the time series.
- `test::String`: The name of the test to use for selecting the seasonal integration order.

# Returns
The selected seasonal integration order.

# Errors
Throws an ArgumentError if the specified test is not supported.

"""
function selectSeasonalIntegrationOrder(
    y::Vector{Fl},
    seasonality::Int,
    test::String,
) where {Fl<:AbstractFloat}
    test in ("seas", "ch", "ocsb") || throw(ArgumentError("The test $test is not supported"))
    # Mirror R's forecast::nsdiffs: if the chosen seasonal test errors out, warn and
    # fall back to D = 0 instead of aborting the model search.
    try
        if test == "seas"
            return seasonalStrengthTest(y, seasonality)["seasonal_difference"]
        elseif test == "ch"
            return StateSpaceModels.canova_hansen_test(y, seasonality)
        else
            return ocsb_test(y; m=seasonality)["seasonal_difference"]
        end
    catch e
        @warn "Seasonal unit root test '$test' failed; assuming D = 0" exception = e
        return 0
    end
end

"""
    selectIntegrationOrder(y, maxd, D, seasonality, test) where Fl<:AbstractFloat

Selects the integration order for a time series based on the specified test.

# Arguments
- `y::Vector{Fl}`: The time series data.
- `maxd::Int`: The maximum order of differencing to consider.
- `D::Int`: The maximum seasonal order of differencing to consider.
- `seasonality::Int`: The seasonal period of the time series.
- `test::String`: The name of the test to use for selecting the integration order.

# Returns
The selected integration order.

# Errors
Throws an ArgumentError if the specified test is not supported.

"""
function selectIntegrationOrder(
    y::Vector{Fl},
    maxd::Int,
    D::Int,
    seasonality::Int,
    test::String,
) where {Fl<:AbstractFloat}
    if test == "kpssStateSpace"
        return StateSpaceModels.repeated_kpss_test(y, maxd, D, seasonality)
    elseif test in ("kpss", "kpssShort")
        # "kpss": Hobijn et al. automatic lag selection (statsmodels-compatible).
        # "kpssShort": the intent of this mode is to match R's forecast::ndiffs (and
        # therefore auto.arima's differencing decisions). ndiffs does NOT use urca's
        # lags = "short": it fixes use.lag = trunc(3*sqrt(n)/13) (verified against
        # forecast 8.23.0), which the :ndiffs bandwidth reproduces. The mode keeps its
        # historical name; only the bandwidth was corrected.
        lagMethod = (test == "kpssShort") ? :ndiffs : :auto
        for i in 0:maxd
            diffSeries = differentiate(y, i, D, seasonality)
            result = kpss_test(diffSeries; nlags=lagMethod)
            if result["p_value"] > 0.05
                return i
            end
        end
        return maxd
    end

    throw(ArgumentError("The test $test is not supported"))
end

"""
    automatic_differentiation(series; seasonalPeriod=1, seasonalIntegrationTest="seas", integrationTest="kpss", maxd=2)

Automatically applies differentiation to each series in a TimeArray.

# Arguments
- `series::TimeArray`: The input TimeArray containing the time series data.
- `seasonalPeriod::Int=1`: The seasonal period of the time series.
- `seasonalIntegrationTest::String="ocsb"`: The test used to select the seasonal integration order.
- `integrationTest::String="kpss"`: The test used to select the integration order.
- `maxd::Int=2`: The maximum order of differencing to consider.

# Returns
A tuple `(diffSeries, diffSeriesMetadata)` containing:
- `diffSeries::Vector{TimeArray}`: The differentiated time series.
- `diffSeriesMetadata::Vector{Dict{Symbol, Any}}`: Metadata containing the integration orders used for differentiation.

# Errors
Throws an AssertionError if invalid test options or seasonal period are provided.

"""
function automatic_differentiation(
    series::TimeArray;
    seasonalPeriod::Int = 1,
    seasonalIntegrationTest::String = "ocsb",
    integrationTest::String = "kpss",
    maxd::Int = 2,
)
    @assert integrationTest ∈ ["kpss"]
    @assert seasonalIntegrationTest ∈ ["seas", "ch", "ocsb"]
    @assert seasonalPeriod ≥ 1

    diffSeriesVector::Array{TimeArray} = []
    diffSeriesMetadata = Dict{Symbol,Any}()

    for col in colnames(series)
        if startswith(string(col), "outlier")
            push!(diffSeriesVector, series[col])
            diffSeriesMetadata[col] = Dict(:d => 0, :D => 0)
            continue
        end
        # Identify seasonal integration order
        y = series[col]
        seasonalIntegrationOrder = 0
        if seasonalPeriod ≠ 1
            seasonalIntegrationOrder = selectSeasonalIntegrationOrder(
                values(y),
                seasonalPeriod,
                seasonalIntegrationTest,
            )
        end

        # Identify integration order
        integrationOrder = Sarimax.selectIntegrationOrder(
            values(y),
            maxd,
            seasonalIntegrationOrder,
            seasonalPeriod,
            integrationTest,
        )

        # Apply the integration orders to differentiate the time series
        diffSeriesAux =
            differentiate(y, integrationOrder, seasonalIntegrationOrder, seasonalPeriod)
        push!(diffSeriesVector, diffSeriesAux)
        diffSeriesMetadata[col] =
            Dict(:d => integrationOrder, :D => seasonalIntegrationOrder)
    end

    diffSeries = merge(diffSeriesVector)
    return diffSeries, diffSeriesMetadata
end

"""
    isConstant(
        series::TimeArray,
    )

Check if a time series is constant.

# Arguments
- `series::TimeArray`: The time series data.

# Returns
A boolean indicating whether the time series is constant.
"""
function isConstant(series::TimeArray)
    seriesValues = values(series)
    # Iterate over the columns of the time series
    for i = 1:size(seriesValues, 2)
        if length(unique(seriesValues[:, i])) == 1
            return true
        end
    end
    return false
end


"""
Tolerância *relativa* abaixo da qual a amplitude interquartil é considerada degenerada,
isto é, indistinguível de zero na escala dos próprios dados. Ver `identifyOutliers`.

O valor acompanha a tolerância default do Ipopt (1e-8): os resíduos que alimentam
`identifyOutliers` não são subtrações, são variáveis de um modelo JuMP amarradas por
restrição de igualdade e satisfeitas apenas até a tolerância do solver. Fica ~8 ordens de
grandeza acima do piso de ruído de ponto flutuante (ULP) e ~8 abaixo de qualquer dispersão
com significado estatístico.
"""
const DEGENERATE_IQR_RTOL = 1e-8

"""
    identifyOutliers(series::Vector{Fl}, method::String="iqr", threshold::Float64=1.5) where Fl<:AbstractFloat

Identify outliers in a time series using the specified method.

# Arguments
- `series::Vector{Fl}`: The time series data.
- `method::String="iqr"`: The method used to identify outliers. Supported methods are "iqr".
- `threshold::Float64=1.5`: The threshold used to identify outliers.

# Returns
A boolean vector indicating the outliers in the time series.

# Dispersão degenerada

Quando a amplitude interquartil é nula — ou pequena demais na escala dos dados, abaixo de
`DEGENERATE_IQR_RTOL` vezes `max(|q1|, |q3|)` — a regra do IQR perde sentido: as cercas
colapsam sobre os próprios quartis e *tudo* que não for bit-idêntico a eles é sinalizado.
Nesse regime a resposta é nenhum outlier. Dispersão zero é evidência zero de atipicidade,
e a diferença entre "igual" e "quase igual" ao quartil é ruído de tolerância numérica, não
sinal. Sem essa guarda a saída da função depende do último bit dos resíduos e, portanto,
do runner em que o solver rodou.
"""
function identifyOutliers(
    series::Vector{Fl},
    method::String = "iqr",
    threshold::Float64 = 1.5,
) where {Fl<:AbstractFloat}
    if length(series) == 0
        return BitVector()
    end

    if method == "iqr"
        q1 = quantile(series, 0.25)
        q3 = quantile(series, 0.75)
        iqr = q3 - q1
        # Degenerate-dispersion guard. The comparison is relative to the scale of the
        # quartiles rather than to an absolute constant, so that the guard means the same
        # thing on series of any magnitude. The `<=` also covers `q1 == q3 == 0`, where the
        # scale is zero.
        scale = max(abs(q1), abs(q3))
        if iqr <= DEGENERATE_IQR_RTOL * scale
            return falses(length(series))
        end
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        # make a list where 1 indicates an outlier and 0 indicates no outlier
        return (series .< lower) .| (series .> upper)
    end
    throw(ArgumentError("The method $method is not supported"))
end

"""
    createOutliersDummies(outliers::BitVector, initialOffset::Int=0, endOffset::Int=0)

Create dummy variables for the outliers in a time series.

# Arguments
- `outliers::BitVector`: A boolean vector indicating the outliers in the time series.
- `initialOffset::Int=0`: The initial offset for the dummy variables.
- `endOffset::Int=0`: The end offset for the dummy variables.

# Returns
A DataFrame containing dummy variables for the outliers.
"""
function createOutliersDummies(
    outliers::BitVector,
    initialOffset::Int = 0,
    endOffset::Int = 0,
)
    outliersDict = Dict()
    for (i, value) in enumerate(outliers)
        if value
            auxArray = zeros(length(outliers) + initialOffset + endOffset)
            auxArray[i+initialOffset] = 1
            outliersDict["outlier_$i"] = auxArray
        end
    end

    return DataFrame(outliersDict)
end

"""
    loglike(model::SarimaxModel)

Conditional (CSS) Gaussian log-likelihood of a fitted model, evaluated at the
maximum-likelihood variance `σ̂² = RSS / n`, with full Gaussian constants:

`ℓ = -n/2 * (log(2π) + 1 + log(RSS / n))`

where `n` is the number of effective residuals (the observations used after
differencing and conditioning on the pre-sample values). This is the likelihood
convention of conditional least squares — comparable to R's
`arima(..., method = "CSS")`, NOT to the exact (Kalman-filter) likelihood
reported by default by R and statsmodels.

# Arguments
- `model::SarimaxModel`: A fitted SARIMAModel object.

# Returns
- The conditional Gaussian log-likelihood.

# Errors
- `MissingMethodImplementation("fit!")`: Thrown if the `fit!` method is not implemented for the given model type.
- `ModelNotFitted()`: Thrown if the model has not been fitted.
"""
function loglike(model::SarimaxModel)
    !has_fit_methods(typeof(model)) && throw(MissingMethodImplementation("fit!"))
    !isFitted(model) && throw(ModelNotFitted())
    r = observedResiduals(model)
    n = length(r)
    rss = sum(abs2, r)
    σ² = rss / n
    return -n / 2 * (log(2π) + 1 + log(σ²))
end

"""
    observedResiduals(model::SarimaxModel)

Residuals corresponding to actually observed data points. When the series was
fitted with missing observations, the smoothed pseudo-residuals at the imputed
indices are excluded (they are not real innovations); otherwise this returns the
full residual vector.
"""
function observedResiduals(model::SarimaxModel)
    ϵ = model.ϵ
    mask = hasproperty(model, :metadata) ? get(model.metadata, "missingResidualMask", nothing) : nothing
    return isnothing(mask) ? ϵ : ϵ[.!mask]
end

"""
    loglikelihood(model::SarimaxModel)

Alias for [`loglike`](@ref): the conditional (CSS) Gaussian log-likelihood.
"""
StatsAPI.loglikelihood(model::SarimaxModel) = loglike(model)
