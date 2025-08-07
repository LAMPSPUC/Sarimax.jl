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

# Errors
- This method only works with `d` and `D` in the set {0,1}.

# Example
```jldoctest
julia> airPassengers = loadDataset(AIR_PASSENGERS)

julia> stationaryAirPassengers = differentiate(airPassengers, d=1, D=1, s=12)
```
"""
function differentiate(series::TimeArray, d::Int = 0, D::Int = 0, s::Int = 1)
    Fl = eltype(values(series))
    copiedValues::Vector{Fl} = values(series)
    coeffs = differentiatedCoefficients(d, D, s, Fl)
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
    coeffs = differentiatedCoefficients(d, D, s, Fl)
    lenCoeffs = length(coeffs)
    diffValues::Vector{Fl} = Vector{Fl}()
    for i = lenCoeffs:length(series)
        y_diff = coeffs'series[i:-1:i-lenCoeffs+1]
        push!(diffValues, y_diff)
    end
    return diffValues
end

"""
    differentiatedCoefficients(d::Int, D::Int, s::Int, Fl::DataType=Float64)

Compute the coefficients for differentiating a time series.

# Arguments
- `d::Int`: Order of non-seasonal differencing.
- `D::Int`: Order of seasonal differencing.
- `s::Int`: Seasonal period.
- `Fl`: The type of the coefficients. Default is `Float64`.

# Returns
- `coeffs::Vector{AbstractFloat}`: Coefficients for differentiation.
"""
function differentiatedCoefficients(d::Int, D::Int, s::Int, Fl::DataType = Float64)
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
    coeffs = differentiatedCoefficients(d, D, s, Fl)
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
    if test == "seas"
        return StateSpaceModels.seasonal_strength_test(y, seasonality)
    elseif test == "ch"
        return StateSpaceModels.canova_hansen_test(y, seasonality)
    elseif test == "ocsb"
        return ocsb_test(y;m=seasonality)["seasonal_difference"]
    end
    throw(ArgumentError("The test $test is not supported"))
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
    elseif test == "kpss"
        for i in 0:maxd
            diffSeries = differentiate(y, i, D, seasonality)
            result = kpss_test(diffSeries)
            if result["p_value"] > 0.05
                return i
            end
        end
        return maxd
    end

    throw(ArgumentError("The test $test is not supported"))
end

"""
    automaticDifferentiation(series; seasonalPeriod=1, seasonalIntegrationTest="seas", integrationTest="kpss", maxd=2)

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
function automaticDifferentiation(
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
    identifyOutliers(series::Vector{Fl}, method::String="iqr", threshold::Float64=1.5) where Fl<:AbstractFloat

Identify outliers in a time series using the specified method.

# Arguments
- `series::Vector{Fl}`: The time series data.
- `method::String="iqr"`: The method used to identify outliers. Supported methods are "iqr".
- `threshold::Float64=1.5`: The threshold used to identify outliers.

# Returns
A boolean vector indicating the outliers in the time series.
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
        lower = q1 - threshold * (q3 - q1)
        upper = q3 + threshold * (q3 - q1)
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
    loglikelihood(model::SarimaxModel)

Calculate the log-likelihood of a SARIMAModel.
The log-likelihood is calculated based on the formula
`-0.5 * (T * log(2π) + T * log(σ²) + sum(ϵ.^2 ./ σ²))`
where:
- T is the length of the residuals vector (ϵ).
- σ² is the estimated variance of the model.

# Arguments
- `model::SarimaxModel`: A SARIMAModel object.

# Returns
- The log-likelihood of the SARIMAModel.

# Errors
- `MissingMethodImplementation("fit!")`: Thrown if the `fit!` method is not implemented for the given model type.
- `ModelNotFitted()`: Thrown if the model has not been fitted.

"""
function loglikelihood(model::SarimaxModel)
    !hasFitMethods(typeof(model)) && throw(MissingMethodImplementation("fit!"))
    !isFitted(model) && throw(ModelNotFitted())
    T = length(model.ϵ)
    return -0.5 * (T * log(2π) + T * log(model.σ²) + sum(model.ϵ .^ 2 ./ model.σ²))
end

"""
    loglike(model::SarimaxModel)

Calculate the log-likelihood of a SARIMAModel using the normal probability density function.
The log-likelihood is calculated by summing the logarithm of the probability density function (PDF) of each data point under the assumption of a normal distribution with mean 0 and standard deviation equal to the square root of the estimated variance (σ²) of the model.

# Arguments
- `model::SarimaxModel`: A SARIMAModel object.

# Returns
- The log-likelihood of the SARIMAModel.

# Errors
- `MissingMethodImplementation("fit!")`: Thrown if the `fit!` method is not implemented for the given model type.
- `ModelNotFitted()`: Thrown if the model has not been fitted.

"""
function loglike(model::SarimaxModel)
    !hasFitMethods(typeof(model)) && throw(MissingMethodImplementation("fit!"))
    !isFitted(model) && throw(ModelNotFitted())
    return sum(logpdf.(Normal(0, sqrt(model.σ²)), model.ϵ))
end
