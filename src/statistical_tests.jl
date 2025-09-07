"""
    interpolate(x::Vector{T}, y::Vector{T}) where T <: AbstractFloat

Create a linear interpolation function for the points (x,y). The function handles
out-of-bounds values by returning the first/last y value.

# Arguments
- `x::Vector{T}`: x coordinates, must be sorted in ascending order
- `y::Vector{T}`: y coordinates

# Returns
- Function that takes a value and returns its interpolated value
"""
function interpolate(x::Vector{T}, y::Vector{T}) where T <: AbstractFloat
    if !issorted(x)
        throw(ArgumentError("x values must be sorted in ascending order"))
    end
    if length(x) != length(y)
        throw(ArgumentError("x and y must have the same length"))
    end

    return function(val::T)
        if val <= x[1]
            return y[1]
        elseif val >= x[end]
            return y[end]
        end

        # Find the interval containing val
        for i in 1:(length(x)-1)
            if x[i] <= val <= x[i+1]
                # Linear interpolation
                slope = (y[i+1] - y[i]) / (x[i+1] - x[i])
                return y[i] + slope * (val - x[i])
            end
        end

        # Should never reach here if val is in bounds
        error("Interpolation failed")
    end
end

"""
    _kpss_autolag(resids::Vector{T}, nobs::Int) where T <: AbstractFloat

Computes the number of lags for covariance matrix estimation in KPSS test
using method of Hobijn et al (1998). See also Andrews (1991), Newey & West
(1994), and Schwert (1989). Assumes Bartlett / Newey-West kernel.

# Arguments
- `resids::Vector{T}`: Residuals from the KPSS regression
- `nobs::Int`: Number of observations

# Returns
- `Int`: Number of lags to use in the KPSS test
"""
function _kpss_autolag(resids::Vector{T}, nobs::Int) where T <: AbstractFloat
    covlags = Int(floor(nobs^(2.0/9.0)))
    s0 = sum(resids.^2) / nobs
    s0 = s0 > 0 ? s0 : 1e-5
    s1 = 0.0

    for i in 1:covlags
        # Calculate dot product of shifted residuals
        resids_prod = dot(resids[(i+1):end], resids[1:(nobs-i)])
        resids_prod /= nobs / 2.0
        s0 += resids_prod
        s1 += i * resids_prod
    end

    s_hat = s1 / s0
    pwr = 1.0 / 3.0
    gamma_hat = 1.1447 * (s_hat * s_hat)^pwr
    return max(Int(floor(gamma_hat * nobs^pwr)),1)
end

"""
    kpss_test(y::Vector{T}; regression::Symbol=:c, nlags::Union{Symbol,Int}=:legacy) where T <: AbstractFloat

Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.

# Arguments
- `y::Vector{T}`: Time series data where T is a floating point type
- `regression::Symbol=:c`: The null hypothesis for the KPSS test
    - `:c`: The data is stationary around a constant (default)
    - `:ct`: The data is stationary around a trend
- `nlags::Union{Symbol,Int}=:legacy`: Number of lags to use
    - `:legacy`: Uses int(12 * (n/100)^(1/4)) as in Schwert (1989)
    - Integer value: Uses specified number of lags

# Returns
- `Dict`: Dictionary containing test results including:
    - test_statistic: KPSS test statistic
    - p_value: p-value of the test
    - critical_values: Dictionary of critical values at different significance levels
    - lags: Number of lags used

# Notes
To estimate σ² the Newey-West estimator is used. The p-values are interpolated from
Table 1 of Kwiatkowski et al. (1992).

# References
- Kwiatkowski et al. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. Journal of Econometrics, 54: 159-178.
- Newey & West (1994). Automatic lag selection in covariance matrix estimation. Review of Economic Studies, 61: 631-653.
- Schwert (1989). Tests for unit roots: A Monte Carlo investigation. Journal of Business and Economic Statistics, 7(2): 147-159.
"""
function kpss_test(y::Vector{Fl}; regression::Symbol=:c, nlags::Union{Symbol,Int}=:auto) where Fl
    nobs = length(y)

    # Set critical values based on regression type
    critical_values = Dict()
    if regression == :ct
        # Trend stationarity
        critical_values = Dict(0.10 => 0.119, 0.05 => 0.146, 0.025 => 0.176, 0.01 => 0.216)
    elseif regression == :c
        # Level stationarity
        critical_values = Dict(0.10 => 0.347, 0.05 => 0.463, 0.025 => 0.574, 0.01 => 0.739)
    else
        throw(ArgumentError("regression must be :c or :ct"))
    end

    # Calculate residuals
    residuals = y

    if regression == :ct
        # Detrend data: regress on constant and trend
        t = collect(1:nobs)
        X = hcat(ones(nobs), t)
        β = X \ y
        residuals = y - X * β
    else
        # Remove mean
        residuals = y .- mean(y)
    end

    # Compute number of lags
    if nlags == :legacy
        nlags = min(Int(ceil(12.0 * (nobs/100.0)^0.25)), nobs - 1)

    elseif nlags == :auto
        nlags = _kpss_autolag(residuals, nobs)
    elseif isa(nlags, Integer)
        if nlags >= nobs
            throw(ArgumentError("nlags must be < number of observations"))
        end
    else
        throw(ArgumentError("nlags must be :legacy, :auto or an integer"))
    end

    # Compute KPSS test statistic
    partial_sums = cumsum(residuals)
    eta = sum(partial_sums.^2) / (nobs^2)

    # Compute s^2 using Newey-West estimator
    s2 = sum(residuals.^2) # Initial variance (γ₀)

    # Add autocovariance terms
    for lag in 1:nlags
        w = 1.0 - lag/(nlags + 1)  # Bartlett kernel weights
        γₖ = sum(residuals[lag+1:end] .* residuals[1:end-lag])
        s2 += 2.0 * w * γₖ
    end

    s2 = s2 / nobs

    # Compute test statistic
    kpss_stat::Float64 = (s2 == 0.0) ? 0.0 : eta / s2

    # Compute p-value using interpolation
    crit_vals::Vector{Float64} = [critical_values[0.10], critical_values[0.05], critical_values[0.025], critical_values[0.01]]
    pvals::Vector{Float64} = [0.10, 0.05, 0.025, 0.01]

    # compute p-value as kpss_stat evaluated at the interpolation of crit_vals and pvals
    p_value::Float64 = interpolate(crit_vals, pvals)(kpss_stat)

    return Dict(
        "test_statistic" => kpss_stat,
        "p_value" => p_value,
        "critical_values" => critical_values,
        "lags" => nlags
    )
end

#######

# Generate lagged matrix (like Python _do_lag/_gen_lags)
function gen_lags(x::Vector{T}, lag::Int) where T <: AbstractFloat
    n = length(x)

    if lag <= 0
        return ones(n) * 0
    end

    if lag == 1
        return reshape(x, n, 1)
    end
    # Each column is x lagged by k (k=1:lag)
    out::Matrix{T} = ones(n + lag - 1, lag) * NaN
    for k in 1:lag
        # out[i: i + n, i] = y
        out[k: k + n-1, k] = x
    end
    # Remove rows with NaN
    return out[all(.!isnan.(out), dims=2)[:], :]
end

# --- Helper functions ---
# Critical value as in Python/R
function calc_ocsb_crit_val(m::Int)
    log_m = log(m)
    return  -0.2937411 * exp(
            -0.2850853 * (log_m - 0.7656451) + (-0.05983644) * (
                (log_m - 0.7656451)^2
            )
        ) - 1.652202
end

# --- OCSB regression logic ---
function fit_ocsb(x::Vector{T}, m::Int, lag::Int, max_lag::Int) where T <: AbstractFloat
    # Step 1: seasonal difference
    y_seas = Sarimax.differentiate(x, 0, 1, m)
    if isempty(y_seas)
        throw(ArgumentError("No samples after seasonal differencing"))
    end

    # Step 2: regular difference
    y = Sarimax.differentiate(y_seas, 1, 0, m)
    # Step 3: lag matrix for y
    ylag = gen_lags(y, lag)
    if max_lag > -1
        y = y[max_lag+1:end]
    end

    # Step 4: AR fit (with constant)
    # mf = ylag[: y.shape[0]]
    mf = ylag[1:length(y), :]
    X_ar = hcat(ones(length(mf[:,1])), mf)
    β_ar = X_ar \ y

    pred(A) = A * β_ar

    # Step 5: Z4 (residuals from AR on seasonal diff)
    z4_y = y_seas[lag+1:end]
    z4_lag = gen_lags(y_seas, lag)[1:size(z4_y, 1), :]
    z4_preds = pred(hcat(ones(size(z4_lag, 1)), z4_lag))
    z4 = z4_y - z4_preds

    # Step 6: Z5 (residuals from AR on regular diff)
    z5_y = Sarimax.differentiate(x, 1, 0, m)
    z5_lag = gen_lags(z5_y, lag)
    z5_y = z5_y[lag+1:end]
    z5_lag = z5_lag[1:size(z5_y, 1), :]
    z5_preds = pred(hcat(ones(size(z5_lag, 1)), z5_lag))
    z5 = z5_y - z5_preds

    #####
    data = hcat(
        mf,
        vec(z4[1:size(mf, 1)]),
        vec(z5[1:size(mf, 1)]),
    )

    β_final = data \ y

    residuals = y - data * β_final
    σ² = sum(residuals.^2) / (length(y) - size(data, 2))
    covβ = σ² * LinearAlgebra.pinv(data' * data)
    std_errors = sqrt.(LinearAlgebra.diag(covβ))
    t_values = β_final ./ std_errors
    return (t_values[end], sum(residuals.^2), length(y), size(data)[2])
end

"""
    ocsb_test(y::Vector{T}; m::Int=12, lag_method::Symbol=:aic, max_lag::Int=3) where T <: AbstractFloat

Perform an OCSB test of seasonality (Osborn, Chui, Smith, and Birchenhall test).

# Arguments
- `y::Vector{T}`: Time series data
- `m::Int=12`: The seasonal differencing term (e.g., 12 for monthly, 4 for quarterly)
- `lag_method::Symbol=:aic`: Method for lag selection, one of [:fixed, :aic, :bic, :aicc]
- `max_lag::Int=3`: Maximum lag order to consider

# Returns
- `Dict`: Dictionary containing test results including:
    - test_statistic: OCSB test statistic
    - critical_value: Critical value for the test
    - seasonal_difference: Recommended seasonal differencing (0 or 1)

# References
- Osborn DR, Chui APL, Smith J, and Birchenhall CR (1988). Seasonality and the order of integration for consumption.
  Oxford Bulletin of Economics and Statistics 50(4):361-377.
"""
function ocsb_test(y::Vector{T}; m::Int=12, lag_method::Symbol=:aic, max_lag::Int=3) where T <: AbstractFloat
    if m <= 1
        throw(ArgumentError("m must be greater than 1"))
    end
    if !(lag_method in [:fixed, :aic, :bic, :aicc])
        throw(ArgumentError("lag_method must be one of [:fixed, :aic, :bic, :aicc]"))
    end

    # --- Information criteria ---
    ic_funcs = Dict(
        :aic => (n, k, rss) -> n * log(rss / n) + 2 * k,
        :bic => (n, k, rss) -> n * log(rss / n) + k * log(n),
        :aicc => (n, k, rss) -> n * log(rss / n) + 2 * k * (n / (n - k - 1))
    )

    if max_lag > 0 && lag_method != :fixed
        fits = []
        icvals = []
        for lag in 1:max_lag
            try
                result = fit_ocsb(y, m, lag, max_lag)
                ic = ic_funcs[lag_method](result[2], result[3], result[4])
                push!(fits, result[1])
                push!(icvals, ic)
            catch e
                println(e)
                push!(fits, nothing)
                push!(icvals, Inf)
                continue
            end
        end

        max_lag = argmin(icvals)
        index_min = findmin(icvals)[2]
        best_stat = fits[index_min]
    end

    try
        result = fit_ocsb(y, m, max_lag, max_lag)
        best_stat = result[1]
    catch
        throw(ErrorException("Could not find a solution. Try a longer "))
    end

    crit_val = calc_ocsb_crit_val(m)
    return Dict(
        "test_statistic" => best_stat,
        "critical_value" => crit_val,
        "seasonal_difference" => Int(best_stat > crit_val)
    )
end
