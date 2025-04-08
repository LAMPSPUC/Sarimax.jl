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
function kpss_test(y::Vector{Fl}; regression::Symbol=:c, nlags::Union{Symbol,Int}=:legacy) where Fl
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
    elseif isa(nlags, Integer)
        if nlags >= nobs
            throw(ArgumentError("nlags must be < number of observations"))
        end
    else
        throw(ArgumentError("nlags must be :legacy or an integer"))
    end

    # Compute KPSS test statistic
    partial_sums = cumsum(residuals)
    eta = sum(partial_sums.^2) / (nobs^2)

    # Compute s^2 using Newey-West estimator
    s2 = var(residuals)  # Initial variance (γ₀)

    # Add autocovariance terms
    for lag in 1:nlags
        w = 1.0 - lag/(nlags + 1)  # Bartlett kernel weights
        γₖ = sum(residuals[lag+1:end] .* residuals[1:end-lag]) / nobs
        s2 += 2.0 * w * γₖ
    end

    # Compute test statistic
    kpss_stat::Float32 = eta / s2

    # Compute p-value using interpolation
    crit_vals::Vector{Float32} = [critical_values[0.10], critical_values[0.05], critical_values[0.025], critical_values[0.01]]
    pvals::Vector{Float32} = [0.10, 0.05, 0.025, 0.01]

    # compute p-value as kpss_stat evaluated at the interpolation of crit_vals and pvals
    p_value = interpolate(crit_vals, pvals)(kpss_stat)

    return Dict(
        "test_statistic" => kpss_stat,
        "p_value" => p_value,
        "critical_values" => critical_values,
        "lags" => nlags
    )
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

    # Helper function to calculate OCSB critical value
    function calc_ocsb_crit_val(m::Int)
        log_m = log(m)
        return -0.2937411 * exp(-0.2850853 * (log_m - 0.7656451) + 
               (-0.05983644) * ((log_m - 0.7656451)^2)) - 1.652202
    end

    # Helper function to create lags
    function gen_lags(x::Vector{T}, lag::Int) where T <: AbstractFloat
        n = length(x)
        if lag == 0
            return zeros(T, n)
        end
        
        result = Matrix{T}(undef, n - lag, lag)
        for i in 1:lag
            result[:, i] = x[lag-i+1:n-i]
        end
        return result
    end

    # Helper function to fit OCSB model
    function fit_ocsb(x::Vector{T}, m::Int, lag::Int, max_lag::Int) where T <: AbstractFloat
        # First order seasonal difference
        y_seasonal_diff = diff(x, m)
        
        if isempty(y_seasonal_diff)
            throw(ArgumentError("No samples after seasonal differencing"))
        end
        
        # Regular difference
        y = diff(y_seasonal_diff)
        ylag = gen_lags(y, lag)
        
        if max_lag > -1
            y = y[max_lag+1:end]
        end
        
        # Fit initial model with constant term
        X = hcat(ones(size(ylag, 1)), ylag)
        β = X \ y
        
        # Create Z4 (seasonal residuals)
        z4_y = y_seasonal_diff[lag+1:end]
        z4_lag = gen_lags(y_seasonal_diff, lag)
        z4_preds = hcat(ones(size(z4_lag, 1)), z4_lag) * β
        z4 = z4_y - z4_preds
        
        # Create Z5 (regular difference residuals)
        z5_y = diff(x)
        z5_lag = gen_lags(z5_y, lag)
        z5_y = z5_y[lag+1:end]
        z5_preds = hcat(ones(size(z5_lag, 1)), z5_lag) * β
        z5 = z5_y - z5_preds
        
        # Final regression
        X_final = hcat(ylag, z4[1:size(ylag,1)], z5[1:size(ylag,1)])
        β_final = X_final \ y
        
        # Compute t-values
        residuals = y - X_final * β_final
        σ² = sum(residuals.^2) / (length(y) - size(X_final, 2))
        std_errors = sqrt.(σ² * diag(inv(X_final'X_final)))
        t_values = β_final ./ std_errors
        
        return t_values[end]  # Return t-value for z5 coefficient
    end

    # Information criteria functions
    ic_funcs = Dict(
        :aic => (n, k, rss) -> n * log(rss/n) + 2k,
        :bic => (n, k, rss) -> n * log(rss/n) + k * log(n),
        :aicc => (n, k, rss) -> n * log(rss/n) + 2k * (n/(n-k-1))
    )

    # Main test logic
    crit_val = calc_ocsb_crit_val(m)
    
    # Determine optimal lag if not fixed
    if max_lag > 0 && lag_method != :fixed
        best_lag = 1
        best_ic = Inf
        best_stat = nothing
        
        for lag in 1:max_lag
            try
                stat = fit_ocsb(y, m, lag, max_lag)
                
                # Calculate residuals and IC
                n = length(y) - max_lag
                k = lag + 2  # number of parameters including constant and z4,z5
                rss = sum((diff(y) .- stat).^2)
                
                ic = ic_funcs[lag_method](n, k, rss)
                
                if ic < best_ic
                    best_ic = ic
                    best_lag = lag
                    best_stat = stat
                end
            catch
                continue
            end
        end
        
        if best_stat === nothing
            throw(ErrorException("Could not find valid lag order"))
        end
        
        test_stat = best_stat
    else
        test_stat = fit_ocsb(y, m, max_lag, max_lag)
    end
    
    # Return results
    return Dict(
        "test_statistic" => test_stat,
        "critical_value" => crit_val,
        "seasonal_difference" => Int(test_stat > crit_val)
    )
end
