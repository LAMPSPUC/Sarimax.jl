"""
    copyTimeArray(y::TimeSeries.TimeArray)

Create a shallow copy of a TimeArray object.

This function creates a new TimeArray with copies of the timestamp and values
from the original TimeArray. The new TimeArray is independent of the original,
but the elements themselves are not deeply copied.

# Arguments
- `y::TimeSeries.TimeArray`: The TimeArray to copy.

# Returns
- `TimeSeries.TimeArray`: A new TimeArray with copies of the timestamp and values.

# Examples
```julia
original = TimeArray(Date.(2021:2023), [1, 2, 3])
copied = copyTimeArray(original)
```
"""
function copyTimeArray(y::TimeSeries.TimeArray)
    return TimeArray(copy(timestamp(y)), copy(values(y)))
end

"""
    deepcopyTimeArray(y::TimeSeries.TimeArray)

Create a deep copy of a TimeArray object.

This function creates a new TimeArray with deep copies of the timestamp and values
from the original TimeArray. The new TimeArray and all its elements are completely
independent of the original.

# Arguments
- `y::TimeSeries.TimeArray`: The TimeArray to deep copy.

# Returns
- `TimeSeries.TimeArray`: A new TimeArray with deep copies of the timestamp and values.

# Examples
```julia
original = TimeArray(Date.(2021:2023), [1, 2, 3])
deepCopied = deepcopyTimeArray(original)
```
"""
function deepcopyTimeArray(y::TimeSeries.TimeArray)
    return TimeArray(deepcopy(timestamp(y)), deepcopy(values(y)))
end

"""
    buildDatetimes(startDatetime, granularity, weekDaysOnly, datetimesLength)

Builds an array of DateTime objects based on a given starting DateTime, granularity, and length.

# Arguments
- `startDatetime::T`: The DateTime from which the dateTime array will be computed. It won't be included in the final array
- `granularity::Dates.Period`: The granularity by which to increment the timestamps.
- `weekDaysOnly::Bool`: Whether to include only weekdays (Monday to Friday) in the timestamps.
- `datetimesLength::Int`: The length of the array of DateTime objects to build.

# Returns
An array of DateTime objects.

"""
function buildDatetimes(
    startDatetime::T,
    granularity::P where {P<:Dates.Period},
    weekDaysOnly::Bool,
    datetimesLength::Int,
) where {T}
    if datetimesLength == 0
        return DateTime[]
    end

    # Initialize the array with the starting datetime
    datetimes = []

    currentDatetime = startDatetime

    # Loop to generate timestamps based on granularity
    for _ = 1:datetimesLength
        if weekDaysOnly && dayofweek(currentDatetime) == 5
            currentDatetime += Dates.Day(3)
        else
            currentDatetime += granularity
        end
        push!(datetimes, currentDatetime)
    end

    return datetimes
end



"""
    identifyGranularity(datetimes::Vector{T})

Identifies the granularity of an array of timestamps.

# Arguments
- `datetimes::Vector{T}`: An array of TimeType objects.

# Returns
A tuple `(granularity, frequency, weekdays)` where:
- `granularity`: The identified granularity, which could be one of [:Second, :Minute, :Hour, :Day, :Week, :Month, :Year].
- `frequency`: The frequency of the identified granularity.
- `weekdays`: A boolean indicating whether the series uses weekdays only.

# Errors
Throws an error if the timestamps do not follow a consistent pattern.

"""
function identifyGranularity(datetimes::Vector{T}) where {T}
    # Define base units and periods
    baseUnits = [Second(1), Minute(1), Hour(1), Day(1), Week(1)]
    basePeriods = [:Second, :Minute, :Hour, :Day, :Week, :Month, :Year]

    unitPeriod = nothing
    diffBetweenTimestamps = nothing
    weekDaysSeries = false

    for (i, unit) in enumerate(baseUnits)
        differences = diff(datetimes) ./ unit
        min_difference = minimum(differences)

        lessThanOne = all(differences .< 1)
        if lessThanOne
            break
        end

        # Check if all elements are equal
        regularDistribution = all(differences .== differences[1])
        if regularDistribution
            unitPeriod = basePeriods[i]
            diffBetweenTimestamps = differences[1]

            if unit in [Minute(1), Second(1)]
                if diffBetweenTimestamps < 60
                    break
                else
                    continue
                end
            elseif unit == Hour(1)
                if diffBetweenTimestamps < 24
                    break
                else
                    continue
                end
            elseif unit == Day(1)
                if diffBetweenTimestamps < 7
                    break
                elseif diffBetweenTimestamps % 7 != 0
                    break
                else
                    continue
                end
            end

            break
        else
            if unit == Day(1)
                # Check if the series uses weekdays only
                allWeekdays = all([dayofweek(dt) <= 5 for dt in datetimes])
                if allWeekdays && min_difference < 7
                    unitPeriod = :Day
                    diffBetweenTimestamps = 1
                    weekDaysSeries = true
                    break
                end
            end
        end

        amplitude = maximum(differences) - min_difference
        if amplitude < 1
            unitPeriod = basePeriods[i]
            diffBetweenTimestamps = round(Int, min_difference)
            break
        end
    end

    if isnothing(unitPeriod)
        throw(InconsistentDatePattern())
    end

    # Analyze the minimum difference to determine granularity
    if unitPeriod == :Week
        if diffBetweenTimestamps % 52 == 0
            unitPeriod = :Year
            diffBetweenTimestamps = diffBetweenTimestamps / 52
        elseif diffBetweenTimestamps % 13 == 0
            unitPeriod = :Month
            diffBetweenTimestamps = 3
        elseif diffBetweenTimestamps % 4 == 0
            unitPeriod = :Month
            diffBetweenTimestamps = diffBetweenTimestamps / 4
        end
    end

    return (
        granularity = unitPeriod,
        frequency = diffBetweenTimestamps,
        weekdays = weekDaysSeries,
    )
end

"""
    merge(timeArrayVector::Vector{TimeArray},modelFl::DataType=Float64)

Merge multiple `TimeArray` objects into a single `TimeArray`. The function aligns the time series based on their timestamps, preserving only the common time span.

# Arguments
- `timeArrayVector::Vector{TimeArray}`: Vector of `TimeArray` objects to be merged.
- `modelFl::DataType`: The type of the values in the `TimeArray`. Default is `Float64`.

# Returns
A `TimeArray` object representing the merged time series.

"""
function merge(timeArrayVector::Vector{TimeArray}, modelFl::DataType = Float64)
    if length(timeArrayVector) == 0
        return TimeArray(DateTime[], [])
    end

    if length(timeArrayVector) == 1
        return timeArrayVector[1]
    end

    initialTimestamp = timestamp(timeArrayVector[1])[1]
    finalTimestamp = timestamp(timeArrayVector[1])[end]

    for ta in timeArrayVector[2:end]
        initialTimestampAux = timestamp(ta)[1]
        finalTimestampAux = timestamp(ta)[end]
        initialTimestamp = max(initialTimestamp, initialTimestampAux)
        finalTimestamp = min(finalTimestamp, finalTimestampAux)
    end

    newTimeArrayVector = []
    for ta in timeArrayVector
        newTimeArray = from(ta, initialTimestamp)
        newTimeArray = to(newTimeArray, finalTimestamp)
        push!(newTimeArrayVector, newTimeArray)
    end

    auxiliarDf = DataFrame((:timestamp => timestamp(newTimeArrayVector[1])))
    for ta in newTimeArrayVector
        # Add a column with ta colname and values
        for (i,colname) in enumerate(colnames(ta))
            valuesTa::Array{modelFl} = Array{modelFl}(values(ta)[:,i])
            auxiliarDf[!, colname] = valuesTa
        end
    end

    return TimeArray(auxiliarDf, timestamp = :timestamp)
end
