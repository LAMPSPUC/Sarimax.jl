"""
The Datasets Enum is used to identify the dataset used in the `load_dataset` function.

The Datasets Enum is defined as follows:

- `AIR_PASSENGERS = 1`
- `GDPC1 = 2`
- `NROU = 3`

The `load_dataset` function uses this Enum to determine the dataset to be loaded.
"""
@enum Datasets begin
    AIR_PASSENGERS = 1
    GDPC1 = 2
    NROU = 3
end

@doc("
The AirPassengers dataset, it contains the classic Box-Jenkins airline data.
", AIR_PASSENGERS)

@doc("
The GDPC1 dataset, it contains the US Real Gross Domestic Product (GDPC1) from 1947 to 2019.
", GDPC1)

@doc("
The NROU dataset: the U.S. Noncyclical Rate of Unemployment (FRED series NROU), quarterly, including FRED's projections.
", NROU)


const AIR_PASSENGERS = instances(Datasets)[1]
const GDPC1 = instances(Datasets)[2]
const NROU = instances(Datasets)[3]
export AIR_PASSENGERS, GDPC1, NROU

datasetsPaths = [
    joinpath(dirname(@__DIR__()), "datasets", "airpassengers.csv"),
    joinpath(dirname(@__DIR__()), "datasets", "GDPC1.csv"),
    joinpath(dirname(@__DIR__()), "datasets", "NROU.csv"),
]



"""
    load_dataset(
        dataset::Datasets
    )

Loads a dataset from the `Datasets` enum.

# Example
```jldoctest
julia> airPassengers = load_dataset(AIR_PASSENGERS)
204×1 TimeArray{Float64, 1, Date, Vector{Float64}} 1991-07-01 to 2008-06-01
│            │ value   │
├────────────┼─────────┤
│ 1991-07-01 │ 3.5266  │
│ 1991-08-01 │ 3.1809  │
│ ⋮          │ ⋮       │
│ 2008-06-01 │ 19.4317 │

```
"""
function load_dataset(dataset::Datasets)
    datasetIndex = Int(dataset)
    seriesData = CSV.read(datasetsPaths[datasetIndex], DataFrame)
    y = TimeArray(seriesData, timestamp=:date)
    return y
end

"""
    load_dataset(
        df::DataFrame,
        showLogs::Bool=false
    )

Loads a dataset from a Dataframe. If the DataFrame does not have a column named
`date` a new column will be created with the index of the DataFrame.

# Arguments
- `df::DataFrame`: The DataFrame to be converted to a TimeArray.
- `showLogs::Bool=false`: If true, logs will be shown.

# Example
```jldoctest
julia> airPassengersDf = CSV.File("datasets/airpassengers.csv") |> DataFrame
julia> airPassengers = load_dataset(airPassengersDf)
204×1 TimeArray{Float64, 1, Date, Vector{Float64}} 1991-07-01 to 2008-06-01
│            │ value   │
├────────────┼─────────┤
│ 1991-07-01 │ 3.5266  │
│ 1991-08-01 │ 3.1809  │
│ ⋮          │ ⋮       │
│ 2008-06-01 │ 19.4317 │

```
"""
function load_dataset(df::DataFrame, showLogs::Bool=false)
    auxiliarDF = deepcopy(df)
    if !(:date in propertynames(auxiliarDF))
        showLogs && @info("The DataFrame does not have a column named 'date'.")
        showLogs && @info("Creating a date column with the index of the DataFrame")
        auxiliarDF[!, :date] = [Date(i) for i = 1:size(auxiliarDF, 1)]
    end
    y = TimeArray(auxiliarDF, timestamp=:date)
    return y
end

"""
    split_train_test(
        data::TimeArray;
        trainSize::Fl=0.8
    ) where Fl<:AbstractFloat

Splits the time series in training and testing sets.
"""
function split_train_test(
    data::TimeArray;
    trainPercentage::AbstractFloat=0.8,
)
    trainingSetEndIndex = floor(Int, trainPercentage * length(data))
    trainingSet = TimeArray(
        timestamp(data)[1:trainingSetEndIndex],
        values(data)[1:trainingSetEndIndex],
    )
    testingSet = TimeArray(
        timestamp(data)[trainingSetEndIndex+1:end],
        values(data)[trainingSetEndIndex+1:end],
    )
    return trainingSet, testingSet
end
