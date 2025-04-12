# API Reference {#Reference}

## Types

### Structs

```@docs
Sarimax.SARIMAModel
```

### Enums

```@docs
Sarimax.Datasets
```

### Exceptions

```@docs
Sarimax.ModelNotFitted
Sarimax.MissingMethodImplementation
Sarimax.MissingExogenousData
Sarimax.InconsistentDatePattern
Sarimax.InvalidParametersCombination
```

## Constructors

```@docs
Sarimax.SARIMA(
    y::TimeSeries.TimeArray,
    p::Int,
    d::Int,
    q::Int;
    seasonality::Int = 1,
    P::Int = 0,
    D::Int = 0,
    Q::Int = 0,
    silent::Bool = true,
    allowMean::Bool = true,
    allowDrift::Bool = false)
```

```@docs
Sarimax.SARIMA(
    y::TimeSeries.TimeArray;
    exog::Union{TimeSeries.TimeArray,Nothing} = nothing,
    arCoefficients::Union{Vector{Fl},Nothing} = nothing,
    maCoefficients::Union{Vector{Fl},Nothing} = nothing,
    seasonalARCoefficients::Union{Vector{Fl},Nothing} = nothing,
    seasonalMACoefficients::Union{Vector{Fl},Nothing} = nothing,
    mean::Union{Fl,Nothing} = nothing,
    trend::Union{Fl,Nothing} = nothing,
    exogCoefficients::Union{Vector{Fl},Nothing} = nothing,
    d::Int = 0,
    D::Int = 0,
    seasonality::Int = 1,
    silent::Bool = true,
    allowMean::Bool = true,
    allowDrift::Bool = false)
```

```@docs
Sarimax.SARIMA(
    y::TimeSeries.TimeArray,
    exog::Union{TimeSeries.TimeArray,Nothing},
    p::Int,
    d::Int,
    q::Int;
    seasonality::Int = 1,
    P::Int = 0,
    D::Int = 0,
    Q::Int = 0,
    silent::Bool = true,
    allowMean::Bool = true,
    allowDrift::Bool = false)
```

## Model Functions

### Model Fitting and Prediction

```@docs
Sarimax.fit!
Sarimax.predict!
Sarimax.auto
Sarimax.simulate
```

### Model Evaluation

```@docs
Sarimax.loglikelihood
Sarimax.loglike
Sarimax.aic
Sarimax.aicc
Sarimax.bic
```

### Time Series Operations

```@docs
Sarimax.differentiate
Sarimax.integrate
Sarimax.differentiatedCoefficients
Sarimax.toMA
```

### Dataset and Utility Functions

```@docs
Sarimax.loadDataset
Sarimax.splitTrainTest
Sarimax.identifyGranularity
Sarimax.buildDatetimes
```

### Model Information

```@docs
Sarimax.hasFitMethods
Sarimax.hasHyperparametersMethods
Sarimax.getHyperparametersNumber
```

### Model Manipulation

```@docs
Sarimax.print
```

```@docs
Sarimax.copyTimeArray(y::TimeSeries.TimeArray)
Sarimax.deepcopyTimeArray(y::TimeSeries.TimeArray)
```
