# API Reference {#Reference}

## Types

### Structs

```@docs
SARIMAX.SARIMAModel
```

### Enums

```@docs
SARIMAX.Datasets
```

### Exceptions

```@docs
SARIMAX.ModelNotFitted
SARIMAX.MissingMethodImplementation
SARIMAX.MissingExogenousData
SARIMAX.InconsistentDatePattern
SARIMAX.InvalidParametersCombination
```

## Constructors

```@docs
SARIMAX.SARIMA(
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
SARIMAX.SARIMA(
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
SARIMAX.SARIMA(
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
SARIMAX.fit!
SARIMAX.predict!
SARIMAX.auto
SARIMAX.simulate
```

### Model Evaluation

```@docs
SARIMAX.loglikelihood
SARIMAX.loglike
SARIMAX.aic
SARIMAX.aicc
SARIMAX.bic
```

### Time Series Operations

```@docs
SARIMAX.differentiate
SARIMAX.integrate
SARIMAX.differentiatedCoefficients
SARIMAX.toMA
```

### Dataset and Utility Functions

```@docs
SARIMAX.loadDataset
SARIMAX.splitTrainTest
SARIMAX.identifyGranularity
SARIMAX.buildDatetimes
```

### Model Information

```@docs
SARIMAX.hasFitMethods
SARIMAX.hasHyperparametersMethods
SARIMAX.getHyperparametersNumber
```

### Model Manipulation

```@docs
SARIMAX.print
```

```@docs
TimeSeries.copy(y::TimeSeries.TimeArray)
TimeSeries.deepcopy(y::TimeSeries.TimeArray)
```
