# API Reference

## Types

```@docs
Sarimax.SARIMAModel
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

## Model specification

```@docs
Sarimax.SARIMA
```

## Estimation

```@docs
Sarimax.fit!
Sarimax.auto
```

## Forecasting and simulation

```@docs
Sarimax.predict!
Sarimax.simulate
Sarimax.forecastErrors
```

## Coefficients, residuals and inference (StatsAPI)

```@docs
Sarimax.coef(::Sarimax.SARIMAModel)
Sarimax.coefnames(::Sarimax.SARIMAModel)
Sarimax.residuals(::Sarimax.SARIMAModel)
Sarimax.fitted(::Sarimax.SARIMAModel)
Sarimax.nobs(::Sarimax.SARIMAModel)
Sarimax.vcov(::Sarimax.SARIMAModel)
Sarimax.stderror(::Sarimax.SARIMAModel)
Sarimax.cssResiduals
```

## Likelihood and information criteria

```@docs
Sarimax.loglike
Sarimax.aic
Sarimax.aicc
Sarimax.bic
```

## Residual diagnostics

```@docs
Sarimax.ljung_box_test
Sarimax.jarque_bera_test
```

## Transformations

```@docs
Sarimax.boxcox_transform
Sarimax.inverse_boxcox
Sarimax.boxcox_lambda
```

## Model evaluation

```@docs
Sarimax.cross_validation
```

## Stationarity and seasonality tests

```@docs
Sarimax.kpss_test
Sarimax.ocsb_test
Sarimax.automatic_differentiation
```

## Differencing utilities

```@docs
Sarimax.differentiate
Sarimax.integrate
Sarimax.differentiated_coefficients
Sarimax.to_ma
```

## Parameterizations

```@docs
Sarimax.reflectionToMA
Sarimax.reflectionToAR
```

## Data handling

```@docs
Sarimax.load_dataset
Sarimax.split_train_test
Sarimax.build_datetimes
Sarimax.identify_granularity
Sarimax.copy_time_array
Sarimax.deepcopy_time_array
```

## MLJ integration

```@docs
Sarimax.SARIMAForecaster
```

## Display

```@docs
Sarimax.print
```

## Introspection

```@docs
Sarimax.has_fit_methods
Sarimax.has_hyperparameters_methods
Sarimax.get_hyperparameters_number
```
