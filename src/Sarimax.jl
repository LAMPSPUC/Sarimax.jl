"""
Sarimax

A module for Seasonal Autoregressive Integrated Moving Average with eXogenous regressors
(Sarimax) modeling in Julia.

This module provides functionality for time series analysis and forecasting using Sarimax
models. It includes tools for model fitting, prediction, and various statistical tests.

Main features:
- SARIMA model implementation
- Automatic model selection
- Exogenous variables support
- Time series differentiation and integration
- Statistical tests (e.g., KPSS test)
- Dataset handling utilities

For more information, see the documentation of individual functions and types.
"""
module Sarimax


import Base: show, print, showerror

using Alpine
using CSV
using DataFrames
using Dates
using Distributions
using Ipopt
using JuMP
using LinearAlgebra
using MathOptInterface
using OffsetArrays
using Printf
using Optim
using Random
using SCIP
using SeasonalTrendLoess
using StateSpaceModels
using Statistics
using TimeSeries
using Tables

import MLJModelInterface
import RecipesBase

import StatsAPI
import StatsAPI: coef, coefnames, residuals, nobs, fitted, stderror, vcov, loglikelihood

abstract type SarimaxModel end

include("datasets.jl")
include("datetime_utils.jl")
include("exceptions.jl")
include("fit.jl")
include("models/sarima.jl")
include("utils.jl")
include("stl_r.jl")
include("statistical_tests.jl")
include("diagnostics.jl")
include("transformations.jl")
include("cross_validation.jl")
include("integrations.jl")
include("mlj.jl")
include("statsapi.jl")
include("deprecated.jl")


# Export types
export SARIMAModel

# Export Exceptions/Errors
export ModelNotFitted
export MissingMethodImplementation
export MissingExogenousData
export InconsistentDatePattern
export InvalidParametersCombination

# Export enums
export Datasets

# Export functions
export automatic_differentiation
export split_train_test
export print
export copy_time_array
export deepcopy_time_array
export fit!
export predict!
export SARIMA
export differentiate
export identify_granularity
export integrate
export simulate
export load_dataset
export loglikelihood
export loglike
export has_fit_methods
export has_hyperparameters_methods
export get_hyperparameters_number
export auto
export aic
export aicc
export bic
export ljung_box_test
export jarque_bera_test
export boxcox_transform
export inverse_boxcox
export boxcox_lambda
export cross_validation
export SARIMAForecaster
export build_datetimes
export to_ma
export differentiated_coefficients

# StatsAPI interface
export coef
export coefnames
export residuals
export nobs
export fitted
export stderror
export vcov

# Deprecated camelCase names (kept exported until v1.0; see src/deprecated.jl)
export loadDataset
export splitTrainTest
export hasFitMethods
export hasHyperparametersMethods
export getHyperparametersNumber
export automaticDifferentiation
export identifyGranularity
export buildDatetimes
export copyTimeArray
export deepcopyTimeArray
export toMA
export differentiatedCoefficients


end # module Sarimax
