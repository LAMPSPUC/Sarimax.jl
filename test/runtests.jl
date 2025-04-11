using CSV
using DataFrames
using Dates
using SARIMAX
using Statistics
using Test
using Random
using TimeSeries
using Revise
using JSON

# Testes dos modelos

include("models/sarima_base.jl")

include("models/sarima.jl")

include("models/sarima_auto.jl")

include("models/sarima_fit.jl")

include("models/sarima_predict.jl")

include("datetime_utils.jl")

include("utils.jl")

include("exceptions.jl")

include("datasets.jl")

include("fit.jl")

include("test_statistical_tests.jl")
