using CSV
using DataFrames
using Dates
using Sarimax
using Statistics
using Test
using Random
using TimeSeries
using JSON
using LinearAlgebra

# Testes dos modelos


@testset "Sarimax.jl" begin
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

    include("statsapi.jl")

    include("reference_values.jl")

    include("diagnostics.jl")

    include("integrations.jl")

    include("statistical_properties.jl")

    include("numerical_conditioning.jl")

    include("objective_functions.jl")

    include("warm_start.jl")

    include("solver_interface.jl")

    include("stl_parity.jl")

    include("missing_data.jl")

    include("exact_likelihood.jl")

    include("objective_guardrails.jl")

    include("ridge_determinante.jl")

    include("deterministic_term.jl")

    include("order_guards.jl")

    include("exact_ml_determinant.jl")

    include("exog_dynamics.jl")

    include("exog_penalty_position.jl")

    include("multistart.jl")

    include("aqua.jl")
end
