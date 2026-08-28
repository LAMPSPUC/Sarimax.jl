# Deprecated camelCase API (renamed to snake_case in v0.3).
# These shims keep old code working with a deprecation warning; they are scheduled
# for removal in v2.0. Keyword-argument names are unchanged.

Base.@deprecate loadDataset(dataset::Datasets) load_dataset(dataset)
Base.@deprecate loadDataset(df::DataFrame, showLogs::Bool = false) load_dataset(df, showLogs)
Base.@deprecate splitTrainTest(data::TimeArray; kwargs...) split_train_test(data; kwargs...)
Base.@deprecate hasFitMethods(modelType::Type{<:SarimaxModel}) has_fit_methods(modelType)
Base.@deprecate hasHyperparametersMethods(modelType::Type{<:SarimaxModel}) has_hyperparameters_methods(
    modelType,
)
Base.@deprecate getHyperparametersNumber(model::SARIMAModel) get_hyperparameters_number(model)
Base.@deprecate getHyperparametersNumber(model::JuMP.Model) get_hyperparameters_number(model)
Base.@deprecate automaticDifferentiation(series::TimeArray; kwargs...) automatic_differentiation(
    series;
    kwargs...,
)
Base.@deprecate identifyGranularity(datetimes::Vector) identify_granularity(datetimes)
Base.@deprecate buildDatetimes(startDatetime, granularity, weekDaysOnly::Bool, datetimesLength::Int) build_datetimes(
    startDatetime,
    granularity,
    weekDaysOnly,
    datetimesLength,
)
Base.@deprecate copyTimeArray(y::TimeSeries.TimeArray) copy_time_array(y)
Base.@deprecate deepcopyTimeArray(y::TimeSeries.TimeArray) deepcopy_time_array(y)
Base.@deprecate toMA(model::SARIMAModel, maxLags::Int = 12) to_ma(model, maxLags)
Base.@deprecate differentiatedCoefficients(d::Int, D::Int, s::Int, Fl::DataType = Float64) differentiated_coefficients(
    d,
    D,
    s,
    Fl,
)
