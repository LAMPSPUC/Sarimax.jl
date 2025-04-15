var documenterSearchIndex = {"docs":
[{"location":"reference/#API-Reference-{#Reference}","page":"API Reference","title":"API Reference {#Reference}","text":"","category":"section"},{"location":"reference/#Types","page":"API Reference","title":"Types","text":"","category":"section"},{"location":"reference/#Structs","page":"API Reference","title":"Structs","text":"","category":"section"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Sarimax.SARIMAModel","category":"page"},{"location":"reference/#Main.Sarimax.SARIMAModel","page":"API Reference","title":"Main.Sarimax.SARIMAModel","text":"The SARIMAModel struct represents a SARIMA model. It contains the following fields:\n\ny: The time series data.\np: The autoregressive order for the non-seasonal part.\nd: The degree of differencing.\nq: The moving average order for the non-seasonal part.\nseasonality: The seasonality period.\nP: The autoregressive order for the seasonal part.\nD: The degree of seasonal differencing.\nQ: The moving average order for the seasonal part.\nmetadata: A dictionary containing model metadata.\nexog: Optional exogenous variables.\nc: The constant term.\ntrend: The trend term.\nϕ: The autoregressive coefficients for the non-seasonal part.\nθ: The moving average coefficients for the non-seasonal part.\nΦ: The autoregressive coefficients for the seasonal part.\nΘ: The moving average coefficients for the seasonal part.\nϵ: The residuals.\nexogCoefficients: The coefficients of the exogenous variables.\nσ²: The variance of the residuals.\nfitInSample: The in-sample fit.\nforecast: The forecast.\nsilent: Whether to suppress output.\nallowMean: Whether to include a mean term in the model.\nallowDrift: Whether to include a drift term in the model.\nkeepProvidedCoefficients: Whether to keep the provided coefficients.\n\n\n\n\n\n","category":"type"},{"location":"reference/#Enums","page":"API Reference","title":"Enums","text":"","category":"section"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Sarimax.Datasets","category":"page"},{"location":"reference/#Main.Sarimax.Datasets","page":"API Reference","title":"Main.Sarimax.Datasets","text":"The Datasets Enum is used to identify the dataset used in the loadDataset function.\n\nThe Datasets Enum is defined as follows:\n\nAIR_PASSENGERS = 1\nGDPC1 = 2\nNROU = 3\n\nThe loadDataset function uses this Enum to determine the dataset to be loaded.\n\n\n\n\n\n","category":"type"},{"location":"reference/#Exceptions","page":"API Reference","title":"Exceptions","text":"","category":"section"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Sarimax.ModelNotFitted\nSarimax.MissingMethodImplementation\nSarimax.MissingExogenousData\nSarimax.InconsistentDatePattern\nSarimax.InvalidParametersCombination","category":"page"},{"location":"reference/#Main.Sarimax.ModelNotFitted","page":"API Reference","title":"Main.Sarimax.ModelNotFitted","text":"struct ModelNotFitted <: Exception\n\nAn exception type that indicates the model has not been fitted yet.\n\nUsage\n\nThis exception can be thrown when an operation that requires a fitted model is attempted on an unfitted model.\n\n\n\n\n\n","category":"type"},{"location":"reference/#Main.Sarimax.MissingMethodImplementation","page":"API Reference","title":"Main.Sarimax.MissingMethodImplementation","text":"MissingMethodImplementation <: Exception\n\nCustom exception type for indicating that a required method is not implemented in the model.\n\nFields\n\nmethod::String: The name of the method that is missing.\n\n\n\n\n\n","category":"type"},{"location":"reference/#Main.Sarimax.MissingExogenousData","page":"API Reference","title":"Main.Sarimax.MissingExogenousData","text":"MissingExogenousData <: Exception\n\nAn exception type that indicates the absence of exogenous data required for forecasting the requested horizon.\n\n\n\n\n\n","category":"type"},{"location":"reference/#Main.Sarimax.InconsistentDatePattern","page":"API Reference","title":"Main.Sarimax.InconsistentDatePattern","text":"struct InconsistentDatePattern <: Exception\n\nAn exception type to indicate that the timestamps do not follow a consistent pattern.\n\n\n\n\n\n","category":"type"},{"location":"reference/#Main.Sarimax.InvalidParametersCombination","page":"API Reference","title":"Main.Sarimax.InvalidParametersCombination","text":"InvalidParametersCombination <: Exception\n\nA custom exception type to indicate that the combination of parameters provided to the model is invalid.\n\nFields\n\nmsg::String: A message describing why the parameters are invalid.\n\n\n\n\n\n","category":"type"},{"location":"reference/#Constructors","page":"API Reference","title":"Constructors","text":"","category":"section"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Sarimax.SARIMA(\n    y::TimeSeries.TimeArray,\n    p::Int,\n    d::Int,\n    q::Int;\n    seasonality::Int = 1,\n    P::Int = 0,\n    D::Int = 0,\n    Q::Int = 0,\n    silent::Bool = true,\n    allowMean::Bool = true,\n    allowDrift::Bool = false)","category":"page"},{"location":"reference/#Main.Sarimax.SARIMA-Tuple{TimeArray, Int64, Int64, Int64}","page":"API Reference","title":"Main.Sarimax.SARIMA","text":"SARIMA constructor.\n\nParameters:\n- y: TimeArray with the time series.\n- p: Int with the autoregressive order for the non-seasonal part.\n- d: Int with the degree of differencing.\n- q: Int with the moving average order for the non-seasonal part.\n- seasonality: Int with the seasonality period.\n- P: Int with the autoregressive order for the seasonal part.\n- D: Int with the degree of seasonal differencing.\n- Q: Int with the moving average order for the seasonal part.\n- silent: Bool to supress output.\n- allowMean: Bool to include a mean term in the model.\n- allowDrift: Bool to include a drift term in the model.\n\n\n\n\n\n","category":"method"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Sarimax.SARIMA(\n    y::TimeSeries.TimeArray;\n    exog::Union{TimeSeries.TimeArray,Nothing} = nothing,\n    arCoefficients::Union{Vector{Fl},Nothing} = nothing,\n    maCoefficients::Union{Vector{Fl},Nothing} = nothing,\n    seasonalARCoefficients::Union{Vector{Fl},Nothing} = nothing,\n    seasonalMACoefficients::Union{Vector{Fl},Nothing} = nothing,\n    mean::Union{Fl,Nothing} = nothing,\n    trend::Union{Fl,Nothing} = nothing,\n    exogCoefficients::Union{Vector{Fl},Nothing} = nothing,\n    d::Int = 0,\n    D::Int = 0,\n    seasonality::Int = 1,\n    silent::Bool = true,\n    allowMean::Bool = true,\n    allowDrift::Bool = false)","category":"page"},{"location":"reference/#Main.Sarimax.SARIMA-Tuple{TimeArray}","page":"API Reference","title":"Main.Sarimax.SARIMA","text":"SARIMA constructor to initialize model with provided coefficients.\n\nParameters:\n\ny: TimeArray with the time series.\nexog: TimeArray with the exogenous variables.\narCoefficients: Vector with the autoregressive coefficients.\nmaCoefficients: Vector with the moving average coefficients.\nseasonalARCoefficients: Vector with the autoregressive coefficients for the seasonal component.\nseasonalMACoefficients: Vector with the moving average coefficients for the seasonal component.\nmean: Float with the mean term.\ntrend: Float with the trend term.\nexogCoefficients: Vector with the exogenous coefficients.\nd: Int with the degree of differencing.\nD: Int with the degree of seasonal differencing.\nseasonality: Int with the seasonality period.\nsilent: Bool to supress output.\nallowMean: Bool to include a mean term in the model.\nallowDrift: Bool to include a drift term in the model.\n\n\n\n\n\n","category":"method"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Sarimax.SARIMA(\n    y::TimeSeries.TimeArray,\n    exog::Union{TimeSeries.TimeArray,Nothing},\n    p::Int,\n    d::Int,\n    q::Int;\n    seasonality::Int = 1,\n    P::Int = 0,\n    D::Int = 0,\n    Q::Int = 0,\n    silent::Bool = true,\n    allowMean::Bool = true,\n    allowDrift::Bool = false)","category":"page"},{"location":"reference/#Main.Sarimax.SARIMA-Tuple{TimeArray, Union{Nothing, TimeArray}, Int64, Int64, Int64}","page":"API Reference","title":"Main.Sarimax.SARIMA","text":"SARIMA constructor.\n\nParameters:\n- y: TimeArray with the time series.\n- exog: TimeArray with the exogenous variables.\n- p: Int with the order of the AR component.\n- d: Int with the degree of differencing.\n- q: Int with the order of the MA component.\n- seasonality: Int with the seasonality period.\n- P: Int with the order of the seasonal AR component.\n- D: Int with the degree of seasonal differencing.\n- Q: Int with the order of the seasonal MA component.\n- silent: Bool to supress output.\n- allowMean: Bool to include a mean term in the model.\n- allowDrift: Bool to include a drift term in the model.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Model-Functions","page":"API Reference","title":"Model Functions","text":"","category":"section"},{"location":"reference/#Model-Fitting-and-Prediction","page":"API Reference","title":"Model Fitting and Prediction","text":"","category":"section"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Sarimax.fit!\nSarimax.predict!\nSarimax.auto\nSarimax.simulate","category":"page"},{"location":"reference/#Main.Sarimax.fit!","page":"API Reference","title":"Main.Sarimax.fit!","text":"fit!(\n    model::SARIMAModel;\n    silent::Bool=true,\n    optimizer::DataType=Ipopt.Optimizer,\n    objectiveFunction::String=\"mse\"\n    automaticExogDifferentiation::Bool=false\n)\n\nEstimate the SARIMA model parameters via non-linear least squares. The resulting optimal parameters as well as the residuals and the model's σ² are stored within the model. The default objective function used to estimate the parameters is the mean squared error (MSE), but it can be changed to the maximum likelihood (ML) by setting the objectiveFunction parameter to \"ml\".\n\nArguments\n\nmodel::SARIMAModel: The SARIMA model to be fitted.\nsilent::Bool: Whether to suppress solver output. Default is true.\noptimizer::DataType: The optimizer to be used for optimization. Default is Ipopt.Optimizer.\nobjectiveFunction::String: The objective function used for estimation. Default is \"mse\".\nautomaticExogDifferentiation::Bool: Whether to automatically differentiate the exogenous variables. Default is false.\n\nExample\n\njulia> airPassengers = loadDataset(AIR_PASSENGERS)\n\njulia> model = SARIMA(airPassengers,0,1,1;seasonality=12,P=0,D=1,Q=1)\n\njulia> fit!(model)\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.predict!","page":"API Reference","title":"Main.Sarimax.predict!","text":"predict!(\n    model::SARIMAModel;\n    stepsAhead::Int = 1\n    seed::Int = 1234,\n    isSimulation::Bool = false,\n    displayConfidenceIntervals::Bool = false,\n    confidenceLevel::Fl = 0.95\n    automaticExogDifferentiation::Bool=false\n) where Fl<:AbstractFloat\n\nPredicts the SARIMA model for the next stepsAhead periods. The resulting forecast is stored within the model in the forecast field.\n\nArguments\n\nmodel::SARIMAModel: The SARIMA model to make predictions.\nstepsAhead::Int: The number of periods ahead to forecast (default: 1).\nseed::Int: Seed for random number generation when simulating forecasts (default: 1234).\nisSimulation::Bool: Whether to perform a simulation-based forecast (default: false).\ndisplayConfidenceIntervals::Bool: Whether to display confidence intervals (default: false).\nconfidenceLevel::Fl: The confidence level for the confidence intervals (default: 0.95).\nautomaticExogDifferentiation::Bool: Whether to automatically differentiate the exogenous variables. Default is false.\n\nExample\n\n```julia julia> airPassengers = loadDataset(AIR_PASSENGERS)\n\njulia> model = SARIMA(airPassengers, 0, 1, 1; seasonality=12, P=0, D=1, Q=1)\n\njulia> fit!(model)\n\njulia> predict!(model; stepsAhead=12)\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.auto","page":"API Reference","title":"Main.Sarimax.auto","text":"auto(\n    y::TimeArray;\n    exog::Union{TimeArray,Nothing}=nothing,\n    seasonality::Int=1,\n    d::Int = -1,\n    D::Int = -1,\n    maxp::Int = 5,\n    maxd::Int = 2,\n    maxq::Int = 5,\n    maxP::Int = 2,\n    maxD::Int = 1,\n    maxQ::Int = 2,\n    maxOrder::Int = 5,\n    informationCriteria::String = \"aicc\",\n    allowMean:Union{Bool,Nothing} = nothing,\n    allowDrift::Union{Bool,Nothing} = nothing,\n    integrationTest::String = \"kpss\",\n    seasonalIntegrationTest::String = \"seas\",\n    objectiveFunction::String = \"mse\",\n    assertStationarity::Bool = true,\n    assertInvertibility::Bool = true,\n    showLogs::Bool = false,\n    outlierDetection::Bool = false\n    searchMethod::String = \"stepwise\"\n)\n\nAutomatically fits the best SARIMA model according to the specified parameters.\n\nArguments\n\ny::TimeArray: The time series data.\nexog::Union{TimeArray,Nothing}: Optional exogenous variables. If Nothing, no exogenous variables are used.\nseasonality::Int: The seasonality period. Default is 1 (non-seasonal).\nd::Int: The degree of differencing for the non-seasonal part. Default is -1 (auto-select).\nD::Int: The degree of differencing for the seasonal part. Default is -1 (auto-select).\nmaxp::Int: The maximum autoregressive order for the non-seasonal part. Default is 5.\nmaxd::Int: The maximum integration order for the non-seasonal part. Default is 2.\nmaxq::Int: The maximum moving average order for the non-seasonal part. Default is 5.\nmaxP::Int: The maximum autoregressive order for the seasonal part. Default is 2.\nmaxD::Int: The maximum integration order for the seasonal part. Default is 1.\nmaxQ::Int: The maximum moving average order for the seasonal part. Default is 2.\nmaxOrder::Int: The maximum order for the non-seasonal part. Default is 5.\ninformationCriteria::String: The information criteria to be used for model selection. Options are \"aic\", \"aicc\", or \"bic\". Default is \"aicc\".\nallowMean::Union{Bool,Nothing}: Whether to include a mean term in the model. Default is nothing.\nallowDrift::Union{Bool,Nothing}: Whether to include a drift term in the model. Default is nothing.\nintegrationTest::String: The integration test to be used for determining the non-seasonal integration order. Default is \"kpss\".\nseasonalIntegrationTest::String: The integration test to be used for determining the seasonal integration order. Default is \"seas\".\nobjectiveFunction::String: The objective function to be used for model selection. Options are \"mse\", \"ml\", or \"bilevel\". Default is \"mse\".\nassertStationarity::Bool: Whether to assert stationarity of the fitted model. Default is true.\nassertInvertibility::Bool: Whether to assert invertibility of the fitted model. Default is true.\nshowLogs::Bool: Whether to suppress output. Default is false.\noutlierDetection::Bool: Whether to perform outlier detection. Default is false.\nsearchMethod::String = \"stepwise\"\n\nReferences\n\nHyndman, RJ and Khandakar. \"Automatic time series forecasting: The forecast package for R.\" Journal of Statistical Software, 26(3), 2008.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.simulate","page":"API Reference","title":"Main.Sarimax.simulate","text":"simulate(\n    model::SARIMAModel,\n    stepsAhead::Int = 1,\n    numScenarios::Int = 200,\n    seed::Int = 1234\n)\n\nSimulates the SARIMA model for the next stepsAhead periods assuming that the model's estimated σ². Returns a vector of numScenarios scenarios of the forecasted values.\n\nArguments\n\nmodel::SARIMAModel: The SARIMA model to simulate.\nstepsAhead::Int: The number of periods ahead to simulate. Default is 1.\nnumScenarios::Int: The number of simulation scenarios. Default is 200.\nseed::Int: The seed of the simulation. Default is 1234.\n\nReturns\n\nVector{Vector{AbstractFloat}}: A vector of scenarios, each containing the forecasted values for the next stepsAhead periods.\n\nExample\n\njulia> airPassengers = loadDataset(AIR_PASSENGERS)\n\njulia> model = SARIMA(airPassengers, 0, 1, 1; seasonality=12, P=0, D=1, Q=1)\n\njulia> fit!(model)\n\njulia> scenarios = simulate(model, stepsAhead=12, numScenarios=1000)\n\n\n\n\n\n","category":"function"},{"location":"reference/#Model-Evaluation","page":"API Reference","title":"Model Evaluation","text":"","category":"section"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Sarimax.loglikelihood\nSarimax.loglike\nSarimax.aic\nSarimax.aicc\nSarimax.bic","category":"page"},{"location":"reference/#Main.Sarimax.loglikelihood","page":"API Reference","title":"Main.Sarimax.loglikelihood","text":"loglikelihood(model::SarimaxModel)\n\nCalculate the log-likelihood of a SARIMAModel. The log-likelihood is calculated based on the formula -0.5 * (T * log(2π) + T * log(σ²) + sum(ϵ.^2 ./ σ²)) where:\n\nT is the length of the residuals vector (ϵ).\nσ² is the estimated variance of the model.\n\nArguments\n\nmodel::SarimaxModel: A SARIMAModel object.\n\nReturns\n\nThe log-likelihood of the SARIMAModel.\n\nErrors\n\nMissingMethodImplementation(\"fit!\"): Thrown if the fit! method is not implemented for the given model type.\nModelNotFitted(): Thrown if the model has not been fitted.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.loglike","page":"API Reference","title":"Main.Sarimax.loglike","text":"loglike(model::SarimaxModel)\n\nCalculate the log-likelihood of a SARIMAModel using the normal probability density function. The log-likelihood is calculated by summing the logarithm of the probability density function (PDF) of each data point under the assumption of a normal distribution with mean 0 and standard deviation equal to the square root of the estimated variance (σ²) of the model.\n\nArguments\n\nmodel::SarimaxModel: A SARIMAModel object.\n\nReturns\n\nThe log-likelihood of the SARIMAModel.\n\nErrors\n\nMissingMethodImplementation(\"fit!\"): Thrown if the fit! method is not implemented for the given model type.\nModelNotFitted(): Thrown if the model has not been fitted.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.aic","page":"API Reference","title":"Main.Sarimax.aic","text":"aic(K::Int, loglikeVal::Fl) where Fl<:AbstractFloat -> Fl\n\nCalculate the Akaike Information Criterion (AIC) for a given number of parameters and log-likelihood value.\n\nArguments\n\nK::Int: Number of parameters in the model.\nloglikeVal::Fl: Log-likelihood value of the model.\n\nReturns\n\nThe AIC value calculated using the formula: AIC = 2K - 2loglikeVal.\n\n\n\n\n\naic(model::SarimaxModel; offset::Fl) -> Fl where Fl<:AbstractFloat\n\nCalculate the Akaike Information Criterion (AIC) for a Sarimax model.\n\nArguments\n\nmodel::SarimaxModel: The Sarimax model for which AIC is calculated.\noffset::Fl=0.0: Offset value to be added to the AIC value.\n\nReturns\n\nThe AIC value calculated using the number of parameters and log-likelihood value of the model.\n\nErrors\n\nThrows a MissingMethodImplementation if the getHyperparametersNumber method is not implemented for the given model type.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.aicc","page":"API Reference","title":"Main.Sarimax.aicc","text":"aicc(T::Int, K::Int, loglikeVal::Fl) where Fl<:AbstractFloat -> Fl\n\nCalculate the corrected Akaike Information Criterion (AICc) for a given number of observations, number of parameters, and log-likelihood value.\n\nArguments\n\nT::Int: Number of observations in the data.\nK::Int: Number of parameters in the model.\nloglikeVal::Fl: Log-likelihood value of the model.\n\nReturns\n\nThe AICc value calculated using the formula: AICc = AIC(K, loglikeVal) + ((2KK + 2*K) / (T - K - 1)).\n\n\n\n\n\naicc(model::SarimaxModel; offset::Fl) -> Fl where Fl<:AbstractFloat\n\nCalculate the Corrected Akaike Information Criterion (AICc) for a Sarimax model.\n\nArguments\n\nmodel::SarimaxModel: The Sarimax model for which AICc is calculated.\noffset::Fl=0.0: Offset value to be added to the AICc value.\n\nReturns\n\nThe AICc value calculated using the number of parameters, sample size, and log-likelihood value of the model.\n\nErrors\n\nThrows a MissingMethodImplementation if the getHyperparametersNumber method is not implemented for the given model type.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.bic","page":"API Reference","title":"Main.Sarimax.bic","text":"bic(T::Int, K::Int, loglikeVal::Fl) -> Fl\n\nCalculate the Bayesian Information Criterion (BIC) for a given number of observations, number of parameters, and log-likelihood value.\n\nArguments\n\nT::Int: Number of observations in the data.\nK::Int: Number of parameters in the model.\nloglikeVal::Fl: Log-likelihood value of the model.\n\nReturns\n\nThe BIC value calculated using the formula: BIC = log(T) * K - 2 * loglikeVal.\n\n\n\n\n\nbic(model::SarimaxModel; offset::Fl) -> Fl where Fl<:AbstractFloat\n\nCalculate the Bayesian Information Criterion (BIC) for a Sarimax model.\n\nArguments\n\nmodel::SarimaxModel: The Sarimax model for which BIC is calculated.\noffset::Fl=0.0: Offset value to be added to the BIC value.\n\nReturns\n\nThe BIC value calculated using the number of parameters, sample size, and log-likelihood value of the model.\n\nErrors\n\nThrows a MissingMethodImplementation if the getHyperparametersNumber method is not implemented for the given model type.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Time-Series-Operations","page":"API Reference","title":"Time Series Operations","text":"","category":"section"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Sarimax.differentiate\nSarimax.integrate\nSarimax.differentiatedCoefficients\nSarimax.toMA","category":"page"},{"location":"reference/#Main.Sarimax.differentiate","page":"API Reference","title":"Main.Sarimax.differentiate","text":"differentiate(\n    series::TimeArray,\n    d::Int=0,\n    D::Int=0,\n    s::Int=1\n)\n\nDifferentiates a TimeArray series d times and D times with a seasonal difference of s periods.\n\nArguments\n\nseries::TimeArray: The time series data to differentiate.\nd::Int=0: The number of non-seasonal differences to take.\nD::Int=0: The number of seasonal differences to take.\ns::Int=1: The seasonal period for the differences.\n\nReturns\n\nA differentiated TimeArray.\n\nErrors\n\nThis method only works with d and D in the set {0,1}.\n\nExample\n\njulia> airPassengers = loadDataset(AIR_PASSENGERS)\n\njulia> stationaryAirPassengers = differentiate(airPassengers, d=1, D=1, s=12)\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.integrate","page":"API Reference","title":"Main.Sarimax.integrate","text":"integrate(initialValues::Vector{Fl}, diffSeries::Vector{Fl}, d::Int, D::Int, s::Int) where Fl<:AbstractFloat\n\nConverts a differentiated time series back to its original scale.\n\nArguments\n\ninitialValues::Vector{Fl}: Initial values of the original time series.\ndiffSeries::Vector{Fl}: Differentiated time series.\nd::Int: Order of non-seasonal differencing.\nD::Int: Order of seasonal differencing.\ns::Int: Seasonal period.\n\nReturns\n\norigSeries::Vector{Fl}: Time series in the original scale.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.differentiatedCoefficients","page":"API Reference","title":"Main.Sarimax.differentiatedCoefficients","text":"differentiatedCoefficients(d::Int, D::Int, s::Int, Fl::DataType=Float64)\n\nCompute the coefficients for differentiating a time series.\n\nArguments\n\nd::Int: Order of non-seasonal differencing.\nD::Int: Order of seasonal differencing.\ns::Int: Seasonal period.\nFl: The type of the coefficients. Default is Float64.\n\nReturns\n\ncoeffs::Vector{AbstractFloat}: Coefficients for differentiation.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.toMA","page":"API Reference","title":"Main.Sarimax.toMA","text":"toMA(model::SARIMAModel, maxLags::Int=12)\n\nConvert a SARIMA model to a Moving Average (MA) model.\n\n# Arguments\n- `model::SARIMAModel`: The SARIMA model to convert.\n- `maxLags::Int=12`: The maximum number of lags to include in the MA model.\n\n# Returns\n- `MAmodel::MAModel`: The coefficients of the lagged errors in the MA model.\n\n# References\n- Brockwell, P. J., & Davis, R. A. Time Series: Theory and Methods (page 92). Springer(2009)\n\n\n\n\n\n","category":"function"},{"location":"reference/#Dataset-and-Utility-Functions","page":"API Reference","title":"Dataset and Utility Functions","text":"","category":"section"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Sarimax.loadDataset\nSarimax.splitTrainTest\nSarimax.identifyGranularity\nSarimax.buildDatetimes","category":"page"},{"location":"reference/#Main.Sarimax.loadDataset","page":"API Reference","title":"Main.Sarimax.loadDataset","text":"loadDataset(\n    dataset::Datasets\n)\n\nLoads a dataset from the Datasets enum.\n\nExample\n\njulia> airPassengers = loadDataset(AIR_PASSENGERS)\n204×1 TimeArray{Float64, 1, Date, Vector{Float64}} 1991-07-01 to 2008-06-01\n│            │ value   │\n├────────────┼─────────┤\n│ 1991-07-01 │ 3.5266  │\n│ 1991-08-01 │ 3.1809  │\n│ ⋮          │ ⋮       │\n│ 2008-06-01 │ 19.4317 │\n\n\n\n\n\n\nloadDataset(\n    df::DataFrame,\n    showLogs::Bool=false\n)\n\nLoads a dataset from a Dataframe. If the DataFrame does not have a column named date a new column will be created with the index of the DataFrame.\n\nArguments\n\ndf::DataFrame: The DataFrame to be converted to a TimeArray.\nshowLogs::Bool=false: If true, logs will be shown.\n\nExample\n\njulia> airPassengersDf = CSV.File(\"datasets/airpassengers.csv\") |> DataFrame\njulia> airPassengers = loadDataset(airPassengersDf)\n204×1 TimeArray{Float64, 1, Date, Vector{Float64}} 1991-07-01 to 2008-06-01\n│            │ value   │\n├────────────┼─────────┤\n│ 1991-07-01 │ 3.5266  │\n│ 1991-08-01 │ 3.1809  │\n│ ⋮          │ ⋮       │\n│ 2008-06-01 │ 19.4317 │\n\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.splitTrainTest","page":"API Reference","title":"Main.Sarimax.splitTrainTest","text":"splitTrainTest(\n    data::TimeArray;\n    trainSize::Fl=0.8\n) where Fl<:AbstractFloat\n\nSplits the time series in training and testing sets.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.identifyGranularity","page":"API Reference","title":"Main.Sarimax.identifyGranularity","text":"identifyGranularity(datetimes::Vector{T})\n\nIdentifies the granularity of an array of timestamps.\n\nArguments\n\ndatetimes::Vector{T}: An array of TimeType objects.\n\nReturns\n\nA tuple (granularity, frequency, weekdays) where:\n\ngranularity: The identified granularity, which could be one of [:Second, :Minute, :Hour, :Day, :Week, :Month, :Year].\nfrequency: The frequency of the identified granularity.\nweekdays: A boolean indicating whether the series uses weekdays only.\n\nErrors\n\nThrows an error if the timestamps do not follow a consistent pattern.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.buildDatetimes","page":"API Reference","title":"Main.Sarimax.buildDatetimes","text":"buildDatetimes(startDatetime, granularity, weekDaysOnly, datetimesLength)\n\nBuilds an array of DateTime objects based on a given starting DateTime, granularity, and length.\n\nArguments\n\nstartDatetime::T: The DateTime from which the dateTime array will be computed. It won't be included in the final array\ngranularity::Dates.Period: The granularity by which to increment the timestamps.\nweekDaysOnly::Bool: Whether to include only weekdays (Monday to Friday) in the timestamps.\ndatetimesLength::Int: The length of the array of DateTime objects to build.\n\nReturns\n\nAn array of DateTime objects.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Model-Information","page":"API Reference","title":"Model Information","text":"","category":"section"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Sarimax.hasFitMethods\nSarimax.hasHyperparametersMethods\nSarimax.getHyperparametersNumber","category":"page"},{"location":"reference/#Main.Sarimax.hasFitMethods","page":"API Reference","title":"Main.Sarimax.hasFitMethods","text":"hasFitMethods(modelType::Type{<:SarimaxModel}) -> Bool\n\nCheck if a given SarimaxModel type has the fit! method implemented.\n\nArguments\n\nmodelType::Type{<:SarimaxModel}: Type of the Sarimax model to check.\n\nReturns\n\nA boolean indicating whether the fit! method is implemented for the specified model type.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.hasHyperparametersMethods","page":"API Reference","title":"Main.Sarimax.hasHyperparametersMethods","text":"hasHyperparametersMethods(modelType::Type{<:SarimaxModel}) -> Bool\n\nChecks if a given SarimaxModel type has methods related to hyperparameters.\n\nArguments\n\nmodelType::Type{<:SarimaxModel}: Type of the Sarimax model to check.\n\nReturns\n\nA boolean indicating whether the hyperparameter-related methods are implemented for the specified model type.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Main.Sarimax.getHyperparametersNumber","page":"API Reference","title":"Main.Sarimax.getHyperparametersNumber","text":"getHyperparametersNumber(model::SARIMAModel)\n\nReturns the number of hyperparameters of a SARIMA model.\n\nArguments\n\nmodel::SARIMAModel: The SARIMA model.\n\nReturns\n\nInt: The number of hyperparameters.\n\n\n\n\n\n","category":"function"},{"location":"reference/#Model-Manipulation","page":"API Reference","title":"Model Manipulation","text":"","category":"section"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Sarimax.print","category":"page"},{"location":"reference/#Base.print","page":"API Reference","title":"Base.print","text":"print(model::SARIMAModel)\n\nPrints the SARIMA model.\n\nArguments\n\nmodel::SARIMAModel: The SARIMA model to print.\n\nExample\n\njulia> model = SARIMA(1, 0, 1; P=1, D=0, Q=1, seasonality=12, allowMean=true, allowDrift=false)\n\njulia> print(model)\n\n\n\n\n\n","category":"function"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Sarimax.copyTimeArray(y::TimeSeries.TimeArray)\nSarimax.deepcopyTimeArray(y::TimeSeries.TimeArray)","category":"page"},{"location":"reference/#Main.Sarimax.copyTimeArray-Tuple{TimeArray}","page":"API Reference","title":"Main.Sarimax.copyTimeArray","text":"copyTimeArray(y::TimeSeries.TimeArray)\n\nCreate a shallow copy of a TimeArray object.\n\nThis function creates a new TimeArray with copies of the timestamp and values from the original TimeArray. The new TimeArray is independent of the original, but the elements themselves are not deeply copied.\n\nArguments\n\ny::TimeSeries.TimeArray: The TimeArray to copy.\n\nReturns\n\nTimeSeries.TimeArray: A new TimeArray with copies of the timestamp and values.\n\nExamples\n\noriginal = TimeArray(Date.(2021:2023), [1, 2, 3])\ncopied = copyTimeArray(original)\n\n\n\n\n\n","category":"method"},{"location":"reference/#Main.Sarimax.deepcopyTimeArray-Tuple{TimeArray}","page":"API Reference","title":"Main.Sarimax.deepcopyTimeArray","text":"deepcopyTimeArray(y::TimeSeries.TimeArray)\n\nCreate a deep copy of a TimeArray object.\n\nThis function creates a new TimeArray with deep copies of the timestamp and values from the original TimeArray. The new TimeArray and all its elements are completely independent of the original.\n\nArguments\n\ny::TimeSeries.TimeArray: The TimeArray to deep copy.\n\nReturns\n\nTimeSeries.TimeArray: A new TimeArray with deep copies of the timestamp and values.\n\nExamples\n\noriginal = TimeArray(Date.(2021:2023), [1, 2, 3])\ndeepCopied = deepcopyTimeArray(original)\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"<div style=\"width:100%; height:150px;border-width:4px;border-style:solid;padding-top:25px;\n        border-color:#000;border-radius:10px;text-align:center;background-color:#99DDFF;\n        color:#000\">\n    <h3 style=\"color: black;\">Star us on GitHub!</h3>\n    <a class=\"github-button\" href=\"https://github.com/LAMPSPUC/Sarimax.jl\" data-icon=\"octicon-star\" data-size=\"large\" data-show-count=\"true\" aria-label=\"Star LAMPSPUC/Sarimax.jl on GitHub\" style=\"margin:auto\">Star</a>\n    <script async defer src=\"https://buttons.github.io/buttons.js\"></script>\n</div>","category":"page"},{"location":"#Sarimax.jl-Documentation","page":"Home","title":"Sarimax.jl Documentation","text":"","category":"section"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Sarimax.jl is a groundbreaking Julia package that revolutionizes SARIMA (Seasonal Autoregressive Integrated Moving Average) modeling by seamlessly integrating with the JuMP framework — a powerful optimization modeling language. Unlike traditional SARIMA implementations, Sarimax.jl leverages JuMP's optimization capabilities to provide precise and highly customizable SARIMA models.","category":"page"},{"location":"#Key-Features","page":"Home","title":"Key Features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Fit models using various objective functions:\nMean Squared Errors\nMaximum Likelihood estimation\nBilevel objective function\nAuto SARIMA model selection\nSupport for exogenous variables (Sarimax)\nScenario simulation capabilities\nTime series integration and differentiation\nModel evaluation criteria (AIC, AICc, BIC)","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Sarimax.jl can be installed using Julia's built-in package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add Sarimax","category":"page"},{"location":"","page":"Home","title":"Home","text":"Or, you can install it by using Pkg directly:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(\"Sarimax\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"To use the development version, you can install directly from the GitHub repository:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pkg.add(url = \"https://github.com/LAMPSPUC/Sarimax.jl.git\")","category":"page"},{"location":"#Quick-Start","page":"Home","title":"Quick Start","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To start using Sarimax.jl, simply import the package:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Sarimax","category":"page"},{"location":"","page":"Home","title":"Home","text":"Check out our Tutorial section for detailed examples of how to use the package.","category":"page"},{"location":"#License","page":"Home","title":"License","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Sarimax.jl is licensed under the MIT License. This means you are free to use, modify, and distribute the code, subject to the terms and conditions of the MIT license.","category":"page"},{"location":"#Contributing","page":"Home","title":"Contributing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Contributions are welcome! If you find a bug or have a feature request, please open an issue on the GitHub repository. Pull requests for bug fixes and new features are also appreciated.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For more detailed information about the package functionality, please refer to the following sections:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pages = [\n    \"tutorial.md\",\n    \"api.md\",\n    \"examples.md\"\n]\nDepth = 2","category":"page"},{"location":"tutorial/#Tutorial-{#tutorial}","page":"Tutorial","title":"Tutorial {#tutorial}","text":"","category":"section"},{"location":"tutorial/#","page":"Tutorial","title":"","text":"","category":"section"}]
}
