var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"<!-- Ensure that raw HTML is properly formatted -->\n<div style=\"width:100%; height:150px;border-width:4px;border-style:solid;padding-top:25px;\n    border-color:#000;border-radius:10px;text-align:center;background-color:#99DDFF;\n    color:#000\">\n    <h3 style=\"color: black;\">Star us on GitHub!</h3>\n    <a class=\"github-button\" href=\"https://github.com/LAMPSPUC/SARIMAX.jl\" data-icon=\"octicon-star\" data-size=\"large\" data-show-count=\"true\" aria-label=\"Star LAMPSPUC/SARIMAX.jl on GitHub\" style=\"margin:auto\">Star</a>\n    <script async defer src=\"https://buttons.github.io/buttons.js\"></script>\n</div>","category":"page"},{"location":"#Sarimax.jl","page":"Home","title":"Sarimax.jl","text":"","category":"section"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package is registered so you can simply add it using Julia's Pkg manager:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(\"SARIMAX\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"Auto SARIMA implementation","category":"page"},{"location":"","page":"Home","title":"Home","text":"Sarimax.auto","category":"page"},{"location":"#Main.Sarimax.auto","page":"Home","title":"Main.Sarimax.auto","text":"auto(\n    y::TimeArray;\n    exog::Union{TimeArray,Nothing}=nothing,\n    seasonality::Int=1,\n    d::Int = -1,\n    D::Int = -1,\n    maxp::Int = 5,\n    maxd::Int = 2,\n    maxq::Int = 5,\n    maxP::Int = 2,\n    maxD::Int = 1,\n    maxQ::Int = 2,\n    maxOrder::Int = 5,\n    informationCriteria::String = \"aicc\",\n    allowMean:Union{Bool,Nothing} = nothing,\n    allowDrift::Union{Bool,Nothing} = nothing,\n    integrationTest::String = \"kpss\",\n    seasonalIntegrationTest::String = \"seas\",\n    objectiveFunction::String = \"mse\",\n    assertStationarity::Bool = true,\n    assertInvertibility::Bool = true,\n    showLogs::Bool = false,\n    outlierDetection::Bool = false\n    searchMethod::String = \"stepwise\"\n)\n\nAutomatically fits the best SARIMA model according to the specified parameters.\n\nArguments\n\ny::TimeArray: The time series data.\nexog::Union{TimeArray,Nothing}: Optional exogenous variables. If Nothing, no exogenous variables are used.\nseasonality::Int: The seasonality period. Default is 1 (non-seasonal).\nd::Int: The degree of differencing for the non-seasonal part. Default is -1 (auto-select).\nD::Int: The degree of differencing for the seasonal part. Default is -1 (auto-select).\nmaxp::Int: The maximum autoregressive order for the non-seasonal part. Default is 5.\nmaxd::Int: The maximum integration order for the non-seasonal part. Default is 2.\nmaxq::Int: The maximum moving average order for the non-seasonal part. Default is 5.\nmaxP::Int: The maximum autoregressive order for the seasonal part. Default is 2.\nmaxD::Int: The maximum integration order for the seasonal part. Default is 1.\nmaxQ::Int: The maximum moving average order for the seasonal part. Default is 2.\nmaxOrder::Int: The maximum order for the non-seasonal part. Default is 5.\ninformationCriteria::String: The information criteria to be used for model selection. Options are \"aic\", \"aicc\", or \"bic\". Default is \"aicc\".\nallowMean::Union{Bool,Nothing}: Whether to include a mean term in the model. Default is nothing.\nallowDrift::Union{Bool,Nothing}: Whether to include a drift term in the model. Default is nothing.\nintegrationTest::String: The integration test to be used for determining the non-seasonal integration order. Default is \"kpss\".\nseasonalIntegrationTest::String: The integration test to be used for determining the seasonal integration order. Default is \"seas\".\nobjectiveFunction::String: The objective function to be used for model selection. Options are \"mse\", \"ml\", or \"bilevel\". Default is \"mse\".\nassertStationarity::Bool: Whether to assert stationarity of the fitted model. Default is true.\nassertInvertibility::Bool: Whether to assert invertibility of the fitted model. Default is true.\nshowLogs::Bool: Whether to suppress output. Default is false.\noutlierDetection::Bool: Whether to perform outlier detection. Default is false.\nsearchMethod::String = \"stepwise\"\n\nReferences\n\nHyndman, RJ and Khandakar. \"Automatic time series forecasting: The forecast package for R.\" Journal of Statistical Software, 26(3), 2008.\n\n\n\n\n\n","category":"function"}]
}
