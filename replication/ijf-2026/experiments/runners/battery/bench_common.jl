# Shared helpers for the ported July-2026 battery runners.
# JSON Lines raw records; no fabricated values; failures recorded as status="failed".
#
# Port of original-scripts/battery-2026-07/scripts/bench_common.jl (archived battery, run on
# Sarimax v0.3.0 = 3f68017) to the release API (v1.0.0 = 14dd581). Adaptations:
#   * RAW is $IJF_RUN_DIR/raw/battery-2026-07 (the original used <script dir>/../results/raw).
#     The runners write ONLY below it; `outfile` creates the block directory.
#   * `append_record` writes non-finite numbers as JSON null. The original passed NaN/Inf to
#     JSON.json, which throws: that is how the archived SCIP scaling run lost a record and had
#     to be re-invoked (artifacts/raw/battery-2026-07/solver/scip_scaling_run.log).
#   * `append_record` adds the field "sarimax_version" (provenance; no table reads it).
#   * New: V030_INITIALIZATION / V030_STATIONARY, the two `fit!` defaults of v0.3.0 that the
#     release changed (initialization :zeroed -> :innovations, stationary false -> true). The
#     archived scripts relied on the defaults; the ports pass these values explicitly so that
#     the same estimator is requested.
#   * New: `css_criteria` (AIC/BIC under the v0.3.0 definition, recorded next to the release's
#     values) and `moi_diagnostics` (what a solver reports through MOI: statuses, objective,
#     dual bound, relative gap, version), used for the SCIP records.
#   * extract_estimates, trymetric, as_monthly, the accuracy metrics, seasonal_naive and
#     rolling_origins are unchanged.

using Sarimax
using TimeSeries, DataFrames, Dates, Random, Statistics, JSON
import JuMP     # qualified only (JuMP.MOI.get ...): scripts that need JuMP's names `using` it

const RUN_DIR = let d = get(ENV, "IJF_RUN_DIR", "")
    isempty(d) && error("IJF_RUN_DIR is not set. The battery runners write their records to " *
                        "\$IJF_RUN_DIR/raw/battery-2026-07/<block>/ and nowhere else; run them " *
                        "through `./reproduce.sh full --experiment battery_<block>` or set it.")
    abspath(d)
end
const RAW = joinpath(RUN_DIR, "raw", "battery-2026-07")

# fit! defaults of Sarimax v0.3.0 (3f68017) that the release changed. See the header.
const V030_INITIALIZATION = :zeroed
const V030_STATIONARY = false

"Path of a record file of `block` in this run (the directory is created)."
function outfile(block::AbstractString, name::AbstractString)
    d = joinpath(RAW, block)
    mkpath(d)
    return joinpath(d, name)
end

"Path of an input data file regenerated in this run; fails clearly when the step that makes it did not run."
function datafile(block::AbstractString, name::AbstractString; producer::AbstractString)
    p = joinpath(RAW, block, "data", name)
    isfile(p) || error("$(p) not found. It is generated in this run directory by $(producer); " *
                       "run that step first (it is the first step of the experiment's configuration).")
    return p
end

"Replace non-finite floats by `nothing` (JSON null), recursively."
jsonsafe(x::AbstractFloat) = isfinite(x) ? x : nothing
jsonsafe(x::AbstractDict) = Dict{String,Any}(string(k) => jsonsafe(v) for (k, v) in x)
jsonsafe(x::AbstractVector) = Any[jsonsafe(v) for v in x]
jsonsafe(x::Tuple) = Any[jsonsafe(v) for v in x]
jsonsafe(x::Symbol) = String(x)
jsonsafe(x) = x

"Append one run record (a Dict) as a JSON line."
function append_record(path::AbstractString, rec::AbstractDict)
    r = jsonsafe(rec)
    r["sarimax_version"] = string(pkgversion(Sarimax))
    open(path, "a") do io
        println(io, JSON.json(r))
    end
end

"Safely extract fitted coefficients from a SARIMAModel into a plain Dict."
function extract_estimates(m)
    est = Dict{String,Any}()
    for (name, f) in (("phi", :ϕ), ("theta", :θ), ("Phi", :Φ), ("Theta", :Θ),
                      ("exog", :exogCoefficients))
        v = getfield(m, f)
        est[name] = v === nothing ? nothing : collect(v)
    end
    est["mean"] = m.c
    est["trend"] = m.trend
    est["sigma2"] = m.σ²
    return est
end

"Try metric functions, returning nothing on failure (definitions may not align)."
trymetric(f, m) = try f(m) catch; nothing end

"""
AIC and BIC as Sarimax v0.3.0 defined them: the conditional (CSS) log-likelihood `loglike`,
`K = get_hyperparameters_number`, `n = length(m.ϵ)`. The release's `aic`/`bic`/`aicc` score the
EXACT Gaussian likelihood when it is computable (falling back to CSS), so their values differ
from the archived ones by definition; these are recorded alongside for comparison.
"""
function css_criteria(m)
    ll = Sarimax.loglike(m)
    K = Sarimax.get_hyperparameters_number(m)
    n = length(m.ϵ)
    return Dict{String,Any}("aic_css" => 2K - 2ll, "bic_css" => K * log(n) - 2ll,
                            "aicc_css" => 2K - 2ll + (2K^2 + 2K) / (n - K - 1), "n_css" => n, "K" => K)
end

"Solver status recorded by the release's fit! (a string; empty when absent)."
fit_status(m) = string(get(m.metadata, "solverStatus", ""))

"`statuses` summarised as one string: the status when all agree, else `A x n, B x m`."
function status_summary(statuses)
    u = unique(string.(statuses))
    length(u) == 1 && return u[1]
    return join(["$(s) x $(count(==(s), string.(statuses)))" for s in u], ", ")
end

"""
    moi_diagnostics(o) -> Dict

What the solver behind `o` (a JuMP backend or a raw MOI optimizer) reports after a solve:
termination and raw status, primal status, result count, objective value, dual bound
(`objective_bound`), relative gap, solver time, node count, solver name and version. Each
attribute is read on its own and recorded as `nothing` when the solver cannot supply it;
nothing is inferred or hard-coded.
"""
function moi_diagnostics(o)
    get_or_nothing(attr) = try
        v = JuMP.MOI.get(o, attr)
        v isa Enum ? string(v) : v
    catch
        nothing
    end
    nres = get_or_nothing(JuMP.MOI.ResultCount())
    return Dict{String,Any}(
        "termination_status" => get_or_nothing(JuMP.MOI.TerminationStatus()),
        "raw_status" => get_or_nothing(JuMP.MOI.RawStatusString()),
        "primal_status" => get_or_nothing(JuMP.MOI.PrimalStatus()),
        "result_count" => nres,
        "solver_objective_value" => (nres isa Integer && nres > 0) ? get_or_nothing(JuMP.MOI.ObjectiveValue()) : nothing,
        "objective_bound" => get_or_nothing(JuMP.MOI.ObjectiveBound()),
        "relative_gap" => get_or_nothing(JuMP.MOI.RelativeGap()),
        "solver_time_s" => get_or_nothing(JuMP.MOI.SolveTimeSec()),
        "node_count" => get_or_nothing(JuMP.MOI.NodeCount()),
        "solver_name" => get_or_nothing(JuMP.MOI.SolverName()),
        "solver_version" => get_or_nothing(JuMP.MOI.SolverVersion()),
    )
end
moi_diagnostics(m::JuMP.Model) = moi_diagnostics(JuMP.backend(m))

"Build a TimeArray from a numeric vector with a monthly date index (for simulation)."
function as_monthly(values::AbstractVector; start::Date = Date(2000, 1, 1))
    ts = [start + Month(i - 1) for i in 1:length(values)]
    return TimeArray(ts, collect(float.(values)), ["value"])
end

# --- Forecast accuracy metrics (actual vs forecast over horizon) ---
mae(a, f) = mean(abs.(a .- f))
rmse(a, f) = sqrt(mean((a .- f) .^ 2))
smape(a, f) = mean(2 .* abs.(a .- f) ./ (abs.(a) .+ abs.(f) .+ eps())) * 100
"MASE scaled by in-sample seasonal-naive MAE of the training series."
function mase(a, f, train, m::Int)
    denom = m < length(train) ? mean(abs.(train[(m+1):end] .- train[1:(end-m)])) : NaN
    return (denom == 0 || isnan(denom)) ? NaN : mean(abs.(a .- f)) / denom
end
"Seasonal-naive forecast: repeat last season of training over the horizon."
function seasonal_naive(train::AbstractVector, h::Int, m::Int)
    m = max(m, 1)
    return [train[end - m + 1 + ((i - 1) % m)] for i in 1:h]
end

"Expanding-window rolling origins: train-end indices k with k+H <= n."
function rolling_origins(n::Int; init_frac::Float64 = 0.7, H::Int = 12, step::Int = 12)
    k0 = floor(Int, init_frac * n)
    return [k for k in k0:step:(n - H)]
end
