# Worker process of the M4 campaign runner (scripts/ijflib/campaign.py): fits one arm of an
# experiment configuration, one series at a time.
#
# Protocol (JSON lines):
#   stdin  <- {"series": 17, "y": [...], "s": 12, "H": 18}
#   stdout -> {"series": 17, "status": "OK", "forecast": [...], "p": ..., ...}
# A line {"ready": true, ...} is printed once the package is loaded and warmed up. Anything
# else the package prints goes to stderr, so stdout carries the protocol only.
#
# Every keyword argument of `Sarimax.auto` that affects estimation comes from the
# configuration file ([auto_defaults] and [arms.auto] tables) and is passed explicitly: the package's defaults
# have moved between releases, so a run that inherits a default does not denote a fixed
# computation. One BLAS thread per worker: parallelism is between processes.
#
# Usage: julia --project=<env> m4_worker.jl <config.toml> <arm name>

using Sarimax
using DataFrames, JSON, LinearAlgebra, TOML, TimeSeries

BLAS.set_num_threads(1)
const PROTOCOL = stdout
redirect_stdout(stderr)          # package output must not corrupt the protocol stream

const SYMBOL_KEYS = Set(["initialization", "seasonalForm", "exogDynamics", "penaltyTarget", "penalty"])

function armKeywords(config, armName)
    arms = [a for a in config["arms"] if a["name"] == armName]
    length(arms) == 1 || error("arm $armName not found (or not unique) in the configuration")
    arm = arms[1]
    arm["engine"] == "julia" || error("arm $armName is not a Julia arm")
    kw = Dict{Symbol,Any}()
    # [auto_defaults] holds the keyword arguments shared by every arm; an arm's [arms.auto]
    # overrides them. Both are written out in full in the configuration file.
    for (k, v) in merge(get(config, "auto_defaults", Dict{String,Any}()), get(arm, "auto", Dict{String,Any}()))
        kw[Symbol(k)] = v == "nothing" ? nothing :
                        k in SYMBOL_KEYS ? Symbol(v) :
                        k == "optimizer" ? (v == "Ipopt" ? Sarimax.Ipopt.Optimizer : error("optimizer $v")) : v
    end
    haskey(kw, :seasonality) && error("seasonality is set per frequency by the runner, not in [arms.auto]")
    return kw
end

function fitOne(y, s, H, kw)
    t0 = time()
    # The series enters exactly as in the archived runners: `load_dataset(DataFrame(y = y))`,
    # which indexes it by Date(1), Date(2), ... (the timestamps do not enter the estimation).
    m = Sarimax.auto(Sarimax.load_dataset(DataFrame(y = y)); seasonality = s, kw...)
    Sarimax.predict!(m; stepsAhead = H)
    f = Float64.(values(m.forecast)[:, 1])
    return Dict("status" => "OK", "forecast" => f,
                "p" => m.p, "d" => m.d, "q" => m.q, "P" => m.P, "D" => m.D, "Q" => m.Q,
                "has_mean" => isnothing(m.c) ? 0 : Int(abs(m.c) > 1e-10),
                "has_drift" => isnothing(m.trend) ? 0 : Int(abs(m.trend) > 1e-10),
                "solver_status" => string(get(m.metadata, "solverStatus", "")),
                "fallback" => haskey(m.metadata, "huberFallback") ? Int(m.metadata["huberFallback"]) : -1,
                "fit_seconds" => round(time() - t0, digits = 3))
end

function main()
    config = TOML.parsefile(ARGS[1])
    kw = armKeywords(config, ARGS[2])
    # Warm-up on a fixed toy series so that compilation time is not charged to the first
    # series of the run (it is reported separately).
    tw = @elapsed try
        fitOne(collect(1.0:30.0) .+ sin.(1:30), 1, 3, kw)
    catch
    end
    println(PROTOCOL, JSON.json(Dict("ready" => true, "warmup_seconds" => round(tw, digits = 2),
                                     "julia" => string(VERSION), "sarimax" => string(pkgversion(Sarimax)))))
    flush(PROTOCOL)
    for line in eachline(stdin)
        isempty(strip(line)) && continue
        job = JSON.parse(line)
        out = try
            fitOne(Float64.(job["y"]), Int(job["s"]), Int(job["H"]), kw)
        catch err
            Dict("status" => "ERROR:" * first(replace(sprint(showerror, err), '\n' => ' '), 300))
        end
        out["series"] = job["series"]
        println(PROTOCOL, JSON.json(out))
        flush(PROTOCOL)
    end
end

main()
