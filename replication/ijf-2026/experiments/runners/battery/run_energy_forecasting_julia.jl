# Energy dataset (PJME daily) - rolling-origin forecasting, SARIMAX.jl side.
# Same expanding-window protocol as the other datasets. Daily aggregation -> weekly seasonality.
# A bounded recent window keeps the optimization-based fits (one residual variable per obs) tractable.
#
# Port of original-scripts/battery-2026-07/scripts/run_energy_forecasting_julia.jl to the
# release API. Window (540 days), origins (init_frac 0.7, step 28), horizon 14, order
# SARIMA(1,1,1)(1,0,1)_7, objective "mse", untimed JIT warm-up, metrics and record layout
# unchanged. Adaptations:
#   * fit! receives initialization = :zeroed and stationary = false (the v0.3.0 defaults the
#     archived script relied on; the release defaults are :innovations and true).
#   * input: pjme_daily.csv derived in this run by setup/gen_energy_data.py from
#     $IJF_PJME_DIR/PJME_hourly.csv; output
#     $IJF_RUN_DIR/raw/battery-2026-07/energy/julia_results.jsonl; load_dataset instead of
#     loadDataset.
#   * Extra field: "solver_status_counts" (the release's metadata["solverStatus"] per origin).
#   * Two top-level assignments are marked `global`. In the original, `jit = @elapsed ...`
#     inside a top-level `try` created a new local (Julia's soft-scope rule for files), so the
#     warm-up RAN but the archived record says jit_warmup_s = 0.0; the port records the measured
#     time (the timed per-fit runtimes are unaffected). `nfail += 1` in the `catch` of the
#     top-level loop would have thrown UndefVarError on the first failed origin; it never fired
#     in the archived run (n_failures = 0).
#   * Release-internal change no keyword can undo: fit! solves on the differenced series
#     divided by its standard deviation (here ~1e3 MW) and maps the estimates back.
include(joinpath(@__DIR__, "bench_common.jl"))
using CSV

const OUT = outfile("energy", "julia_results.jsonl")
isfile(OUT) && rm(OUT)

daily_full = load_dataset(CSV.read(datafile("energy", "pjme_daily.csv";
                                            producer = "experiments/runners/battery/setup/gen_energy_data.py (needs ./reproduce.sh fetch pjme)"),
                                   DataFrame))
WINDOW = 540
ta = daily_full[end-WINDOW+1:end]
yv = vec(values(ta))
n = length(yv)
m = 7; H = 14; step = 28; init_frac = 0.7
origins = rolling_origins(n; init_frac=init_frac, H=H, step=step)

emit(rec) = (append_record(OUT, rec); println(rec["implementation"], " / ", rec["dataset"],
             " -> ", rec["status"], " (origins=", get(rec,"n_origins",0), " fail=", get(rec,"n_failures",0), ")"))

base_rec(impl, order) = Dict{String,Any}("block"=>"energy_forecasting", "implementation"=>impl,
    "dataset"=>"PJME_daily", "order"=>order, "horizon"=>H, "seasonality"=>m,
    "protocol"=>"rolling-origin", "n_origins"=>length(origins), "window"=>WINDOW, "seed"=>1234)

function aggregate!(rec, maes, rmses, smapes, mases, rts, nfail)
    rec["n_failures"] = nfail
    if isempty(maes); rec["status"]="failed"; return rec; end
    rec["status"]="ok"; rec["mae"]=mean(maes); rec["rmse"]=mean(rmses)
    rec["smape"]=mean(smapes); rec["mase"]=mean(filter(!isnan, mases)); rec["runtime_s"]=sum(rts)
    rec
end

# JIT warm-up (UNTIMED): compile fit!/predict! so the reported runtime is steady-state per-fit,
# not Julia's one-time time-to-first-execution.
mkmodel(train) = (mm = SARIMA(train, 1, 1, 1; seasonality=7, P=1, D=0, Q=1);
                  fit!(mm; objectiveFunction="mse", initialization=V030_INITIALIZATION,
                       stationary=V030_STATIONARY); mm)
jit = 0.0
try
    global jit = @elapsed (wm = mkmodel(ta[1:origins[1]]); predict!(wm; stepsAhead=H))
catch
end

# SARIMAX.jl SARIMA(1,1,1)(1,0,1)_7 (timed, warm)
maes=Float64[]; rmses=Float64[]; smapes=Float64[]; mases=Float64[]; rts=Float64[]; nfail=0
statuses = String[]
for k in origins
    train = ta[1:k]; trainv = yv[1:k]; act = yv[(k+1):(k+H)]
    try
        t = @elapsed begin
            mdl = mkmodel(train)
            predict!(mdl; stepsAhead=H)
        end
        push!(statuses, fit_status(mdl))
        fc = vec(values(mdl.forecast))[1:H]
        push!(maes, mae(act,fc)); push!(rmses, rmse(act,fc))
        push!(smapes, smape(act,fc)); push!(mases, mase(act,fc,trainv,m)); push!(rts,t)
    catch; global nfail += 1; end
end
let r = aggregate!(base_rec("SARIMAX.jl", "(1,1,1)(1,0,1)_7"), maes,rmses,smapes,mases,rts,nfail)
    r["jit_warmup_s"] = jit
    r["runtime_note"] = "warm per-fit (one-time JIT warm-up of $(round(jit,digits=2)) s excluded)"
    r["solver_status_counts"] = Dict(s => count(==(s), statuses) for s in unique(statuses))
    emit(r)
end

# Seasonal-naive baseline (weekly)
smaes=Float64[]; srmses=Float64[]; sps=Float64[]; sms=Float64[]
for k in origins
    trainv = yv[1:k]; act = yv[(k+1):(k+H)]
    fc = seasonal_naive(trainv, H, m)
    push!(smaes, mae(act,fc)); push!(srmses, rmse(act,fc)); push!(sps, smape(act,fc)); push!(sms, mase(act,fc,trainv,m))
end
snrec = aggregate!(base_rec("seasonal-naive","snaive_s7"), smaes,srmses,sps,sms,[0.0],0)
delete!(snrec, "runtime_s"); emit(snrec)

println("energy_forecasting_julia DONE -> ", OUT)
