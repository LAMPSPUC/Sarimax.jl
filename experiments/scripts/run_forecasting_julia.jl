# Block 2 - Rolling-origin forecasting (SARIMAX.jl side).
# Expanding-window rolling origins on AirPassengers (monthly, s=12) and GDPC1 (quarterly).
# Refits the model at each origin; reports mean MAE/RMSE/sMAPE/MASE + #origins + #failures.
include(joinpath(@__DIR__, "bench_common.jl"))

const OUT = joinpath(RAW, "forecasting", "julia_results.jsonl")
isfile(OUT) && rm(OUT)

# (dataset id, TimeArray, seasonality m, horizon H, step, init_frac, model spec)
airp = loadDataset(AIR_PASSENGERS)
gdp  = loadDataset(GDPC1)

emit(rec) = (append_record(OUT, rec); println(rec["implementation"], " / ",
             rec["dataset"], " -> ", rec["status"], " (origins=", get(rec, "n_origins", 0),
             " fail=", get(rec, "n_failures", 0), ")"))

function base_rec(impl, dataset, order, H, m, n_origins)
    Dict{String,Any}("block"=>"forecasting", "implementation"=>impl,
        "dataset"=>dataset, "order"=>order, "horizon"=>H, "seasonality"=>m,
        "protocol"=>"rolling-origin", "n_origins"=>n_origins, "seed"=>1234)
end

function aggregate!(rec, maes, rmses, smapes, mases, rts, nfail)
    rec["n_failures"] = nfail
    if isempty(maes)
        rec["status"] = "failed"
    else
        rec["status"] = "ok"
        rec["mae"] = mean(maes); rec["rmse"] = mean(rmses)
        rec["smape"] = mean(smapes); rec["mase"] = mean(filter(!isnan, mases))
        rec["runtime_s"] = sum(rts)
    end
    rec
end

function run_dataset(id, ta; m, H, step, init_frac, order, fitfun)
    yv = vec(values(ta))
    n = length(yv)
    origins = rolling_origins(n; init_frac=init_frac, H=H, step=step)

    # JIT warm-up (UNTIMED): compile the fit!/predict! code paths once, so the reported runtime
    # reflects steady-state per-fit cost rather than Julia's one-time time-to-first-execution.
    jit = 0.0
    try
        jit = @elapsed begin
            wm = fitfun(ta[1:origins[1]]); predict!(wm; stepsAhead = H)
        end
    catch
    end

    # SARIMAX.jl model (timed, warm)
    maes=Float64[]; rmses=Float64[]; smapes=Float64[]; mases=Float64[]; rts=Float64[]; nfail=0
    for k in origins
        train = ta[1:k]; trainv = yv[1:k]; act = yv[(k+1):(k+H)]
        try
            t = @elapsed begin
                mdl = fitfun(train)
                predict!(mdl; stepsAhead = H)
            end
            fc = vec(values(mdl.forecast))[1:H]
            push!(maes, mae(act, fc)); push!(rmses, rmse(act, fc))
            push!(smapes, smape(act, fc)); push!(mases, mase(act, fc, trainv, m))
            push!(rts, t)
        catch
            nfail += 1
        end
    end
    rec = aggregate!(base_rec("SARIMAX.jl", id, order, H, m, length(origins)),
                     maes, rmses, smapes, mases, rts, nfail)
    rec["jit_warmup_s"] = jit
    rec["runtime_note"] = "warm per-fit (one-time JIT warm-up of $(round(jit,digits=2)) s excluded)"
    emit(rec)

    # Seasonal-naive baseline (m=1 reduces to random-walk naive)
    smaes=Float64[]; srmses=Float64[]; sps=Float64[]; sms=Float64[]
    for k in origins
        trainv = yv[1:k]; act = yv[(k+1):(k+H)]
        fc = seasonal_naive(trainv, H, m)
        push!(smaes, mae(act, fc)); push!(srmses, rmse(act, fc))
        push!(sps, smape(act, fc)); push!(sms, mase(act, fc, trainv, m))
    end
    snrec = aggregate!(base_rec(m > 1 ? "seasonal-naive" : "naive", id,
                                m > 1 ? "snaive" : "naive", H, m, length(origins)),
                       smaes, srmses, sps, sms, [0.0], 0)
    delete!(snrec, "runtime_s")
    emit(snrec)
end

run_dataset("airpassengers", airp; m=12, H=12, step=12, init_frac=0.7,
            order="(1,0,1)(1,0,1)_12",
            fitfun = tr -> (mm = SARIMA(tr, 1, 0, 1; seasonality=12, P=1, D=0, Q=1);
                            fit!(mm; objectiveFunction="mse"); mm))

run_dataset("GDPC1", gdp; m=1, H=8, step=20, init_frac=0.7,
            order="(1,1,1) drift",
            fitfun = tr -> (mm = SARIMA(tr, 1, 1, 1; allowDrift=true);
                            fit!(mm; objectiveFunction="mse"); mm))

println("forecasting_julia (rolling-origin) DONE -> ", OUT)
