#!/usr/bin/env python
"""Energy (PJME daily) rolling-origin forecasting - statsmodels baseline.
Same window/origins/order as the SARIMAX.jl energy script.

Port of original-scripts/battery-2026-07/scripts/run_energy_forecasting_python.py. Window,
origins, order, statsmodels options, metrics and record layout unchanged. Adaptations:
  * input: pjme_daily.csv derived in this run by setup/gen_energy_data.py from
    $IJF_PJME_DIR/PJME_hourly.csv ($IJF_RUN_DIR/raw/battery-2026-07/energy/data/);
  * output: $IJF_RUN_DIR/raw/battery-2026-07/energy/python_statsmodels_results.jsonl.
"""
import os, json, time, warnings
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if not os.environ.get("IJF_RUN_DIR"):
    raise SystemExit("IJF_RUN_DIR is not set: the battery runners write only below "
                     "$IJF_RUN_DIR/raw/battery-2026-07/")
RAW = os.path.join(os.path.abspath(os.environ["IJF_RUN_DIR"]), "raw", "battery-2026-07")
OUT = os.path.join(RAW, "energy", "python_statsmodels_results.jsonl")
SRC = os.path.join(RAW, "energy", "data", "pjme_daily.csv")
if not os.path.isfile(SRC):
    raise SystemExit(f"{SRC} not found: it is derived in this run by "
                     "experiments/runners/battery/setup/gen_energy_data.py, which needs "
                     "./reproduce.sh fetch pjme")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
if os.path.exists(OUT):
    os.remove(OUT)

WINDOW, M, H, STEP, INIT = 540, 7, 14, 28, 0.7


def rolling_origins(n, init_frac, H, step):
    return list(range(int(np.floor(init_frac * n)), n - H + 1, step))


def metrics(act, fc, train, m):
    act, fc = np.asarray(act), np.asarray(fc)
    mae = float(np.mean(np.abs(act - fc)))
    rmse = float(np.sqrt(np.mean((act - fc) ** 2)))
    smape = float(np.mean(2 * np.abs(act - fc) / (np.abs(act) + np.abs(fc) + 1e-12)) * 100)
    denom = np.mean(np.abs(train[m:] - train[:-m])) if len(train) > m else np.nan
    mase = float(mae / denom) if denom and not np.isnan(denom) else np.nan
    return mae, rmse, smape, mase


y = pd.read_csv(SRC)["value"].to_numpy()[-WINDOW:]
n = len(y)
origins = rolling_origins(n, INIT, H, STEP)
maes, rmses, smapes, mases, rts, nfail = [], [], [], [], [], 0
for k in origins:
    train, act = y[:k], y[k:k + H]
    try:
        t = time.time()
        res = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), trend="n",
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = np.asarray(res.forecast(steps=H))
        a, r, s, ms = metrics(act, fc, train, M)
        maes.append(a); rmses.append(r); smapes.append(s); mases.append(ms); rts.append(time.time() - t)
    except Exception:  # noqa
        nfail += 1
rec = dict(block="energy_forecasting", implementation="statsmodels", dataset="PJME_daily",
           order="(1,1,1)(1,0,1)_7", horizon=H, seasonality=M, protocol="rolling-origin",
           n_origins=len(origins), window=WINDOW, n_failures=nfail, seed=1234)
if maes:
    rec.update(status="ok", mae=float(np.mean(maes)), rmse=float(np.mean(rmses)),
               smape=float(np.mean(smapes)), mase=float(np.nanmean(mases)), runtime_s=float(np.sum(rts)))
else:
    rec["status"] = "failed"
with open(OUT, "a") as f:
    f.write(json.dumps(rec) + "\n")
print("statsmodels PJME_daily ->", rec["status"], "origins", len(origins), "fail", nfail)
print("energy_forecasting_python DONE ->", OUT)
