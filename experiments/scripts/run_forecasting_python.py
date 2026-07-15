#!/usr/bin/env python
"""Block 2 - Rolling-origin forecasting (statsmodels baseline).
Expanding window on AirPassengers (s=12) and GDPC1 (quarterly). Refit at each origin."""
import os, json, time, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "results", "raw"))
PKG = os.path.normpath(os.path.join(HERE, "..", "..", "datasets"))
OUT = os.path.join(RAW, "forecasting", "python_statsmodels_results.jsonl")
if os.path.exists(OUT):
    os.remove(OUT)


def rolling_origins(n, init_frac, H, step):
    return list(range(int(np.floor(init_frac * n)), n - H + 1, step))


def metrics(act, fc, train, m):
    act, fc = np.asarray(act), np.asarray(fc)
    mae = np.mean(np.abs(act - fc))
    rmse = float(np.sqrt(np.mean((act - fc) ** 2)))
    smape = float(np.mean(2 * np.abs(act - fc) / (np.abs(act) + np.abs(fc) + 1e-12)) * 100)
    denom = np.mean(np.abs(train[m:] - train[:-m])) if len(train) > m else np.nan
    mase = float(mae / denom) if denom and not np.isnan(denom) else np.nan
    return float(mae), rmse, smape, mase


def rec_append(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_dataset(dataset, y, m, H, step, init_frac, order, seasonal_order):
    n = len(y)
    origins = rolling_origins(n, init_frac, H, step)
    maes, rmses, smapes, mases, rts, nfail = [], [], [], [], [], 0
    for k in origins:
        train, act = y[:k], y[k:k + H]
        try:
            t = time.time()
            res = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                          trend="c", enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
            fc = np.asarray(res.forecast(steps=H))
            a, r, s, ms = metrics(act, fc, train, m)
            maes.append(a); rmses.append(r); smapes.append(s); mases.append(ms)
            rts.append(time.time() - t)
        except Exception:  # noqa
            nfail += 1
    rec = dict(block="forecasting", implementation="statsmodels", dataset=dataset,
               order=order_label(order, seasonal_order), horizon=H, seasonality=m,
               protocol="rolling-origin", n_origins=len(origins), n_failures=nfail,
               seed=1234)
    if maes:
        rec.update(status="ok", mae=float(np.mean(maes)), rmse=float(np.mean(rmses)),
                   smape=float(np.mean(smapes)),
                   mase=float(np.nanmean(mases)), runtime_s=float(np.sum(rts)))
    else:
        rec["status"] = "failed"
    rec_append(rec)
    print("statsmodels", dataset, "->", rec["status"], "origins", len(origins), "fail", nfail)


def order_label(order, seasonal_order):
    s = f"{order}"
    if seasonal_order and seasonal_order[-1] > 0:
        s += f"{seasonal_order}"
    return s


def main():
    airp = pd.read_csv(os.path.join(PKG, "airpassengers.csv")).iloc[:, 1].to_numpy()
    gdp = pd.read_csv(os.path.join(PKG, "GDPC1.csv"), sep=";").iloc[:, 1].to_numpy()
    run_dataset("airpassengers", airp, 12, 12, 12, 0.7, (1, 0, 1), (1, 0, 1, 12))
    run_dataset("GDPC1", gdp, 1, 8, 20, 0.7, (1, 1, 1), (0, 0, 0, 0))
    print("forecasting_python (rolling-origin) DONE ->", OUT)


if __name__ == "__main__":
    main()
