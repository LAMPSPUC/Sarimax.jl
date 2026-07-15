#!/usr/bin/env python
"""Block 1 - Validation (statsmodels baseline). Writes JSONL run records."""
import os, json, time, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "results", "raw"))
DATA = os.path.join(RAW, "validation", "data")
OUT = os.path.join(RAW, "validation", "python_statsmodels_results.jsonl")
PKG_DATASETS = os.path.normpath(os.path.join(HERE, "..", "..", "datasets"))

if os.path.exists(OUT):
    os.remove(OUT)


def rec_append(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_one(dataset, y, order, seasonal_order=(0, 0, 0, 0), exog=None):
    rec = dict(block="validation", implementation="statsmodels",
               dataset=dataset, order=f"{order}{seasonal_order}",
               objective="ml", solver="statespace-mle", seed=1234)
    try:
        t = time.time()
        mod = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order,
                      trend="c", enforce_stationarity=False,
                      enforce_invertibility=False)
        res = mod.fit(disp=False)
        rec["runtime_s"] = time.time() - t
        rec["status"] = "ok"
        rec["estimates"] = {str(k): float(v)
                            for k, v in zip(res.param_names, np.asarray(res.params))}
        rec["loglike"] = float(res.llf)
        rec["aic"] = float(res.aic)
        rec["bic"] = float(res.bic)
        rec["rss"] = float(np.sum(res.resid ** 2))
        rec["converged"] = bool(res.mle_retvals.get("converged", True))
    except Exception as e:  # noqa
        rec["status"] = "failed"
        rec["error"] = repr(e)
    rec_append(rec)
    print(dataset, order, seasonal_order, "->", rec["status"])


def main():
    arma = pd.read_csv(os.path.join(DATA, "sim_arma.csv"))["value"].to_numpy()
    sx = pd.read_csv(os.path.join(DATA, "sim_sarimax.csv"))
    airp = pd.read_csv(os.path.join(PKG_DATASETS, "airpassengers.csv"))
    airp_y = airp.iloc[:, 1].to_numpy()

    run_one("sim_arma", arma, (1, 0, 0))
    run_one("sim_arma", arma, (0, 0, 1))
    run_one("sim_arma", arma, (1, 0, 1))
    run_one("airpassengers", airp_y, (1, 0, 1))
    run_one("airpassengers", airp_y, (1, 0, 1), (1, 0, 1, 12))
    # sim_sarimax is a dynamic-regression (ARX) DGP: y_t = c + phi*y_{t-1} + Xb + e.
    # SARIMAX.jl models exog in ARX form, so we fit the comparable ARX here
    # (lagged y as a regressor, no AR error structure). See diagnose_exog.py / B4.
    run_arx_sim_sarimax(sx)
    print("validation_python DONE ->", OUT)


def run_arx_sim_sarimax(sx):
    y = sx["value"].to_numpy()
    ylag = np.empty_like(y); ylag[0] = np.nan; ylag[1:] = y[:-1]
    msk = ~np.isnan(ylag)
    exog = np.column_stack([sx["x1"].to_numpy()[msk], sx["x2"].to_numpy()[msk], ylag[msk]])
    rec = dict(block="validation", implementation="statsmodels",
               dataset="sim_sarimax", order="ARX(1)+2exog", objective="ml",
               solver="statespace-mle", seed=1234, model_family="ARX")
    try:
        t = time.time()
        res = SARIMAX(y[msk], exog=exog, order=(0, 0, 0), trend="c").fit(disp=False)
        pn = list(res.param_names)
        rec["runtime_s"] = time.time() - t
        rec["status"] = "ok"
        # expose lagged-y coefficient (x3) as the AR coefficient for the table
        rec["estimates"] = {"ar.L1": float(res.params[pn.index("x3")]),
                            "x1": float(res.params[pn.index("x1")]),
                            "x2": float(res.params[pn.index("x2")])}
        rec["loglike"] = float(res.llf)
        rec["aic"] = float(res.aic); rec["bic"] = float(res.bic)
        rec["rss"] = float(np.sum(res.resid ** 2))
    except Exception as e:  # noqa
        rec["status"] = "failed"; rec["error"] = repr(e)
    rec_append(rec)
    print("sim_sarimax ARX ->", rec["status"])


if __name__ == "__main__":
    main()
