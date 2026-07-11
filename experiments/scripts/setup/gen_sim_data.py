#!/usr/bin/env python
"""Generate deterministic simulated series shared by Julia/Python/R benchmarks.

Single source of truth so every implementation fits the *same* data.
Outputs CSV (date,value[,x1,x2]) under results/raw/<block>/data/.
"""
import os
import numpy as np
import pandas as pd

SEED = 20240627
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "..", "results", "raw"))


def monthly_index(n, start="2000-01-01"):
    return pd.date_range(start=start, periods=n, freq="MS")


def sim_arma(n, phi, theta, sigma=1.0, seed=SEED, burn=200):
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n + burn)
    y = np.zeros(n + burn)
    for t in range(1, n + burn):
        ar = phi * y[t - 1] if phi else 0.0
        ma = theta * e[t - 1] if theta else 0.0
        y[t] = ar + ma + e[t]
    return y[burn:]


def sim_sarimax(n, beta, phi=0.5, sigma=1.0, seed=SEED, burn=200):
    rng = np.random.default_rng(seed + 1)
    k = len(beta)
    X = rng.normal(0.0, 1.0, size=(n + burn, k))
    e = rng.normal(0.0, sigma, size=n + burn)
    y = np.zeros(n + burn)
    for t in range(1, n + burn):
        y[t] = phi * y[t - 1] + X[t] @ np.asarray(beta) + e[t]
    return y[burn:], X[burn:]


def write(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print("wrote", path, df.shape)


def main():
    n = 300
    # ARMA(1,1) series for ARIMA(1,0,0)/(0,0,1)/(1,0,1) validation
    y = sim_arma(n, phi=0.6, theta=0.4)
    write(os.path.join(RAW, "validation", "data", "sim_arma.csv"),
          pd.DataFrame({"date": monthly_index(n), "value": y}))
    # SARIMAX with 2 exogenous regressors (true betas 1.5, -0.8)
    ys, X = sim_sarimax(n, beta=[1.5, -0.8], phi=0.5)
    write(os.path.join(RAW, "validation", "data", "sim_sarimax.csv"),
          pd.DataFrame({"date": monthly_index(n), "value": ys,
                        "x1": X[:, 0], "x2": X[:, 1]}))
    # Clean ARMA + injected outliers for the objective-swap (architecture) block
    yo = sim_arma(n, phi=0.6, theta=0.0, seed=SEED + 7)
    yo = yo.copy()
    for idx in (60, 120, 180, 240):  # fixed, documented outliers
        yo[idx] += 8.0
    write(os.path.join(RAW, "architecture", "data", "sim_outliers.csv"),
          pd.DataFrame({"date": monthly_index(n), "value": yo}))
    # Regularization design: many exogenous regressors, only 2 informative
    rng = np.random.default_rng(SEED + 3)
    p = 8
    Xr = rng.normal(size=(n, p))
    beta = np.zeros(p); beta[0] = 1.5; beta[1] = -1.0
    yr = Xr @ beta + rng.normal(0, 1.0, size=n)
    cols = {"date": monthly_index(n), "value": yr}
    for j in range(p):
        cols[f"x{j+1}"] = Xr[:, j]
    write(os.path.join(RAW, "architecture", "data", "sim_regularization.csv"),
          pd.DataFrame(cols))


if __name__ == "__main__":
    main()
