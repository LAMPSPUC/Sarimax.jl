#!/usr/bin/env python
"""Focused diagnostic for the sim_sarimax exogenous discrepancy (B4).

Tests the hypothesis: SARIMAX.jl fits a dynamic-regression / ARX model
  y_t = c + phi*y_{t-1} + b1*x1_t + b2*x2_t + e_t          (AR acts on observed y)
while statsmodels SARIMAX(exog) and R Arima(xreg) fit regression-with-ARIMA-errors
  y_t = c + b1*x1_t + b2*x2_t + u_t,  u_t = phi*u_{t-1} + e_t   (AR acts on residual)
which are different models and differ by a -phi*X_{t-1}*b cross term.

The shared DGP (gen_sim_data.py) is ARX, so SARIMAX.jl matches it; statsmodels/R do not.
We confirm by emulating ARX inside statsmodels/OLS (lagged y as a regressor): it should
reproduce SARIMAX.jl's estimates (phi~0.52, b~[1.53,-0.68], RSS~296).
"""
import os, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "results", "raw"))
DATA = os.path.join(RAW, "validation", "data")
OUT = os.path.join(RAW, "validation", "exog_diagnostic.jsonl")

df = pd.read_csv(os.path.join(DATA, "sim_sarimax.csv"))
y = df["value"].to_numpy()
X = df[["x1", "x2"]].to_numpy()
TRUE = {"phi": 0.5, "b1": 1.5, "b2": -0.8, "sigma2": 1.0}

records = []


def rss_of(resid):
    return float(np.sum(np.asarray(resid) ** 2))


# 1) statsmodels regression-with-ARIMA-errors (the default exog behavior)
res_rae = SARIMAX(y, exog=X, order=(1, 0, 0), trend="c",
                  enforce_stationarity=False).fit(disp=False)
records.append(dict(model="statsmodels reg+ARIMA-errors", family="reg-w-ARIMA-errors",
                    phi=float(res_rae.params[res_rae.param_names.index("ar.L1")]),
                    b1=float(res_rae.params[res_rae.param_names.index("x1")]),
                    b2=float(res_rae.params[res_rae.param_names.index("x2")]),
                    rss=rss_of(res_rae.resid), sigma2=float(res_rae.params[-1])))

# 2) statsmodels ARX emulation: lagged y as an extra regressor, no AR error structure
ylag = np.empty_like(y); ylag[0] = np.nan; ylag[1:] = y[:-1]
m = ~np.isnan(ylag)
Xarx = np.column_stack([X[m], ylag[m]])
res_arx = SARIMAX(y[m], exog=Xarx, order=(0, 0, 0), trend="c").fit(disp=False)
pn = res_arx.param_names
records.append(dict(model="statsmodels ARX (lagged-y regressor)", family="ARX",
                    phi=float(res_arx.params[pn.index("x3")]),
                    b1=float(res_arx.params[pn.index("x1")]),
                    b2=float(res_arx.params[pn.index("x2")]),
                    rss=rss_of(res_arx.resid), sigma2=float(res_arx.params[-1])))

# 3) Plain OLS ARX (closed form) as an independent check
Xols = sm.add_constant(np.column_stack([X[m], ylag[m]]))
ols = sm.OLS(y[m], Xols).fit()
records.append(dict(model="OLS ARX (closed form)", family="ARX",
                    phi=float(ols.params[3]), b1=float(ols.params[1]),
                    b2=float(ols.params[2]), rss=rss_of(ols.resid),
                    sigma2=float(np.var(ols.resid, ddof=Xols.shape[1]))))

# 4) Reference: SARIMAX.jl estimates already on disk (ARX native)
jl = None
for line in open(os.path.join(RAW, "validation", "julia_results.jsonl")):
    r = json.loads(line)
    if r["dataset"] == "sim_sarimax":
        e = r["estimates"]
        jl = dict(model="SARIMAX.jl (native ARX)", family="ARX",
                  phi=e["phi"][0], b1=e["exog"][0], b2=e["exog"][1],
                  rss=r["rss"], sigma2=e["sigma2"])
if jl:
    records.append(jl)

with open(OUT, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"{'model':40} {'family':20} {'phi':>7} {'b1':>7} {'b2':>7} {'RSS':>8} {'sig2':>6}")
print(f"{'TRUE DGP':40} {'ARX':20} {TRUE['phi']:7.3f} {TRUE['b1']:7.3f} "
      f"{TRUE['b2']:7.3f} {'--':>8} {TRUE['sigma2']:6.2f}")
for r in records:
    print(f"{r['model']:40} {r['family']:20} {r['phi']:7.3f} {r['b1']:7.3f} "
          f"{r['b2']:7.3f} {r['rss']:8.2f} {r['sigma2']:6.2f}")
print("\nwrote", OUT)
