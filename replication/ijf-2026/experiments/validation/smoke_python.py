"""Smoke test, Python side: the statsmodels SARIMAX baseline and the forecast metrics.

Prints one JSON object per measured quantity (engine, test, quantity, value); the orchestrator
compares them with smoke_expected.json. The statsmodels fit is the one the July 2026 battery
archived (statsmodels 0.14.6) for the same data.

Usage: python smoke_python.py <fixtures directory>
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path


def emit(test, quantity, value, note=""):
    print(json.dumps({"engine": "python", "test": test, "quantity": quantity, "value": value,
                      "note": note}), flush=True)


def run(name, fn):
    t0 = time.perf_counter()
    try:
        fn()
    except Exception as exc:  # reported, never swallowed
        emit(name, "error", f"{type(exc).__name__}: {exc}")
    emit(name, "seconds", round(time.perf_counter() - t0, 2))


def statsmodels_ar1(x):
    import numpy as np
    import statsmodels
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    emit("py_runtime", "statsmodels_version", statsmodels.__version__)
    emit("py_runtime", "numpy_version", np.__version__)
    res = SARIMAX(np.asarray(x), order=(1, 0, 0), trend="c", enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)
    params = dict(zip(res.model.param_names, res.params))
    emit("py_sarimax_ar1", "ar_L1", float(params["ar.L1"]))
    emit("py_sarimax_ar1", "intercept", float(params["intercept"]))
    emit("py_sarimax_ar1", "loglik", float(res.llf))
    emit("py_sarimax_ar1", "rss", float(np.sum(res.resid ** 2)))


def metrics():
    # The table generators' conventions on a hand-checkable example (two series, horizon 2):
    #   series A: actual (100, 100), forecast (110, 90), Naive2 (120, 120), in-sample scale 10
    #   series B: actual (10, 10),  forecast (10, 12),  Naive2 (8, 8),     in-sample scale 1
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from ijftables.common import owa

    def smape(a, f):
        return 200 / len(a) * sum(abs(x - y) / (abs(x) + abs(y)) for x, y in zip(a, f))

    def mase(a, f, scale):
        return sum(abs(x - y) for x, y in zip(a, f)) / len(a) / scale

    A, B = ((100, 100), (110, 90), (120, 120), 10.0), ((10, 10), (10, 12), (8, 8), 1.0)
    s = [smape(a, f) for a, f, _, _ in (A, B)]
    m = [mase(a, f, k) for a, f, _, k in (A, B)]
    sn = [smape(a, n) for a, _, n, _ in (A, B)]
    mn = [mase(a, n, k) for a, _, n, k in (A, B)]
    emit("metrics", "smape_A", s[0])           # 200/2 * (10/210 + 10/190) = 10.0250626...
    emit("metrics", "mase_B", m[1])            # (0 + 2)/2 / 1 = 1
    ratio_of_means = owa(sum(m) / 2, sum(mn) / 2, sum(s) / 2, sum(sn) / 2)
    mean_of_ratios = sum(owa(m[i], mn[i], s[i], sn[i]) for i in range(2)) / 2
    emit("metrics", "owa_ratio_of_means", ratio_of_means)
    emit("metrics", "owa_mean_of_ratios", mean_of_ratios,
         note="NOT the M4 OWA; shown to document that the two differ")
    emit("metrics", "owa_definitions_differ", abs(ratio_of_means - mean_of_ratios) > 1e-6)


def main() -> int:
    fixtures = Path(sys.argv[1])
    with open(fixtures / "sim_arma.csv", newline="") as f:
        x = [float(r["value"]) for r in csv.DictReader(f)]
    run("py_sarimax_ar1", lambda: statsmodels_ar1(x))
    run("metrics", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
