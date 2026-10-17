"""M4 data access and the forecast-accuracy metrics used by every M4 runner.

The series id is the 1-based ROW index of the official `<Freq>-train.csv` / `<Freq>-test.csv`
files, as in the ForecastTester harness (`build_train_test_dict`) and in every archived
campaign and baseline file. The metric definitions are the harness's (`src/metrics.jl`),
which the stored auto.arima/Naive2 baselines also used:

    sMAPE = (200/h) * sum_i |a_i - f_i| / (|a_i| + |f_i|)
    MASE  = mean_i |a_i - f_i| / d,  d = (1/(T-s)) * sum_{j=s+1..T} |y_j - y_{j-s}|

with d computed on the whole training window and shared by every horizon block.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

FREQ_FILE = {"yearly": "Yearly", "quarterly": "Quarterly", "monthly": "Monthly",
             "weekly": "Weekly", "daily": "Daily", "hourly": "Hourly"}
SEASONALITY = {"yearly": 1, "quarterly": 4, "monthly": 12, "weekly": 1, "daily": 1, "hourly": 24}
HORIZON = {"yearly": 6, "quarterly": 8, "monthly": 18, "weekly": 13, "daily": 14, "hourly": 48}
BLOCKS = {
    "monthly": {"short": (1, 6), "medium": (7, 12), "long": (13, 18), "total": (1, 18)},
    "quarterly": {"short": (1, 2), "medium": (3, 5), "long": (6, 8), "total": (1, 8)},
    "yearly": {"short": (1, 2), "medium": (3, 4), "long": (5, 6), "total": (1, 6)},
    "weekly": {"short": (1, 4), "medium": (5, 9), "long": (10, 13), "total": (1, 13)},
    "daily": {"short": (1, 4), "medium": (5, 9), "long": (10, 14), "total": (1, 14)},
    "hourly": {"short": (1, 16), "medium": (17, 32), "long": (33, 48), "total": (1, 48)},
}
BLOCK_NAMES = ("short", "medium", "long", "total")


def _read(path: Path) -> list[list[float]]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            vals = [float(x) for x in row[1:] if x.strip() not in ("", "NA")]
            out.append(vals)
    return out


def load(m4_dir: Path, freq: str) -> tuple[list[list[float]], list[list[float]]]:
    stem = FREQ_FILE[freq]
    return _read(m4_dir / f"{stem}-train.csv"), _read(m4_dir / f"{stem}-test.csv")


def mase_denominator(y: list[float], s: int) -> float:
    T = len(y)
    return sum(abs(y[j] - y[j - s]) for j in range(s, T)) / (T - s)


def block_metrics(freq: str, y: list[float], actual: list[float], forecast: list[float]) -> dict:
    s = SEASONALITY[freq]
    den = mase_denominator(y, s)
    out = {"mase_den": den}
    h = min(len(actual), len(forecast))
    for name, (lo, hi) in BLOCKS[freq].items():
        idx = [i for i in range(lo - 1, hi) if i < h]
        if not idx:
            out[f"smape_{name}"] = out[f"mase_{name}"] = math.nan
            continue
        a = [actual[i] for i in idx]
        f = [forecast[i] for i in idx]
        # 0/0 terms give NaN exactly as in the harness (M4 values are positive, so it does not
        # occur in practice); a zero scale gives an infinite MASE rather than a silent number.
        sm = 200 / len(a) * sum(abs(x - z) / (abs(x) + abs(z)) if (abs(x) + abs(z)) > 0 else math.nan
                                for x, z in zip(a, f))
        ma = (sum(abs(x - z) for x, z in zip(a, f)) / len(a)) / den if den > 0 else math.inf
        out[f"smape_{name}"], out[f"mase_{name}"] = sm, ma
    return out
