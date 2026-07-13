#!/usr/bin/env python
"""Deterministic preprocessing of the PJM Hourly Energy Consumption (PJME) dataset.

Source: PJM Interconnection (PJM East / PJME zone), public ISO operational load data,
popularized as the Kaggle "Hourly Energy Consumption" dataset (CC0 1.0, R. Mulla).
Downloaded from a public GitHub mirror for unauthenticated reproducibility.

Outputs (no manual preprocessing; all steps below are programmatic):
- pjme_daily.csv        : daily mean load (MW), full history -> weekly seasonality (s=7)
- pjme_hourly_recent.csv : last 1500 hourly obs -> daily seasonality (s=24), reduced task/solver
- pjme_preprocessing.json: provenance + cleaning stats
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "..", "results", "raw"))
DATA = os.path.join(RAW, "energy", "data")
SRC = os.path.join(DATA, "PJME_hourly.csv")

df = pd.read_csv(SRC)
n_raw = len(df)
df["Datetime"] = pd.to_datetime(df["Datetime"])
# sort chronologically and drop duplicate timestamps (DST fall-back duplicates), keep first
df = df.sort_values("Datetime")
n_dups = int(df.duplicated(subset="Datetime").sum())
df = df.drop_duplicates(subset="Datetime", keep="first").set_index("Datetime")
# reindex onto a regular hourly grid; count and fill missing hours by time interpolation
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
n_missing = int(len(full_idx) - len(df))
df = df.reindex(full_idx)
df["PJME_MW"] = df["PJME_MW"].interpolate(method="time").bfill().ffill()
df.index.name = "date"

# Daily aggregation (mean MW per day) -> weekly seasonality
daily = df["PJME_MW"].resample("D").mean().rename("value").reset_index()
daily.to_csv(os.path.join(DATA, "pjme_daily.csv"), index=False)

# Recent hourly slice (last 1500 hours) -> daily seasonality, reduced task + solver
recent = df["PJME_MW"].iloc[-1500:].rename("value").reset_index()
recent.columns = ["date", "value"]
recent.to_csv(os.path.join(DATA, "pjme_hourly_recent.csv"), index=False)

meta = {
    "source": "PJM Interconnection (PJME / PJM East zone) hourly load, public ISO data",
    "distribution": "Kaggle 'Hourly Energy Consumption' dataset (CC0 1.0, R. Mulla); "
                    "downloaded from a public GitHub mirror for unauthenticated reproducibility",
    "license": "CC0 1.0 (public domain dedication)",
    "url_mirror": "https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv",
    "frequency_raw": "hourly (MW)",
    "n_raw_rows": n_raw,
    "duplicate_timestamps_removed": n_dups,
    "missing_hours_interpolated": n_missing,
    "span": [str(df.index.min()), str(df.index.max())],
    "n_hourly_clean": int(len(df)),
    "transformations": [
        "parse Datetime; sort chronologically",
        "drop duplicate timestamps (DST fall-back), keep first",
        "reindex onto regular hourly grid; time-interpolate missing hours (then bfill/ffill edges)",
        "daily aggregation = mean MW per calendar day",
    ],
    "outputs": {
        "pjme_daily.csv": {"freq": "daily", "seasonality": 7, "n": int(len(daily))},
        "pjme_hourly_recent.csv": {"freq": "hourly", "seasonality": 24, "n": int(len(recent))},
    },
}
with open(os.path.join(DATA, "pjme_preprocessing.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(json.dumps(meta, indent=2))
