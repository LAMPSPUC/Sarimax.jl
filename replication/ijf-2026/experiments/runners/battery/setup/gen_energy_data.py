#!/usr/bin/env python
"""Deterministic preprocessing of the PJM Hourly Energy Consumption (PJME) dataset.

Source: PJM Interconnection (PJM East / PJME zone), public ISO operational load data,
popularized as the Kaggle "Hourly Energy Consumption" dataset (CC0 1.0, R. Mulla).
Downloaded from a public GitHub mirror for unauthenticated reproducibility.

Outputs (no manual preprocessing; all steps below are programmatic):
- pjme_daily.csv        : daily mean load (MW), full history -> weekly seasonality (s=7)
- pjme_hourly_recent.csv : last 1500 hourly obs -> daily seasonality (s=24), reduced task/solver
- pjme_preprocessing.json: provenance + cleaning stats

Port of original-scripts/battery-2026-07/scripts/setup/gen_energy_data.py. The preprocessing
(sort, de-duplication, hourly reindexing, time interpolation, daily mean, last 1500 hours) is
unchanged. Adaptations:
  * input: $IJF_PJME_DIR/PJME_hourly.csv, verified against the archived SHA-256 (the original
    read results/raw/energy/data/PJME_hourly.csv, checked by `shasum -c` in its reproduce.sh);
  * outputs: $IJF_RUN_DIR/raw/battery-2026-07/energy/data/ (nothing is written elsewhere), with
    "\\n" line endings whatever the platform (pandas writes os.linesep by default; the archived
    files, written on macOS, have "\\n");
  * pjme_preprocessing.json also records the input path and digest of this run;
  * the derived files are not in the bundle, but the battery committed them to the repository
    (experiments/results/raw/energy/data/ at 8ffc512, the v0.3.0 re-run; identical blobs at
    ecc4878 and 308461e). Their SHA-256 digests and cleaning statistics are recorded below;
    the regenerated files are compared with them and the result printed as MATCH / MISMATCH
    and written to energy/data/input_check.json. On a MISMATCH the run CONTINUES WITH THE DATA
    DERIVED HERE and says so loudly.
"""
import hashlib
import json
import os
import sys

import numpy as np  # noqa: F401  (imported by the original)
import pandas as pd

PJME_SHA256 = "4eb2b16d42bf07ec41ab55cb842191594cb69452725a6d3c0991658a628fde84"
# experiments/results/raw/energy/data/ at commit 8ffc512 (not part of the artifact bundle)
ARCHIVED_SHA256 = {
    "pjme_daily.csv": "2b365b9e14476661a8de6a8b7ba5f726630c218fe204063405c5ce87420b0632",
    "pjme_hourly_recent.csv": "0772e828e787b11057a789e2e97f52640d6185fcd375bfc95ffb654629e7d002",
}
ARCHIVED_STATS = {  # from the archived pjme_preprocessing.json (same commit)
    "n_raw_rows": 145366, "duplicate_timestamps_removed": 4, "missing_hours_interpolated": 30,
    "span": ["2002-01-01 01:00:00", "2018-08-03 00:00:00"], "n_hourly_clean": 145392,
    "n_daily": 6059, "n_hourly_recent": 1500,
}


def _paths():
    run_dir = os.environ.get("IJF_RUN_DIR", "")
    pjme_dir = os.environ.get("IJF_PJME_DIR", "")
    if not run_dir:
        sys.exit("IJF_RUN_DIR is not set: the battery runners write only below "
                 "$IJF_RUN_DIR/raw/battery-2026-07/")
    src = os.path.join(pjme_dir, "PJME_hourly.csv") if pjme_dir else ""
    if not src or not os.path.isfile(src):
        print(f"ERROR: PJME_hourly.csv not found (IJF_PJME_DIR={pjme_dir!r}); "
              "run ./reproduce.sh fetch pjme")
        sys.exit(4)
    return (os.path.join(os.path.abspath(run_dir), "raw", "battery-2026-07", "energy", "data"), src)


DATA, SRC = _paths()


def _sha256(path, strip_cr=False):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk.replace(b"\r", b"") if strip_cr else chunk)
    return h.hexdigest()


src_sha = _sha256(SRC)
if src_sha != PJME_SHA256 and _sha256(SRC, strip_cr=True) != PJME_SHA256:
    print(f"ERROR: {SRC} has sha256 {src_sha}, not the archived {PJME_SHA256}; "
          "run ./reproduce.sh fetch pjme")
    sys.exit(4)
os.makedirs(DATA, exist_ok=True)

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
daily.to_csv(os.path.join(DATA, "pjme_daily.csv"), index=False, lineterminator="\n")

# Recent hourly slice (last 1500 hours) -> daily seasonality, reduced task + solver
recent = df["PJME_MW"].iloc[-1500:].rename("value").reset_index()
recent.columns = ["date", "value"]
recent.to_csv(os.path.join(DATA, "pjme_hourly_recent.csv"), index=False, lineterminator="\n")

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
    # added by the port: the input actually read in this run
    "input_path": SRC,
    "input_sha256": src_sha,
    "pandas": pd.__version__,
}
with open(os.path.join(DATA, "pjme_preprocessing.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(meta, f, indent=2)

print(json.dumps(meta, indent=2))

# ---- comparison with the archived derived files (digests) and statistics ----------------------
checks = []
for name, ref in ARCHIVED_SHA256.items():
    got = _sha256(os.path.join(DATA, name))
    checks.append({"file": f"raw/battery-2026-07/energy/data/{name}", "sha256": got,
                   "sha256_archived": ref, "result": "MATCH" if got == ref else "MISMATCH"})
stats = {"n_raw_rows": n_raw, "duplicate_timestamps_removed": n_dups,
         "missing_hours_interpolated": n_missing, "span": meta["span"],
         "n_hourly_clean": meta["n_hourly_clean"], "n_daily": int(len(daily)),
         "n_hourly_recent": int(len(recent))}
bad_stats = {k: {"regenerated": stats[k], "archived": v} for k, v in ARCHIVED_STATS.items() if stats[k] != v}
checks.append({"file": "cleaning statistics", "result": "MISMATCH" if bad_stats else "MATCH",
               "differences": bad_stats})
with open(os.path.join(DATA, "input_check.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump({"generator": "experiments/runners/battery/setup/gen_energy_data.py",
               "reference": "experiments/results/raw/energy/data/ at commit 8ffc512 (not in the bundle)",
               "input_sha256": src_sha, "checks": checks}, f, indent=2)
mismatches = [c for c in checks if c["result"] == "MISMATCH"]
for c in checks:
    line = f"[input check] {c['file']} vs archived (8ffc512): {c['result']}"
    if c["result"] == "MISMATCH":
        bar = "!" * 100
        print(f"{bar}\n{line} {c.get('differences') or ''}\nTHE RUN CONTINUES WITH THE DATA DERIVED HERE, "
              f"WHICH DIFFER FROM THE ARCHIVED INPUT.\n{bar}", flush=True)
    else:
        print(line, flush=True)
print("[input check] " + (f"{len(mismatches)} MISMATCH(ES) - see above" if mismatches
                          else "no mismatch against the archived derived files"))
