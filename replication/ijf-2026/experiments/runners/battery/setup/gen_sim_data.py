#!/usr/bin/env python
"""Generate deterministic simulated series shared by Julia/Python/R benchmarks.

Single source of truth so every implementation fits the *same* data.
Outputs CSV (date,value[,x1,x2]) under $IJF_RUN_DIR/raw/battery-2026-07/<block>/data/.

Port of original-scripts/battery-2026-07/scripts/setup/gen_sim_data.py. The generator itself
(seeds, DGPs, sizes, injected outliers) is unchanged. Adaptations:
  * outputs go to $IJF_RUN_DIR/raw/battery-2026-07/<block>/data/ (the original wrote to
    <script dir>/../../results/raw/); nothing is written anywhere else;
  * CSVs are written with "\\n" line endings. pandas >= 1.5 writes os.linesep by default, so the
    original produced "\\n" on macOS but would produce "\\r\\n" on Windows; fixing the terminator
    makes the bytes platform-independent and comparable with the archived copies;
  * after writing, each file is compared with the archived copy in the bundle
    ($IJF_ARTIFACTS_DIR/raw/battery-2026-07/<block>/data/<file>, and for sim_arma.csv also
    $IJF_FIXTURES/sim_arma.csv): byte-wise SHA-256 first, then value by value. The result is
    printed as MATCH / MISMATCH / NOT COMPARED and written to <block>/data/input_check.json.
    On a MISMATCH the run CONTINUES WITH THE REGENERATED DATA (never with the archived copy),
    and the mismatch is reported loudly; the exit code stays 0 so the remaining steps run.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

SEED = 20240627


def _run_raw():
    run_dir = os.environ.get("IJF_RUN_DIR", "")
    if not run_dir:
        sys.exit("IJF_RUN_DIR is not set: the battery runners write only below "
                 "$IJF_RUN_DIR/raw/battery-2026-07/")
    return os.path.join(os.path.abspath(run_dir), "raw", "battery-2026-07")


RAW = _run_raw()
ARTIFACTS = os.environ.get("IJF_ARTIFACTS_DIR", "")
FIXTURES = os.environ.get("IJF_FIXTURES", "")


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


WRITTEN = []


def write(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    print("wrote", path, df.shape)
    WRITTEN.append(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(path, reference, label):
    """Compare a regenerated file with an archived copy: SHA-256, then values."""
    out = {"reference": reference, "reference_kind": label}
    if not reference or not os.path.isfile(reference):
        out["result"] = "NOT COMPARED"
        out["detail"] = f"{label} copy not available"
        return out
    new_sha, old_sha = sha256(path), sha256(reference)
    out.update(sha256_regenerated=new_sha, sha256_reference=old_sha)
    if new_sha == old_sha:
        out["result"] = "MATCH"
        out["detail"] = "byte-identical (SHA-256)"
        return out
    out["result"] = "MISMATCH"
    try:
        a, b = pd.read_csv(path), pd.read_csv(reference)
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            out["detail"] = f"layout differs: {list(a.columns)} {a.shape} vs {list(b.columns)} {b.shape}"
        else:
            num = [c for c in a.columns if c != "date"]
            dates_equal = bool((a["date"].astype(str) == b["date"].astype(str)).all()) if "date" in a else True
            diff = float(np.max(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float))))
            out.update(max_abs_value_difference=diff, dates_equal=dates_equal)
            out["detail"] = (f"bytes differ; max |value difference| = {diff:.3e}, "
                             f"dates {'equal' if dates_equal else 'DIFFER'}")
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        out["detail"] = f"bytes differ; value comparison failed: {exc!r}"
    return out


def check_against_archive():
    by_block = {}
    mismatches = 0
    for path in WRITTEN:
        block = os.path.basename(os.path.dirname(os.path.dirname(path)))
        name = os.path.basename(path)
        rel = f"raw/battery-2026-07/{block}/data/{name}"
        checks = [compare(path, os.path.join(ARTIFACTS, rel) if ARTIFACTS else "", "archived bundle")]
        if name == "sim_arma.csv":
            checks.append(compare(path, os.path.join(FIXTURES, name) if FIXTURES else "", "fixture"))
        rec = {"file": rel, "sha256": sha256(path), "checks": checks}
        by_block.setdefault(block, []).append(rec)
        for c in checks:
            line = f"[input check] {rel} vs {c['reference_kind']}: {c['result']} ({c['detail']})"
            if c["result"] == "MISMATCH":
                mismatches += 1
                bar = "!" * 100
                print(f"{bar}\n{line}\nTHE RUN CONTINUES WITH THE REGENERATED DATA, WHICH DIFFER FROM THE "
                      f"ARCHIVED INPUT: results built on it are NOT comparable input-for-input with the "
                      f"archived battery.\n{bar}", flush=True)
            else:
                print(line, flush=True)
    for block, recs in by_block.items():
        with open(os.path.join(RAW, block, "data", "input_check.json"), "w", encoding="utf-8",
                  newline="\n") as f:
            json.dump({"generator": "experiments/runners/battery/setup/gen_sim_data.py",
                       "seed": SEED, "numpy": np.__version__, "pandas": pd.__version__,
                       "files": recs}, f, indent=2)
    if mismatches:
        print(f"[input check] {mismatches} MISMATCH(ES) against archived inputs - see above", flush=True)
    else:
        print("[input check] no mismatch against the archived inputs", flush=True)


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
    check_against_archive()


if __name__ == "__main__":
    main()
