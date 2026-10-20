"""Shared loaders and conventions for the revision result package.

Every convention below is copied from the replication package's own table generator,
`experiments/tables/make_table_m4.jl`, which reproduces the published M4 table byte for byte
on this machine. Where this module departs from it, the departure is named.

    sMAPE_i = mean_h 200 |f - a| / (|f| + |a| + 1e-12)          (percent)
    MASE_i  = mean_h |f - a| / den_i                             (den_i of the WHOLE series)
    OWA     = 0.5 mean(sMAPE)/mean(sMAPE_N2) + 0.5 mean(MASE)/mean(MASE_N2)

OWA is a RATIO OF MEANS over a set of series. It is an aggregate quantity and has no
per-series analogue; nothing in this package computes or reports a per-series "OWA".
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent                                     # paper_results_revision/
FT = PKG.parent                                       # ForecastTester/
REPL = PKG / "_replication" / "experiments"
RAW = REPL / "results" / "raw"
REF = Path(os.environ.get("M4_REFERENCE_DIR", FT / "Results_autoarima_R"))
M4 = Path(os.environ.get("M4_DATA_DIR", FT / "datasets"))

DATA, FIG, TAB, REP = PKG / "data", PKG / "figures", PKG / "tables", PKG / "reports"
for _d in (DATA, FIG, TAB, REP):
    _d.mkdir(parents=True, exist_ok=True)

FREQS = ["yearly", "quarterly", "monthly", "weekly", "daily", "hourly"]
FREQ_LABEL = {f: f.capitalize() for f in FREQS}
M4_PREFIX = {"yearly": "Y", "quarterly": "Q", "monthly": "M",
             "weekly": "W", "daily": "D", "hourly": "H"}
HORIZON = {"yearly": 6, "quarterly": 8, "monthly": 18,
           "weekly": 13, "daily": 14, "hourly": 48}

# A paired difference is counted as a tie when |Delta| is below this value, in the metric's
# own units (percentage points for sMAPE, scale-free units for MASE). Ties are real here:
# when both procedures select the same model they produce the same forecast, and the
# residual difference is floating-point noise. Sensitivity to the choice is reported.
TIE_TOL = 1e-4
TIE_TOL_SENSITIVITY = (1e-6, 1e-4, 1e-2)

QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
# numpy's default quantile estimator (linear interpolation between order statistics,
# Hyndman & Fan type 7). Declared because the tail quantiles of heavy-tailed differences
# depend visibly on the estimator.
QUANTILE_METHOD = "linear"


def smape(f: np.ndarray, a: np.ndarray) -> float:
    return float(np.mean(2.0 * np.abs(f - a) / (np.abs(f) + np.abs(a) + 1e-12)) * 100.0)


def mase(f: np.ndarray, a: np.ndarray, den: float) -> float:
    return float(np.mean(np.abs(f - a)) / max(den, 1e-12))


def _read_commented_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def load_campaign_a(freq: str) -> pd.DataFrame:
    """Sarimax.jl, campaign A (`initialization = :innovations`, objective `mse`, no time cap).

    sMAPE and MASE are RECOMPUTED over the full horizon from the stored forecast and actual
    vectors, exactly as the table generator does, and the stored columns are kept alongside
    so the two can be checked against each other.
    """
    df = _read_commented_csv(RAW / f"m4_innovations_{freq}.csv")
    df = df[df["status"].astype(str).str.startswith("OK")].copy()
    fc = df["forecast"].astype(str).str.split(";")
    ac = df["actual"].astype(str).str.split(";")
    sm, ms, hf, ha = [], [], [], []
    for fv, av, den in zip(fc, ac, df["mase_den"].to_numpy(float)):
        f = np.array([float(x) for x in fv if x != ""])
        a = np.array([float(x) for x in av if x != ""])
        sm.append(smape(f, a))
        ms.append(mase(f, a, den))
        hf.append(f.size)
        ha.append(a.size)
    df["smape_recomputed"] = sm
    df["mase_recomputed"] = ms
    df["h_forecast"] = hf
    df["h_actual"] = ha
    return df.drop(columns=["forecast", "actual"])


def load_reference(freq: str, model: str = "autoarima_R", block: str = "total") -> pd.DataFrame:
    """Per-series reference metrics, read BY COLUMN NAME (the files carry a header)."""
    df = pd.read_csv(REF / freq / model / f"{block}.csv")
    df = df[["series", "sMAPE", "MASE"]].copy()
    df = df[np.isfinite(df["sMAPE"]) & np.isfinite(df["MASE"])]
    return df.rename(columns={"sMAPE": "smape", "MASE": "mase"})


def owa(smape_m, mase_m, smape_n2, mase_n2) -> float:
    return 0.5 * np.mean(smape_m) / np.mean(smape_n2) + 0.5 * np.mean(mase_m) / np.mean(mase_n2)


def quantile_row(x: np.ndarray, qs=QUANTILES) -> dict:
    x = np.asarray(x, float)
    out = {"N": int(x.size), "mean": float(np.mean(x))}
    for q in qs:
        out[f"q{int(round(q * 100)):02d}"] = float(np.quantile(x, q, method=QUANTILE_METHOD))
    return out


def load_m4_train(freq: str) -> dict[int, np.ndarray]:
    """Training series keyed by the integer id the result files use (`W304` -> 304)."""
    path = M4 / f"{FREQ_LABEL[freq]}-train.csv"
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].astype(str).str[1:].astype(int)
    vals = df.iloc[:, 1:].to_numpy(float)
    return {int(i): row[~np.isnan(row)] for i, row in zip(ids, vals)}


# ---- figure conventions -----------------------------------------------------------------
# Reference categorical palette (validated order; see the dataviz reference instance).
# Colour is never the only identity channel: every series also gets a line style, so a
# figure printed in greyscale remains readable.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#4a3aa7", "#e34948"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def setup_matplotlib():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,          # embed TrueType: text stays text in the PDF
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
    })


def save(fig, stem: str):
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=200)


def write_table(df: pd.DataFrame, stem: str, latex: str):
    df.to_csv(TAB / f"{stem}.csv", index=False)
    (TAB / f"{stem}.tex").write_text(latex, encoding="utf-8")
