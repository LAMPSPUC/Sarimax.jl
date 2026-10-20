"""GAVEA campaigns of September 2026 (raw archive brutos_gavea.zip, bundle part 2), with the
yearly/weekly stable runs of RECREIO campaign F (part 1): the three comparisons tabulated on
2026-10-17 (`y_updated_results.py`, results/historical/updated-2026-10-17/):

* `gavea.stable_objective`: MSE against the `stable` (CVaR) objective, alpha 0.5 and/or 0.9,
  on six M4 frequencies (monthly: a 5,000-series sample and the 923 series of the alpha=0.5 arm);
* `gavea.admissibility_grid`: the 2x2x2x2 census of M4 monthly, stationary x invertible x MA
  normalization (Section 5.5 of the manuscript), baseline: stationary and normalized (`gs`);
* `gavea.require_terms_guard`: requireTermsWhenOverDifferenced off against on (hourly,
  quarterly, daily).

For each: `.summary` (N, mean sMAPE, mean MASE, OWA, auto.arima's OWA on the same series) and
`.paired` (per-series differences against the baseline arm and against auto.arima: mean,
median, q05, q95, shares lower / tied / higher at 1e-4, share with the same selected order).

Definitions of 2026-10-17: each arm's rows are concatenated in file-name order and the LAST
row of a series is kept; a series counts when that row's status is LOCALLY_SOLVED, OK or
ALMOST_LOCALLY_SOLVED and its sMAPE and MASE are finite; every comparison uses the series all
its arms fitted and whose two stored references are finite. The hourly and quarterly stable
campaigns wrote MASE with the lag-1 in-sample naive denominator; it is converted exactly to the
M4 (seasonal) definition with the per-series ratio in derived/stable-admissibility-gavea-2026-09/.
Quantiles are linear (type 7). OWA is a ratio of means.

The reference values are parsed from the tables written on 2026-10-17 (archived in part 2);
the arms ran on different revisions and hosts (see the manifest), so rows of different
frequencies are not one controlled experiment.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from .common import Cell, Context, fmean, owa, read_csv, write_csv

TITLE = "GAVEA stable (CVaR), 2x2x2x2 admissibility and guard campaigns, Sep 2026 (tables of 2026-10-17)"
GV = "raw/stable-admissibility-gavea-2026-09/brutos_gavea"
GVD = "derived/stable-admissibility-gavea-2026-09"
F_FILE = "raw/m4-recreio-2026-08/stable_weekly_yearly.csv.gz"
BASE = "raw/m4-baselines-r-2026-07"
FREQS = ("yearly", "quarterly", "monthly", "weekly", "daily", "hourly")
OK_STATUS = {"LOCALLY_SOLVED", "OK", "ALMOST_LOCALLY_SOLVED"}
TIE = 1e-4
ORDER = ("p", "d", "q", "P", "D", "Q")
CENSUS = ["gs", "gff", "gi", "gsi", "fs", "f1", "fi", "fsi"]


def _gavea_inventory() -> list[str]:
    inv = Path(__file__).resolve().parents[2] / "artifacts-part2.sha256"
    out = []
    for line in inv.read_text(encoding="utf-8").splitlines():
        rel = line.partition("  ")[2].strip()
        if rel.startswith(GV + "/") and rel.endswith((".csv", ".csv.gz")) and "_prev." not in rel:
            out.append(rel)
    return out


GAVEA_FILES = _gavea_inventory()
INPUTS = sorted(GAVEA_FILES + [F_FILE]
                + [f"{GVD}/mase_lag1_to_seasonal_{f}.csv.gz" for f in ("hourly", "quarterly")]
                + [f"{BASE}/{f}/{m}/total.csv.gz" for f in FREQS for m in ("Naive", "autoarima_R")])


def _stem(rel: str) -> str:
    return rel[len(GV) + 1:].removesuffix(".gz")


def load(ctx: Context, pattern: str) -> list[dict]:
    """Rows of every GAVEA file whose name (inside the archive) matches `pattern`, in file-name
    order, as the 2026-10-17 script read them (glob of the original names, then sorted)."""
    rx = re.compile(pattern + r"$")
    files = sorted((r for r in GAVEA_FILES if rx.match(_stem(r))), key=_stem)
    if not files:
        raise ValueError(f"no GAVEA file matches {pattern}")
    rows = []
    for rel in files:
        rows += read_csv(ctx.path(rel))
    return rows


def keep_last_ok(rows: list[dict], status_map=None) -> dict[int, dict]:
    last = {}
    for r in rows:
        last[int(float(r["sid"]))] = r
    out = {}
    for sid, r in last.items():
        st = str(r["status"]).strip('"')
        if status_map:
            st = status_map(st)
        if st not in OK_STATUS:
            continue
        sm, ma = float(r["smape"]), float(r["mase"])
        if math.isfinite(sm) and math.isfinite(ma):
            out[sid] = {"smape": sm, "mase": ma, "order": tuple(str(int(float(r[k]))) for k in ORDER)}
    return out


def rescale(d: dict[int, dict], ratio: dict[int, float]) -> dict[int, dict]:
    return {s: {**v, "mase": v["mase"] * ratio[s]} for s, v in d.items()}


def reference(ctx: Context, freq: str, method: str) -> dict[int, dict]:
    out = {}
    for r in read_csv(ctx.path(f"{BASE}/{freq}/{method}/total.csv.gz")):
        sm, ma = float(r["sMAPE"]), float(r["MASE"])
        if math.isfinite(sm) and math.isfinite(ma):
            out[int(r["series"])] = {"smape": sm, "mase": ma, "order": None}
    return out


def quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def block(ctx: Context, stem: str, freq_label: str, freq: str, arms: dict[str, dict], base: str,
          summary_rows: list, paired_rows: list) -> list[Cell]:
    nv, ar = reference(ctx, freq, "Naive"), reference(ctx, freq, "autoarima_R")
    ids = set(nv) & set(ar)
    for d in arms.values():
        ids &= set(d)
    ids = sorted(ids)
    allm = {**arms, "auto.arima": ar}
    n2 = (fmean(nv[i]["mase"] for i in ids), fmean(nv[i]["smape"] for i in ids))
    o_ar = owa(fmean(ar[i]["mase"] for i in ids), n2[0], fmean(ar[i]["smape"] for i in ids), n2[1])
    cells, t = [], f"gavea.{stem}"
    for k, d in arms.items():
        s, m = fmean(d[i]["smape"] for i in ids), fmean(d[i]["mase"] for i in ids)
        o = owa(m, n2[0], s, n2[1])
        rk = f"{freq_label}|{k}"
        cells += [Cell(f"{t}.summary", rk, "N", len(ids), 0), Cell(f"{t}.summary", rk, "sMAPE", s, 6),
                  Cell(f"{t}.summary", rk, "MASE", m, 6), Cell(f"{t}.summary", rk, "OWA", o, 6),
                  Cell(f"{t}.summary", rk, "OWA_autoarima_same_series", o_ar, 6)]
        summary_rows.append([stem, freq_label, k, len(ids), s, m, o, o_ar])
    for b in (base, "auto.arima"):
        for k in list(arms) + ["auto.arima"]:
            if k == b:
                continue
            rk = f"{freq_label}|{k}|{b}"
            row = [stem, freq_label, k, b, len(ids)]
            for metric in ("MASE", "sMAPE"):
                key = metric.lower()
                x = [allm[k][i][key] - allm[b][i][key] for i in ids]
                vals = {"mean_diff": fmean(x), "median_diff": quantile(x, 0.5), "q05": quantile(x, 0.05),
                        "q95": quantile(x, 0.95), "pct_lower": 100.0 * sum(v < -TIE for v in x) / len(x),
                        "pct_tie": 100.0 * sum(abs(v) <= TIE for v in x) / len(x),
                        "pct_higher": 100.0 * sum(v > TIE for v in x) / len(x)}
                if allm[k][ids[0]]["order"] is not None and allm[b][ids[0]]["order"] is not None:
                    vals["pct_same_order"] = 100.0 * sum(allm[k][i]["order"] == allm[b][i]["order"] for i in ids) / len(ids)
                for c, v in vals.items():
                    cells.append(Cell(f"{t}.paired", rk, f"{metric}_{c}", v, 6))
                row += [vals["mean_diff"], vals["median_diff"], vals["q05"], vals["q95"], vals["pct_lower"]]
            paired_rows.append(row)
    return cells


def generate(ctx: Context) -> list[Cell]:
    ratio = {f: {int(r["series"]): float(r["ratio_lag1_to_seasonal"])
                 for r in read_csv(ctx.path(f"{GVD}/mase_lag1_to_seasonal_{f}.csv.gz"))}
             for f in ("hourly", "quarterly")}
    F = read_csv(ctx.path(F_FILE))
    f_status = lambda s: "OK" if s.startswith("OK") else s       # campaign F writes OK:<detail>

    def campaign_f(freq, arm):
        return keep_last_ok([{**r, "sid": r["series"]} for r in F if r["freq"] == freq and r["arm"] == arm], f_status)

    def g(pattern, freq=None):
        d = keep_last_ok(load(ctx, pattern))
        return rescale(d, ratio[freq]) if freq in ratio else d

    cells, summary_rows, paired_rows = [], [], []
    # ---- stable objective, six frequencies (order of the 2026-10-17 table) ----------------------
    cells += block(ctx, "stable_objective", "yearly", "yearly",
                   {"mse": campaign_f("yearly", "mse"), "stable_050": campaign_f("yearly", "stable_050"),
                    "stable_090": campaign_f("yearly", "stable_090")}, "mse", summary_rows, paired_rows)
    cells += block(ctx, "stable_objective", "quarterly", "quarterly",
                   {"mse": g(r"stable_quarterly/quarterly_mse_[0-9]+\.csv", "quarterly"),
                    "stable_050": g(r"stable_quarterly/quarterly_stable_[0-9]+\.csv", "quarterly")},
                   "mse", summary_rows, paired_rows)
    mm = g(r"stable_monthly/stm4_mse_0\.9_[0-9]+\.csv")
    cells += block(ctx, "stable_objective", "monthly", "monthly",
                   {"mse": mm, "stable_090": g(r"stable_monthly/stm4_stable_0\.9_[0-9]+\.csv")},
                   "mse", summary_rows, paired_rows)
    cells += block(ctx, "stable_objective", "monthly (alpha=0.5 subset)", "monthly",
                   {"mse": mm, "stable_050": g(r"stable_monthly/stm4_stable_0\.5_[0-9]+\.csv")},
                   "mse", summary_rows, paired_rows)
    cells += block(ctx, "stable_objective", "weekly", "weekly",
                   {"mse": campaign_f("weekly", "mse"), "stable_050": campaign_f("weekly", "stable_050"),
                    "stable_090": campaign_f("weekly", "stable_090")}, "mse", summary_rows, paired_rows)
    cells += block(ctx, "stable_objective", "daily", "daily",
                   {"mse": g(r"stable_daily/daily_mse_[0-9]+\.csv"),
                    "stable_050": g(r"stable_daily/daily_stable_[0-9]+\.csv")}, "mse", summary_rows, paired_rows)
    cells += block(ctx, "stable_objective", "hourly", "hourly",
                   {"mse": g(r"stable_hourly/hourly_mse_[0-9]+\.csv", "hourly"),
                    "stable_050": g(r"stable_hourly/hourly_stable_[0-9]+\.csv", "hourly")},
                   "mse", summary_rows, paired_rows)
    # ---- 2x2x2x2 admissibility census, monthly ------------------------------------------------
    cells += block(ctx, "admissibility_grid", "monthly", "monthly",
                   {c: g(rf"censo_2x2x2x2/{c}_[0-9]+\.csv") for c in CENSUS}, "gs", summary_rows, paired_rows)
    # ---- over-differencing guard --------------------------------------------------------------
    for freq in ("hourly", "quarterly", "daily"):
        cells += block(ctx, "require_terms_guard", freq, freq,
                       {"guard_off": g(rf"stable_{freq}/{freq}_mse_[0-9]+\.csv", freq),
                        "guard_on": g(rf"requireterms/{freq}_mse_[0-9]+\.csv")},
                       "guard_off", summary_rows, paired_rows)

    write_csv(ctx.out / "summary.csv", ["comparison", "frequency", "arm", "N", "sMAPE", "MASE", "OWA",
                                        "OWA_autoarima_same_series"], summary_rows)
    write_csv(ctx.out / "paired_differences.csv",
              ["comparison", "frequency", "objective", "baseline", "N"]
              + [f"{m}_{c}" for m in ("MASE", "sMAPE") for c in ("mean_diff", "median_diff", "q05", "q95", "pct_lower")],
              paired_rows)
    return cells
