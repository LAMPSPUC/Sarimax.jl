"""Expected cells of the `m4_objectives_controlled` group, parsed from the tables that the
controlled campaign's independent scorer (original-scripts/m4-objectives-controlled-2026-10/
scoring/score_campaign.py) wrote on 2026-10-19, archived in bundle part 2 under
derived/m4-objectives-controlled-2026-10/:

* table5_controlled.tex (OWA at the 4 printed decimals, N) and table5_controlled.csv (mean
  sMAPE/MASE, kept here to 6 decimals);
* table6_controlled.tex (N, mean, q05, median, q95 of the MASE difference, 3 printed decimals)
  and objective_paired_differences.csv (shares lower / tied / higher and same order, 2 decimals);
* objective_paired_differences.csv, again, for the mean and quantiles of the paired MASE and
  sMAPE differences against MSE (6 decimals);
* objective_distribution_quantiles.csv (mean and quantiles of MASE and sMAPE per method, 6 decimals);
* table3_weekly_reconciliation.csv (weekly MSE: the controlled cell and the Table 3 protocol run).

The scorer computed sMAPE and MASE from the archived forecasts and the M4 test values with its
own code; the generator checked against these values uses the per-series metrics of the bundle.
"""

from __future__ import annotations

import csv
from pathlib import Path

D = "derived/m4-objectives-controlled-2026-10"
FREQS = ("monthly", "quarterly", "yearly", "weekly")
LABELS = ("MSE", "Huber", "MAE", "Ridge")


def cell(table, row, column, value, decimals, source):
    return {"table": table, "row": row, "column": column, "expected": value,
            "decimals": "" if decimals is None else str(decimals), "source": source}


def tex_rows(path: Path) -> list[list[str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.rstrip().endswith(r"\\") and "&" in line and not line.lstrip().startswith(("Frequency", r"\toprule")):
            out.append([t.strip().replace("{,}", "") for t in line.rstrip()[:-2].split("&")])
    return out


def table5(bundle: Path) -> list[dict]:
    src = f"{D}/table5_controlled.tex"
    rows = []
    for tok in tex_rows(bundle / src):
        freq = tok[0].lower()
        rows.append(cell("controlled.table5", freq, "N", tok[1], 0, src))
        for col, v in zip(("Reference",) + LABELS, tok[2:]):
            rows.append(cell("controlled.table5", freq, col, v, len(v.split(".")[1]), src))
    src = f"{D}/table5_controlled.csv"
    with open(bundle / src, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for m in ("sMAPE", "MASE"):
                for who in ("Reference", "Naive2") + LABELS:
                    rows.append(cell("controlled.table5_means", r["frequency"], f"{who}.{m}",
                                     f"{float(r[f'{who}_{m}']):.6f}", 6, src))
    return rows


def table6(bundle: Path) -> list[dict]:
    src = f"{D}/table6_controlled.tex"
    rows, freq = [], None
    for tok in tex_rows(bundle / src):
        freq = tok[0].lower() or freq
        rk = f"{freq}|{tok[1]}"
        rows.append(cell("controlled.table6", rk, "N", tok[2], 0, src))
        for col, v in zip(("Mean", "q05", "Median", "q95"), tok[3:]):
            rows.append(cell("controlled.table6", rk, col, v, len(v.split(".")[1]), src))
    src = f"{D}/objective_paired_differences.csv"
    with open(bundle / src, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["baseline"] != "MSE" or r["objective"] not in LABELS:
                continue
            rk = f"{r['frequency']}|{r['objective']}"
            for metric in ("MASE", "sMAPE"):
                for k in ("pct_lower", "pct_tie", "pct_higher"):
                    rows.append(cell("controlled.table6", rk, f"{metric}.{k}", f"{float(r[f'{metric}_{k}']):.2f}", 2, src))
            rows.append(cell("controlled.table6", rk, "pct_same_order", f"{float(r['MASE_pct_same_order']):.2f}", 2, src))
    return rows


def paired(bundle: Path) -> list[dict]:
    src = f"{D}/objective_paired_differences.csv"
    rows = []
    with open(bundle / src, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["baseline"] != "MSE" or r["objective"] not in LABELS:
                continue
            for metric in ("MASE", "sMAPE"):
                rk = f"{r['frequency']}|{r['objective']}|{metric}"
                rows.append(cell("controlled.paired", rk, "N", r["N"], 0, src))
                rows.append(cell("controlled.paired", rk, "mean", f"{float(r[f'{metric}_mean_diff']):.6f}", 6, src))
                for q in ("q01", "q05", "q25", "q75", "q95", "q99"):
                    rows.append(cell("controlled.paired", rk, q, f"{float(r[f'{metric}_{q}']):.6f}", 6, src))
                rows.append(cell("controlled.paired", rk, "q50", f"{float(r[f'{metric}_median_diff']):.6f}", 6, src))
    return rows


def quantiles(bundle: Path) -> list[dict]:
    src = f"{D}/objective_distribution_quantiles.csv"
    rows = []
    with open(bundle / src, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rk = f"{r['frequency']}|{r['metric']}|{r['method']}"
            for c in ("mean", "q01", "q05", "q25", "q50", "q75", "q95", "q99"):
                rows.append(cell("controlled.quantiles", rk, c, f"{float(r[c]):.6f}", 6, src))
    return rows


def weekly(bundle: Path) -> list[dict]:
    src = f"{D}/table3_weekly_reconciliation.csv"
    names = {"controlled P2 (14dd581)": "controlled_P2",
             "controlled-environment P1 extra run (14dd581)": "extra_run_P1_table3_protocol"}
    rows = []
    with open(bundle / src, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source"] not in names:
                continue
            k = names[r["source"]]
            rows += [cell("controlled.weekly", k, "N", r["N"], 0, src),
                     cell("controlled.weekly", k, "OWA", f"{float(r['OWA']):.4f}", 4, src),
                     cell("controlled.weekly", k, "same_order_as_Table3", r["same_order_as_Table3"], 0, src),
                     cell("controlled.weekly", k, "identical_to_Table3", r["identical_to_Table3"], 0, src)]
    if len(rows) != 8:
        raise SystemExit(f"{src}: expected the controlled and the extra-run rows")
    return rows


def extract(bundle: Path) -> list[dict]:
    return table5(bundle) + table6(bundle) + paired(bundle) + quantiles(bundle) + weekly(bundle)
