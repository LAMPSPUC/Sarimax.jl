"""Expected cells of the `stable_admissibility_gavea` group, parsed from the tables written on
2026-10-17 by y_updated_results.py (pandas, from the raw archive and campaign F), archived
in bundle part 2 under derived/stable-admissibility-gavea-2026-09/tables-2026-10-17/:
<stem>_summary.csv and <stem>_paired_differences.csv for the stable objective, the 2x2x2x2
admissibility census and the over-differencing guard. The CSVs carry full precision; the cells
are compared at 6 decimals.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

D = "derived/stable-admissibility-gavea-2026-09/tables-2026-10-17"
STEMS = ("stable_objective", "admissibility_grid", "require_terms_guard")
PAIRED = ("mean_diff", "median_diff", "q05", "q95", "pct_lower", "pct_tie", "pct_higher", "pct_same_order")


def cell(table, row, column, value, decimals, source):
    return {"table": table, "row": row, "column": column, "expected": value, "decimals": str(decimals),
            "source": source}


def extract(bundle: Path) -> list[dict]:
    rows = []
    for stem in STEMS:
        src = f"{D}/{stem}_summary.csv"
        with open(bundle / src, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rk = f"{r['frequency']}|{r['arm']}"
                rows.append(cell(f"gavea.{stem}.summary", rk, "N", r["N"], 0, src))
                for c in ("sMAPE", "MASE", "OWA", "OWA_autoarima_same_series"):
                    rows.append(cell(f"gavea.{stem}.summary", rk, c, f"{float(r[c]):.6f}", 6, src))
        src = f"{D}/{stem}_paired_differences.csv"
        with open(bundle / src, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rk = f"{r['frequency']}|{r['objective']}|{r['baseline']}"
                for metric in ("MASE", "sMAPE"):
                    for c in PAIRED:
                        v = r.get(f"{metric}_{c}", "")
                        if v in ("", None) or math.isnan(float(v)):
                            continue          # no selected order for auto.arima
                        rows.append(cell(f"gavea.{stem}.paired", rk, f"{metric}_{c}", f"{float(v):.6f}", 6, src))
    return rows
