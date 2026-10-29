"""Expected cells of the `stable_admissibility` group, parsed from the tables written on
2026-10-17 (pandas, from the raw archive of the September 2026 campaigns and campaign F),
archived under historical-tables/stable-admissibility-2026-09/: <stem>_summary.csv and
<stem>_paired_differences.csv for the stable objective and the 2x2x2x2 admissibility grid.
The CSVs carry full precision; the cells are compared at 6 decimals.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

D = "historical-tables/stable-admissibility-2026-09"
STEMS = ("stable_objective", "admissibility_grid")
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
                rows.append(cell(f"stable_admissibility.{stem}.summary", rk, "N", r["N"], 0, src))
                for c in ("sMAPE", "MASE", "OWA", "OWA_autoarima_same_series"):
                    rows.append(cell(f"stable_admissibility.{stem}.summary", rk, c, f"{float(r[c]):.6f}", 6, src))
        src = f"{D}/{stem}_paired_differences.csv"
        with open(bundle / src, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rk = f"{r['frequency']}|{r['objective']}|{r['baseline']}"
                for metric in ("MASE", "sMAPE"):
                    for c in PAIRED:
                        v = r.get(f"{metric}_{c}", "")
                        if v in ("", None) or math.isnan(float(v)):
                            continue          # no selected order for auto.arima
                        rows.append(cell(f"stable_admissibility.{stem}.paired", rk, f"{metric}_{c}", f"{float(v):.6f}", 6, src))
    return rows
