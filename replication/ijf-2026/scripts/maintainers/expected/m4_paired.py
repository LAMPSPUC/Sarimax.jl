"""Expected cells of the `m4_paired` group, parsed from the tables of the September 2026
revision analysis (`b_m4_distributional.py`, 2026-09-28), which are archived in this repository
(results/historical/m4-paired-2026-09/; origin in results/historical/PROVENANCE.md), not in bundle v1:

* m4_paired_differences_full.csv: paired differences Sarimax.jl - auto.arima;
* m4_distribution_quantiles_full.csv: marginal distributions of both methods.

The CSVs print full float precision; values are compared at 6 decimals, as for the other
full-precision historical tables. The `bundle` argument is not used.
"""

from __future__ import annotations

import csv
from pathlib import Path

HIST = Path(__file__).resolve().parents[3] / "results" / "historical" / "m4-paired-2026-09"
REL = "results/historical/m4-paired-2026-09"
PAIRED = ["N_paired", "mean_diff", "median_diff", "q01", "q05", "q25", "q75", "q95", "q99",
          "frac_lt0", "frac_gt0", "frac_tie", "frac_tie_tol1e-06", "frac_tie_tol0.01",
          "share_abs_disagreement_top1pct", "share_abs_disagreement_top5pct", "mean_diff_trimmed_1_99"]
MARGINAL = ["N", "mean", "q01", "q05", "q25", "q50", "q75", "q95", "q99"]


def extract(bundle: Path) -> list[dict]:
    rows = []
    src = f"{REL}/m4_paired_differences_full.csv"
    with open(HIST / "m4_paired_differences_full.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = f"{r['frequency']}|{r['metric']}"
            for c in PAIRED:
                rows.append({"table": "m4_paired.paired", "row": row, "column": c, "expected": r[c],
                             "decimals": "0" if c == "N_paired" else "6", "source": src})
    src = f"{REL}/m4_distribution_quantiles_full.csv"
    with open(HIST / "m4_distribution_quantiles_full.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = f"{r['frequency']}|{r['metric']}|{r['method']}"
            for c in MARGINAL:
                rows.append({"table": "m4_paired.marginal", "row": row, "column": c, "expected": r[c],
                             "decimals": "0" if c == "N" else "6", "source": src})
    return rows
