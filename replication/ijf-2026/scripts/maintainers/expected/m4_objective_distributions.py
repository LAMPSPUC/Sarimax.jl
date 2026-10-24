"""Expected cells of the `m4_objective_distributions` group, parsed from the tables of the
October 2026 objective-distribution pipeline as they were generated (2026-10, ForecastTester
`table7_results/tables/`, now historical-tables/m4-objective-distributions-2026-10/tables/):

* reproduction.csv (s3_reproduce.py): both candidate sets against the published totals;
* 4a_marginal.csv, 4b_paired.csv, 4c_tail.csv, 4c_tail_ratios.csv, 4d_union_tail.csv
  (s4_compile.py): the `archived` set, the only one compiled.

Row keys are the identifying columns joined by `|`; column keys are the CSV column names.

The original scripts wrote these files with csv.DictWriter, which prints floats at full
precision (Python repr): there is no printed rounding to compare at. Float cells are therefore
compared at 6 decimals (|generated - expected| <= 5e-7, far below the 4 decimals of the
published totals); integer cells (N, N_published, U_size) exactly; text cells (`yes`/`no`,
`n/a`, `not recorded`) as text.

recompute_check.csv (s2: needs the per-series forecasts and the M4 test values) and
inventory.csv (s1: files on the original host) are not expected cells of this group.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

DIR = "historical-tables/m4-objective-distributions-2026-10/tables"
TABLES = {
    "reproduction": ("set", "frequency", "method"),
    "4a_marginal": ("set", "frequency", "metric", "method"),
    "4b_paired": ("set", "frequency", "metric", "objective", "reference"),
    "4c_tail": ("set", "frequency", "metric", "method"),
    "4c_tail_ratios": ("set", "frequency", "metric", "method", "reference"),
    "4d_union_tail": ("set", "frequency", "metric", "method"),
}
FULL_PRECISION_DECIMALS = "6"
INT = re.compile(r"^[+-]?\d+$")
FLOAT = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$|^[+-]?(nan|inf)$", re.IGNORECASE)


def decimals(tok: str) -> str:
    if INT.match(tok):
        return "0"
    if FLOAT.match(tok):
        return FULL_PRECISION_DECIMALS
    return ""


def table(bundle: Path, name: str, keys: tuple[str, ...]) -> list[dict]:
    rel = f"{DIR}/{name}.csv"
    rows = []
    with open(bundle / rel, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = "|".join(r[k] for k in keys)
            for column, tok in r.items():
                if column in keys:
                    continue
                rows.append({"table": f"distributions.{name}", "row": row, "column": column,
                             "expected": tok, "decimals": decimals(tok), "source": rel})
    return rows


def extract(bundle: Path) -> list[dict]:
    return [r for name, keys in TABLES.items() for r in table(bundle, name, keys)]
