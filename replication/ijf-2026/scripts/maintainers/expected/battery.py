"""Expected cells of the `battery` group, parsed from the tables of the July 2026 battery as
they were generated at the time (combine_results.py, pandas `to_csv`):

    historical-tables/battery-2026-07/table_{validation,forecasting,architecture_checks,
        solver_comparison,solver_scaling,global_value,energy_forecasting}.csv

(the .tex next to each CSV is the same rows rendered for the manuscript).

One expected cell per non-key column of every row. The row key joins the columns that
identify the row (ROW_KEY, the keys scripts/ijftables/battery.py also uses); the column key
is the CSV column name; the expected value is the printed text. Fixed-point numbers carry their
printed decimals and integers 0 decimals. Everything else is compared as text: "--", "NA",
labels, statuses, and the solver-scaling RelGap, which combine_results printed with its own
formatter ("0" for an exact zero, "%.1e" otherwise).
"""

from __future__ import annotations

import csv
import re
import string
from pathlib import Path

DIR = "historical-tables/battery-2026-07"
TABLES = [  # (table id, historical CSV)
    ("battery.validation", "table_validation.csv"),
    ("battery.forecasting", "table_forecasting.csv"),
    ("battery.architecture", "table_architecture_checks.csv"),
    ("battery.solver_comparison", "table_solver_comparison.csv"),
    ("battery.solver_scaling", "table_solver_scaling.csv"),
    ("battery.global_value", "table_global_value.csv"),
    ("battery.energy", "table_energy_forecasting.csv"),
]
ROW_KEY = {
    "battery.validation": "{Dataset}|{Order}|{Implementation}",
    "battery.forecasting": "{Task}|{Implementation}|{Order}",
    "battery.architecture": "{Experiment}|{Setting}",
    "battery.solver_comparison": "{Level}|{Model}|{Solver}|{Obj}",
    "battery.solver_scaling": "T={T}",
    "battery.global_value": "{Instance}|{Solver}",
    "battery.energy": "{Dataset}|{Model}|{Implementation}",
}
TEXT_COLUMNS = {("battery.solver_scaling", "RelGap")}
FIXED = re.compile(r"^[+-]?\d+\.(\d+)$")
INTEGER = re.compile(r"^[+-]?\d+$")


def decimals(text: str) -> str:
    m = FIXED.match(text)
    if m:
        return str(len(m.group(1)))
    return "0" if INTEGER.match(text) else ""


def extract(bundle: Path) -> list[dict]:
    rows = []
    for table, name in TABLES:
        rel = f"{DIR}/{name}"
        key_cols = [f for _, f, _, _ in string.Formatter().parse(ROW_KEY[table]) if f]
        with open(bundle / rel, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in key_cols if c not in (reader.fieldnames or [])]
            if missing:
                raise SystemExit(f"{rel}: no key column(s) {missing}")
            for rec in reader:
                if None in rec or None in rec.values():
                    raise SystemExit(f"{rel}: row with a field count other than the header's: {rec}")
                row = ROW_KEY[table].format_map(rec)
                for column, text in rec.items():
                    if column in key_cols:
                        continue
                    dec = "" if (table, column) in TEXT_COLUMNS else decimals(text)
                    rows.append({"table": table, "row": row, "column": column,
                                 "expected": text, "decimals": dec, "source": rel})
    return rows
