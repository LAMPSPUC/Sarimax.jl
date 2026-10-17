"""Expected cells of the `m4_benchmark` group, parsed from
historical-tables/m4-recreio-2026-08/table_m4.txt (recreio make_table_m4.jl, 2026-08-28)."""

from __future__ import annotations

import re
from pathlib import Path

REL = "historical-tables/m4-recreio-2026-08/table_m4.txt"
COLS = ["sMAPE.ours", "sMAPE.ref", "MASE.ours", "MASE.ref", "OWA.ours", "OWA.ref", "OWA.gap"]


def extract(bundle: Path) -> list[dict]:
    rows, freq = [], None
    for line in (bundle / REL).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z]+)\s+\(m4_innovations_(\w+)\.csv\)", line)
        if m:
            freq = m.group(2)
            continue
        tok = line.split()
        if freq is None or len(tok) != 9 or tok[0] not in ("short", "medium", "long", "total"):
            continue
        row = f"{freq}|{tok[0]}"
        rows.append({"table": "m4_benchmark.table_m4", "row": row, "column": "n",
                     "expected": tok[1], "decimals": "0", "source": REL})
        for c, v in zip(COLS, tok[2:]):
            rows.append({"table": "m4_benchmark.table_m4", "row": row, "column": c,
                         "expected": v.lstrip("+"), "decimals": str(len(v.split(".")[1])), "source": REL})
    return rows
