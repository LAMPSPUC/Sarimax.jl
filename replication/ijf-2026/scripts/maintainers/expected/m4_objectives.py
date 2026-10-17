"""Expected cells of the `m4_objectives` group, parsed from:

* historical-tables/m4-objectives-icarai-2026-08/table1_horizon_total.txt and
  table2_owa_by_block.txt (ICARAI make_tables.py, 2026-08-28);
* historical-tables/forecasttester-2026-08/tabelas_paper.txt (tabelas_paper.py, 2026-08-26).
"""

from __future__ import annotations

import re
from pathlib import Path

FREQS = ("monthly", "quarterly", "yearly", "weekly")
BLOCKS = ("short", "medium", "long", "total")
NUM = re.compile(r"^[+-]?\d+\.\d+$")


def decimals(tok: str) -> str:
    return str(len(tok.split(".")[1]))


def cell(table, row, column, tok, source):
    return {"table": table, "row": row, "column": column, "expected": tok.lstrip("+"),
            "decimals": decimals(tok), "source": source}


def table1(bundle: Path) -> list[dict]:
    rel = "historical-tables/m4-objectives-icarai-2026-08/table1_horizon_total.txt"
    rows, freq = [], None
    for line in (bundle / rel).read_text(encoding="utf-8").splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] in FREQS:
            freq, tok = tok[0], tok[1:]
        if freq is None or not tok:
            continue
        name, vals = tok[0], tok[1:]
        if name in ("mse", "huber", "mae", "ridge"):
            cols = [f"{i}.{m}" for i in ("zeroed", "innovations") for m in ("MASE", "sMAPE", "OWA", "dOWA")]
            for c, v in zip(cols, vals):
                if NUM.match(v):
                    rows.append(cell("icarai.table1", f"{freq}|{name}", c, v, rel))
        elif name == "auto.arima":
            for c, v in zip(("MASE", "sMAPE", "OWA"), vals):
                rows.append(cell("icarai.table1", f"{freq}|auto.arima", c, v, rel))
        elif name == "Naive2":
            for c, v in zip(("MASE", "sMAPE"), vals):
                rows.append(cell("icarai.table1", f"{freq}|Naive2", c, v, rel))
            n = re.search(r"\(n=(\d+)\)", line).group(1)
            rows.append({"table": "icarai.table1", "row": f"{freq}|Naive2", "column": "n",
                         "expected": n, "decimals": "0", "source": rel})
    return rows


def table2(bundle: Path) -> list[dict]:
    rel = "historical-tables/m4-objectives-icarai-2026-08/table2_owa_by_block.txt"
    rows, freq = [], None
    for line in (bundle / rel).read_text(encoding="utf-8").splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] in FREQS:
            freq, tok = tok[0], tok[1:]
        if freq is None or not tok:
            continue
        name, vals = tok[0], tok[1:]
        if name in ("mse", "huber", "mae", "ridge"):
            cols = [f"{i}.{b}" for i in ("zeroed", "innovations") for b in BLOCKS]
            for c, v in zip(cols, vals):
                if NUM.match(v):
                    rows.append(cell("icarai.table2", f"{freq}|{name}", c, v, rel))
        elif name == "auto.arima":
            for b, v in zip(BLOCKS, vals):
                rows.append(cell("icarai.table2", f"{freq}|auto.arima", b, v, rel))
    return rows


def tabelas_paper(bundle: Path) -> list[dict]:
    rel = "historical-tables/forecasttester-2026-08/tabelas_paper.txt"
    rows, section, freq, current = [], None, None, None
    head = re.compile(r"^([AB])\) .*\((\w+), n=(\d+)\)")
    for line in (bundle / rel).read_text(encoding="utf-8").splitlines():
        m = head.match(line)
        if m:
            section, freq = m.group(1), m.group(2)
            rows.append({"table": f"forecasttester.{section}", "row": freq, "column": "n",
                         "expected": m.group(3), "decimals": "0", "source": rel})
            continue
        tok = line.split()
        if not tok or section is None or tok[0] in ("celula",) or set(line.strip()) == {"-"}:
            continue
        if tok[0] not in BLOCKS:
            current, tok = tok[0], tok[1:]
        block, vals = tok[0], tok[1:]
        tab, row = f"forecasttester.{section}", f"{freq}|{current}|{block}"
        if current == "Naive":
            names = ("MASE", "sMAPE")
        elif current == "auto.arima":
            names = ("MASE", "sMAPE", "OWA")
        else:
            names = ("MASE", "sMAPE", "OWA", "dOWA")
        for c, v in zip(names, vals):
            rows.append(cell(tab, row, c, v, rel))
    return rows


def extract(bundle: Path) -> list[dict]:
    return table1(bundle) + table2(bundle) + tabelas_paper(bundle)
