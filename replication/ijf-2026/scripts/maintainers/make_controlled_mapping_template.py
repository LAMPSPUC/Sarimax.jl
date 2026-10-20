"""Write the transcription template for the items the 2026-10-19 revision takes from the
controlled objective campaign (experiment m4_objectives).

The 2026-10-16 revision prints Tables 5, 6, S9, S10, Figure 3 and the sample sizes of Section 5.4
from the historical objective campaigns (m4_objectives_icarai_*). The 2026-10-19 revision is
the controlled one: those items come from m4_objectives, whose cells the table group
m4_objectives_controlled regenerates. This script re-points every 2026-10-16 row of those items
to the controlled cell that holds the same statistic, with the printed value left EMPTY: it is
entered from the 2026-10-19 source, never copied from the regenerated values. Rows whose
printed value is empty are reported as AWAITING TRANSCRIPTION; once entered they are compared
with the reference values like every other row.

Usage: python make_controlled_mapping_template.py    (writes results/manuscript/pending/...)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPL = Path(__file__).resolve().parents[2]
SRC = REPL / "results" / "manuscript" / "manuscript_values.csv"
OUT = REPL / "results" / "manuscript" / "pending" / "manuscript_values_2026-10-19_controlled.csv"
GROUP = "m4_objectives_controlled"
LABEL = {"huber": "Huber", "mae": "MAE", "ridge": "Ridge"}


def target(m: dict) -> tuple[str, str, str] | None:
    """(source, source_row, source_column) of the controlled cell for a 2026-10-16 row."""
    item, row, col = m["item"], m["row"], m["column"]
    if item == "Table 5":
        freq = row.split("(")[-1].rstrip(")") if row.startswith("caption") else row
        return f"{GROUP}/controlled.table5", freq.lower(), col
    if item == "Table 6":
        freq, obj = row.split("|")
        return f"{GROUP}/controlled.table6", f"{freq.lower()}|{obj}", col
    if item in ("Table S9", "Table S10"):
        freq, obj = row.split("|")
        metric = "MASE" if item == "Table S9" else "sMAPE"
        return f"{GROUP}/controlled.paired", f"{freq.lower()}|{obj}|{metric}", "q50" if col == "Median" else col
    if item == "Figure 3":
        freq, obj = row.split("|")
        return f"{GROUP}/controlled.table6", f"{freq}|{LABEL[obj]}", "N"
    if item == "Section 5.4":
        return f"{GROUP}/controlled.table5", row.rsplit(", ", 1)[1], "N"
    return None


def main() -> int:
    with open(SRC, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        cols = list(rows[0])
    expected = {}
    with open(REPL / "results" / "expected" / f"{GROUP}.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            expected[(r["table"], r["row"], r["column"])] = r
    out, seen = [], set()
    for m in rows:
        t = target(m)
        if t is None:
            continue
        src, srow, scol = t
        key = (src.split("/", 1)[1], srow, scol)
        if key not in expected:
            raise SystemExit(f"{m['item']} {m['row']} / {m['column']}: no controlled cell {key}")
        # Table 5 is mapped twice in the 2026-10-16 file (two historical sources); keep one row.
        ident = (m["item"], m["row"], m["column"])
        if ident in seen:
            continue
        seen.add(ident)
        out.append({**m, "printed": "", "decimals": m["decimals"], "source_kind": "regenerated",
                    "source": src, "source_row": srow, "source_column": scol,
                    "note": (f"2026-10-19 controlled revision: printed value to be transcribed from its source; "
                             f"the 2026-10-16 revision printed {m['printed']} from the historical campaigns")})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    counts: dict[str, int] = {}
    for r in out:
        counts[r["item"]] = counts.get(r["item"], 0) + 1
    print(f"{len(out)} rows -> {OUT.relative_to(REPL)}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
