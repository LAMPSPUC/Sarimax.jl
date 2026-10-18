"""Check results/manuscript/manuscript_values.csv against the historical reference values.

Every `regenerated` manuscript value points at a cell of a table group. `./reproduce.sh tables`
compares it with the REGENERATED cell, which needs the artifact bundle. This script needs only
the repository: it compares each printed value with the reference value of the same cell
(results/expected/<group>.csv, parsed from the tables printed at the time), so that a wrong
mapping or a printed value that no archived table supports is found without the bundle. Run it
after editing the mapping or the expected values; CI runs it on every change.

Usage: python check_manuscript_mapping.py [--report FILE]
Exit codes: 0 every mapped value agrees; 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPL / "scripts"))

from ijflib import manuscript  # noqa: E402


class Ref:
    def __init__(self, value: str):
        self.value = value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, help="write the cell-level comparison to this CSV")
    args = ap.parse_args()
    expected: dict[str, dict] = {}
    for f in sorted((REPL / "results" / "expected").glob("*.csv")):
        with open(f, newline="", encoding="utf-8") as fh:
            expected[f.stem] = {(r["table"], r["row"], r["column"]): Ref(r["expected"])
                                for r in csv.DictReader(fh)}
    rows = manuscript.reconcile(REPL, expected, set())
    # Cells of a table that has no historical counterpart (e.g. m4_paired.text, the range quoted
    # in Section 5.3) can only be checked against the regenerated values, at run time.
    tables = {t for cells in expected.values() for (t, _, _) in cells}
    for r in rows:
        if r["source_kind"] == "regenerated" and r["status"] == manuscript.DIFFER \
                and r["source"].partition("/")[2] not in tables:
            r["status"], r["difference"] = "RUN-TIME CHECK ONLY", "no historical table for this cell"
    if args.report:
        manuscript.write_cells(args.report, rows)
    bad = [r for r in rows if r["status"] in (manuscript.DIFFER, manuscript.DERIVED_DIFFER)]
    # A manuscript value that is reconciled at run time only (no historical table of its own)
    # points at a group without reference values; it is listed, not failed.
    for item, d in manuscript.summarize(rows).items():
        print(f"{item:12s} {d['label'] or '(no label)':34s} " +
              ", ".join(f"{k} {v}" for k, v in d["counts"].items()))
    for r in bad:
        print(f"DIFFER {r['item']} {r['row']} / {r['column']}: printed {r['printed']}, "
              f"reference {r['source_value'] or '(cell not found)'} ({r['difference']})")
    total = {}
    for r in rows:
        total[r["status"]] = total.get(r["status"], 0) + 1
    print(f"{len(rows)} manuscript values: " + ", ".join(f"{k} {v}" for k, v in total.items()))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
