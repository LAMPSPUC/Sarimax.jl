"""Write results/expected/<group>.csv from the historical tables in the artifact bundle.

The expected values are PARSED from the tables as they were printed at the time
(historical-tables/ in the bundle). They are never computed by the generators that are
checked against them; that would make the comparison circular.

Each module of scripts/maintainers/expected/ handles one group and returns rows with
table, row, column, expected (the printed text), decimals (printed precision; empty for
text) and source (the bundle file the value was read from).

Usage (maintainers):
    python extract_expected.py --bundle ../../artifacts [--group m4_benchmark]
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "expected"
GROUPS = ["battery", "m4_benchmark", "m4_paired", "m4_objectives_controlled", "stable_admissibility"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--group", action="append")
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    OUT.mkdir(parents=True, exist_ok=True)
    for group in args.group or GROUPS:
        try:
            mod = importlib.import_module(f"expected.{group}")
        except ModuleNotFoundError:
            print(f"{group}: no extractor yet")
            continue
        rows = mod.extract(args.bundle)
        keys = set()
        for r in rows:
            k = (r["table"], r["row"], r["column"])
            if k in keys:
                raise SystemExit(f"{group}: duplicate expected cell {k}")
            keys.add(k)
        with open(OUT / f"{group}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ["table", "row", "column", "expected", "decimals", "source"],
                               lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"{group}: {len(rows)} expected cells -> {OUT / (group + '.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
