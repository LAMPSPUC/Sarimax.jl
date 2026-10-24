"""Derive the printed quantities of the manuscript that no table group or archived table holds.

The submitted manuscript prints a few values that are simple functions of archived records but
are not a cell of any archived table: the Ipopt times of Table 7 at the precision of the record,
the Stable-minus-MSE OWA column of Table S15, and the shares of series improved by an
alternative objective at the tolerance the text declares (Section 5.4). This script computes
them, with the SHA-256 of every input, into results/manuscript/derived_quantities.csv, the source
the manuscript mapping compares those printed values with (source_kind = derived).

    python derive_manuscript_quantities.py [--bundle DIR] [--write]

`--bundle` is the extracted artifact bundle (both parts, as `./reproduce.sh fetch artifacts`
writes them): part 1 holds the archived Table 7 record
raw/battery-2026-07/solver/global_value_results.jsonl, part 2 the controlled campaign's
per-series records raw/m4-objectives-controlled-2026-10/per_series/.

Without `--bundle` the committed rows of those two inputs are kept, with the SHA-256 of their
inputs listed; without `--write` the script reports whether the committed file follows from
the inputs that were given. The Stable column is always recomputed: its input is in the
repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import gzip
import math
from pathlib import Path

REPL = Path(__file__).resolve().parents[2]
OUT = REPL / "results" / "manuscript" / "derived_quantities.csv"
STABLE = REPL / "results" / "historical" / "updated-2026-10-17" / "tables" / "stable_objective_summary.csv"
T7 = "raw/battery-2026-07/solver/global_value_results.jsonl"
PER_SERIES = "raw/m4-objectives-controlled-2026-10/per_series"
TOL = 1e-6
FIELDS = ["quantity", "value", "input", "input_sha256", "rule"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table7(bundle: Path) -> list[dict]:
    path = bundle / T7
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("solver") == "Ipopt" and r.get("experiment") == "global_value":
            out.append({"quantity": f"t7_ipopt_time|{r['dataset']}", "value": repr(r["runtime_s"]),
                        "input": f"bundle:{T7}", "input_sha256": sha(path),
                        "rule": "Ipopt runtime_s of the archived Table 7 record (first solve includes compilation)"})
    return out


def stable() -> list[dict]:
    rows = list(csv.DictReader(open(STABLE, newline="", encoding="utf-8")))
    mse = {r["frequency"]: float(r["OWA"]) for r in rows if r["arm"] == "mse"}
    out = []
    for r in rows:
        if r["arm"] == "mse":
            continue
        out.append({"quantity": f"s15_delta_owa|{r['frequency']}|{r['arm']}",
                    "value": repr(float(r["OWA"]) - mse[r["frequency"]]),
                    "input": STABLE.relative_to(REPL).as_posix(), "input_sha256": sha(STABLE),
                    "rule": "OWA(stable arm) - OWA(paired MSE arm) of the same campaign"})
    return out


def mase(path: Path) -> dict[str, float]:
    """MASE of every series fitted with status OK and a finite MASE."""
    out = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["status"] == "OK" and r["mase_total"] and math.isfinite(float(r["mase_total"])):
                out[r["series"]] = float(r["mase_total"])
    return out


def shares(bundle: Path) -> list[dict]:
    out = []
    for freq in ("monthly", "quarterly", "yearly", "weekly"):
        ref = bundle / PER_SERIES / f"mse__{freq}.csv.gz"
        b = mase(ref)
        for obj in ("huber", "mae", "ridge"):
            path = bundle / PER_SERIES / f"{obj}__{freq}.csv.gz"
            a = mase(path)
            ids = sorted(set(a) & set(b))
            d = [a[i] - b[i] for i in ids]
            out.append({"quantity": f"s54_pct_lower_tol1e-06|{freq}|{obj}",
                        "value": repr(100 * sum(x < -TOL for x in d) / len(d)),
                        "input": f"bundle:{PER_SERIES}/{obj}__{freq}.csv.gz;bundle:{PER_SERIES}/mse__{freq}.csv.gz",
                        "input_sha256": f"{sha(path)};{sha(ref)}",
                        "rule": f"percentage of the {len(d)} common series with MASE(objective) - MASE(MSE) < -{TOL:g}"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    committed = list(csv.DictReader(open(OUT, newline="", encoding="utf-8"))) if OUT.exists() else []
    keep = lambda prefix: [r for r in committed if r["quantity"].startswith(prefix)]
    rows = (table7(a.bundle) if a.bundle else keep("t7_")) + stable() + \
           (shares(a.bundle) if a.bundle else keep("s54_"))
    buf = io.StringIO()
    w = csv.DictWriter(buf, FIELDS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    if a.write:
        OUT.write_text(buf.getvalue(), encoding="utf-8", newline="\n")
        print(f"wrote {OUT.relative_to(REPL)}: {len(rows)} quantities")
        return 0
    ok = OUT.exists() and OUT.read_text(encoding="utf-8") == buf.getvalue()
    print(f"{OUT.relative_to(REPL)}: " + ("follows from the inputs given" if ok else "DIFFERS from the inputs given"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
