"""Check that the derived tables of the September 2026 simulation scripts follow from the recovered outputs.

results/historical/revision-2026-09/ holds three derived tables that the manuscript prints
(Table S10 from quantile_controlled_example.csv, Table S11 from multistart_kstart.csv, and the
coefficients behind Figure S1 in weighted_regularization_demo.csv). The scripts that produced
them (efg_julia_experiments.jl, parts E, F and G, and efg_figures.py) and the outputs the Julia
script wrote were recovered and are archived in recovered/. This script verifies, without the
artifact bundle:

* every recovered file against recovered/SHA256SUMS;
* Table S10 and the Figure S1 table: equal, value for value, to the summaries the Julia script
  wrote (the derived copies were re-written by pandas, so floats are compared to 1e-12 relative);
* Table S11: recomputed from the per-start records (outputs/multistart_kstart.csv) with the rule of
  efg_figures.py part G (objective values clustered greedily within 1e-6 of the cluster's lowest
  value; converged = LOCALLY_SOLVED, OPTIMAL or ALMOST_LOCALLY_SOLVED), every column.

It does not re-run the simulations; that would be a new execution (efg_julia_experiments.jl runs
against the package). Exit 0 when everything agrees, 1 otherwise.
"""

from __future__ import annotations

import csv
import hashlib
import statistics
import sys
from pathlib import Path

REPL = Path(__file__).resolve().parents[2]
D = REPL / "results" / "historical" / "revision-2026-09"
REC = D / "recovered"
REL_TOL = 1e-6
OK_STATUS = ("LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED")


def rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def close(a: str, b: str, rel: float = 1e-12) -> bool:
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a.strip('"') == b.strip('"')
    if x != x and y != y:
        return True
    return abs(x - y) <= rel * max(abs(x), abs(y), 1e-300) or abs(x - y) <= 1e-15


def same_table(derived: Path, original: Path, problems: list[str]) -> int:
    a, b = rows(derived), rows(original)
    if [list(r) for r in a[:1]] != [list(r) for r in b[:1]] or len(a) != len(b):
        problems.append(f"{derived.name}: header or row count differs from {original.name}")
        return 0
    n = 0
    for i, (ra, rb) in enumerate(zip(a, b)):
        for k in ra:
            n += 1
            if not close(ra[k], rb[k]):
                problems.append(f"{derived.name} row {i + 1} {k}: {ra[k]} != {rb[k]}")
    return n


def cluster(vals: list[float]) -> list[int]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    lab = [0] * len(vals)
    c, anchor = -1, None
    for i in order:
        if anchor is None or (vals[i] - anchor) > REL_TOL * max(abs(anchor), 1e-12):
            c += 1
            anchor = vals[i]
        lab[i] = c
    return lab


def multistart(problems: list[str]) -> int:
    starts = rows(REC / "outputs" / "multistart_kstart.csv")
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in starts:
        groups.setdefault((r["case"], r["initialization"]), []).append(r)
    derived = {(r["case"], r["initialization"]): r for r in rows(D / "multistart_kstart.csv")}
    if set(derived) != set(groups):
        problems.append("multistart_kstart.csv: cases differ from the per-start records")
        return 0
    n = 0
    for key, d in groups.items():
        v = [float(r["objective"]) for r in d]
        lab = cluster(v)
        best = min(v)
        at_best = [l == 0 for l in lab]
        coefs = [[float(x) for x in r["coefficients"].split(";") if x not in ("", "nan")] for r in d]
        best_coefs = [c for c, b in zip(coefs, at_best) if b]
        spread = (max(max(col) - min(col) for col in zip(*best_coefs)) if len(best_coefs) > 1 else 0.0)
        zero = [x for x, r in zip(v, d) if r["start_kind"] == "zero"]
        its = [float(r["iterations"]) for r in d if r["iterations"] not in ("", "nan")]
        got = {
            "starts": len(d), "converged": sum(r["solver_status"] in OK_STATUS for r in d),
            "distinct_objective_values": max(lab) + 1, "share_at_best": sum(at_best) / len(d),
            "rel_range_objective": (max(v) - best) / abs(best),
            "zero_start_rel_gap": (zero[0] - best) / abs(best) if zero else float("nan"),
            "max_coef_spread_within_best": spread, "median_iterations": statistics.median(its),
        }
        for k, x in got.items():
            n += 1
            if not close(repr(float(x)), derived[key][k], rel=1e-9):
                problems.append(f"multistart_kstart.csv {key} {k}: derived {derived[key][k]}, recomputed {x!r}")
    return n


def main() -> int:
    problems: list[str] = []
    listed = 0
    for line in (REC / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        listed += 1
        p = REC / name.strip()
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            problems.append(f"recovered/{name.strip()}: missing or SHA-256 differs")
    n6 = same_table(D / "quantile_controlled_example.csv", REC / "outputs" / "quantile_example_summary.csv", problems)
    nf = same_table(D / "weighted_regularization_demo.csv", REC / "outputs" / "weighted_penalty_table.csv", problems)
    n7 = multistart(problems)
    print(f"recovered files: {listed} listed; Table S10 table: {n6} values; Figure S1 table: {nf} values; "
          f"Table S11 table: {n7} values recomputed from the per-start records")
    for p in problems:
        print("DIFFER", p)
    print("OK" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
