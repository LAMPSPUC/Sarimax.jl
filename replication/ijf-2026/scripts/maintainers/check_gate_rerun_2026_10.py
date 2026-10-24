"""Check the archived re-run of the paired 1,500-series initialization gate (Table S9).

results/reruns/gate-1500-f90e554-2026-10-22/ holds a new execution of the gate of 2026-08-21 at
commit f90e554 (RUN.md). This script verifies, without the artifact bundle:

* every file of the directory against its SHA256SUMS;
* that the aggregate the run printed (output/agregado.txt, the August reader le_gate.jl) follows
  from the per-series records m1500_pen.csv and m1500_inov.csv, recomputed here independently:
  mean sMAPE and MASE over the paired series, the number of series with status OK, and the share
  of SEARCHES at the iteration ceiling. Each record is one automatic model-selection search (one
  series); its `iterMax` is the largest Ipopt iteration count among the candidate models that
  search fitted, so `iterMax >= 200` means that at least one candidate reached the 200-iteration
  limit, not that the search or its selected model failed;
* that the per-series validation of the `:innovations` arm against the August per-series record
  (output/valida_inov.csv) counts what RUN.md states;
* that manuscript_sources.csv, the values the manuscript mapping compares printed numbers with,
  follows from these files (`--write` regenerates it), together with the August aggregate kept
  in received/referencia/.

It re-runs nothing. Exit 0 when everything agrees, 1 otherwise.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import statistics
import sys
from pathlib import Path

REPL = Path(__file__).resolve().parents[2]
RUN = REPL / "results" / "reruns" / "gate-1500-f90e554-2026-10-22"
SOURCES = RUN / "manuscript_sources.csv"
CEILING = 200


def records(name: str) -> dict[str, dict]:
    with open(RUN / "output" / name, newline="", encoding="utf-8") as fh:
        return {r["sid"]: r for r in csv.DictReader(fh)}


def check_sums(problems: list[str]) -> int:
    n = 0
    for line in (RUN / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        p = RUN / name.lstrip("*")
        n += 1
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            problems.append(f"{name}: missing or SHA-256 differs from SHA256SUMS")
    return n


def arm(recs: dict[str, dict], ids: list[str]) -> dict:
    it = [float(recs[i]["iterMax"]) for i in ids]
    return {"smape": statistics.fmean(float(recs[i]["smape"]) for i in ids),
            "mase": statistics.fmean(float(recs[i]["mase"]) for i in ids),
            "ceiling_pct": 100 * sum(x >= CEILING for x in it) / len(it),
            "ok": sum(recs[i]["status"] == "OK" for i in ids),
            "wall": sum(float(recs[i]["wall"]) for i in ids)}


def derive(problems: list[str]) -> list[dict]:
    pen, inov = records("m1500_pen.csv"), records("m1500_inov.csv")
    ids = sorted(set(pen) & set(inov), key=int)
    if len(ids) != 1500:
        problems.append(f"{len(ids)} paired series, 1,500 expected")
    a = {"penalized": arm(pen, ids), "innovations": arm(inov, ids)}
    printed = (RUN / "output" / "agregado.txt").read_text(encoding="utf-8")
    for key, v in a.items():
        m = re.search(rf"^:{key}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+(\d+)", printed, re.M)
        if not m:
            problems.append(f"agregado.txt: no row for :{key}")
            continue
        got = (f"{v['smape']:.4f}", f"{v['mase']:.4f}", f"{v['ceiling_pct']:.1f}", str(v["ok"]))
        if got != (m.group(1), m.group(2), m.group(4), m.group(5)):
            problems.append(f":{key}: recomputed {got} != printed {(m.group(1), m.group(2), m.group(4), m.group(5))}")
    with open(RUN / "output" / "valida_inov.csv", newline="", encoding="utf-8") as fh:
        val = list(csv.DictReader(fh))
    same = sum(r["igual_6casas"] == "True" for r in val)
    differ = sorted((r["sid"] for r in val if r["igual_6casas"] != "True"), key=int)
    out: list[dict] = []

    def put(quantity, value, rule):
        out.append({"quantity": quantity, "value": value if isinstance(value, str) else repr(value), "rule": rule})

    for key, v in a.items():
        put(f"gate_rerun_{key}_smape", v["smape"], f"mean sMAPE over the {len(ids)} paired series, :{key}")
        put(f"gate_rerun_{key}_mase", v["mase"], f"mean MASE over the {len(ids)} paired series, :{key}")
        put(f"gate_rerun_{key}_ceiling_pct", v["ceiling_pct"],
            f"percentage of searches (series) whose iterMax, the largest candidate iteration count, is >= {CEILING}")
        put(f"gate_rerun_{key}_ok", v["ok"], "series with status OK")
    put("gate_rerun_paired", len(ids), "series in both arms")
    put("gate_rerun_innovations_identical", same,
        "series whose order and sMAPE/MASE (6 decimals) equal the August per-series record")
    put("gate_rerun_innovations_divergent", len(differ), "series that differ: " + " ".join(differ))
    hist = (RUN / "received" / "referencia" / "agosto_gate_1500_agregado.txt").read_text(encoding="utf-8")
    for key in a:
        m = re.search(rf"^:{key}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+(\d+)", hist, re.M)
        if not m:
            problems.append(f"August aggregate: no row for :{key}")
            continue
        put(f"gate_august_{key}_smape", m.group(1), f"mean sMAPE, :{key}, August aggregate (per-series outputs lost)")
        put(f"gate_august_{key}_mase", m.group(2), f"mean MASE, :{key}, August aggregate")
        put(f"gate_august_{key}_ceiling_pct", m.group(4),
            f"percentage of searches at the {CEILING}-iteration ceiling, :{key}, August aggregate")
    return out


def main() -> int:
    write = "--write" in sys.argv[1:]
    problems: list[str] = []
    n = check_sums(problems)
    buf = io.StringIO()
    w = csv.DictWriter(buf, ["quantity", "value", "rule"], lineterminator="\n")
    w.writeheader()
    w.writerows(derive(problems))
    if write:
        SOURCES.write_text(buf.getvalue(), encoding="utf-8", newline="\n")
    elif not SOURCES.exists() or SOURCES.read_text(encoding="utf-8") != buf.getvalue():
        problems.append(f"{SOURCES.name} does not follow from the records (run with --write)")
    for p in problems:
        print("FAIL", p)
    print(f"gate re-run 2026-10-22: {n} files checked; aggregates recomputed from the per-series records; "
          f"{SOURCES.name}: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
