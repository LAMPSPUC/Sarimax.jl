"""Compare a re-run of the August 2026 certification batch with the batch's record.

Reads $IJF_RUN_DIR/raw/certification-2026-08/records.csv (run_certification_2026_08.jl) and the
reconstructed record of the batch (results/historical/certification-2026-08/received/
tabela_scip_lotes1-2_RECONSTRUIDA.csv), and writes next to the records:

* comparison.csv: per instance, the record's and the re-run's status, time, certified gap,
  Ipopt - SCIP, objectives, and the outcome of each comparison;
* summary.md: the outcomes, and the Table S5 counts by cell and the Section 5.6 seasonal-grid
  numbers computed from the re-run with the rules of scripts/maintainers/check_certification_2026_08.py,
  beside the record's. The manuscript's printed values are compared with these records by the
  manuscript mapping, not here.

Objectives the record holds are compared at the precision it printed: Ipopt's always, SCIP's
when both runs certified the instance, and batch 1's Ipopt - SCIP (differences below 1e-10 are
rounding at the Ipopt point and agree). A case killed by the external clock is the same outcome
whether the record wrote it in Portuguese or the runner in English. Statuses, times and the gaps of instances stopped at the
time limit depend on the machine and its load, and are reported, not judged. Exit 1 only when
an Ipopt objective differs (the data or the model are not the batch's) or the records are
missing; otherwise 0.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
import sys
from pathlib import Path

REPL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPL / "scripts" / "maintainers"))
from check_certification_2026_08 import SEASONAL  # noqa: E402

RECORD = REPL / "results" / "historical" / "certification-2026-08" / "received" / "tabela_scip_lotes1-2_RECONSTRUIDA.csv"


def f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def outcome(status: str) -> str:
    """The record writes a case killed by the external clock in Portuguese, the runner in English."""
    s = status.strip()
    return "KILLED_EXTERNAL_CLOCK" if s.startswith(("MORTO_RELOGIO_EXTERNO", "KILLED_EXTERNAL_CLOCK")) else s


def decimals(s: str) -> int:
    s = s.strip().lower()
    if "e" in s:
        mant, exp = s.split("e")
        d = len(mant.split(".")[1]) if "." in mant else 0
        return d - int(exp)
    return len(s.split(".")[1]) if "." in s else 0


def same_printed(recorded: str, value: float) -> bool:
    """`value` prints as `recorded` at the precision `recorded` was printed with."""
    r = f(recorded)
    if math.isnan(r) or math.isnan(value):
        return False
    return abs(value - r) <= 0.5 * 10 ** (-decimals(recorded)) + 1e-12 * max(1.0, abs(r))


def main() -> int:
    run = Path(os.environ.get("IJF_RUN_DIR", ".")) / "raw" / "certification-2026-08"
    recs = run / "records.csv"
    if not recs.exists():
        print(f"no records at {recs}")
        return 1
    new = {int(r["id"]): r for r in csv.DictReader(open(recs, newline="", encoding="utf-8"))}
    old = {int(r["id"]): r for r in csv.DictReader(open(RECORD, newline="", encoding="utf-8"))}
    rows, bad = [], []
    count = {"status same": 0, "status differs": 0, "ipopt agrees": 0, "ipopt differs": 0,
             "scip optimum agrees": 0, "scip optimum differs": 0,
             "ipopt - scip agrees": 0, "ipopt - scip differs": 0}
    for i in sorted(new):
        n, o = new[i], old.get(i, {})
        out = {"id": i, "cell": n["cel"], "label": n["rotulo"], "T": n["T"],
               "status_record": o.get("status", ""), "status_rerun": n["status"],
               "t_record": o.get("t_s", ""), "t_rerun": n["t_scip"],
               "gap_record": o.get("gap_cert", ""), "gap_rerun": n["gap_certificado"],
               "ipopt_minus_scip_record": o.get("ipopt_menos_scip", ""),
               "ipopt_minus_scip_rerun": repr(f(n["obj_ipopt"]) - f(n["obj_scip"])),
               "obj_ipopt_record": o.get("obj_ipopt", ""), "obj_ipopt_rerun": n["obj_ipopt"],
               "obj_scip_record": o.get("obj_scip", ""), "obj_scip_rerun": n["obj_scip"],
               "dual_rerun": n["dual_scip"], "nodes_rerun": n["scip_nodes"], "outcome": ""}
        notes = []
        st = "same" if outcome(o.get("status", "")) == outcome(n["status"]) else "differs"
        count[f"status {st}"] += 1
        if st == "differs":
            notes.append(f"status {o.get('status')} -> {n['status']}")
        if o.get("obj_ipopt"):
            ok = same_printed(o["obj_ipopt"], f(n["obj_ipopt"]))
            count["ipopt agrees" if ok else "ipopt differs"] += 1
            if not ok:
                notes.append("Ipopt objective differs")
                bad.append(i)
        if o.get("obj_scip") and o.get("status") == "OPTIMAL" and n["status"] == "OPTIMAL":
            ok = same_printed(o["obj_scip"], f(n["obj_scip"]))
            count["scip optimum agrees" if ok else "scip optimum differs"] += 1
            if not ok:
                notes.append("certified optimum differs")
        if o.get("ipopt_menos_scip"):
            # batch 1 printed Ipopt - SCIP to 3 significant digits; below 1e-10 both runs ended at
            # the Ipopt point and the digits are rounding, so those agree whatever they print
            d = f(n["obj_ipopt"]) - f(n["obj_scip"])
            ok = same_printed(o["ipopt_menos_scip"], d) or max(abs(d), abs(f(o["ipopt_menos_scip"]))) < 1e-10
            count["ipopt - scip agrees" if ok else "ipopt - scip differs"] += 1
            if not ok:
                notes.append(f"Ipopt - SCIP {o['ipopt_menos_scip']} -> {d:.2e}")
        out["outcome"] = "; ".join(notes) or "agrees"
        rows.append(out)
    with open(run / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    def s5_cells(tab):
        """OPTIMAL / TIME_LIMIT / killed by the external clock, by the harness's cells."""
        out = []
        for cell in sorted({r["cel"] for r in tab.values()}) + ["all"]:
            st = [outcome(r["status"]) for r in tab.values() if cell in ("all", r["cel"])]
            out.append((cell, "/".join(str(st.count(k)) for k in ("OPTIMAL", "TIME_LIMIT", "KILLED_EXTERNAL_CLOCK"))))
        return out

    def seasonal(tab, gapcol, diffcol):
        g = [f(tab[i][gapcol]) for i in SEASONAL if i in tab and not math.isnan(f(tab[i][gapcol]))]
        d = [f(tab[i][diffcol]) for i in SEASONAL if i in tab and not math.isnan(f(tab[i][diffcol]))]
        return {"s56_seasonal_ipopt_minus_scip_max": max(d) if d else math.nan,
                "s56_seasonal_gap_min": min(g) if g else math.nan,
                "s56_seasonal_gap_max": max(g) if g else math.nan,
                "s56_seasonal_gap_median": statistics.median(g) if g else math.nan}

    for i, n in new.items():
        n["ipopt_menos_scip"] = repr(f(n["obj_ipopt"]) - f(n["obj_scip"]))
    lines = [f"# Re-run of the August 2026 certification batch: {run.parent.parent.name}", "",
             "| comparison | instances |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in count.items()]
    lines += ["", "## Table S5 (by the harness's cells; OPTIMAL / TIME_LIMIT / killed by the external clock)", "",
              "| cell | record | re-run |", "|---|---|---|"]
    for (k, o), (_, n) in zip(s5_cells(old), s5_cells(new)):
        lines.append(f"| {k} | {o} | {n} |")
    so = seasonal(old, "gap_cert", "ipopt_menos_scip")
    sn = seasonal(new, "gap_certificado", "ipopt_menos_scip")
    lines += ["", "## Section 5.6, seasonal grid (cell B)", "", "| quantity | record | re-run |", "|---|---|---|"]
    lines += [f"| {k} | {so[k]:.3e} | {sn[k]:.3e} |" for k in so]
    lines += ["", "## Section 5.6 instances", "", "| id | record Ipopt / SCIP | re-run Ipopt / SCIP / status / gap |",
              "|---|---|---|"]
    for i in (26, 44, 45):
        if i in new:
            n = new[i]
            lines.append(f"| {i} | {old[i]['obj_ipopt']} / {old[i]['obj_scip']} | {f(n['obj_ipopt']):.6f} / "
                         f"{f(n['obj_scip']):.6f} / {n['status']} / {f(n['gap_certificado']):.2e} |")
    lines += ["", "Per instance: `comparison.csv`. Times and the gaps of time-limited instances depend on "
              "the machine and its load."]
    (run / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    if bad:
        print(f"FAIL: the Ipopt objective differs from the record for ids {bad}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
