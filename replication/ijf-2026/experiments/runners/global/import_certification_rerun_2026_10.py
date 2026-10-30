"""Convert the batch author's re-run of the August 2026 certification batch into the runner's records.

The batch's author re-ran the 46 instances on 2026-10-21 with the batch's harness (model, bounds,
warm start and 600 s limit unchanged, recording extended; `received/diff_vs_snapshot_22-08_1435.patch`)
in the batch's environment, one case per process under a 900 s external clock. This writes that
run in the format of `run_certification_2026_08.jl`, so that `compare_certification_2026_08.py`
reads it like a run of this package:

    python import_certification_rerun_2026_10.py RERUN_DIR [OUT_DIR]

RERUN_DIR is the archived re-run (results/reruns/certification-2026-08-rerun-2026-10-21); the
received files are read from RERUN_DIR/received and the records written to
OUT_DIR/raw/certification-2026-08/records.csv (OUT_DIR defaults to RERUN_DIR).

* A finished case is its `saida/caso_<id>.csv` row, column for column (`ipopt_t` -> `t_ipopt`,
  `scip_nos` -> `scip_nodes`); SCIP's primal status was not recorded and stays empty; SCIP's and
  Julia's versions are read from the case's log.
* A case killed by the external clock (ids 34, 35, 38: `saida/` empty) keeps a row with the
  status the runner gives it, `KILLED_EXTERNAL_CLOCK_900s`, and the Ipopt objective and time its
  log printed (4 decimals). The second attempt with a 3600 s clock is noted, not used.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPL = Path(__file__).resolve().parents[3]
INSTANCES = REPL / "experiments" / "configs" / "certification_2026_08_instances.csv"
COLUMNS = ["id", "cel", "rotulo", "P", "Q", "T", "rep", "modo", "sid",
           "obj_ipopt", "obj_scip", "dual_scip", "status", "t_scip", "gap_rel", "gap_certificado", "piso_ok",
           "ipopt_status", "t_ipopt", "ipopt_phi", "ipopt_theta", "ipopt_Phi", "ipopt_Theta",
           "scip_raw_status", "scip_primal_status", "scip_nodes", "scip_solve_time",
           "scip_phi", "scip_theta", "scip_Phi", "scip_Theta", "scip_version", "julia_version", "note"]
RENAMED = {"t_ipopt": "ipopt_t", "scip_nodes": "scip_nos"}
HARD = 900


def versions(log: str) -> tuple[str, str]:
    scip = re.search(r"limite \d+s, SCIP (\S+)", log)
    julia = re.search(r"julia=(\S+)", log)
    return (scip.group(1) if scip else "", julia.group(1) if julia else "")


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    run = Path(sys.argv[1])
    rec = run / "received"
    out_dir = Path(sys.argv[2]) if len(sys.argv) == 3 else run
    inst = list(csv.DictReader(open(INSTANCES, newline="", encoding="utf-8")))
    rows = []
    for c in inst:
        i = int(c["id"])
        out = rec / "saida" / f"caso_{i}.csv"
        lines = out.read_text(encoding="utf-8").splitlines() if out.exists() else []
        if len(lines) >= 2:
            s = next(csv.DictReader(lines))
            log = (rec / "logs" / f"caso_{i}.log").read_text(encoding="utf-8")
            sv, jv = versions(log)
            row = {k: s.get(RENAMED.get(k, k), "") for k in COLUMNS}
            row.update(scip_primal_status="", scip_version=sv, julia_version=jv,
                       note=f"received/saida/caso_{i}.csv")
        else:
            log = (rec / "logs" / f"caso_{i}_clock900.log").read_text(encoding="utf-8")
            sv, jv = versions(log)
            m = re.search(r"ipopt OK em ([\d.]+)s \(obj ([\d.]+)\)", log)
            if not m or f"MORTO POR RELOGIO EXTERNO apos {HARD}s" not in log:
                print(f"case {i}: neither a record nor a {HARD} s kill in its log")
                return 1
            row = dict.fromkeys(COLUMNS, "")
            row.update(id=c["id"], cel=c["cell"], rotulo=c["label"], P=c["P"], Q=c["Q"], T=c["T"],
                       rep=c["draw"], modo=c["mode"], sid=c["sid"], obj_ipopt=m.group(2),
                       status=f"KILLED_EXTERNAL_CLOCK_{HARD}s", t_ipopt=m.group(1), ipopt_status="",
                       scip_version=sv, julia_version=jv,
                       note=(f"killed by the {HARD} s clock (received/logs/caso_{i}_clock900.log); Ipopt objective "
                             f"and time as the log printed them; a second attempt under a 3600 s clock was "
                             f"killed too (received/logs/caso_{i}.log)"))
        for k, v in (("cel", "cell"), ("T", "T"), ("modo", "mode"), ("sid", "sid")):
            if c[v] and str(row[k]) != c[v]:  # the M4 series' T is their length, left empty in the instances
                print(f"case {i}: {k} = {row[k]} in the re-run, {c[v]} in the instances")
                return 1
        rows.append(row)
    dest = out_dir / "raw" / "certification-2026-08"
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "records.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    done = sum(not r["status"].startswith("KILLED") for r in rows)
    print(f"records: {dest / 'records.csv'} ({len(rows)} cases, {done} finished)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
