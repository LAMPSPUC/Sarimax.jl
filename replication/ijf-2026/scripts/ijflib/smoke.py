"""`smoke`: small but real computations in Julia (Ipopt and SCIP), R and Python.

Each engine runs in its own process (SCIP in a process of its own, because a solver crash
cannot be caught from inside Julia). The measured quantities are compared with
experiments/validation/smoke_expected.json. A quantity that is expected but not produced,
an engine that exits non-zero, a reported error, or a value outside its tolerance is a
FAIL. Quantities without an expected value are kept in the report for information.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from . import scip
from .common import (EXIT_FAIL, EXIT_INPUT, EXIT_OK, FAIL, INFO, MISSING, PASS, Paths, Report, julia_cmd,
                     run_logged, tail, which)

TIMEOUT = {"julia": 2400, "julia-scip": 1200, "r": 600, "python": 600}


def _records(log: Path) -> list[dict]:
    out = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{") and '"engine"' in line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _compare(check: dict, value) -> tuple[bool, str]:
    exp, how = check["expected"], check.get("compare", "equal")
    if how == "equal":
        return value == exp, ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, "not a number"
    if not math.isfinite(v):
        return False, "not finite"
    diff = v - float(exp)
    if how == "abs":
        return abs(diff) <= check["tol"], f"{diff:+.3e}"
    if how == "rel":
        return abs(diff) <= check["tol"] * max(abs(float(exp)), 1e-300), f"{diff / float(exp):+.3e} rel"
    return False, f"unknown comparison {how}"


def run(paths: Paths, args) -> int:
    report = Report("smoke", paths)
    val = paths.repl / "experiments" / "validation"
    fixtures = paths.repl / "data" / "fixtures"
    outdir = paths.output / "smoke"
    expected = json.loads((val / "smoke_expected.json").read_text(encoding="utf-8"))["checks"]
    py = os.environ.get("IJF_PYTHON", "python3" if which("python3") else "python")

    engines = [
        ("julia", julia_cmd(paths) + [str(val / "smoke.jl"), "airline_ipopt", "ar1_css", "objectives", "admissibility"],
         os.environ.get("IJF_JULIA", "julia")),
        ("julia-scip", julia_cmd(paths) + [str(val / "smoke.jl"), "scip_global"], os.environ.get("IJF_JULIA", "julia")),
        ("r", ["Rscript", "--vanilla", str(val / "smoke.R"), str(fixtures)], "Rscript"),
        ("python", [py, str(val / "smoke_python.py"), str(fixtures)], py),
    ]
    records: list[dict] = []
    absent: set[str] = set()      # engines not run because an input is missing
    for name, cmd, exe in engines:
        print(f"== smoke: {name}", flush=True)
        # Sarimax.jl loads SCIP, so neither Julia engine runs before SCIP has been fetched.
        if name.startswith("julia") and scip.status()[0] is False:
            report.add("smoke", f"{name} engine", MISSING, note=scip.FETCH_HINT)
            absent.add("julia")
            continue
        if not which(exe):
            report.add("smoke", f"{name} engine", FAIL, note=f"{exe} not found")
            continue
        log = outdir / f"{name}.log"
        code, secs = run_logged(cmd, log, cwd=paths.repl, timeout=TIMEOUT[name])
        recs = _records(log)
        records += recs
        for r in recs:
            if r.get("quantity") == "error":
                report.add("smoke", f"{r['engine']}:{r['test']}", FAIL, note=f"raised: {r['value']}")
        if code != 0:
            why = {124: "timed out", 127: "could not start"}.get(code, f"exited with code {code}")
            report.add("smoke", f"{name} engine", FAIL, generated=f"{len(recs)} values",
                       note=f"{why} after {secs:.0f}s; last lines: {tail(log, 4)}")
        else:
            report.add("smoke", f"{name} engine", PASS, generated=f"{len(recs)} values in {secs:.0f}s",
                       note=f"log: {log.relative_to(paths.output)}")

    produced = {(r["engine"], r["test"], r["quantity"]): r["value"] for r in records}
    for c in expected:
        key = (c["engine"], c["test"], c["quantity"])
        label = f"{c['engine']}:{c['test']}:{c['quantity']}"
        if key not in produced:
            report.add("smoke", label, MISSING if c["engine"] in absent else FAIL,
                       expected=c["expected"], generated="(not produced)", note=c.get("source", ""))
            continue
        ok, diff = _compare(c, produced[key])
        tol = f" (tol {c['compare']} {c['tol']})" if "tol" in c else ""
        report.add("smoke", label, PASS if ok else FAIL, expected=c["expected"], generated=produced[key],
                   difference=diff, note=c.get("source", "") + tol)
    checked = {(c["engine"], c["test"], c["quantity"]) for c in expected}
    for key, value in produced.items():
        if key not in checked and key[2] != "error":
            report.add("smoke", ":".join(key), INFO, generated=value, note="recorded, not compared")

    (outdir / "values.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    out = report.write()
    print(f"smoke report: {out}")
    if not report.failed:
        return EXIT_OK
    return EXIT_INPUT if all(r.status == MISSING for r in report.rows if r.status in (FAIL, MISSING)) else EXIT_FAIL
