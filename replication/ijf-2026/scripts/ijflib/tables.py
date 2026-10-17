"""`tables`: rebuild the published tables from archived per-series results and compare.

Each generator group (scripts/ijftables/<group>.py) declares the bundle files it reads. If any
is missing or fails its checksum, every expected cell of that group is reported MISSING; a
group is never reported as reproduced from incomplete inputs. Regenerated cells are compared
with the values printed in the historical tables (results/expected/<group>.csv): a cell
PASSES when it agrees to the precision the table printed (|generated - expected| <= half a
unit in the last printed decimal).
"""

from __future__ import annotations

import csv
import importlib
import math
import sys
import time
import traceback
from pathlib import Path

from .common import (EXIT_FAIL, EXIT_INPUT, EXIT_OK, FAIL, INFO, MISSING, PASS, SKIPPED, WARN,
                     Paths, Report, sha256_file)
from .check import artifact_inventory

# Order = order in the report. Each entry is a module in scripts/ijftables.
GROUPS = [
    "battery",
    "m4_benchmark",
    "m4_objectives",
    "m4_objective_distributions",
    "arms",
    "robust_cvar",
    "scip_ipopt",
]


def load_expected(paths: Paths, group: str) -> list[dict]:
    f = paths.repl / "results" / "expected" / f"{group}.csv"
    if not f.exists():
        return []
    with open(f, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _close(gen, exp: str, decimals: str) -> tuple[bool, str]:
    if decimals in ("", None):
        return str(gen).strip() == exp.strip(), ""
    try:
        e, g = float(exp), float(gen)
    except (TypeError, ValueError):
        return str(gen).strip() == exp.strip(), ""
    if math.isnan(e) and math.isnan(g):
        return True, "0"
    tol = 0.5 * 10 ** (-int(decimals)) + 1e-9 * max(1.0, abs(e))
    return abs(g - e) <= tol, f"{g - e:+.2e}"


def run(paths: Paths, args) -> int:
    sys.path.insert(0, str(paths.repl / "scripts"))
    report = Report("tables", paths)
    detail_rows: list[list] = []
    inventory = dict((p, h) for h, p in artifact_inventory(paths))
    m4 = paths.m4 if any(paths.m4.glob("*-test.csv")) else None
    only = set(getattr(args, "group", None) or [])
    input_failure = False

    for group in GROUPS:
        if only and group not in only:
            continue
        mod = importlib.import_module(f"ijftables.{group}")
        expected = load_expected(paths, group)
        print(f"== tables: {group} — {mod.TITLE}")

        # 1. inputs: present and intact
        problems = []
        for rel in mod.INPUTS:
            f = paths.artifacts / rel
            if not f.exists():
                problems.append(f"missing {rel}")
            elif rel in inventory and sha256_file(f) != inventory[rel]:
                problems.append(f"checksum mismatch {rel}")
        if problems:
            input_failure = True
            report.add(group, mod.TITLE, MISSING, expected=f"{len(expected)} cells",
                       generated="not computed", note="; ".join(problems[:5]) +
                       (f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""))
            for e in expected:
                detail_rows.append([group, e["table"], e["row"], e["column"], e["expected"], "",
                                    "", MISSING, e.get("source", "")])
            continue

        # 2. regenerate
        from ijftables.common import Context
        ctx = Context(bundle=paths.artifacts, out=paths.output / "tables" / group,
                      figures=paths.output / "figures" / group, m4=m4)
        t0 = time.monotonic()
        try:
            cells = mod.generate(ctx)
        except Exception as exc:  # a generator crash is a failure, never a skip
            report.add(group, mod.TITLE, FAIL, note=f"generator raised {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        secs = time.monotonic() - t0
        generated = {(c.table, c.row, c.column): c for c in cells}

        # 3. compare
        n_pass = n_fail = 0
        failed = []
        for e in expected:
            key = (e["table"], e["row"], e["column"])
            c = generated.get(key)
            if c is None:
                n_fail += 1
                failed.append(f"{key} not generated")
                detail_rows.append([group, *key, e["expected"], "", "", FAIL, e.get("source", "")])
                continue
            ok, diff = _close(c.value, e["expected"], e.get("decimals", ""))
            gtxt = _fmt(c.value, e.get("decimals"))
            detail_rows.append([group, *key, e["expected"], gtxt, diff, PASS if ok else FAIL,
                                e.get("source", "")])
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                failed.append(f"{'/'.join(key)}: expected {e['expected']}, got {gtxt}")
        unreferenced = len(set(generated) - {(e["table"], e["row"], e["column"]) for e in expected})
        if not expected:
            report.add(group, mod.TITLE, WARN, generated=f"{len(cells)} cells",
                       note="regenerated, but no archived reference values exist to compare with")
        else:
            report.add(group, mod.TITLE, PASS if n_fail == 0 else FAIL,
                       expected=f"{len(expected)} cells", generated=f"{n_pass} agree",
                       difference=f"{n_fail} differ",
                       note=(f"{secs:.0f}s; tables in {ctx.out}"
                             + (f"; {unreferenced} further cells without a reference" if unreferenced else "")
                             + ("; first differences: " + " | ".join(failed[:3]) if failed else "")))

    detail = paths.reports / "tables_cells.csv"
    detail.parent.mkdir(parents=True, exist_ok=True)
    with open(detail, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["group", "table", "row", "column", "expected", "generated", "difference",
                    "status", "source"])
        w.writerows(detail_rows)
    out = report.write()
    print(f"tables report: {out} (cell-level comparison: {detail})")
    if input_failure:
        return EXIT_INPUT
    return EXIT_FAIL if report.failed else EXIT_OK


def _fmt(v, decimals) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if decimals not in ("", None):
            return f"{v:.{int(decimals) + 2}f}"
        return repr(v)
    return str(v)
