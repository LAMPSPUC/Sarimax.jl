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
import json
import math
import sys
import time
import traceback
from pathlib import Path

from . import manuscript
from .common import (EXIT_FAIL, EXIT_INPUT, EXIT_OK, FAIL, INFO, MISSING, PASS, SKIPPED, WARN,
                     Paths, Report, sha256_file)
from .check import artifact_inventory

# Order = order in the report. Each entry is a module in scripts/ijftables.
GROUPS = [
    "battery",
    "m4_benchmark",
    "m4_paired",
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
    # Generators that can also check stored metrics against the M4 test values do so only when
    # every M4 test file is present (`./reproduce.sh fetch m4`); the check is optional.
    m4 = paths.m4 if all((paths.m4 / f"{s}-test.csv").exists() for s in
                         ("Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly")) else None
    only = set(getattr(args, "group", None) or [])
    input_failure = False
    generated_by_group: dict[str, dict] = {}
    unavailable: set[str] = set()

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
            unavailable.add(group)
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
            unavailable.add(group)
            continue
        secs = time.monotonic() - t0
        generated = {(c.table, c.row, c.column): c for c in cells}
        generated_by_group[group] = generated

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

    reconcile_manuscript(paths, report, generated_by_group, unavailable, only or None)

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


def reconcile_manuscript(paths: Paths, report: Report, generated: dict, unavailable: set,
                         only: set | None) -> None:
    """Every number printed in the manuscript against the regenerated cell it comes from."""
    print("== tables: manuscript — printed values against their sources")
    rows = manuscript.reconcile(paths.repl, generated, unavailable, only)
    manuscript.write_cells(paths.reports / "manuscript_cells.csv", rows)
    totals: dict[str, int] = {}
    for item, d in manuscript.summarize(rows).items():
        counts = d["counts"]
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        n = sum(counts.values())
        note = "; ".join(f"{k} {v}" for k, v in counts.items())
        report.add("manuscript", f"{item} ({d['label'] or 'no label'})", manuscript.item_status(counts),
                   expected=f"{n} printed values", generated=f"{counts.get(manuscript.MATCH, 0)} match",
                   note=note)
    with open(paths.reports / "manuscript.json", "w", encoding="utf-8") as f:
        json.dump({"values": len(rows), "counts": totals}, f, indent=2)
    print(f"manuscript: {len(rows)} printed values; " + ", ".join(f"{k} {v}" for k, v in totals.items())
          + f" (cell-level: {paths.reports / 'manuscript_cells.csv'})")


def _fmt(v, decimals) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if decimals not in ("", None):
            return f"{v:.{int(decimals) + 2}f}"
        return repr(v)
    return str(v)
