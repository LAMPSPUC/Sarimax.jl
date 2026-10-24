"""Static consistency checks of the replication package (no artifact bundle, no Docker needed).

* experiment_manifest.csv: required columns, unique ids, readiness values, existing runner
  configurations that parse and declare a known kind;
* table groups: the same list in the orchestrator and in the expected-value extractor, a
  generator module and an expected-value file for each;
* results/manuscript/manuscript_values.csv: every regenerated source names a known group, every
  derived source file exists;
* data/external_files.csv: a URL and a SHA-256 for every third-party file.

Usage: python check_package.py        exit 0 when consistent, 1 otherwise (problems are listed)
"""

from __future__ import annotations

import csv
import importlib
import re
import sys
import tomllib
from pathlib import Path

REPL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPL / "scripts"))
sys.path.insert(0, str(REPL / "scripts" / "maintainers"))

STATUSES = ("VERIFIED", "EXECUTABLE", "PARTIAL", "BLOCKED", "PENDING", "NOT APPLICABLE")


def main() -> int:
    problems: list[str] = []

    from ijflib import manifest, tables
    import extract_expected

    rows = list(csv.DictReader(open(REPL / "experiment_manifest.csv", newline="", encoding="utf-8")))
    missing = [c for c in manifest.REQUIRED if c not in rows[0]]
    if missing:
        problems.append(f"manifest lacks columns {missing}")
    ids = [r["experiment_id"] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate experiment ids in the manifest")
    for r in rows:
        for col in ("table_status", "full_status"):
            if not r[col].startswith(STATUSES):
                problems.append(f"{r['experiment_id']}: {col} '{r[col]}' is not one of {STATUSES}")
        runner = r["full_runner"]
        if runner not in ("", "none"):
            f = REPL / runner
            if not f.exists():
                problems.append(f"{r['experiment_id']}: runner {runner} does not exist")
                continue
            cfg = tomllib.loads(f.read_text(encoding="utf-8"))
            if cfg.get("kind") not in ("m4", "script"):
                problems.append(f"{runner}: unknown kind {cfg.get('kind')!r}")
            if cfg.get("id") not in (None, r["experiment_id"]):
                problems.append(f"{runner}: id {cfg.get('id')!r} differs from {r['experiment_id']}")

    if list(tables.GROUPS) != list(extract_expected.GROUPS):
        problems.append(f"table groups differ: tables.py {tables.GROUPS} vs extract_expected.py {extract_expected.GROUPS}")
    for g in tables.GROUPS:
        try:
            mod = importlib.import_module(f"ijftables.{g}")
        except ImportError as exc:      # numpy may be absent on a bare host: report, do not crash
            problems.append(f"group {g}: cannot import its generator ({exc})")
            continue
        if not getattr(mod, "INPUTS", None):
            problems.append(f"group {g}: generator declares no INPUTS")
        if not (REPL / "results" / "expected" / f"{g}.csv").exists():
            problems.append(f"group {g}: results/expected/{g}.csv missing")

    ms = list(csv.DictReader(open(REPL / "results" / "manuscript" / "manuscript_values.csv", newline="",
                                  encoding="utf-8")))
    for m in ms:
        if m["source_kind"] == "regenerated":
            group = m["source"].partition("/")[0]
            if group not in tables.GROUPS:
                problems.append(f"manuscript {m['item']} {m['row']}: unknown group {group}")
        elif m["source_kind"] == "derived":
            if not (REPL / m["source"]).exists():
                problems.append(f"manuscript {m['item']} {m['row']}: {m['source']} does not exist")
        elif m["source_kind"] != "none":
            problems.append(f"manuscript {m['item']} {m['row']}: unknown source_kind {m['source_kind']}")
        if m["decimals"] and not re.fullmatch(r"-?[\d,]*\.?\d*([eE][-+]?\d+)?", m["printed"].lstrip("+")):
            problems.append(f"manuscript {m['item']} {m['row']}/{m['column']}: printed value {m['printed']!r} is not a number")

    for e in csv.DictReader(open(REPL / "data" / "external_files.csv", newline="", encoding="utf-8")):
        if not re.fullmatch(r"[0-9a-f]{64}", e["sha256"]) or not e["url"].startswith("https://"):
            problems.append(f"external file {e['file']}: missing URL or SHA-256")

    for p in problems:
        print("PROBLEM:", p)
    print(f"{len(rows)} experiments, {len(tables.GROUPS)} table groups, {len(ms)} manuscript values: "
          + ("consistent" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
