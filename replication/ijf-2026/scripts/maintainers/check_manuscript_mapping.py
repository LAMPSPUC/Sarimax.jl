"""Release check of the manuscript's printed values against the historical reference values.

Every `regenerated` manuscript value points at a cell of a table group. `./reproduce.sh tables`
compares it with the REGENERATED cell, which needs the artifact bundle. This script needs only
the repository: it compares each printed value with the reference value of the same cell
(results/expected/<group>.csv, parsed from the tables printed at the time), so that a wrong
mapping or a printed value that no archived table supports is found without the bundle. CI runs
it on every change as a required check.

It reports every value in one of five categories:

  raw-reproduced     regenerated from archived raw records by a table group, and agrees
  derived-only       agrees with a derived table whose raw inputs are not archived
  differing          the printed value differs from its source (fails the check)
  no evidence        no archived source, or a template row whose printed value is not entered
  textual claims     statements of the text that need verification (textual_claims.csv)

and states, as missing evidence, which manuscript revision is transcribed and whether it is the
canonical one (results/manuscript/REVISION.toml). The template for the canonical revision
(pending/...) is checked too: its rows with printed values are compared like any other.

Two modes. A differing value is an error of the manuscript or of the mapping. When it has been
established as an error of the manuscript, it is entered in results/manuscript/errata.csv with
the archived value, the report that documents it and the proposed correction:

  default    (pull requests) fails on every differing value that is NOT in errata.csv, and on
             every errata.csv entry that no longer differs (a stale register); registered errata
             are reported as warnings
  --release  (release gates: tags, image publication) fails on EVERY differing value, registered
             or not: a release needs the manuscript corrected

Usage: python check_manuscript_mapping.py [--release] [--report FILE] [--markdown FILE]
Exit codes: 0 the check passes in its mode; 1 otherwise. Missing evidence is reported, never a
failure.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tomllib
from pathlib import Path

REPL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPL / "scripts"))

from ijflib import manuscript  # noqa: E402

RUNTIME_ONLY = "RUN-TIME CHECK ONLY"
ORDER = ["raw-reproduced", "derived-only", "differing", "no evidence", "not checked"]


class Ref:
    def __init__(self, value: str):
        self.value = value


def check(expected: dict, tables: set, values: Path) -> list[dict]:
    rows = manuscript.reconcile(REPL, expected, set(), values=values)
    # Cells of a table that has no historical counterpart (e.g. m4_paired.text, the range quoted
    # in Section 5.3) can only be checked against the regenerated values, at run time.
    for r in rows:
        if r["source_kind"] == "regenerated" and r["status"] == manuscript.DIFFER \
                and r["source"].partition("/")[2] not in tables:
            r["status"], r["difference"] = RUNTIME_ONLY, "no historical table for this cell"
    return rows


def annotate(kind: str, text: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{kind}::{text}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, help="write the cell-level comparison to this CSV")
    ap.add_argument("--markdown", type=Path, help="write the traceability report to this file")
    ap.add_argument("--release", action="store_true",
                    help="release gate: registered errata fail too")
    args = ap.parse_args()
    expected: dict[str, dict] = {}
    for f in sorted((REPL / "results" / "expected").glob("*.csv")):
        with open(f, newline="", encoding="utf-8") as fh:
            expected[f.stem] = {(r["table"], r["row"], r["column"]): Ref(r["expected"])
                                for r in csv.DictReader(fh)}
    tables = {t for cells in expected.values() for (t, _, _) in cells}
    rev = tomllib.loads((REPL / "results" / "manuscript" / "REVISION.toml").read_text(encoding="utf-8"))

    rows = check(expected, tables, manuscript.VALUES)
    pending_path = Path("results") / "manuscript" / rev["pending"]["file"]
    pending = check(expected, tables, pending_path) if (REPL / pending_path).exists() else []
    with open(REPL / "results" / "manuscript" / "textual_claims.csv", newline="", encoding="utf-8") as f:
        claims = list(csv.DictReader(f))
    with open(REPL / "results" / "manuscript" / "errata.csv", newline="", encoding="utf-8") as f:
        errata = {(e["item"], e["row"], e["column"], e["printed"]): e for e in csv.DictReader(f)}
    if args.report:
        manuscript.write_cells(args.report, rows + pending)

    def cat(r):
        return "not checked" if r["status"] == RUNTIME_ONLY else manuscript.category(r["status"])

    out: list[str] = []
    say = out.append
    tr, can = rev["transcribed"], rev["canonical"]
    say(f"# Manuscript traceability: revision {tr['revision']} (canonical: {can['revision']})")
    say("")
    if tr["revision"] != can["revision"]:
        missing = ", ".join(can["inputs"]) if not can.get("inputs_available") else "(available)"
        msg = (f"MISSING EVIDENCE: the canonical revision {can['revision']} ({can['title']}) is not "
               f"transcribed; the values below are those of {tr['revision']}. Missing inputs: {missing}")
        say(f"**{msg}**")
        print(msg)
        annotate("warning", msg)
        say("")
    for label, rs in ((f"{tr['revision']} transcription ({tr['file']})", rows),
                      (f"{can['revision']} template for the controlled items ({rev['pending']['file']})", pending)):
        if not rs:
            continue
        counts = {c: 0 for c in ORDER}
        items: dict[str, dict] = {}
        for r in rs:
            counts[cat(r)] += 1
            d = items.setdefault(r["item"], {"label": r["label"], **{c: 0 for c in ORDER}})
            d[cat(r)] += 1
        say(f"## {label}: {len(rs)} values")
        say("")
        say("| item | label | " + " | ".join(ORDER) + " |")
        say("|---|---|" + "---|" * len(ORDER))
        for item, d in items.items():
            say(f"| {item} | {d['label'] or ''} | " + " | ".join(str(d[c] or "") for c in ORDER) + " |")
        say("| **total** | | " + " | ".join(f"**{counts[c]}**" for c in ORDER) + " |")
        say("")
        print(f"{label}: " + ", ".join(f"{c} {counts[c]}" for c in ORDER))
    differing = [r for r in rows + pending if r["status"] in (manuscript.DIFFER, manuscript.DERIVED_DIFFER)]
    key = lambda r: (r["item"], r["row"], r["column"], r["printed"])
    known = [r for r in differing if key(r) in errata]
    new = [r for r in differing if key(r) not in errata]
    stale = [e for k, e in errata.items() if k not in {key(r) for r in differing}]
    bad = differing if args.release else new
    mode = "release" if args.release else "pull request"
    if differing or stale:
        say(f"## Differing values ({mode} mode)")
        say("")
        for r in differing:
            e = errata.get(key(r))
            line = (f"{r['item']} ({r['label']}) {r['row']} / {r['column']}: printed {r['printed']}, "
                    f"reference {r['source_value'] or '(cell not found)'} ({r['difference']})")
            if e is None:
                say(f"* **NOT REGISTERED** {line}")
                print("DIFFER " + line)
                annotate("error", "manuscript value differs from its source: " + line)
            else:
                tag = "fails the release gate" if args.release else "known manuscript erratum"
                say(f"* {tag}: {line}. Archived value {e['archived']}; correction proposed: "
                    f"{e['proposed_correction']} ({e['report']})")
                print(f"ERRATUM {line}")
                annotate("error" if args.release else "warning",
                         f"known manuscript erratum, correction pending: {line}")
        for e in stale:
            line = f"{e['item']} {e['row']} / {e['column']} printed {e['printed']}"
            say(f"* **STALE ERRATUM** (no longer differs; remove it from errata.csv): {line}")
            print("STALE " + line)
            annotate("error", "errata.csv entry no longer differs: " + line)
        say("")
    none = [r for r in rows if cat(r) == "no evidence"]
    if none:
        say("## No evidence")
        say("")
        by_note: dict[tuple[str, str], int] = {}
        for r in none:
            k = (r["item"], r["note"].split(";")[0][:200])
            by_note[k] = by_note.get(k, 0) + 1
        for (item, note), n in by_note.items():
            say(f"* {item}: {n} values. {note}")
        say("")
        annotate("warning", f"{len(none)} printed values of the {tr['revision']} revision have no archived evidence")
    waiting = [r for r in pending if r["status"] == manuscript.AWAITING]
    if waiting:
        annotate("warning", f"{len(waiting)} values of the {can['revision']} controlled items await transcription")
    open_claims = [c for c in claims if c["status"] != "supported"]
    say(f"## Textual claims needing verification: {len(open_claims)} of {len(claims)}")
    say("")
    say("| id | item | status | claim |")
    say("|---|---|---|---|")
    for c in claims:
        say(f"| {c['id']} | {c['item']} | {c['status']} | {c['claim']} |")
    say("")
    print(f"textual claims: {len(claims)} listed, {len(open_claims)} need verification")
    if args.markdown:
        args.markdown.write_text("\n".join(out) + "\n", encoding="utf-8")
    failed = bool(bad or stale)
    print(f"{mode} mode: {len(differing)} differing values ({len(known)} registered errata, "
          f"{len(new)} not registered), {len(stale)} stale errata: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
