"""The scientific experiment manifest (experiment_manifest.csv) and `list`."""

from __future__ import annotations

import csv
import textwrap

from .common import EXIT_OK, Paths

REQUIRED = ["experiment_id", "manuscript_label", "manuscript_label_status", "title", "table_status",
            "full_runner", "full_status", "full_status_note"]


def load_manifest(paths: Paths) -> list[dict]:
    with open(paths.repl / "experiment_manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in REQUIRED if rows and c not in rows[0]]
    if missing:
        raise SystemExit(f"experiment_manifest.csv lacks columns {missing}")
    return rows


def list_experiments(paths: Paths) -> int:
    rows = load_manifest(paths)
    print(f"{'experiment':<36} {'tables (Level B)':<18} {'full re-run (Level C)':<22} manuscript label")
    print("-" * 110)
    for r in rows:
        label = r["manuscript_label"] or "-"
        if r["manuscript_label_status"] != "confirmed":
            label += f" ({r['manuscript_label_status']})"
        print(f"{r['experiment_id']:<36} {r['table_status']:<18} {r['full_status']:<22} {label}")
    print()
    print("Run one with:   ./reproduce.sh full --experiment <experiment> [--jobs N] [--resume] [--limit N]")
    print("Details per experiment: replication/ijf-2026/experiment_manifest.csv and REPRODUCIBILITY_STATUS.md")
    blocked = [r for r in rows if r["full_runner"] in ("", "none")]
    if blocked:
        print("\nNot re-runnable from this package (reason):")
        for r in blocked:
            print("  " + r["experiment_id"] + ": " + textwrap.shorten(r["full_status_note"], 150))
    return EXIT_OK
