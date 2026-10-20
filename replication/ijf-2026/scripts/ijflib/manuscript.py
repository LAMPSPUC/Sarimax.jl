"""Reconciliation of the manuscript's printed numbers with their sources.

`results/manuscript/manuscript_values.csv` lists every number printed in the tables, figures
and experimental text of the manuscript (revision of 2026-10-16), with its LaTeX label and its
source, of one of three kinds:

* `regenerated`: a cell that a table group rebuilds from archived per-series results
  (`source` = "<group>/<table>", `source_row`, `source_column`). `tables` compares the printed
  value with the REGENERATED value, at the precision printed in the manuscript;
* `derived`: a value of a derived table archived in this repository whose raw inputs are not
  archived (`source` = a CSV path under replication/ijf-2026, `source_row` = "col=value;..."
  selectors). The printed value is compared with that table: a consistency check, never
  reported as a reproduction;
* `none`: no archived source was identified; the value is listed with the reason.

`scale` multiplies the source value before the comparison (100 for percentages). A text cell
(empty `decimals`) is compared on its first word.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

MATCH = "MATCH"
DIFFER = "DIFFER"
NOT_REGENERATED = "NOT REGENERATED"
CONSISTENT = "CONSISTENT (derived table only)"
DERIVED_DIFFER = "DIFFER (derived table)"
NO_SOURCE = "NO ARCHIVED SOURCE"
NOT_SELECTED = "NOT SELECTED"
AWAITING = "AWAITING TRANSCRIPTION"

VALUES = Path("results") / "manuscript" / "manuscript_values.csv"

# The five categories of the traceability report, from the cell statuses. A value is
# "raw-reproduced" only when a table group regenerated it from archived raw records and it
# agrees at the printed precision; a value checked against a derived table is "derived-only"
# however well it agrees.
CATEGORIES = {
    MATCH: "raw-reproduced",
    CONSISTENT: "derived-only",
    DIFFER: "differing",
    DERIVED_DIFFER: "differing",
    NO_SOURCE: "no evidence",
    NOT_REGENERATED: "no evidence",
    AWAITING: "no evidence",
}


def category(status: str) -> str:
    return CATEGORIES.get(status, "not checked")


def load(repl: Path, values: Path = VALUES) -> list[dict]:
    with open(repl / values, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare(printed: str, decimals: str, value, scale: str) -> tuple[bool, str]:
    """True when `value` (times `scale`) prints as `printed` at the printed precision."""
    if decimals in ("", None):
        return (str(value).strip().lower().split()[:1] == printed.strip().lower().split()[:1]), ""
    try:
        p = float(printed.replace(",", "").lstrip("+"))
        v = float(str(value).strip()) * (float(scale) if scale else 1.0)
    except (TypeError, ValueError):
        return False, "not a number"
    if math.isnan(v):
        return False, "NaN"
    tol = 0.5 * 10 ** (-int(decimals)) + 1e-9 * max(1.0, abs(p))
    return abs(v - p) <= tol, f"{v - p:+.2e}"


def _derived_value(repl: Path, source: str, selector: str, column: str) -> str | None:
    want = dict(kv.split("=", 1) for kv in selector.split(";") if kv)
    path = repl / source
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if all(r.get(k) == v for k, v in want.items()):
                return r.get(column)
    return None


def reconcile(repl: Path, generated: dict[str, dict], unavailable: set[str],
              selected: set[str] | None = None, values: Path = VALUES) -> list[dict]:
    """Compare every manuscript value with its source.

    `generated` maps a group name to its regenerated cells {(table, row, column): Cell};
    `unavailable` lists the groups that could not be regenerated (missing inputs or a failed
    generator); `selected`, when given, the groups the run was restricted to.
    """
    out = []
    for m in load(repl, values):
        kind, value, diff = m["source_kind"], "", ""
        if m["printed"] == "":
            # a row of a transcription template whose printed value is not entered yet
            out.append({**m, "source_value": "", "difference": "printed value not transcribed",
                        "status": AWAITING})
            continue
        if kind == "regenerated":
            group, _, table = m["source"].partition("/")
            cells = generated.get(group)
            if selected is not None and group not in selected:
                status = NOT_SELECTED
            elif cells is None:
                status = NOT_REGENERATED
                why = "inputs missing or generator failed" if group in unavailable else "group not run"
                diff = why
            else:
                c = cells.get((table, m["source_row"], m["source_column"]))
                if c is None:
                    status, diff = DIFFER, "source cell not generated"
                else:
                    value = c.value
                    ok, diff = compare(m["printed"], m["decimals"], value, m["scale"])
                    status = MATCH if ok else DIFFER
        elif kind == "derived":
            value = _derived_value(repl, m["source"], m["source_row"], m["source_column"])
            if value is None:
                status, diff = DERIVED_DIFFER, "derived value not found"
            else:
                ok, diff = compare(m["printed"], m["decimals"], value, m["scale"])
                status = CONSISTENT if ok else DERIVED_DIFFER
        else:
            status = NO_SOURCE
        out.append({**m, "source_value": _fmt(value), "difference": diff, "status": status})
    return out


def item_status(counts: dict[str, int]) -> str:
    """Report status of a manuscript item: a difference fails, an unverifiable cell warns."""
    if counts.get(DIFFER) or counts.get(DERIVED_DIFFER):
        return "FAIL"
    if counts.get(NOT_REGENERATED) or counts.get(AWAITING):
        return "MISSING"
    if counts.get(NO_SOURCE) or counts.get(CONSISTENT):
        return "WARN"
    if counts.get(MATCH):
        return "PASS"
    return "SKIPPED"


def summarize(rows: list[dict]) -> dict[str, dict]:
    """Per manuscript item (Table 3, Section 5.5, ...): counts by status, in manuscript order."""
    items: dict[str, dict] = {}
    for r in rows:
        d = items.setdefault(r["item"], {"label": r["label"], "counts": {}})
        d["counts"][r["status"]] = d["counts"].get(r["status"], 0) + 1
    return items


def write_cells(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["item", "label", "row", "column", "printed", "decimals", "scale", "source_kind", "source",
            "source_row", "source_column", "source_value", "difference", "status", "note"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _fmt(v) -> str:
    if isinstance(v, float):
        return repr(v)
    return "" if v is None else str(v)
