"""Expected cells of the `arms` group, parsed from the seven paired-arm tables that recreio
`tables/make_table_arms.jl` printed (historical-tables/m4-recreio-2026-08/, committed as
experiments/tables/generated/ on paper/replication-staging-recreio):

  table_objectives.txt, table_objectives_penalized.txt, table_multistart.txt,
  table_isolation.txt, table_axes_weekly.txt, table_axes_weekly_earlier_commit.txt,
  table_stable.txt

Every printed value becomes a cell, keyed as in scripts/ijftables/arms.py:

* row `<freq>` (the block label lowercased; `-`, printed for a file without a `freq` column,
  becomes `all`): series, complete, arms, dropped_pct, and `reference` (the arm named in
  "paired deltas against <arm>", compared as text);
* row `<freq>|<arm>`: sMAPE, MASE, vs_ref, wins_pct (the percentage without `%`), errors,
  non_LS (`n/r` compared as text, a count otherwise), ge_120s, and delta.mean, delta.median,
  delta.p05, delta.p95 from the paired-deltas block. The `-` that the reference arm prints
  for vs_ref and wins is not a value and gives no cell.

`expected` is the printed text (a leading `+` dropped) and `decimals` the printed precision.
"""

from __future__ import annotations

import re
from pathlib import Path

DIR = "historical-tables/m4-recreio-2026-08"
TABLES = ["objectives", "objectives_penalized", "multistart", "isolation", "axes_weekly",
          "axes_weekly_earlier_commit", "stable"]

HEAD = re.compile(r"^(\S+)   (\d+) series, (\d+) complete across all (\d+) arms "
                  r"\((\d+\.\d+)% dropped as incomplete\)$")
COLS = re.compile(r"^  arm\s+sMAPE\s+MASE\s+vs \S+\s+wins\s+errors\s+non-LS\s+>=120s$")
ARM = re.compile(r"^  (\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\d+)$")
DELTA_HEAD = re.compile(r"^  paired deltas against (\S+) \(negative = better than (\S+)\):$")
DELTA = re.compile(r"^    (\S+)\s+mean\s+(\S+)\s+median\s+(\S+)\s+p05\s+(\S+)\s+p95\s+(\S+)$")
NUM = re.compile(r"^[+-]?\d+\.\d+$")


def _freq_key(label: str) -> str:
    return "all" if label == "-" else label.lower()


def _num(table, row, column, tok, rel):
    if not NUM.match(tok):
        raise SystemExit(f"{rel}: {row}/{column}: not a decimal number: {tok!r}")
    return {"table": table, "row": row, "column": column, "expected": tok.lstrip("+"),
            "decimals": str(len(tok.split(".")[1])), "source": rel}


def _int(table, row, column, tok, rel):
    if not tok.isdigit():
        raise SystemExit(f"{rel}: {row}/{column}: not a count: {tok!r}")
    return {"table": table, "row": row, "column": column, "expected": tok, "decimals": "0",
            "source": rel}


def _text(table, row, column, tok, rel):
    return {"table": table, "row": row, "column": column, "expected": tok, "decimals": "",
            "source": rel}


def parse_table(bundle: Path, name: str) -> list[dict]:
    rel = f"{DIR}/table_{name}.txt"
    table = f"arms.{name}"
    rows: list[dict] = []
    freq = None
    blocks: dict[str, dict] = {}            # per frequency: declared arms, arm rows, delta rows
    for line in (bundle / rel).read_text(encoding="utf-8").splitlines():
        m = HEAD.match(line)
        if m:
            freq = _freq_key(m.group(1))
            if freq in blocks:
                raise SystemExit(f"{rel}: frequency block {freq!r} printed twice")
            blocks[freq] = {"arms": int(m.group(4)), "rows": 0, "deltas": 0, "ref": None}
            rows += [_int(table, freq, "series", m.group(2), rel),
                     _int(table, freq, "complete", m.group(3), rel),
                     _int(table, freq, "arms", m.group(4), rel),
                     _num(table, freq, "dropped_pct", m.group(5), rel)]
            continue
        if freq is None or COLS.match(line) or set(line.strip()) <= {"-"}:
            continue
        m = DELTA_HEAD.match(line)
        if m:
            if m.group(1) != m.group(2):
                raise SystemExit(f"{rel}: inconsistent reference arm in {line!r}")
            blocks[freq]["ref"] = m.group(1)
            rows.append(_text(table, freq, "reference", m.group(1), rel))
            continue
        m = DELTA.match(line)
        if m:
            row = f"{freq}|{m.group(1)}"
            for col, tok in zip(("delta.mean", "delta.median", "delta.p05", "delta.p95"), m.groups()[1:]):
                rows.append(_num(table, row, col, tok, rel))
            blocks[freq]["deltas"] += 1
            continue
        m = ARM.match(line)
        if m:
            arm, smape, mase, vs, wins, errors, nonls, cap = m.groups()
            row = f"{freq}|{arm}"
            rows += [_num(table, row, "sMAPE", smape, rel), _num(table, row, "MASE", mase, rel)]
            if (vs == "-") != (wins == "-"):
                raise SystemExit(f"{rel}: {row}: vs/wins only half printed")
            if vs != "-":
                if not wins.endswith("%"):
                    raise SystemExit(f"{rel}: {row}: win rate without %: {wins!r}")
                rows += [_num(table, row, "vs_ref", vs, rel),
                         _num(table, row, "wins_pct", wins[:-1], rel)]
            rows.append(_int(table, row, "errors", errors, rel))
            rows.append(_text(table, row, "non_LS", nonls, rel) if nonls == "n/r"
                        else _int(table, row, "non_LS", nonls, rel))
            rows.append(_int(table, row, "ge_120s", cap, rel))
            blocks[freq]["rows"] += 1
            continue
        if line.startswith(("non-LS = ", "in this file.", "that failed", "see REPRODUCE.md",
                            ">=120s counts", "CENSORING RATE")):
            freq = None                     # the closing notes
            continue
        if line.strip():
            raise SystemExit(f"{rel}: unparsed line {line!r}")

    if not blocks:
        raise SystemExit(f"{rel}: no frequency block found")
    for f, b in blocks.items():
        if b["rows"] != b["arms"] or b["deltas"] != b["arms"] - 1 or b["ref"] is None:
            raise SystemExit(f"{rel}: block {f!r} incomplete: {b}")
    return rows


def extract(bundle: Path) -> list[dict]:
    rows: list[dict] = []
    for name in TABLES:
        rows += parse_table(bundle, name)
    return rows
