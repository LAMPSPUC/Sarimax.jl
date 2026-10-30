"""Shared definitions for the table generators.

A generator reads archived per-series results from the artifact bundle and returns the
cells of the tables it rebuilds. Each cell is compared with the value printed in the
historical table (results/expected/<group>.csv) at the precision that table printed.

The M4 conventions below are the ones every archived campaign and the stored baselines used
(ForecastTester `src/ForecastTester.jl` and `src/metrics.jl`):

* OWA is a RATIO OF MEANS: 0.5 * mean(MASE)/mean(MASE_Naive2) + 0.5 * mean(sMAPE)/mean(sMAPE_Naive2),
  each mean taken over the same set of series. It is NOT the mean of per-series OWA values,
  which is a different (and not comparable) quantity.
* Horizon blocks are fixed per frequency (BLOCKS) and the MASE denominator is the in-sample
  seasonal-naive MAE of the whole training window, identical for every block.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

BLOCK_NAMES = ("short", "medium", "long", "total")
BLOCKS = {
    "monthly": {"short": (1, 6), "medium": (7, 12), "long": (13, 18), "total": (1, 18)},
    "quarterly": {"short": (1, 2), "medium": (3, 5), "long": (6, 8), "total": (1, 8)},
    "yearly": {"short": (1, 2), "medium": (3, 4), "long": (5, 6), "total": (1, 6)},
    "weekly": {"short": (1, 4), "medium": (5, 9), "long": (10, 13), "total": (1, 13)},
    "daily": {"short": (1, 4), "medium": (5, 9), "long": (10, 14), "total": (1, 14)},
    "hourly": {"short": (1, 16), "medium": (17, 32), "long": (33, 48), "total": (1, 48)},
}
SEASONALITY = {"yearly": 1, "quarterly": 4, "monthly": 12, "weekly": 1, "daily": 1, "hourly": 24}
HORIZON = {f: b["total"][1] for f, b in BLOCKS.items()}
POPULATION = {"yearly": 23000, "quarterly": 24000, "monthly": 48000, "weekly": 359,
              "daily": 4227, "hourly": 414}


@dataclass
class Cell:
    table: str                 # table identifier, e.g. "m4_objectives.table1"
    row: str                   # stable row key, e.g. "monthly|mse"
    column: str                # column key, e.g. "zeroed.OWA"
    value: object              # float, int or str
    decimals: int | None = 4   # printed precision; None compares text exactly


@dataclass
class Context:
    bundle: Path               # root of the extracted artifact bundle
    out: Path                  # where this group writes its tables
    figures: Path              # where this group writes its figures
    m4: Path | None = None     # official M4 files, when present (enables deeper checks)

    def path(self, rel: str) -> Path:
        return self.bundle / rel


def open_text(path: Path):
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


def read_csv(path: Path) -> list[dict]:
    """Rows of a CSV (optionally gzipped), skipping leading `#` comment lines."""
    with open_text(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    return list(csv.DictReader(lines))


def read_jsonl(path: Path) -> list[dict]:
    with open_text(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def fmean(xs) -> float:
    xs = list(xs)
    return math.fsum(xs) / len(xs)


def owa(mase: float, mase_n2: float, smape: float, smape_n2: float) -> float:
    return 0.5 * (mase / mase_n2 + smape / smape_n2)


def num(x) -> float:
    try:
        return float(str(x).strip().strip('"'))
    except ValueError:
        return float("nan")


def status_ok(x: str) -> bool:
    return str(x).strip().strip('"').startswith("OK")


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip("\n") + "\n", encoding="utf-8", newline="\n")


def cells_to_csv(path: Path, cells: list[Cell]) -> None:
    write_csv(path, ["table", "row", "column", "value"],
              [[c.table, c.row, c.column, fmt(c.value, c.decimals)] for c in cells])


def fmt(v, decimals: int | None) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        return f"{v:.{decimals}f}" if decimals is not None else repr(v)
    return str(v)
