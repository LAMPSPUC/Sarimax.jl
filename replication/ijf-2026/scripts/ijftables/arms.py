"""RECREIO paired-arm campaigns of August 2026 (campaigns B-F of the recreio REPRODUCE.md),
as tabulated by `tables/make_table_arms.jl` (original-scripts/m4-recreio-2026-08).

Rebuilds, from the per-series result files of raw/m4-recreio-2026-08/, the seven arm tables of
historical-tables/m4-recreio-2026-08/:

  table id                          raw file                          campaign
  arms.objectives                   objectives_monthly                B, monthly, five cells
  arms.objectives_penalized         objectives_monthly_penalized      B, the three `_pen` cells
  arms.multistart                   multistart_random                 C, monthly
  arms.isolation                    isolation                         D, weekly + yearly
  arms.axes_weekly                  axes_weekly                       E, commit aa68d57
  arms.axes_weekly_earlier_commit   axes_weekly_earlier_commit        E, commit 6e8e0bd
  arms.stable                       stable_weekly_yearly              F, weekly + yearly

This is a port of make_table_arms.jl with its semantics kept, including the ones that are easy
to get wrong:

* PAIRING. Per frequency, only the series for which EVERY arm has an OK row enter the means,
  the deltas and the win rates (`complete`). `series` counts every series id seen in that
  frequency, in any arm.
* REFERENCE ARM = the alphabetically first arm label (Julia `sort` of the labels, i.e. code
  point order), not the control arm: `huber` in arms.objectives, `huber_pen`, `base`,
  `innovations`, `base`, `base`, `mse`.
* sMAPE and MASE are MEANS over the paired series (MASE ignoring NaN); deltas are paired
  (arm - reference, per series); `wins_pct` is the share of paired series where the arm's
  sMAPE is strictly lower; the delta mean, median and p05/p95 (Julia `quantile`, type 7) are
  of those per-series differences. Only sMAPE is compared between arms.
* `errors`, `non_LS` and `ge_120s` count over ALL rows of the arm in that frequency, paired or
  not. `ge_120s` counts fits with `seconds >= 119.5` (at or beyond the 120 s cap where one was
  set; a cost tail where none was). `non_LS` is `n/r` when the file has no `solver_status`
  column, which is the case for all seven files.

METRICS USED: make_table_arms.jl reads the per-series `smape` and `mase` columns that the
campaign runners stored (four decimals). It does NOT recompute them from the `forecast` /
`actual` vectors, so the redaction of `actual` in this bundle does not touch these tables and
the bundle's per-block `smape_<block>` / `mase_<block>` columns are not used here.

Parsing follows the Julia reader: `#` lines skipped, a naive comma split of each stripped line,
rows with fewer fields than the header skipped, rows whose `series` is not an integer skipped,
the last row of a duplicated (arm, freq, series) key kept, quotes stripped from the arm, freq,
status, seconds and solver_status fields only. The arm column is `arm` when present, else
`cell` (the defaults of make_table_arms.jl; the columns were renamed from `braco`/`celula`/
`tempo` by backfill_provenance.jl, and the >=120s counts of the historical tables show the
renamed files were the ones tabulated). A file without a `freq` column is one block, printed
as `-` and keyed here as `all`.

Cell keys: table `arms.<name>`; row `<freq>` (block summary: series, complete, arms,
dropped_pct, reference) or `<freq>|<arm>` (sMAPE, MASE, vs_ref, wins_pct, errors, non_LS,
ge_120s, delta.mean, delta.median, delta.p05, delta.p95; the reference arm has no vs_ref,
wins_pct or delta cells, which the table prints as `-`).

HISTORICAL caveats (recreio REPRODUCE.md): campaigns B and C ran under a 120 s wall-clock cap
and are reproducible only at fixed machine load (finding 6); the `_pen` cells of campaign B
measured `initialization = :free`, not `:penalized` (finding 2); the production arm of campaign
D (`v0_6`) ran an uncommitted wrapper (finding 3); campaigns B and C have no recorded commit.
"""

from __future__ import annotations

import gzip
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .common import Cell, Context, write_text

TITLE = "RECREIO paired-arm tables, Aug 2026 (campaigns B-F, make_table_arms.jl)"
RAW = "raw/m4-recreio-2026-08"

# (table name, raw file stem); the historical table of each is table_<name>.txt
TABLES = [
    ("objectives", "objectives_monthly"),
    ("objectives_penalized", "objectives_monthly_penalized"),
    ("multistart", "multistart_random"),
    ("isolation", "isolation"),
    ("axes_weekly", "axes_weekly"),
    ("axes_weekly_earlier_commit", "axes_weekly_earlier_commit"),
    ("stable", "stable_weekly_yearly"),
]

INPUTS = [f"{RAW}/{stem}.csv.gz" for _, stem in TABLES]

CAP_SECONDS = 119.5       # make_table_arms.jl: "at or beyond the 120 s cap"
NO_FREQ = "-"             # the Julia label of a file without a `freq` column


def freq_key(freq: str) -> str:
    return "all" if freq == NO_FREQ else freq.lower()


# ---------------------------------------------------------------------------------------------
# Reading, as make_table_arms.jl `read` does it
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmRow:
    ok: bool
    smape: float
    mase: float
    seconds: float
    solver: str


_INT = re.compile(r"^\s*[+-]?\d+\s*$")


def _jl_int(s: str) -> int | None:
    """Julia `tryparse(Int, s)` for the decimal ids these files carry."""
    return int(s) if _INT.match(s) else None


def _jl_float(s: str) -> float | None:
    """Julia `tryparse(Float64, s)`: no quote stripping, no digit separators."""
    if "_" in s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_arms(path: Path, armcol: str = "") -> dict[tuple[str, str, int], ArmRow]:
    with gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb") as f:
        text = f.read().decode("utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":          # Julia `eachline` yields no line after the last \n
        lines.pop()
    rows: dict[tuple[str, str, int], ArmRow] = {}
    hdr = ix = col = None
    for line in lines:
        if line.endswith("\r"):
            line = line[:-1]
        if line.startswith("#"):
            continue
        v = line.strip().split(",")
        if hdr is None:
            hdr = v
            ix = {h: i for i, h in enumerate(hdr)}
            col = armcol or ("arm" if "arm" in ix else "cell")
            if col not in ix:
                raise ValueError(f"{path.name}: no arm column found; pass one explicitly")
            continue
        if len(v) < len(hdr):
            continue
        sid = _jl_int(v[ix["series"]])
        if sid is None:
            continue
        arm = v[ix[col]].strip('"')
        freq = v[ix["freq"]].strip('"') if "freq" in ix else NO_FREQ
        status = v[ix["status"]].strip('"')
        sm = _jl_float(v[ix["smape"]])
        ma = _jl_float(v[ix["mase"]]) if "mase" in ix else None
        sec = _jl_float(v[ix["seconds"]].strip('"')) if "seconds" in ix else None
        solver = v[ix["solver_status"]].strip('"') if "solver_status" in ix else "-"
        rows[(arm, freq, sid)] = ArmRow(
            ok=status.startswith("OK"),
            smape=math.nan if sm is None else sm,
            mase=math.nan if ma is None else ma,
            seconds=math.nan if sec is None else sec,
            solver=solver)
    return rows


# ---------------------------------------------------------------------------------------------
# Julia Statistics, as used by the original
# ---------------------------------------------------------------------------------------------
def jl_mean(xs: list[float]) -> float:
    return math.fsum(xs) / len(xs) if xs else math.nan


def jl_median(xs: list[float]) -> float:
    if not xs or any(math.isnan(x) for x in xs):
        return math.nan
    s = sorted(xs)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return s[n // 2 - 1] / 2 + s[n // 2] / 2       # Statistics.middle(a, b)


def jl_quantile(xs: list[float], p: float) -> float:
    """Statistics.quantile(v, p) with the default alpha = beta = 1 (Hyndman-Fan type 7).

    Julia raises on NaN; NaN is returned here instead (none occurs in the archived files)."""
    if not xs or any(math.isnan(x) for x in xs):
        return math.nan
    s = sorted(xs)
    n = len(s)
    if n == 1:
        return s[0]
    m = 1.0 + p * (1.0 - 1.0 - 1.0)
    aleph = n * p + m
    j = min(max(math.trunc(aleph), 1), n - 1)       # 1-based
    g = min(max(aleph - j, 0.0), 1.0)
    a, b = s[j - 1], s[j]
    if math.isfinite(a) and math.isfinite(b):
        return a + g * (b - a)
    return (1 - g) * a + g * b


def _f(x: float, spec: str) -> str:
    """Python format of a float, with Julia's spelling of non-finite values."""
    if math.isnan(x) or math.isinf(x):
        word = "NaN" if math.isnan(x) else ("Inf" if x > 0 else "-Inf")
        width = re.match(r"[+-]?(\d*)", spec).group(1)
        return word.rjust(int(width)) if width else word
    return format(x, spec)


# ---------------------------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------------------------
NOTES = """\
non-LS = returned candidate not LOCALLY_SOLVED; `n/r` = solver status not recorded
in this file. NOTE: the status describes the RETURNED candidate only. Candidates
that failed inside the search are absorbed by `auto` and are not counted anywhere;
see REPRODUCE.md, finding 5.
>=120s counts fits at or beyond the 120 s cap where one was set, and is a
CENSORING RATE for those campaigns. Where no cap was set it is a cost tail only."""


def tabulate(name: str, title: str, rows: dict[tuple[str, str, int], ArmRow]) -> tuple[list[Cell], str]:
    table = f"arms.{name}"
    cells: list[Cell] = []
    out = ["=" * 96, title, "=" * 96]
    arms = sorted({a for a, _, _ in rows})
    freqs = sorted({f for _, f, _ in rows})
    for freq in freqs:
        fk = freq_key(freq)
        ids = sorted({s for _, f, s in rows if f == freq})
        paired = [s for s in ids if all((a, freq, s) in rows and rows[(a, freq, s)].ok for a in arms)]
        total = len(ids)
        dropped = 0.0 if total == 0 else 100 * (total - len(paired)) / total
        out.append("")
        out.append(f"{freq.upper()}   {total} series, {len(paired)} complete across all {len(arms)} arms "
                   f"({dropped:.2f}% dropped as incomplete)")
        cells += [Cell(table, fk, "series", total, 0), Cell(table, fk, "complete", len(paired), 0),
                  Cell(table, fk, "arms", len(arms), 0), Cell(table, fk, "dropped_pct", dropped, 2)]
        if not paired:
            continue

        ref = arms[0]
        ref_s = [rows[(ref, freq, s)].smape for s in paired]
        cells.append(Cell(table, fk, "reference", ref, None))
        out.append(f"  {'arm':<14} {'sMAPE':>9} {'MASE':>9} {'vs ' + ref[:8]:>11} {'wins':>10} "
                   f"{'errors':>9} {'non-LS':>9} {'>=120s':>9}")
        out.append("  " + "-" * 90)
        for a in arms:
            mine = [rows[(a, freq, s)] for s in paired]
            every = [rows[(a, freq, s)] for s in ids if (a, freq, s) in rows]
            S = [c.smape for c in mine]
            M = [c.mase for c in mine]
            d = [x - r for x, r in zip(S, ref_s)]
            wins = sum(1 for x in d if x < 0)
            n_err = sum(1 for c in every if not c.ok)
            n_nonls = sum(1 for c in every if c.ok and c.solver != "-" and c.solver != "LOCALLY_SOLVED")
            n_cap = sum(1 for c in every if not math.isnan(c.seconds) and c.seconds >= CAP_SECONDS)
            s_mean = jl_mean(S)
            m_mean = jl_mean([x for x in M if not math.isnan(x)])
            nonls = "n/r" if n_nonls == 0 and all(c.solver == "-" for c in every) else n_nonls
            row = f"{fk}|{a}"
            cells += [Cell(table, row, "sMAPE", s_mean, 4), Cell(table, row, "MASE", m_mean, 4),
                      Cell(table, row, "errors", n_err, 0),
                      Cell(table, row, "non_LS", nonls, None if isinstance(nonls, str) else 0),
                      Cell(table, row, "ge_120s", n_cap, 0)]
            if a == ref:
                vs_txt = wins_txt = "-"
            else:
                d_mean = jl_mean(d)
                win_pct = 100 * wins / len(paired)
                vs_txt, wins_txt = _f(d_mean, "+.4f"), _f(win_pct, ".1f") + "%"
                cells += [Cell(table, row, "vs_ref", d_mean, 4), Cell(table, row, "wins_pct", win_pct, 1)]
            out.append(f"  {a:<14} {_f(s_mean, '9.4f')} {_f(m_mean, '9.4f')} {vs_txt:>11} {wins_txt:>10} "
                       f"{n_err:9d} {str(nonls):>9} {n_cap:9d}")
        out.append("")
        out.append(f"  paired deltas against {ref} (negative = better than {ref}):")
        for a in arms:
            if a == ref:
                continue
            d = [rows[(a, freq, s)].smape - rows[(ref, freq, s)].smape for s in paired]
            stats = {"delta.mean": (jl_mean(d), 4), "delta.median": (jl_median(d), 4),
                     "delta.p05": (jl_quantile(d, 0.05), 3), "delta.p95": (jl_quantile(d, 0.95), 3)}
            cells += [Cell(table, f"{fk}|{a}", k, v, dec) for k, (v, dec) in stats.items()]
            v = [x for x, _ in stats.values()]
            out.append(f"    {a:<14} mean {_f(v[0], '+8.4f')}   median {_f(v[1], '+8.4f')}   "
                       f"p05 {_f(v[2], '+8.3f')}   p95 {_f(v[3], '+8.3f')}")
    out.append("")
    out.append(NOTES)
    return cells, "\n".join(out)


def generate(ctx: Context) -> list[Cell]:
    cells: list[Cell] = []
    for name, stem in TABLES:
        rel = f"{RAW}/{stem}.csv.gz"
        rows = read_arms(ctx.path(rel))
        # the original was run on the uncompressed file and printed its basename
        tab_cells, text = tabulate(name, f"{stem}.csv", rows)
        cells += tab_cells
        write_text(ctx.out / f"table_{name}.txt", text)
    return cells
