"""Empirical battery of July 2026 (the `experiments/` tree of the Sarimax.jl repository):
validation against statsmodels and R `forecast` (Table 2), solver diagnostics with the MA(1)
global certificates (Section 5.6) and the local-versus-global value experiment (Table 7).

Port of `combine_results.py` (original-scripts/battery-2026-07/scripts/), which built the
tables from the per-run JSONL records. Rebuilds, from raw/battery-2026-07/<block>/*_results.jsonl:

    table id                   historical file             LaTeX label of the battery
    battery.validation         table_validation            tab:validation_implementations
    battery.solver_comparison  table_solver_comparison     tab:solver_comparison
    battery.global_value       table_global_value          tab:global_value

and writes each one to <out>/<historical file>.csv and .tex, formatted as the original did.

Semantics kept from combine_results.py:

* a block's records are the lines of every `*_results.jsonl` in its directory, files in name
  order, as in the original;
* `fmt`: None or NaN -> "--"; int or float -> fixed decimals; anything else verbatim (R's
  jsonlite wrote NA as the string "NA", printed as such);
* phi1/theta1 = the first of Julia's phi/theta vectors, else statsmodels' ar.L1/ma.L1, else
  R's ar1/ma1, chained with Python `or` as in the original;
* row order of pandas `sort_values` (stable, ascending, missing last) or of the original's
  sort keys.

Each cell holds the unrounded value, compared with the printed one at the printed precision;
labels, "--", "NA", statuses and the RelGap strings are compared as text. Row keys join the
identifying columns (ROW_KEY), column keys are the CSV column names.

Provenance (PROVENANCE.csv: every file is `74eb882^:experiments/...`): the battery first ran on
Sarimax.jl v0.1.3 (144fb6e; commit ecc4878) and was re-run on v0.3.0 (3f68017; commit
8ffc512), which regenerated these JSONL files and tables.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass

from .common import Cell, Context, read_jsonl, write_csv

TITLE = ("Empirical battery, Jul 2026: combine_results.py tables (validation, solver comparison, "
         "local against global estimation)")
RAW = "raw/battery-2026-07"
# Every `*_results.jsonl` of each block directory in the archived tree (what the original's
# sorted glob matched); `load` checks that the bundle still holds exactly these.
FILES = {
    "validation": ["julia_results.jsonl", "python_statsmodels_results.jsonl", "r_forecast_results.jsonl"],
    "solver": ["global_value_results.jsonl", "julia_results.jsonl", "jump_results.jsonl",
               "scip_cert_results.jsonl"],
}
INPUTS = [f"{RAW}/{block}/{name}" for block, names in FILES.items() for name in sorted(names)]

# table id -> row key over the identifying columns (the expected-value extractor uses the same)
ROW_KEY = {
    "battery.validation": "{Dataset}|{Order}|{Implementation}",
    "battery.solver_comparison": "{Level}|{Model}|{Solver}|{Obj}",
    "battery.global_value": "{Instance}|{Solver}",
}
DASH = "--"


@dataclass(frozen=True)
class Val:
    """One table entry: the text combine_results printed and the value compared."""
    text: str                    # as printed in the CSV (and, escaped, in the LaTeX)
    value: object                # float/int compared at `decimals`, or text compared exactly
    decimals: int | None = None
    missing: bool = False        # a None field: pandas prints "" in the CSV, "None" in the LaTeX


def fmt(x, nd: int = 4) -> Val:
    """combine_results.fmt."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return Val(DASH, DASH)
    if isinstance(x, (int, float)):
        return Val(f"{x:.{nd}f}", float(x), nd)
    return Val(str(x), str(x))


def lit(x) -> Val:
    """A record field (label, count, status) that the original put in the table unformatted."""
    if x is None:
        return Val("", "", missing=True)
    if isinstance(x, int) and not isinstance(x, bool):
        return Val(str(x), x, 0)
    return Val(str(x), str(x))


def first_coef(est, *keys):
    """combine_results.first_coef: first element of a list value, or a scalar value."""
    if not isinstance(est, dict):
        return None
    for k in keys:
        v = est.get(k)
        if isinstance(v, list) and v:
            return v[0]
        if isinstance(v, (int, float)):
            return v
    return None


def sort_rows(rows: list[dict], by: list[str]) -> list[dict]:
    """pandas DataFrame.sort_values(by): stable, ascending, missing values last."""
    return sorted(rows, key=lambda row: tuple((row[c].missing, row[c].text) for c in by))


def tex_escape(s) -> str:
    s = str(s)
    # theta, phi, epsilon, right arrow, superscript two (as in the original, pdfLaTeX-safe)
    for a, b in (("\u03b8", "theta"), ("\u03c6", "phi"), ("\u03b5", "eps"),
                 ("\u2192", "->"), ("\u00b2", "2")):
        s = s.replace(a, b)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def latex_table(cols: list[str], rows: list[dict], caption: str, label: str, colfmt: str) -> str:
    if colfmt is None or len(colfmt) != len(cols):
        colfmt = "l" * len(cols)
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{colfmt}}}", r"\hline"]
    lines.append(" & ".join(tex_escape(c) for c in cols) + r" \\")
    lines.append(r"\hline")
    for row in rows:
        lines.append(" & ".join(tex_escape("None" if row[c].missing else row[c].text)
                                for c in cols) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def load(ctx: Context, block: str, strict: bool = True) -> list[dict]:
    """Records of one block. Strict (the archive): exactly the archived set of files. Not
    strict (a re-run of some tables): whichever of the archived file names the run wrote."""
    d = ctx.path(f"{RAW}/{block}")
    found = sorted(p.name for p in d.glob("*_results.jsonl"))
    if strict and found != sorted(FILES[block]):
        raise RuntimeError(f"{RAW}/{block}: *_results.jsonl files {found} differ from the "
                           f"archived set {sorted(FILES[block])}")
    records = []
    for name in sorted(FILES[block]):
        if strict or (d / name).exists():
            records += read_jsonl(d / name)
    return records


# ---------------------------------------------------------------------------------------------
# The three tables (builders as in combine_results.py)
# ---------------------------------------------------------------------------------------------
def build_validation(rec: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for r in rec["validation"]:
        est = r.get("estimates", {})
        # statsmodels keys: ar.L1, ma.L1; julia: phi/theta; R: ar1/ma1
        phi = first_coef(est, "phi") or est.get("ar.L1") or est.get("ar1")
        theta = first_coef(est, "theta") or est.get("ma.L1") or est.get("ma1")
        rows.append({
            "Implementation": lit(r["implementation"]), "Dataset": lit(r["dataset"]),
            "Order": lit(r["order"]), "phi1": fmt(phi), "theta1": fmt(theta),
            "logLik": fmt(r.get("loglike"), 2), "AIC": fmt(r.get("aic"), 2),
            "BIC": fmt(r.get("bic"), 2), "RSS": fmt(r.get("rss"), 2),
            "runtime_s": fmt(r.get("runtime_s"), 3), "status": lit(r.get("status")),
        })
    return sort_rows(rows, ["Dataset", "Order", "Implementation"])


def build_solver_comparison(rec: dict[str, list[dict]]) -> list[dict]:
    def detail(r):
        if r.get("n_distinct_optima") is not None:
            return f"{r.get('n_converged', '?')} conv, {r['n_distinct_optima']} optima"
        if r.get("rel_gap") is not None:
            return f"gap={fmt(r['rel_gap']).text}"
        if "warning_emitted" in r:
            return "warning emitted" if r["warning_emitted"] else "no warning"
        return DASH

    rows = [{
        "Level": lit(r.get("implementation", DASH)),
        "Model": lit(r["dataset"]), "Solver": lit(r.get("solver")),
        "Obj": lit(r.get("objective", DASH)),
        "Setting": lit(r.get("setting", DASH)),
        "ObjValue": fmt(r.get("obj_value")),
        "ObjSpread": fmt(r.get("obj_spread")),
        "Detail": lit(detail(r)),
        "Termination": lit(r.get("termination", DASH)),
        "runtime_s": fmt(r.get("runtime_s"), 2), "status": lit(r.get("status")),
    } for r in rec["solver"] if r.get("experiment") not in ("scip_scaling", "global_value")]
    return sort_rows(rows, ["Level", "Model", "Solver", "Obj"])


def build_global_value(rec: dict[str, list[dict]]) -> list[dict]:
    recs = sorted((r for r in rec["solver"] if r.get("experiment") == "global_value"),
                  key=lambda r: (r["dataset"], r["solver"]))
    return [{
        "Instance": lit(r["dataset"]),
        "Solver": lit(r["solver"]),
        "Certified": lit("yes" if r.get("certified") else "no"),
        "Objective": fmt(r.get("obj_value")),
        "phi": fmt(r.get("phi")) if r.get("phi") is not None else Val(DASH, DASH),
        "theta": fmt(r.get("theta")),
        "RMSE_oos": fmt(r.get("rmse_oos")),
        "MAE_oos": fmt(r.get("mae_oos")),
        "Time (s)": fmt(r.get("runtime_s"), 1),
    } for r in recs]


# (table id, builder, file stem, caption, label, LaTeX column format), in the original's order
TABLES = [
    ("battery.validation", build_validation, "table_validation",
     "Validation against established implementations under comparable SARIMA/SARIMAX specifications.",
     "tab:validation_implementations", "lllrrrrrrrl"),
    ("battery.solver_comparison", build_solver_comparison, "table_solver_comparison",
     "Solver comparison and nonconvexity diagnostics on small MA-containing instances.",
     "tab:solver_comparison", "lllllrrllrl"),
    ("battery.global_value", build_global_value, "table_global_value",
     "Local (Ipopt) versus certified global (SCIP) estimation of the same specification: "
     "in-sample objective, coefficients, and out-of-sample forecast errors over a held-out window.",
     "tab:global_value", "llllllrrr"),
]


# Blocks whose records each table reads (used when only some tables are rebuilt).
TABLE_BLOCKS = {"battery.validation": "validation", "battery.solver_comparison": "solver",
                "battery.global_value": "solver"}


def generate(ctx: Context, tables: list[str] | None = None) -> list[Cell]:
    """All three tables from the archive (tables=None, every archived file required), or only
    `tables` from a re-run directory (only the blocks those tables read)."""
    if tables is None:
        rec = {block: load(ctx, block) for block in FILES}
    else:
        rec = {block: (load(ctx, block, strict=False) if block in {TABLE_BLOCKS[t] for t in tables} else [])
               for block in FILES}
    cells: list[Cell] = []
    for table, build, stem, caption, label, colfmt in TABLES:
        if tables is not None and table not in tables:
            continue
        rows = build(rec)
        if not rows:                 # the original skipped the table; its cells stay unmatched
            continue
        cols = list(rows[0])
        write_csv(ctx.out / f"{stem}.csv", cols, [[row[c].text for c in cols] for row in rows])
        tex = ctx.out / f"{stem}.tex"
        tex.parent.mkdir(parents=True, exist_ok=True)
        with open(tex, "w", encoding="utf-8", newline="\n") as f:   # no final newline, as the original
            f.write(latex_table(cols, rows, caption, label, colfmt))

        key_cols = [f for _, f, _, _ in string.Formatter().parse(ROW_KEY[table]) if f]
        seen = set()
        for row in rows:
            key = ROW_KEY[table].format_map({c: row[c].text for c in key_cols})
            if key in seen:
                raise ValueError(f"{table}: duplicate row {key!r}")
            seen.add(key)
            cells += [Cell(table, key, c, row[c].value, row[c].decimals)
                      for c in cols if c not in key_cols]
    return cells
