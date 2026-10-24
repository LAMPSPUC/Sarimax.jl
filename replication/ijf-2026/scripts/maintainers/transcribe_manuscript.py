"""Transcribe the printed values of the submitted manuscript into the manuscript mapping.

    python transcribe_manuscript.py --source DIR [--write]

DIR is the LaTeX source snapshot of the submitted manuscript and its Supplementary Material
(main.tex, supplement.tex, chapters/, figs/), identified in results/manuscript/REVISION.toml by
its SHA-256. The printed values are read from the source itself, never typed: table cells from
the tabular and longtable environments, figure annotations from the figure PDFs (pdftotext),
and numbers quoted in the text by locating their exact LaTeX text in the named file. Each
printed value is paired with the archived source it is compared with (a table-group cell, or a
row of a derived table in the repository). The result is results/manuscript/manuscript_values.csv,
which scripts/maintainers/check_manuscript_mapping.py and ./reproduce.sh tables check.

The rows written out below (numbers quoted in the text, the tables and figure annotations added
for the submitted manuscript) carry their source here. The other table rows, and a few quoted
numbers, keep the source specification of the committed mapping, matched by label, row and
column; their printed values are read again from the source, and a row the source no longer
prints is reported. Without --write the script compares its output with the committed mapping
and exits 1 when they differ.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
from pathlib import Path

REPL = Path(__file__).resolve().parents[2]
OUT = REPL / "results" / "manuscript" / "manuscript_values.csv"
FIELDS = ["item", "label", "row", "column", "printed", "decimals", "scale", "source_kind", "source",
          "source_row", "source_column", "note"]
RECON = "results/historical/certification-2026-08/manuscript_sources.csv"
RERUN = "results/reruns/certification-2026-08-rerun-2026-10-21/manuscript_sources.csv"
GATE = "results/reruns/gate-1500-f90e554-2026-10-22/manuscript_sources.csv"
DERIVED = "results/manuscript/derived_quantities.csv"
INVENTORY = "reports/global_certificate_inventory.csv"
STABLE_SUM = "stable_admissibility_gavea/gavea.stable_objective.summary"
STABLE_PAIR = "stable_admissibility_gavea/gavea.stable_objective.paired"
CONTROLLED = "m4_objectives_controlled/controlled.table5"

# --------------------------------------------------------------------------------------------
# LaTeX reading
# --------------------------------------------------------------------------------------------
INPUT = re.compile(r"\\input\{([^}]*)\}")


def tex_files(src: Path, doc: str) -> list[Path]:
    out = []
    for inc in INPUT.findall((src / doc).read_text(encoding="utf-8")):
        f = src / (inc if inc.endswith(".tex") else inc + ".tex")
        out.append(f)
        out += [src / (s if s.endswith(".tex") else s + ".tex") for s in INPUT.findall(f.read_text(encoding="utf-8"))]
    return out


def clean(cell: str) -> str:
    c = re.sub(r"\$\^\{?[a-z]\}?\$", "", cell)                         # footnote marks yes$^{a}$
    c = re.sub(r"\\(textbf|textit|emph|texttt|mathbf|mathrm|text|underline)\{([^{}]*)\}", r"\2", c)
    c = c.replace("$-$", "-").replace("{,}", ",").replace("$", "").replace("\\%", "%").replace("~", " ")
    c = re.sub(r"\\(cmidrule|midrule|toprule|bottomrule|hline)(\([^)]*\))?(\{[^}]*\})?", "", c)
    c = re.sub(r"\\addlinespace(\[[^]]*\])?", "", c)
    c = re.sub(r"\^\{?\\dagger\}?", "", c)
    c = re.sub(r"\\[a-zA-Z]+\*?", "", c).replace("{", "").replace("}", "")
    return " ".join(c.split())


def environments(text: str, env: str):
    for m in re.finditer(rf"\\begin\{{{env}\}}", text):
        end = text.find(f"\\end{{{env}}}", m.end())
        yield m.start(), text[m.end():end]


def table_rows(src: Path) -> dict[str, list[list[str]]]:
    """label -> rows of cleaned cells, for every labelled tabular/longtable of both documents."""
    out: dict[str, list[list[str]]] = {}
    for doc in ("main.tex", "supplement.tex"):
        for f in tex_files(src, doc):
            if not f.exists():
                continue
            s = f.read_text(encoding="utf-8")
            labels = [(m.start(), m.group(1)) for m in re.finditer(r"\\label\{(tab:[^}]*)\}", s)]
            for env in ("tabular", "longtable"):
                for pos, body in environments(s, env):
                    if not labels:
                        continue
                    label = min(labels, key=lambda l: abs(l[0] - pos))[1]
                    body = re.sub(r"(?m)(?<!\\)%.*$", "", body)                     # comments, not \%
                    body = re.sub(r"^\s*\{[^\n]*\}\s*\n", "\n", body, count=1)          # column spec
                    if env == "longtable":
                        body = body.split("\\endhead", 1)[-1]
                    rows = []
                    for line in re.split(r"\\\\", body):
                        cells = [clean(c) for c in line.split("&")]
                        if len(cells) > 1 and any(cells):
                            rows.append(cells)
                    out.setdefault(label, []).extend(rows)
    return out


def lookup(rows: list[list[str]], keys: int, cols: dict[int, str], carry: bool = True) -> dict:
    """(row key, column name) -> printed cell; the first `keys` cells form the row key, and an
    empty leading cell repeats the previous row's (a group continued on the next line)."""
    out, prev = {}, [""] * keys
    for cells in rows:
        k = [c or (prev[i] if carry else "") for i, c in enumerate(cells[:keys])]
        prev = k
        for i, name in cols.items():
            if i < len(cells):
                out[("|".join(k), name)] = cells[i]
    return out


def flat(src: Path, rel: str) -> str:
    """The prose of a source file on one line, with \\% as % and \\texttt{x} as x."""
    s = re.sub(r"(?m)(?<!\\)%.*$", "", (src / rel).read_text(encoding="utf-8"))
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"\1", s).replace("\\%", "%")
    return " ".join(s.split())


# --------------------------------------------------------------------------------------------
# Specification
# --------------------------------------------------------------------------------------------
ITEM = {  # label -> item name in the submitted manuscript
    "tab:matched_validation": "Table 2", "tab:m4_forecasting": "Table 3", "tab:m4_paired_quantiles": "Table 4",
    "tab:m4_objectives": "Table 5", "tab:objective_paired_quantiles": "Table 6",
    "tab:local_global_forecasts": "Table 7", "fig:m4_pair_ecdf": "Figure 2", "fig:objective_ecdf": "Figure 3",
    "tab:exog_validation": "Table S2", "tab:weekly_mse_protocols": "Table S4", "tab:global_experiment": "Table S5",
    "tab:global_certificates": "Table S6", "tab:global_isolated": "Table S7",
    "tab:ten_seed_initialization": "Table S8", "tab:initialization_gate": "Table S9", "tab:quantile_demo": "Table S10",
    "tab:multistart": "Table S11", "tab:supp_m4_marginal": "Table S12", "tab:supp_obj_mase": "Table S13",
    "tab:supp_obj_smape": "Table S14", "tab:stable_m4": "Table S15", "tab:supp_stable_mase": "Table S16",
}
# tables whose rows keep the source specification of the committed mapping: (key cells, columns)
REUSED = {
    "tab:matched_validation": (3, {3: "phi1", 4: "theta1", 5: "RSS"}),
    "tab:m4_forecasting": (1, {1: "Sarimax.jl", 2: "Reference"}),
    "tab:m4_paired_quantiles": (2, {2: "N", 3: "Mean", 4: "q05", 5: "Median", 6: "q95", 7: "Lower (%)", 8: "Tie (%)"}),
    "tab:local_global_forecasts": (2, {2: "Certified", 3: "Obj.", 4: "phi1", 5: "theta1", 6: "RMSE", 7: "MAE", 8: "Time (s)"}),
    "tab:exog_validation": (1, {1: "RMSE Julia", 2: "RMSE R", 3: "Delta log RMSE"}),
    "tab:quantile_demo": (1, {1: "In-sample proportion", 2: "One-step proportion", 3: "Mean phi1"}),
    "tab:multistart": (1, {1: ":zeroed", 2: ":innovations"}),
    "tab:supp_m4_marginal": (3, {3: "N", 4: "Mean", 5: "q05", 6: "Median", 7: "q95"}),
}
# the same, for the tables of the controlled objective comparison
TEMPLATE = {
    "tab:m4_objectives": (1, {1: "Reference", 2: "MSE", 3: "Huber", 4: "MAE", 5: "Ridge"}),
    "tab:objective_paired_quantiles": (2, {2: "N", 3: "Mean", 4: "q05", 5: "Median", 6: "q95"}),
    "tab:supp_obj_mase": (2, {2: "q01", 3: "q05", 4: "q25", 5: "Median", 6: "q75", 7: "q95", 8: "q99"}),
    "tab:supp_obj_smape": (2, {2: "q01", 3: "q05", 4: "q25", 5: "Median", 6: "q75", 7: "q95", 8: "q99"}),
}


def decimals(printed: str) -> str:
    p = printed.replace(",", "").lstrip("+-")
    if "e" in p.lower():
        mant, exp = p.lower().split("e")
        return str((len(mant.split(".")[1]) if "." in mant else 0) - int(exp))
    return str(len(p.split(".")[1])) if "." in p else "0"


def row(item, label, r, c, printed, kind, source, srow, scol, note="", scale="", dec=None):
    return {"item": item, "label": label, "row": r, "column": c, "printed": printed,
            "decimals": decimals(printed) if dec is None else dec, "scale": scale, "source_kind": kind,
            "source": source, "source_row": srow, "source_column": scol, "note": note}


def load(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def reuse(label, spec_rows, cells, problems):
    """The mapping's rows of `label`, with the printed value read again from the source."""
    keys, cols = (REUSED.get(label) or TEMPLATE.get(label))
    look = lookup(cells.get(label, []), keys, cols)
    out = []
    for r in spec_rows:
        if r["label"] != label or r["row"].startswith("caption"):
            continue
        key = re.sub(r" \[[^]]*\]$", "", r["row"])
        printed = look.get((key, r["column"]))
        if printed is None:
            problems.append(f"{label}: {r['row']} / {r['column']} not found in the submitted table")
            continue
        if printed in ("--", "-", ""):
            continue                                   # an empty cell in the submitted table
        printed = printed.replace("%", "")
        out.append(dict(r, item=ITEM[label], printed=printed,
                        decimals=r["decimals"] if r["decimals"] == "" else decimals(printed)))
    return out


def text_rows(src: Path, problems: list[str]) -> list[dict]:
    """Numbers quoted in the text: (file, the exact LaTeX text, printed value, source)."""
    E, G, S5 = "chapters/experiments.tex", "chapters/supplement_tables.tex", "chapters/experiments.tex"
    spec = [
        # Section 5.1
        ("Section 5.1", "sec:experimental_setup", E, "0.8977 in the six-frequency comparison", "weekly MSE OWA, six-frequency comparison", "0.8977",
         "regenerated", "m4_benchmark/m4_benchmark.table_m4", "weekly|total", "OWA.ours", ""),
        ("Section 5.1", "sec:experimental_setup", E, "0.9002 in the objective comparison", "weekly MSE OWA, controlled objective comparison", "0.9002",
         "regenerated", CONTROLLED, "weekly", "MSE", ""),
        ("Section 5.1", "sec:experimental_setup", E, "reflection coordinates recovers 0.8977", "weekly MSE OWA, matched re-execution with reflection coordinates", "0.8977",
         "regenerated", "m4_objectives_controlled/controlled.weekly", "extra_run_P1_table3_protocol", "OWA",
         "the single extra run of the controlled campaign under the six-frequency protocol (359/359 series identical)"),
        # Section 5.3
        ("Section 5.3", "sec:m4_validation", E, "uses 411 series with paired outputs", "hourly series with paired outputs", "411",
         "regenerated", "m4_paired/m4_paired.paired", "hourly|MASE", "N_paired", ""),
        # Section 5.4
        ("Section 5.4", "sec:m4_objectives", E, "48,000 monthly, 24,000 quarterly", "series evaluated, monthly", "48,000", "regenerated", CONTROLLED, "monthly", "N", ""),
        ("Section 5.4", "sec:m4_objectives", E, "24,000 quarterly, 23,000 yearly", "series evaluated, quarterly", "24,000", "regenerated", CONTROLLED, "quarterly", "N", ""),
        ("Section 5.4", "sec:m4_objectives", E, "23,000 yearly and 359 weekly", "series evaluated, yearly", "23,000", "regenerated", CONTROLLED, "yearly", "N", ""),
        ("Section 5.4", "sec:m4_objectives", E, "and 359 weekly series", "series evaluated, weekly", "359", "regenerated", CONTROLLED, "weekly", "N", ""),
        ("Section 5.4", "sec:m4_objectives", E, "Huber lowers MASE for 45.7% of monthly", "monthly series with lower MASE under Huber than MSE (%)", "45.7",
         "derived", DERIVED, "quantity=s54_pct_lower_tol1e-06|monthly|huber", "value", "Delta < -1e-6, as the text declares"),
        ("Section 5.4", "sec:m4_objectives", E, "and 48.5% of weekly series", "weekly series with lower MASE under Huber than MSE (%)", "48.5",
         "derived", DERIVED, "quantity=s54_pct_lower_tol1e-06|weekly|huber", "value", "Delta < -1e-6"),
        ("Section 5.4", "sec:m4_objectives", E, "MAE for 46.2% of quarterly series", "quarterly series with lower MASE under MAE than MSE (%)", "46.2",
         "derived", DERIVED, "quantity=s54_pct_lower_tol1e-06|quarterly|mae", "value", "Delta < -1e-6"),
        ("Section 5.4", "sec:m4_objectives", E, "reduces annual OWA from 0.8950 to 0.8866", "Stable alpha = 0.5, yearly: paired MSE OWA", "0.8950",
         "regenerated", STABLE_SUM, "yearly|mse", "OWA", ""),
        ("Section 5.4", "sec:m4_objectives", E, "from 0.8950 to 0.8866", "Stable alpha = 0.5, yearly: Stable OWA", "0.8866",
         "regenerated", STABLE_SUM, "yearly|stable_050", "OWA", ""),
        ("Section 5.4", "sec:m4_objectives", E, "the R benchmark (0.8913)", "Stable alpha = 0.5, yearly: auto.arima OWA on the same series", "0.8913",
         "regenerated", STABLE_SUM, "yearly|stable_050", "OWA_autoarima_same_series", ""),
        # Section 5.5
        ("Section 5.5", "sec:estimation_sensitivity", E, "fit was 0.2076 for a penalized", "ten-seed initialization: median coefficient distance, penalized free pre-sample", "0.2076",
         "derived", RECON, "quantity=s55_init_distance_free_presample", "value", "recovered aggregate with its per-seed rows"),
        ("Section 5.5", "sec:estimation_sensitivity", E, "versus 0.0863 for", "ten-seed initialization: median coefficient distance, :innovations", "0.0863",
         "derived", RECON, "quantity=s55_init_distance_innovations", "value", "recovered aggregate with its per-seed rows"),
        ("Section 5.5", "sec:estimation_sensitivity", E, "A separate comparison uses 1,500 M4 series", "series of the paired initialization comparison", "1,500",
         "derived", GATE, "quantity=gate_rerun_paired", "value", ""),
        ("Section 5.5", "sec:estimation_sensitivity", E, "ceiling was 48.3%", "searches with a candidate at the 200-iteration ceiling, :penalized, re-run (%)", "48.3",
         "derived", GATE, "quantity=gate_rerun_penalized_ceiling_pct", "value", "re-run of 2026-10-22, recomputed from its per-series records"),
        ("Section 5.5", "sec:estimation_sensitivity", E, "and 0.2% under :innovations (historically", "same, :innovations, re-run (%)", "0.2",
         "derived", GATE, "quantity=gate_rerun_innovations_ceiling_pct", "value", "re-run of 2026-10-22"),
        ("Section 5.5", "sec:estimation_sensitivity", E, "(historically, 47.4% and 0.2%)", "same, :penalized, August aggregate (%)", "47.4",
         "derived", GATE, "quantity=gate_august_penalized_ceiling_pct", "value", "August aggregate; per-series outputs lost"),
        ("Section 5.5", "sec:estimation_sensitivity", E, "47.4% and 0.2%)", "same, :innovations, August aggregate (%)", "0.2",
         "derived", GATE, "quantity=gate_august_innovations_ceiling_pct", "value", "August aggregate"),
        ("Section 5.5", "sec:estimation_sensitivity", E, "1,494 series reproduced the archived orders", ":innovations series reproducing the August per-series record", "1,494",
         "derived", GATE, "quantity=gate_rerun_innovations_identical", "value", ""),
        ("Section 5.5", "sec:estimation_sensitivity", E, "the other six selected different orders", ":innovations series selecting another order", "6",
         "derived", GATE, "quantity=gate_rerun_innovations_divergent", "value", ""),
        ("Section 5.5", "sec:estimation_sensitivity", E, "contains 1,099 paired fits", "AICc verification: paired fits", "1,099",
         "derived", RECON, "quantity=s55_rank_aicc_parity_fits", "value", ""),
        ("Section 5.5", "sec:estimation_sensitivity", E, "distances of 21.6 and 3.3", "CSS-to-ML distance, seasonal pre-sample extension off", "21.6",
         "derived", RECON, "quantity=s55_rank_distance_prior_off", "value", "unpaired; distinct sets of fits"),
        ("Section 5.5", "sec:estimation_sensitivity", E, "21.6 and 3.3 with the seasonal", "CSS-to-ML distance, seasonal pre-sample extension on", "3.3",
         "derived", RECON, "quantity=s55_rank_distance_prior_on", "value", ""),
        ("Section 5.5", "sec:estimation_sensitivity", E, "distinct sets of 14 and 12 fits", "CSS-to-ML distance: fits with the extension off", "14",
         "derived", RECON, "quantity=s55_rank_distance_prior_off_n", "value", ""),
        ("Section 5.5", "sec:estimation_sensitivity", E, "14 and 12 fits", "CSS-to-ML distance: fits with the extension on", "12",
         "derived", RECON, "quantity=s55_rank_distance_prior_on_n", "value", ""),
        # Section 5.6
        ("Section 5.6", "sec:global_diagnostics", E, "nine clusters with :zeroed", "multistart, near-cancelling ARMA(2,2), :zeroed clusters", "9",
         "derived", "results/historical/revision-2026-09/multistart_kstart.csv", "case=ARMA(2,2), near-cancelling;initialization=zeroed",
         "distinct_objective_values", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "and ten with :innovations", "multistart, near-cancelling ARMA(2,2), :innovations clusters", "10",
         "derived", "results/historical/revision-2026-09/multistart_kstart.csv", "case=ARMA(2,2), near-cancelling;initialization=innovations",
         "distinct_objective_values", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "reconstruction contains 23 OPTIMAL", "certification grid, August record: OPTIMAL", "23", "derived", RECON, "quantity=s5_cell_all_O", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "21 TIME\\_LIMIT and two externally interrupted", "certification grid, August record: TIME_LIMIT", "21", "derived", RECON, "quantity=s5_cell_all_T", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "and two externally interrupted runs", "certification grid, August record: interrupted", "2", "derived", RECON, "quantity=s5_cell_all_X", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "rerun reports 23, 20 and three", "certification grid, 2026-10-21 re-run: OPTIMAL", "23", "derived", RERUN, "quantity=rerun_cell_all_O", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "reports 23, 20 and three", "certification grid, 2026-10-21 re-run: TIME_LIMIT", "20", "derived", RERUN, "quantity=rerun_cell_all_T", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "20 and three, respectively", "certification grid, 2026-10-21 re-run: did not return", "3", "derived", RERUN, "quantity=rerun_cell_all_X", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "All 23 previously reported certificates were reproduced", "certificates reproduced by the re-run", "23",
         "derived", RERUN, "quantity=rerun_certificates_reproduced", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "None of the twelve standard", "seasonal grid cases at the time limit", "12", "derived", RECON, "quantity=s5_cell_B_T", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "objective was $2.52\\times10^{-9}$", "seasonal grid: largest |SCIP incumbent - Ipopt objective|", "2.52e-9",
         "derived", RECON, "quantity=s56_seasonal_ipopt_minus_scip_max", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "from $1.94\\times10^{-4}$", "seasonal grid: smallest remaining normalized gap (August)", "1.94e-4",
         "derived", RECON, "quantity=s56_seasonal_gap_min", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "to $4.90\\times10^{-1}$", "seasonal grid: largest remaining normalized gap (August)", "4.90e-1",
         "derived", RECON, "quantity=s56_seasonal_gap_max", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "(median $1.51\\times10^{-2}$)", "seasonal grid: median remaining normalized gap (August)", "1.51e-2",
         "derived", RECON, "quantity=s56_seasonal_gap_median", "value", "median of 12: mean of the 6th and 7th values"),
        ("Section 5.6", "sec:global_diagnostics", E, "Ipopt returned 46.660913", "case 26: Ipopt objective", "46.660913", "derived", RECON, "quantity=s56_near_cancellation_ipopt", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "feasible objective of 45.314976", "case 26: SCIP feasible objective", "45.314976", "derived", RECON, "quantity=s56_near_cancellation_scip", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "gaps of $1.09\\times10^{-9}$", "case 44: normalized gap (re-run)", "1.09e-9", "derived", RERUN, "quantity=rerun_44_gap", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "and $1.07\\times10^{-8}$", "case 45: normalized gap (re-run)", "1.07e-8", "derived", RERUN, "quantity=rerun_45_gap", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "Ipopt objectives of 57.868044", "case 44: Ipopt objective", "57.868044", "derived", RECON, "quantity=s56_certified_1_ipopt", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "and 54.706978 exceed", "case 45: Ipopt objective", "54.706978", "derived", RECON, "quantity=s56_certified_2_ipopt", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "certified values 57.536983", "case 44: certified objective", "57.536983", "derived", RECON, "quantity=s56_certified_1_scip", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "and 53.681295 by", "case 45: certified objective", "53.681295", "derived", RECON, "quantity=s56_certified_2_scip", "value", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "by 0.575% and", "case 44: local excess over the certified value (%)", "0.575",
         "derived", RECON, "quantity=s56_certified_1_local_excess", "value", "", "100"),
        ("Section 5.6", "sec:global_diagnostics", E, "and 1.911%, using the SCIP objectives", "case 45: local excess over the certified value (%)", "1.911",
         "derived", RECON, "quantity=s56_certified_2_local_excess", "value", "", "100"),
        ("Section 5.6", "sec:global_diagnostics", E, "optimality at 899.958 seconds", "Table 7, PJME: time of the unlogged OPTIMAL record (s)", "899.958",
         "derived", INVENTORY, 'instance=pjme_daily(scaled) ARMA(1,0,1) T=80;run=archive 8ffc512', "runtime_s", ""),
        ("Section 5.6", "sec:global_diagnostics", E, "relative gap $3.4\\times10^{-5}$", "Table 7, PJME: relative gap of the logged re-execution", "3.4e-5",
         "derived", INVENTORY, 'instance=pjme_daily(scaled) ARMA(1,0,1) T=80;run=re-run 2026-10-20', "relative_gap", ""),
        # Supplementary Material, text beside the tables
        ("Table S5", "tab:global_experiment", G, "An identified re-execution returned 43 of 46", "instances that returned in the 2026-10-21 re-run", "43",
         "derived", RERUN, "quantity=rerun_returned", "value", ""),
        ("Table S9", "tab:initialization_gate", G, "Historical aggregates were 13.9000, 0.9528 and 47.4%", "August aggregate, :penalized, mean sMAPE", "13.9000",
         "derived", GATE, "quantity=gate_august_penalized_smape", "value", "August aggregate; per-series outputs lost"),
        ("Table S9", "tab:initialization_gate", G, "13.9000, 0.9528 and", "August aggregate, :penalized, mean MASE", "0.9528", "derived", GATE, "quantity=gate_august_penalized_mase", "value", ""),
        ("Table S9", "tab:initialization_gate", G, "0.9528 and 47.4% under", "August aggregate, :penalized, searches at the ceiling (%)", "47.4", "derived", GATE, "quantity=gate_august_penalized_ceiling_pct", "value", ""),
        ("Table S9", "tab:initialization_gate", G, "versus 13.9395, 0.9471", "August aggregate, :innovations, mean sMAPE", "13.9395", "derived", GATE, "quantity=gate_august_innovations_smape", "value", ""),
        ("Table S9", "tab:initialization_gate", G, "13.9395, 0.9471 and", "August aggregate, :innovations, mean MASE", "0.9471", "derived", GATE, "quantity=gate_august_innovations_mase", "value", ""),
        ("Table S9", "tab:initialization_gate", G, "0.9471 and 0.2% under", "August aggregate, :innovations, searches at the ceiling (%)", "0.2", "derived", GATE, "quantity=gate_august_innovations_ceiling_pct", "value", ""),
        ("Table S9", "tab:initialization_gate", G, "sMAPE/MASE to six decimals for 1,494 series", ":innovations series reproducing the August per-series record", "1,494",
         "derived", GATE, "quantity=gate_rerun_innovations_identical", "value", ""),
        ("Table S9", "tab:initialization_gate", G, "The AICc diagnostic contains 1,099 paired fits", "AICc verification: paired fits", "1,099", "derived", RECON, "quantity=s55_rank_aicc_parity_fits", "value", ""),
        ("Table S9", "tab:initialization_gate", G, "medians of 21.61 and 3.30", "CSS-to-ML distance, extension off", "21.61", "derived", RECON, "quantity=s55_rank_distance_prior_off", "value", ""),
        ("Table S9", "tab:initialization_gate", G, "21.61 and 3.30 log-likelihood", "CSS-to-ML distance, extension on", "3.30", "derived", RECON, "quantity=s55_rank_distance_prior_on", "value", ""),
        ("Table S9", "tab:initialization_gate", G, "distinct sets of 14 and 12 fits. This comparison", "CSS-to-ML distance: fits, extension off", "14", "derived", RECON, "quantity=s55_rank_distance_prior_off_n", "value", ""),
        ("Table S9", "tab:initialization_gate", G, "14 and 12 fits. This comparison", "CSS-to-ML distance: fits, extension on", "12", "derived", RECON, "quantity=s55_rank_distance_prior_on_n", "value", ""),
    ]
    out = []
    for item, label, rel, find, desc, printed, kind, source, srow, scol, note, *scale in spec:
        if find not in flat(src, rel):
            problems.append(f"{item}: '{find}' not found in {rel}")
            continue
        out.append(row(item, label, desc, "value", printed, kind, source, srow, scol, note, scale[0] if scale else ""))
    return out


def figures(src: Path, old: list[dict], problems: list[str]) -> list[dict]:
    out = []
    txt = subprocess.run(["pdftotext", "-layout", str(src / "figs" / "m4_paired_difference_distributions.pdf"), "-"],
                         capture_output=True, text=True, check=True).stdout
    mase, smape = txt.split("ECDF of sMAPE")
    for metric, part in (("MASE", mase), ("sMAPE", smape)):
        for sign in ("<0", "=0", ">0"):
            printed = sorted(int(x) for x in re.findall(rf"{re.escape(sign)} (\d+)%", part))
            rows = [r for r in old if r["label"] == "fig:m4_pair_ecdf" and r["row"].endswith(f"|{metric}")
                    and r["column"] == f"{sign} (%)"]
            if sorted(int(r["printed"]) for r in rows) != printed:
                problems.append(f"Figure 2 {metric} {sign}: panel values {printed} differ from the mapping's rows")
            out += [dict(r, item=ITEM["fig:m4_pair_ecdf"]) for r in rows]
    txt = subprocess.run(["pdftotext", "-layout", str(src / "figs" / "objectives_ecdf_delta_mase_controlled.pdf"), "-"],
                         capture_output=True, text=True, check=True).stdout
    for freq, n in re.findall(r"(Monthly|Quarterly|Yearly|Weekly), N = ([\d,]+)", txt):
        out.append(row(ITEM["fig:objective_ecdf"], "fig:objective_ecdf", freq.lower(), "panel N", n, "regenerated",
                       CONTROLLED, freq.lower(), "N", "common series of the panel"))
    if sum(r["label"] == "fig:objective_ecdf" for r in out) != 4:
        problems.append("Figure 3: four panel sizes expected")
    return out


def new_tables(cells: dict, src: Path, problems: list[str]) -> list[dict]:
    out = []
    # Table 5 caption sizes
    cap = flat(src, "chapters/experiments.tex")
    for freq in ("monthly", "quarterly", "yearly", "weekly"):
        m = re.search(rf"{freq} \$N=([\d{{}},]+)\$", cap)
        if not m:
            problems.append(f"Table 5 caption: no N for {freq}")
            continue
        out.append(row("Table 5", "tab:m4_objectives", f"caption ({freq})", "N", m.group(1).replace("{,}", ","),
                       "regenerated", CONTROLLED, freq, "N", ""))
    # Table S4: weekly MSE protocols
    look = lookup(cells["tab:weekly_mse_protocols"], 1, {1: "six-frequency", 2: "controlled"})
    out.append(row("Table S4", "tab:weekly_mse_protocols", "OWA (MSE)", "six-frequency MSE study", look[("OWA (MSE)", "six-frequency")],
                   "regenerated", "m4_benchmark/m4_benchmark.table_m4", "weekly|total", "OWA.ours", ""))
    out.append(row("Table S4", "tab:weekly_mse_protocols", "OWA (MSE)", "controlled objective study", look[("OWA (MSE)", "controlled")],
                   "regenerated", CONTROLLED, "weekly", "MSE", ""))
    for col, kind, source, srow, scol in (("six-frequency", "regenerated", "m4_paired/m4_paired.paired", "weekly|MASE", "N_paired"),
                                          ("controlled", "regenerated", CONTROLLED, "weekly", "N")):
        m = re.search(r"(\d+)", look[("M4 series", col)])
        out.append(row("Table S4", "tab:weekly_mse_protocols", "M4 series", col, m.group(1), kind, source, srow, scol, ""))
    # Table S5: grid by cell, August and re-run, O/T/X
    for cells_row in cells["tab:global_experiment"]:
        cell = cells_row[0]
        if cell not in "ABCDEFG" and cell != "Total":
            continue
        key = "all" if cell == "Total" else cell
        for col, src_file, prefix in ((3, RECON, "s5_cell"), (4, RERUN, "rerun_cell")):
            for code, v in zip("OTX", cells_row[col].split("/")):
                out.append(row("Table S5", "tab:global_experiment", f"cell {cell}",
                               f"{'August' if col == 3 else '21 Oct.'} {code}", v, "derived", src_file,
                               f"quantity={prefix}_{key}_{code}", "value",
                               "O OPTIMAL, T TIME_LIMIT, X interrupted without return"))
    # Table S6: selected records of the re-run
    for c in cells["tab:global_certificates"]:
        if not c[0].isdigit():
            continue
        for k, col in ((1, "ipopt"), (2, "scip"), (3, "dual"), (4, "gap")):
            v = c[k].replace("\\mathrm", "").replace("mathrme", "e").replace(" ", "")
            v = re.sub(r"e(-?)\{?-?\}?", "e", v) if False else v
            v = v.replace("e−", "e-")
            if v in ("--", "-", ""):
                continue
            out.append(row("Table S6", "tab:global_certificates", f"case {c[0]}", col, v, "derived", RERUN,
                           f"quantity=rerun_{c[0]}_{col}", "value", ""))
    # Table S7: isolated runs
    for c in cells["tab:global_isolated"]:
        if not c[0].isdigit():
            continue
        for k, col in ((2, "gap_pct"), (3, "solver_time"), (4, "call_elapsed")):
            out.append(row("Table S7", "tab:global_isolated", f"case {c[0]}", col, c[k], "derived", RERUN,
                           f"quantity=isolated_{c[0]}_{col}", "value", "isolated run of 2026-10-22"))
    # Table S8: ten-seed initialization diagnostic
    q = {"A": "s55_init_distance_free_presample", "B": "s55_init_distance_free_presample_cross_term",
         "C": "s55_init_distance_innovations", "D": "s55_init_distance_innovations_determinant"}
    for c in cells["tab:ten_seed_initialization"]:
        if c[0] in q:
            out.append(row("Table S8", "tab:ten_seed_initialization", f"arm {c[0]}", "median distance to R ML", c[2],
                           "derived", RECON, f"quantity={q[c[0]]}", "value", "recovered aggregate with its per-seed rows"))
    # Table S9: paired 1,500-series comparison (re-run)
    for c in cells["tab:initialization_gate"]:
        arm = c[0].lstrip(":")
        if arm not in ("penalized", "innovations"):
            continue
        for k, col in ((1, "smape"), (2, "mase"), (3, "ceiling_pct")):
            out.append(row("Table S9", "tab:initialization_gate", c[0], col, c[k], "derived", GATE,
                           f"quantity=gate_rerun_{arm}_{col}", "value", "re-run of 2026-10-22, recomputed from its per-series records"))
    # Table S15: Stable/CVaR OWA
    arms = {"0.5": "stable_050", "0.9": "stable_090"}
    look, prev = [], ""
    for c in cells["tab:stable_m4"]:
        freq = c[0] or prev
        prev = freq
        if c[1] not in arms:
            continue
        f = "monthly (alpha=0.5 subset)" if freq.startswith("Monthly") and c[1] == "0.5" else freq.lower()
        a = arms[c[1]]
        tag = f"{freq}|alpha={c[1]}"
        out += [row("Table S15", "tab:stable_m4", tag, "N", c[2], "regenerated", STABLE_SUM, f"{f}|{a}", "N", ""),
                row("Table S15", "tab:stable_m4", tag, "R", c[3], "regenerated", STABLE_SUM, f"{f}|{a}", "OWA_autoarima_same_series", ""),
                row("Table S15", "tab:stable_m4", tag, "MSE", c[4], "regenerated", STABLE_SUM, f"{f}|mse", "OWA", ""),
                row("Table S15", "tab:stable_m4", tag, "Stable", c[5], "regenerated", STABLE_SUM, f"{f}|{a}", "OWA", ""),
                row("Table S15", "tab:stable_m4", tag, "Delta vs MSE", c[6], "derived", DERIVED,
                    f"quantity=s15_delta_owa|{f}|{a}", "value", "difference of the two OWA cells of the same campaign")]
    # Table S16: Stable/CVaR paired MASE quantiles
    for c in cells["tab:supp_stable_mase"]:
        if c[1] not in arms:
            continue
        f = "monthly (alpha=0.5 subset)" if c[0].startswith("Monthly") and c[1] == "0.5" else c[0].lower()
        a, tag = arms[c[1]], f"{c[0]}|alpha={c[1]}"
        out += [row("Table S16", "tab:supp_stable_mase", tag, "N", c[2], "regenerated", STABLE_SUM, f"{f}|{a}", "N", "")]
        for k, col in ((3, "MASE_q05"), (4, "MASE_median_diff"), (5, "MASE_q95"), (6, "MASE_pct_lower")):
            out.append(row("Table S16", "tab:supp_stable_mase", tag, col, c[k], "regenerated", STABLE_PAIR, f"{f}|{a}|mse", col, ""))
    # Table 7: Ipopt times at the precision of the archived record
    names = {"Sim. MA(1)": "sim_arma MA(1) T=80", "Sim. ARMA(1,0,1)": "sim_arma ARMA(1,0,1) T=80",
             "PJME ARMA(1,0,1)": "pjme_daily(scaled) ARMA(1,0,1) T=80"}
    for c in cells["tab:local_global_forecasts"]:
        if c[1] == "Ipopt" and c[0] in names:
            out.append(row("Table 7", "tab:local_global_forecasts", f"{c[0]}|Ipopt", "Time (s)", c[8], "derived", DERIVED,
                           f"quantity=t7_ipopt_time|{names[c[0]]}", "value",
                           "archived record (8ffc512, bundle part 1); the first solve of the process includes compilation"))
    return out


def build(src: Path, problems: list[str]) -> list[dict]:
    cells = table_rows(src)
    old = load(OUT)                                     # the committed mapping: the source specification
    out: list[dict] = []
    for label in REUSED:
        rows = reuse(label, old, cells, problems)
        if label == "tab:local_global_forecasts":       # Ipopt times and the PJME label are written below
            rows = [r for r in rows if not (r["column"] == "Time (s)" and r["row"].endswith("|Ipopt"))
                    and not (r["column"] == "Certified" and r["row"].startswith("PJME"))]
        out += rows
    for label in TEMPLATE:
        out += reuse(label, old, cells, problems)
    # quoted numbers whose source specification is kept from the mapping, checked against the text
    keep = {("sec:m4_validation", r) for r in ("yearly MASE paired difference, 5th percentile",
                                                 "yearly MASE paired difference, 95th percentile",
                                                 "top-5% share of sum |Delta|, smallest of 12 panels (%)",
                                                 "top-5% share of sum |Delta|, largest of 12 panels (%)")}
    keep |= {("sec:estimation_sensitivity", r["row"]) for r in old if r["label"] == "sec:estimation_sensitivity"
             and r["printed"] in ("0.9095", "0.9038", "0.9094", "0.9115")}
    keep |= {("sec:global_diagnostics", r["row"]) for r in old if r["label"] == "sec:global_diagnostics"
             and r["printed"] in ("1.7", "14.8")}
    exp, sens = flat(src, "chapters/experiments.tex"), None
    for r in old:
        if (r["label"], r["row"]) in keep:
            lit = {"0.460": "-0.460", "41": "41--78\\%", "78": "41--78\\%"}.get(r["printed"].lstrip("-"), r["printed"])
            if lit.replace("\\%", "%") not in exp.replace("\\%", "%") and r["printed"].lstrip("-") not in exp:
                problems.append(f"{r['label']} {r['row']}: {r['printed']} not found in the submitted text")
                continue
            out.append(dict(r, item={"sec:m4_validation": "Section 5.3", "sec:estimation_sensitivity": "Section 5.5",
                                     "sec:global_diagnostics": "Section 5.6"}[r["label"]]))
    out += text_rows(src, problems)
    out += figures(src, old, problems)
    out += new_tables(cells, src, problems)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    problems: list[str] = []
    rows = build(a.source, problems)
    buf = io.StringIO()
    w = csv.DictWriter(buf, FIELDS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    for p in problems:
        print("PROBLEM", p)
    if a.write:
        OUT.write_text(buf.getvalue(), encoding="utf-8", newline="\n")
        print(f"wrote {OUT.relative_to(REPL)}: {len(rows)} printed values")
    else:
        same = OUT.exists() and OUT.read_text(encoding="utf-8") == buf.getvalue()
        print(f"{len(rows)} printed values; " + ("identical to the committed mapping" if same else "DIFFERENT from the committed mapping"))
        return 0 if same and not problems else 1
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
