"""Expected cells of the `scip_ipopt` group, parsed from the summary blocks ("RESUMO") that
the original scripts printed at the end of raw/scip-ipopt-yearly-2026-07/scip_experiment_log.txt
and scip_constr_log.txt (July 2026)."""

from __future__ import annotations

import re
from pathlib import Path

RAW = "raw/scip-ipopt-yearly-2026-07"


def _dec(tok: str) -> str:
    return str(len(tok.split(".")[1])) if "." in tok else "0"


def _row(table, column, tok, source, decimals=None):
    return {"table": table, "row": "summary", "column": column, "expected": tok,
            "decimals": decimals if decimals is not None else _dec(tok), "source": source}


def extract(bundle: Path) -> list[dict]:
    rows = []
    src = f"{RAW}/scip_experiment_log.txt"
    text = (bundle / src).read_text(encoding="utf-8")
    t = "scip_ipopt.free"
    rows.append(_row(t, "n", re.search(r"RESUMO \(n=(\d+)\)", text).group(1), src))
    for col, pat in (("scip_lower_objective", r"objetivo MENOR .*?: (\d+)/"),
                     ("sigma2_equal_rel_1e-6", r"sigma2 praticamente igual .*?: (\d+)/"),
                     ("coefficients_equal_1e-4", r"coeficientes iguais .*?: (\d+)/"),
                     ("forecasts_equal_1e-6", r"previsoes iguais .*?: (\d+)/"),
                     ("scip_status_optimal", r"SCIP OPTIMAL: (\d+)/")):
        rows.append(_row(t, col, re.search(pat, text).group(1), src))
    m = re.search(r"sMAPE\s+scip=([\d.]+)\s+ipopt=([\d.]+)", text)
    rows += [_row(t, "smape_scip", m.group(1), src, "4"), _row(t, "smape_ipopt", m.group(2), src, "4")]
    m = re.search(r"MASE\s+scip=([\d.]+)\s+ipopt=([\d.]+)", text)
    rows += [_row(t, "mase_scip", m.group(1), src, "5"), _row(t, "mase_ipopt", m.group(2), src, "5")]
    m = re.search(r"tempo\s+scip=([\d.]+)s\s+ipopt=([\d.]+)s", text)
    rows += [_row(t, "seconds_scip", m.group(1), src, "2"), _row(t, "seconds_ipopt", m.group(2), src, "3")]

    src = f"{RAW}/scip_constr_log.txt"
    text = (bundle / src).read_text(encoding="utf-8")
    t = "scip_ipopt.constrained"
    rows.append(_row(t, "n", re.search(r"RESUMO RESTRITO \(n=(\d+)", text).group(1), src))
    for col, pat in (("scip_lower_objective_1e-4", r"objetivo MENOR >0\.01%: (\d+)/"),
                     ("sigma2_equal_rel_1e-6", r"sigma2 identico .*?: (\d+)/"),
                     ("scip_status_optimal", r"SCIP OPTIMAL: (\d+)/")):
        rows.append(_row(t, col, re.search(pat, text).group(1), src))
    rows.append(_row(t, "largest_relative_reduction",
                     re.search(r"maior reducao relativa: (-?[\d.]+)", text).group(1), src, "4"))
    m = re.search(r"sMAPE scip=([\d.]+) ipopt=([\d.]+)", text)
    rows += [_row(t, "smape_scip", m.group(1), src, "4"), _row(t, "smape_ipopt", m.group(2), src, "4")]
    m = re.search(r"MASE\s+scip=([\d.]+) ipopt=([\d.]+)", text)
    rows += [_row(t, "mase_scip", m.group(1), src, "5"), _row(t, "mase_ipopt", m.group(2), src, "5")]
    m = re.search(r"tempo scip=([\d.]+)s ipopt=([\d.]+)s", text)
    rows += [_row(t, "seconds_scip", m.group(1), src, "2"), _row(t, "seconds_ipopt", m.group(2), src, "3")]
    return rows
