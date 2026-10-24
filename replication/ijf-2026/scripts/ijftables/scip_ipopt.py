"""SCIP against Ipopt on short M4 yearly series (July 2026, `docker_scip/`).

100 yearly series with T <= 100 (sample drawn with Random.seed!(20260731)), orders taken from
an Ipopt `auto` fit and then held fixed; each model is estimated by Ipopt (local) and by SCIP
(spatial branch-and-bound) with `limits/gap = 0.01` and `limits/time = 600`. Two runs: the
free parameterisation (`stationary = invertible = false`) and the constrained one
(`stationary = invertible = true`; orders with p or q >= 3 are not covered).

Recomputes the summary blocks that the original scripts (scip_experiment.jl,
scip_experiment_constr.jl) printed at the end of their logs, with the same definitions.

GLOBAL-OPTIMALITY EVIDENCE: the archived rows record SCIP's MOI termination status only.
"OPTIMAL" here means SCIP stopped with a relative gap below limits/gap = 1%; no dual bound,
no achieved gap and no SCIP log was archived, so these rows are NOT independently verifiable
certificates of global optimality. See LIMITATIONS.md.
"""

from __future__ import annotations

import math

from .common import Cell, Context, fmean, num, read_csv, write_text

TITLE = "SCIP vs Ipopt on 100 short M4 yearly series (summary of the July 2026 runs)"
RAW = "raw/scip-ipopt-yearly-2026-07"
INPUTS = [f"{RAW}/scip_experiment_results.csv", f"{RAW}/scip_constr_results.csv"]


def _col(rows, name):
    return [num(r[name]) for r in rows]


def generate(ctx: Context) -> list[Cell]:
    cells, lines = [], []
    free = read_csv(ctx.path(f"{RAW}/scip_experiment_results.csv"))
    n = len(free)
    rel = _col(free, "sigma2_reldiff")
    coef = _col(free, "maxdiff_coef")
    fc = _col(free, "maxdiff_fc")
    t = "scip_ipopt.free"
    summary = {
        "n": (n, 0),
        "scip_lower_objective": (sum(r["scip_melhor"].strip().lower() == "true" for r in free), 0),
        "sigma2_equal_rel_1e-6": (sum(abs(x) < 1e-6 for x in rel), 0),
        "coefficients_equal_1e-4": (sum((not math.isnan(x)) and x < 1e-4 for x in coef), 0),
        "forecasts_equal_1e-6": (sum(x < 1e-6 for x in fc), 0),
        "scip_status_optimal": (sum(r["scip_status"] == "OPTIMAL" for r in free), 0),
        "smape_scip": (fmean(_col(free, "smape_scip")), 4),
        "smape_ipopt": (fmean(_col(free, "smape_ipopt")), 4),
        "mase_scip": (fmean(_col(free, "mase_scip")), 5),
        "mase_ipopt": (fmean(_col(free, "mase_ipopt")), 5),
        "seconds_scip": (fmean(_col(free, "s_scip")), 2),
        "seconds_ipopt": (fmean(_col(free, "s_ipopt")), 3),
    }
    for k, (v, d) in summary.items():
        cells.append(Cell(t, "summary", k, v, d))
        lines.append(f"free        {k:<26} {v:.{d}f}" if isinstance(v, float) else f"free        {k:<26} {v}")

    con = read_csv(ctx.path(f"{RAW}/scip_constr_results.csv"))
    n = len(con)
    rel = _col(con, "sigma2_reldiff")
    t = "scip_ipopt.constrained"
    summary = {
        "n": (n, 0),
        "scip_lower_objective_1e-4": (sum(x < -1e-4 for x in rel), 0),
        "sigma2_equal_rel_1e-6": (sum(abs(x) < 1e-6 for x in rel), 0),
        "largest_relative_reduction": (min(rel), 4),
        "scip_status_optimal": (sum(r["scip_status"] == "OPTIMAL" for r in con), 0),
        "smape_scip": (fmean(_col(con, "smape_scip")), 4),
        "smape_ipopt": (fmean(_col(con, "smape_ipopt")), 4),
        "mase_scip": (fmean(_col(con, "mase_scip")), 5),
        "mase_ipopt": (fmean(_col(con, "mase_ipopt")), 5),
        "seconds_scip": (fmean(_col(con, "s_scip")), 2),
        "seconds_ipopt": (fmean(_col(con, "s_ipopt")), 3),
    }
    for k, (v, d) in summary.items():
        cells.append(Cell(t, "summary", k, v, d))
        lines.append(f"constrained {k:<26} {v:.{d}f}" if isinstance(v, float) else f"constrained {k:<26} {v}")
    write_text(ctx.out / "scip_ipopt_summary.txt",
               "SCIP (limits/gap=0.01) vs Ipopt, fixed orders, 100 M4 yearly series with T<=100\n"
               "NOTE: status OPTIMAL = gap below 1%; dual bounds were not archived.\n" + "\n".join(lines))
    return cells
