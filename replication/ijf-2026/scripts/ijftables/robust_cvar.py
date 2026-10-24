"""Robust objectives under contamination (study B1, August 2026, ForecastTester workspace).

200 M4 monthly series (sample from Random.seed!(20260801)), six deterministic contamination
regimes (outlier fraction x magnitude in residual standard deviations, seeded per series), and
the objectives mse, mae and stable (CVaR). The first sweep (exp_robust_sweep_monthly.csv) ran
"stable" with the PRE-FIX CVaR normalisation (alpha effectively 1 - 0.7/n); the re-run
(exp_stable_rerun_monthly.csv) adds the corrected objective at alpha = 0.9 and 0.5 on the same
series and regimes. 30 s cap per fit, 8 workers.

Port of analyze_stable.py: mean sMAPE per regime and objective, degradation against the clean
regime, paired win rates against mse, and the paired effect of the CVaR correction.
"""

from __future__ import annotations

import math
import statistics as st

from .common import Cell, Context, num, read_csv, write_text

TITLE = "Robust objectives (mse/mae/CVaR) under contamination, 200 M4 monthly series"
RAW = "raw/robust-cvar-monthly-2026-08"
INPUTS = [f"{RAW}/exp_robust_sweep_monthly.csv", f"{RAW}/exp_stable_rerun_monthly.csv"]
REGIMES = ["limpo", "1%x5sd", "3%x5sd", "5%x5sd", "3%x10sd", "5%x10sd"]
OBJS = ["mse", "mae", "stable", "stable_a0.9", "stable_a0.5"]
LABEL = {"stable": "stable_prefix"}     # the pre-fix CVaR arm


def _load(rows):
    idx = {}
    for r in rows:
        if r["status"] != "OK":
            continue
        sm = num(r["smape"])
        if math.isnan(sm):
            continue
        idx[(r["series"], r["regime"], r["objetivo"])] = sm
    return idx


def generate(ctx: Context) -> list[Cell]:
    old = _load(read_csv(ctx.path(INPUTS[0])))
    new = _load(read_csv(ctx.path(INPUTS[1])))
    idx = {**old, **new}
    series = sorted({k[0] for k in old})
    cells = [Cell("robust_cvar.sample", "series", "n", len(series), 0)]
    lines = [f"series: {len(series)}", "", "mean sMAPE by regime"]
    level = {}
    for reg in REGIMES:
        vals = []
        for o in OBJS:
            v = [idx[(s, reg, o)] for s in series if (s, reg, o) in idx]
            level[(reg, o)] = st.mean(v) if v else float("nan")
            cells.append(Cell("robust_cvar.level", reg, LABEL.get(o, o), level[(reg, o)], 3))
            vals.append(f"{level[(reg, o)]:12.3f}")
        lines.append(f"{reg:<10}" + "".join(vals))
    lines += ["", "degradation vs own clean regime"]
    for reg in REGIMES[1:]:
        vals = []
        for o in OBJS:
            d = level[(reg, o)] - level[("limpo", o)]
            cells.append(Cell("robust_cvar.degradation", reg, LABEL.get(o, o), d, 3))
            vals.append(f"{d:+12.3f}")
        lines.append(f"{reg:<10}" + "".join(vals))
    lines += ["", "paired wins over mse (%)"]
    best = {}
    for reg in REGIMES:
        vals = []
        for o in OBJS[1:]:
            w = n = 0
            for s in series:
                b, c = idx.get((s, reg, "mse")), idx.get((s, reg, o))
                if b is None or c is None:
                    continue
                n += 1
                w += c < b
            pct = 100 * w / n if n else float("nan")
            best[o] = max(best.get(o, -1.0), pct)
            cells.append(Cell("robust_cvar.wins_vs_mse", reg, LABEL.get(o, o), pct, 1))
            vals.append(f"{pct:11.1f}%")
        lines.append(f"{reg:<10}" + "".join(vals))
    for o, v in best.items():
        cells.append(Cell("robust_cvar.wins_vs_mse", "max over regimes", LABEL.get(o, o), v, 1))
    lines += ["", "effect of the CVaR correction (stable_a0.9 - stable_prefix), paired"]
    for reg in REGIMES:
        d = [idx[(s, reg, "stable_a0.9")] - idx[(s, reg, "stable")] for s in series
             if (s, reg, "stable_a0.9") in idx and (s, reg, "stable") in idx]
        if d:
            improved = sum(1 for x in d if x < 0)
            cells += [Cell("robust_cvar.correction", reg, "mean_delta", st.mean(d), 3),
                      Cell("robust_cvar.correction", reg, "improved", improved, 0),
                      Cell("robust_cvar.correction", reg, "pairs", len(d), 0)]
            lines.append(f"  {reg:<10} mean delta {st.mean(d):+8.3f}  improved in {improved}/{len(d)}")
    write_text(ctx.out / "robust_cvar.txt", "\n".join(lines))
    return cells
