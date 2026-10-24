"""Controlled comparison of estimation objectives (experiment `m4_objectives`, October 2026):
MSE, MAE, Huber and Ridge under one frozen protocol, one revision (v1.0.0, 14dd581) and one
environment, every series of M4 monthly, quarterly, yearly and weekly.

Rebuilds, from the per-series results of the campaign (bundle part 2) and the stored
auto.arima/Naive2 baselines (part 1):

* `controlled.table5`: OWA of auto.arima and of each objective on the jointly evaluable series
  (the replacement Table 5 proposed for the manuscript), with N;
* `controlled.table5_means`: mean sMAPE and MASE of each method and of Naive2 on that set;
* `controlled.table6`: N, mean, q05, median and q95 of MASE(objective) - MASE(MSE) (the
  replacement Table 6), and the shares of series with a lower, tied (|d| <= 1e-4) or higher
  MASE and sMAPE than MSE, and with the same selected order;
* `controlled.paired`: N, mean and quantiles (q01 ... q99) of the paired differences
  MASE(objective) - MASE(MSE) and sMAPE(objective) - sMAPE(MSE) (the controlled counterpart of
  the supplementary paired-quantile tables of the objectives, S13 for MASE and S14 for sMAPE);
* `controlled.quantiles`: mean and quantiles (q01 ... q99) of MASE and sMAPE of each method on
  the joint set (the controlled counterpart of the supplementary distribution tables);
* `controlled.weekly`: the weekly MSE reconciliation: the controlled cell and the single extra
  run under the Table 3 protocol, with how many of its series reproduce the archived Table 3
  cell (same order and |delta sMAPE| <= 1e-3).

and writes the rebuilt tables (CSV and LaTeX) and the ECDF figure of the per-series MASE
differences (`fig:objective_ecdf`, Figure 3).

Definitions (those of the campaign's protocol, [evaluation]): a series counts for an objective
when the fit returned a finite forecast (status OK, any solver status); the joint set is the
series that every objective fitted and whose two stored references are finite. OWA is a ratio
of means over that set. Quantiles are linear (type 7).

The reference values (results/expected/m4_objectives_controlled.csv) are parsed from the
tables the campaign's independent scorer wrote (derived/m4-objectives-controlled-2026-10/),
which computed sMAPE and MASE from the forecasts and the M4 test values with its own code;
this generator uses the per-series metrics computed while the bundle was built.

NOT to be combined with the historical objective campaigns (m4_objectives_icarai_*), which
ran under another protocol on other revisions.
"""

from __future__ import annotations

import math

from .common import BLOCK_NAMES, Cell, Context, fmean, owa, read_csv, write_csv, write_text

TITLE = "M4 objectives, controlled campaign Oct 2026: replacement Tables 5, 6, S9, S10 and objective ECDF figure"
RAW = "raw/m4-objectives-controlled-2026-10"
BASE = "raw/m4-baselines-r-2026-07"
FREQS = [("monthly", 48000), ("quarterly", 24000), ("yearly", 23000), ("weekly", 359)]
OBJECTIVES = [("mse", "MSE"), ("huber", "Huber"), ("mae", "MAE"), ("ridge", "Ridge")]
TIE = 1e-4
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
ORDER = ("p", "d", "q", "P", "D", "Q")

INPUTS = sorted([f"{RAW}/per_series/{o}__{f}.csv.gz" for f, _ in FREQS for o, _ in OBJECTIVES]
                + [f"{RAW}/per_series_p1_table3_protocol/mse__weekly.csv.gz",
                   "raw/m4-recreio-2026-08/m4_innovations_weekly.csv.gz"]
                + [f"{BASE}/{f}/{m}/total.csv.gz" for f, _ in FREQS for m in ("Naive", "autoarima_R")])


def finite(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


def read_cell(ctx: Context, rel: str, expected: int) -> dict[int, dict]:
    rows = read_csv(ctx.path(rel))
    if len(rows) != expected:
        raise ValueError(f"{rel}: {len(rows)} rows, expected {expected} (every attempted series has a row)")
    out = {}
    for r in rows:
        if r["status"] != "OK":
            continue
        sm, ma = float(r["smape_total"]), float(r["mase_total"])
        if finite(sm, ma):
            out[int(r["series"])] = {"smape": sm, "mase": ma, "order": tuple(r[k] for k in ORDER)}
    return out


def read_reference(ctx: Context, freq: str, method: str) -> dict[int, tuple[float, float]]:
    out = {}
    for r in read_csv(ctx.path(f"{BASE}/{freq}/{method}/total.csv.gz")):
        sm, ma = float(r["sMAPE"]), float(r["MASE"])
        if finite(sm, ma):
            out[int(r["series"])] = (sm, ma)
    return out


def quantile(xs: list[float], q: float) -> float:
    """Linear interpolation between order statistics (Hyndman-Fan type 7, numpy's default)."""
    s = sorted(xs)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def pct(n: int, total: int) -> float:
    return 100.0 * n / total


def generate(ctx: Context) -> list[Cell]:
    cells: list[Cell] = []
    t5_rows, t6_rows, paired_rows, deltas = [], [], [], {}
    for freq, expected in FREQS:
        cell = {o: read_cell(ctx, f"{RAW}/per_series/{o}__{freq}.csv.gz", expected) for o, _ in OBJECTIVES}
        nv, ar = read_reference(ctx, freq, "Naive"), read_reference(ctx, freq, "autoarima_R")
        joint = set(nv) & set(ar)
        for d in cell.values():
            joint &= set(d)
        ids = sorted(joint)
        n2_s, n2_m = fmean(nv[i][0] for i in ids), fmean(nv[i][1] for i in ids)
        ref_s, ref_m = fmean(ar[i][0] for i in ids), fmean(ar[i][1] for i in ids)
        row = {"frequency": freq, "N": len(ids), "Reference": owa(ref_m, n2_m, ref_s, n2_s)}
        cells += [Cell("controlled.table5", freq, "N", len(ids), 0),
                   Cell("controlled.table5", freq, "Reference", row["Reference"]),
                   Cell("controlled.table5_means", freq, "Reference.sMAPE", ref_s, 6),
                   Cell("controlled.table5_means", freq, "Reference.MASE", ref_m, 6),
                   Cell("controlled.table5_means", freq, "Naive2.sMAPE", n2_s, 6),
                   Cell("controlled.table5_means", freq, "Naive2.MASE", n2_m, 6)]
        for o, label in OBJECTIVES:
            s = fmean(cell[o][i]["smape"] for i in ids)
            m = fmean(cell[o][i]["mase"] for i in ids)
            row[label] = owa(m, n2_m, s, n2_s)
            row[f"{label}_sMAPE"], row[f"{label}_MASE"] = s, m
            cells += [Cell("controlled.table5", freq, label, row[label]),
                      Cell("controlled.table5_means", freq, f"{label}.sMAPE", s, 6),
                      Cell("controlled.table5_means", freq, f"{label}.MASE", m, 6)]
        t5_rows.append(row)
        # Marginal distributions on the joint set (the controlled counterpart of Tables S9/S10).
        series = {label: cell[o] for o, label in OBJECTIVES}
        series["auto.arima"] = {i: {"smape": ar[i][0], "mase": ar[i][1]} for i in ids}
        for label, d in series.items():
            for metric in ("MASE", "sMAPE"):
                x = [d[i][metric.lower()] for i in ids]
                rk = f"{freq}|{metric}|{label}"
                cells.append(Cell("controlled.quantiles", rk, "mean", fmean(x), 6))
                cells += [Cell("controlled.quantiles", rk, f"q{int(round(q * 100)):02d}", quantile(x, q), 6)
                          for q in QUANTILES]
        for o, label in OBJECTIVES[1:]:
            rk = f"{freq}|{label}"
            dm = [cell[o][i]["mase"] - cell["mse"][i]["mase"] for i in ids]
            ds = [cell[o][i]["smape"] - cell["mse"][i]["smape"] for i in ids]
            same = sum(cell[o][i]["order"] == cell["mse"][i]["order"] for i in ids)
            deltas[(freq, label)] = dm
            st = {"N": len(ids), "Mean": fmean(dm), "q05": quantile(dm, 0.05), "Median": quantile(dm, 0.5),
                  "q95": quantile(dm, 0.95)}
            t6_rows.append({"frequency": freq, "objective": label, **st})
            cells += [Cell("controlled.table6", rk, "N", len(ids), 0)]
            cells += [Cell("controlled.table6", rk, k, st[k], 3) for k in ("Mean", "q05", "Median", "q95")]
            for metric, d in (("MASE", dm), ("sMAPE", ds)):
                pk = f"{rk}|{metric}"
                paired_rows.append([freq, label, metric, len(d), fmean(d)] + [quantile(d, q) for q in QUANTILES])
                cells += [Cell("controlled.paired", pk, "N", len(d), 0),
                          Cell("controlled.paired", pk, "mean", fmean(d), 6)]
                cells += [Cell("controlled.paired", pk, f"q{int(round(q * 100)):02d}", quantile(d, q), 6)
                          for q in QUANTILES]
                cells += [Cell("controlled.table6", rk, f"{metric}.pct_lower", pct(sum(x < -TIE for x in d), len(d)), 2),
                          Cell("controlled.table6", rk, f"{metric}.pct_tie", pct(sum(abs(x) <= TIE for x in d), len(d)), 2),
                          Cell("controlled.table6", rk, f"{metric}.pct_higher", pct(sum(x > TIE for x in d), len(d)), 2)]
            cells.append(Cell("controlled.table6", rk, "pct_same_order", pct(same, len(ids)), 2))

    # ---- weekly MSE reconciliation: controlled P2 cell and the Table 3 protocol extra run -----
    p1 = read_cell(ctx, f"{RAW}/per_series_p1_table3_protocol/mse__weekly.csv.gz", 359)
    p2 = read_cell(ctx, f"{RAW}/per_series/mse__weekly.csv.gz", 359)
    nv = read_reference(ctx, "weekly", "Naive")
    t3 = {int(r["series"]): r for r in read_csv(ctx.path("raw/m4-recreio-2026-08/m4_innovations_weekly.csv.gz"))}
    weekly_lines = ["source N OWA same_order_as_Table3 identical_to_Table3"]
    for name, d in (("controlled_P2", p2), ("extra_run_P1_table3_protocol", p1)):
        ids = sorted(set(d) & set(nv))
        o = owa(fmean(d[i]["mase"] for i in ids), fmean(nv[i][1] for i in ids),
                fmean(d[i]["smape"] for i in ids), fmean(nv[i][0] for i in ids))
        same = [i for i in ids if i in t3 and d[i]["order"] == tuple(str(int(float(t3[i][k]))) for k in ORDER)]
        ident = [i for i in same if abs(d[i]["smape"] - float(t3[i]["smape"])) <= 1e-3]
        cells += [Cell("controlled.weekly", name, "N", len(ids), 0), Cell("controlled.weekly", name, "OWA", o),
                  Cell("controlled.weekly", name, "same_order_as_Table3", len(same), 0),
                  Cell("controlled.weekly", name, "identical_to_Table3", len(ident), 0)]
        weekly_lines.append(f"{name} {len(ids)} {o:.4f} {len(same)} {len(ident)}")

    # ---- outputs ----------------------------------------------------------------------------
    cols5 = ["frequency", "N", "Reference", "MSE", "Huber", "MAE", "Ridge"]
    write_csv(ctx.out / "table5_controlled.csv", cols5 + [f"{l}_{m}" for _, l in OBJECTIVES for m in ("sMAPE", "MASE")],
              [[r[c] for c in cols5] + [r[f"{l}_{m}"] for _, l in OBJECTIVES for m in ("sMAPE", "MASE")] for r in t5_rows])
    tex5 = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"Frequency & $N$ & Reference & MSE & Huber & MAE & Ridge \\", r"\midrule"]
    tex5 += [f"{r['frequency'].capitalize()} & {r['N']:,} & ".replace(",", "{,}") +
             " & ".join(f"{r[c]:.4f}" for c in cols5[2:]) + r" \\" for r in t5_rows]
    tex5 += [r"\bottomrule", r"\end{tabular}"]
    write_text(ctx.out / "table5_controlled.tex", "\n".join(tex5))
    write_csv(ctx.out / "table6_controlled.csv", ["frequency", "objective", "N", "mean", "q05", "median", "q95"],
              [[r["frequency"], r["objective"], r["N"], r["Mean"], r["q05"], r["Median"], r["q95"]] for r in t6_rows])
    tex6 = [r"\begin{tabular}{llrrrrr}", r"\toprule",
            r"Frequency & Objective & $N$ & Mean & $q_{.05}$ & Median & $q_{.95}$ \\", r"\midrule"]
    last = None
    for r in t6_rows:
        if last is not None and r["frequency"] != last:
            tex6.append(r"\addlinespace")
        tex6.append(" & ".join([r["frequency"].capitalize() if r["frequency"] != last else "", r["objective"],
                                f"{r['N']:,}".replace(",", "{,}")] + [f"{r[k]:.3f}" for k in ("Mean", "q05", "Median", "q95")])
                    + r" \\")
        last = r["frequency"]
    tex6 += [r"\bottomrule", r"\end{tabular}"]
    write_text(ctx.out / "table6_controlled.tex", "\n".join(tex6))
    write_csv(ctx.out / "paired_controlled.csv",
              ["frequency", "objective", "metric", "N", "mean"] + [f"q{int(round(q * 100)):02d}" for q in QUANTILES],
              paired_rows)
    write_text(ctx.out / "weekly_reconciliation.txt", "\n".join(weekly_lines))
    figure(ctx, deltas, [f for f, _ in FREQS])
    return cells


def figure(ctx: Context, deltas: dict, freqs: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator

    styles = {"Huber": ("#1f6fd1", "-"), "MAE": ("#e8613c", "--"), "Ridge": ("#1aa37a", "-.")}
    fig, axes = plt.subplots(1, len(freqs), figsize=(6.9, 2.3), sharey=True, squeeze=False)
    for ax, f in zip(axes[0], freqs):
        n = 0
        for label, (color, ls) in styles.items():
            x = sorted(deltas[(f, label)])
            n = len(x)
            ax.step(x, [(k + 1) / n for k in range(n)], where="post", color=color, ls=ls, lw=1.0, label=label)
        ax.axvline(0, color="0.2", lw=0.6, zorder=0)
        ax.axhline(0.5, color="0.85", lw=0.5, zorder=0)
        ax.set_xscale("symlog", linthresh=0.01, linscale=0.6)
        ax.set_xlim(-3, 3)
        ax.xaxis.set_major_locator(FixedLocator([-1, 0, 1]))
        ax.xaxis.set_major_formatter(FixedFormatter(["$-1$", "0", "1"]))
        ax.set_title(f"{f.capitalize()}, N = {n:,}", fontsize=8)
    axes[0][0].set_ylabel("ECDF")
    axes[0][0].legend(loc="upper left", frameon=False, fontsize=7)
    fig.supxlabel(r"$\Delta_i$ = MASE(objective) $-$ MASE(MSE), same series; negative favours the objective "
                  "(symmetric-log axis)", fontsize=7.5, y=0.01)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.24, top=0.88, wspace=0.08)
    ctx.figures.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(ctx.figures / f"figure4_controlled_ecdf.{ext}", **({"dpi": 200} if ext == "png" else {}))
    plt.close(fig)
