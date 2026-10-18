"""Series-level comparison behind the headline M4 table: the paired differences against
forecast::auto.arima and the marginal distributions of both methods (manuscript Table 4,
Table S8 and Figure 2; the paired shares quoted in Section 5.3).

Same archived inputs as `m4_benchmark` (recreio campaign A, `mse`, `:innovations`, no time
cap, and the stored auto.arima per-series metrics), at the total horizon. The original analysis
is `b_m4_distributional.py` of the September 2026 revision workspace; its tables are archived
in `results/historical/m4-paired-2026-09/` and its definitions are the ones below:

* the sample of a frequency is every campaign series with status OK and finite total-horizon
  metrics, paired by series id with the auto.arima series whose stored metrics are finite
  (Naive2 is not needed: no OWA is computed per series);
* Delta_i = metric_i(Sarimax.jl) - metric_i(auto.arima), with the campaign's sMAPE and MASE
  over the total horizon (the `smape_total`/`mase_total` columns of the bundle, computed from
  the stored forecast and test values before the latter were redacted);
* a tie is |Delta_i| < 1e-4; "lower" is Delta_i <= -1e-4 and "higher" Delta_i >= 1e-4;
* quantiles by `np.quantile(x, q, method="linear")` (Hyndman-Fan type 7); the trimmed mean
  averages the Delta_i within [q01, q99];
* the top-k% share is the fraction of sum |Delta_i| carried by the ceil(k/100 * N) series with
  the largest |Delta_i|.

The figure (`m4_paired_difference_distributions.{pdf,png}`) is rebuilt with the layout of
manuscript Figure 2; its panel percentages are the rounded lower/tie/higher shares, which are
compared as cells.
"""

from __future__ import annotations

import math

import numpy as np

from .common import Cell, Context, read_csv, status_ok, write_csv

TITLE = "M4 paired differences and marginal distributions against auto.arima (recreio campaign A)"
RAW = "raw/m4-recreio-2026-08"
BASE = "raw/m4-baselines-r-2026-07"
FREQS = ["yearly", "quarterly", "monthly", "weekly", "daily", "hourly"]   # manuscript order
METRICS = [("MASE", "mase_total", "MASE"), ("sMAPE", "smape_total", "sMAPE")]
METHODS = ["Sarimax.jl", "auto.arima"]
QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
TIE = 1e-4
TIE_EXTRA = [1e-6, 1e-2]
DECIMALS = 6

INPUTS = ([f"{RAW}/m4_innovations_{f}.csv.gz" for f in FREQS]
          + [f"{BASE}/{f}/autoarima_R/total.csv.gz" for f in FREQS])


def _finite(x) -> float | None:
    try:
        v = float(str(x).strip().strip('"'))
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def paired_sample(ctx: Context, freq: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per metric, the aligned (Sarimax.jl, auto.arima) values of the paired series."""
    ours: dict[int, dict] = {}
    for r in read_csv(ctx.path(f"{RAW}/m4_innovations_{freq}.csv.gz")):
        if status_ok(r["status"]) and r["forecast"].strip('"'):
            ours[int(r["series"])] = r          # later rows win, as in make_table_m4.jl
    ref = {int(r["series"]): r for r in read_csv(ctx.path(f"{BASE}/{freq}/autoarima_R/total.csv.gz"))}
    out = {}
    for name, ocol, rcol in METRICS:
        ids = sorted(s for s in ours if s in ref
                     and _finite(ours[s][ocol]) is not None and _finite(ref[s][rcol]) is not None)
        out[name] = (np.array([float(ours[s][ocol]) for s in ids]),
                     np.array([float(ref[s][rcol]) for s in ids]))
    return out


def top_share(d: np.ndarray, k: float) -> float:
    a = np.sort(np.abs(d))[::-1]
    return float(a[:math.ceil(k / 100 * len(a))].sum() / a.sum())


def generate(ctx: Context) -> list[Cell]:
    cells: list[Cell] = []
    paired_rows, marginal_rows = [], []
    panels = {}
    for freq in FREQS:
        sample = paired_sample(ctx, freq)
        for metric, (x, y) in sample.items():
            d = x - y
            n = len(d)
            q = dict(zip(QUANTILES, np.quantile(d, QUANTILES, method="linear")))
            lo, hi = q[0.01], q[0.99]
            row = f"{freq}|{metric}"
            # column names and units (fractions) of the historical m4_paired_differences_full.csv
            vals = {
                "N_paired": n, "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                "q01": float(q[0.01]), "q05": float(q[0.05]), "q25": float(q[0.25]),
                "q75": float(q[0.75]), "q95": float(q[0.95]), "q99": float(q[0.99]),
                "frac_lt0": float(np.mean(d <= -TIE)), "frac_gt0": float(np.mean(d >= TIE)),
                "frac_tie": float(np.mean(np.abs(d) < TIE)),
                "frac_tie_tol1e-06": float(np.mean(np.abs(d) < TIE_EXTRA[0])),
                "frac_tie_tol0.01": float(np.mean(np.abs(d) < TIE_EXTRA[1])),
                "share_abs_disagreement_top1pct": top_share(d, 1),
                "share_abs_disagreement_top5pct": top_share(d, 5),
                "mean_diff_trimmed_1_99": float(d[(d >= lo) & (d <= hi)].mean()),
            }
            for col, v in vals.items():
                cells.append(Cell("m4_paired.paired", row, col, v, 0 if col == "N_paired" else DECIMALS))
            paired_rows.append([freq, metric] + [vals[c] for c in vals])
            panels[(freq, metric)] = (d, vals)
            for method, v in zip(METHODS, (x, y)):
                mq = np.quantile(v, QUANTILES, method="linear")
                mrow = f"{freq}|{metric}|{method}"
                cells.append(Cell("m4_paired.marginal", mrow, "N", len(v), 0))
                cells.append(Cell("m4_paired.marginal", mrow, "mean", float(v.mean()), DECIMALS))
                for p, val in zip(QUANTILES, mq):
                    cells.append(Cell("m4_paired.marginal", mrow, f"q{round(p * 100):02d}", float(val), DECIMALS))
                marginal_rows.append([freq, metric, method, len(v), float(v.mean())] + [float(t) for t in mq])
    # Section 5.3: "the 5% of series with the largest absolute paired disagreements account for
    # roughly 41-78% of the sum of absolute disagreements" (range over the twelve panels)
    shares = [vals["share_abs_disagreement_top5pct"] for _, vals in panels.values()]
    cells.append(Cell("m4_paired.text", "share_abs_disagreement_top5pct", "min", min(shares), DECIMALS))
    cells.append(Cell("m4_paired.text", "share_abs_disagreement_top5pct", "max", max(shares), DECIMALS))

    write_csv(ctx.out / "m4_paired_differences.csv",
              ["frequency", "metric"] + list(next(iter(panels.values()))[1]), paired_rows)
    write_csv(ctx.out / "m4_distribution_quantiles.csv",
              ["frequency", "metric", "method", "N", "mean"] + [f"q{round(p * 100):02d}" for p in QUANTILES],
              marginal_rows)
    figure(ctx, panels)
    return cells


def figure(ctx: Context, panels: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctx.figures.mkdir(parents=True, exist_ok=True)
    with plt.style.context("default"):
        fig, axes = plt.subplots(2, len(FREQS), figsize=(2.6 * len(FREQS), 4.6), squeeze=False)
        for j, freq in enumerate(FREQS):
            for i, (metric, _, _) in enumerate(METRICS):
                ax = axes[i][j]
                d, vals = panels[(freq, metric)]
                s = np.sort(d)
                ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", color="#1f2937", lw=1.1)
                ax.set_xscale("symlog", linthresh=TIE)
                ax.set_xticks([-1, 0, 1] if metric == "MASE" else [-10, 0, 10])   # as in Figure 2
                ax.axvline(0, color="#9ca3af", lw=0.8)
                ax.text(0.03, 0.97, f"<0  {100 * vals['frac_lt0']:.0f}%\n=0  {100 * vals['frac_tie']:.0f}%\n"
                        f">0  {100 * vals['frac_gt0']:.0f}%", transform=ax.transAxes, va="top", fontsize=7)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.set_title(freq.capitalize(), fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"ECDF of Δ{metric}", fontsize=8)
        fig.supxlabel("Δi = Sarimax.jl − auto.arima  (negative favours Sarimax.jl; symmetric-log axis)",
                      fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(ctx.figures / f"m4_paired_difference_distributions.{ext}", dpi=150)
        plt.close(fig)
