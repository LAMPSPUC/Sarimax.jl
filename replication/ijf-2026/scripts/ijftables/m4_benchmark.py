"""Headline M4 comparison (campaign A, August 2026): SARIMAX.jl with
objective `mse`, `initialization = :innovations`, `stationary = true`, no time cap, on all six
M4 frequencies, against forecast::auto.arima, with Naive2 as the OWA reference.

Port of campaign A's `tables/make_table_m4.jl`: per frequency and horizon block, the mean sMAPE and
MASE of both methods over the series common to the campaign and both baselines, and the OWA
as a ratio of means. The archived campaign files stored the forecast and actual vectors; the
artifact bundle replaces `actual` (M4 data, not redistributed) by the per-block metrics
computed from them before redaction, which is what this generator reads.

Campaign A ran at commits aa68d57 (yearly, quarterly, weekly, hourly, daily) and 3d7f883
(monthly), both ancestors of the release, on Julia 1.9.4; see the manifest.
"""

from __future__ import annotations

import math

from .common import BLOCK_NAMES, Cell, Context, fmean, read_csv, status_ok, write_text

TITLE = "M4 headline comparison against auto.arima (campaign A)"
RAW = "raw/m4-recreio-2026-08"
BASE = "raw/m4-baselines-r-2026-07"
FREQS = ["daily", "hourly", "monthly", "quarterly", "weekly", "yearly"]   # make_table_m4 order

INPUTS = ([f"{RAW}/m4_innovations_{f}.csv.gz" for f in FREQS]
          + [f"{BASE}/{f}/{m}/{b}.csv.gz" for f in FREQS for m in ("autoarima_R", "Naive") for b in BLOCK_NAMES])


def _finite(x: str) -> float | None:
    try:
        v = float(x)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def read_reference(ctx: Context, freq: str, model: str, block: str) -> dict[int, tuple[float, float]]:
    out = {}
    for r in read_csv(ctx.path(f"{BASE}/{freq}/{model}/{block}.csv.gz")):
        s, m = _finite(r["sMAPE"]), _finite(r["MASE"])
        if s is not None and m is not None:
            out[int(r["series"])] = (s, m)
    return out


def generate(ctx: Context) -> list[Cell]:
    cells, lines = [], []
    for freq in FREQS:
        ours: dict[int, dict] = {}
        for r in read_csv(ctx.path(f"{RAW}/m4_innovations_{freq}.csv.gz")):
            if not status_ok(r["status"]) or not r["forecast"].strip('"'):
                continue
            ours[int(r["series"])] = r          # later rows win, as in make_table_m4.jl
        lines += ["=" * 86, f"{freq.upper()}   (m4_innovations_{freq}.csv)", "=" * 86,
                  f"  {'block':<8} {'n':>6} {'sMAPE (ours/ref)':>19} {'MASE (ours/ref)':>19} {'OWA (ours/ref/gap)':>19}"]
        for block in BLOCK_NAMES:
            R = read_reference(ctx, freq, "autoarima_R", block)
            N2 = read_reference(ctx, freq, "Naive", block)
            common = [s for s, r in ours.items() if s in R and s in N2
                      and _finite(r[f"smape_{block}"]) is not None and _finite(r[f"mase_{block}"]) is not None]
            if not common:
                continue
            sn = [float(ours[s][f"smape_{block}"]) for s in common]
            mn = [float(ours[s][f"mase_{block}"]) for s in common]
            sr, mr = [R[s][0] for s in common], [R[s][1] for s in common]
            s2, m2 = fmean(N2[s][0] for s in common), fmean(N2[s][1] for s in common)
            on = 0.5 * fmean(sn) / s2 + 0.5 * fmean(mn) / m2
            orr = 0.5 * fmean(sr) / s2 + 0.5 * fmean(mr) / m2
            row = f"{freq}|{block}"
            cells += [Cell("m4_benchmark.table_m4", row, "n", len(common), 0),
                      Cell("m4_benchmark.table_m4", row, "sMAPE.ours", fmean(sn)),
                      Cell("m4_benchmark.table_m4", row, "sMAPE.ref", fmean(sr)),
                      Cell("m4_benchmark.table_m4", row, "MASE.ours", fmean(mn)),
                      Cell("m4_benchmark.table_m4", row, "MASE.ref", fmean(mr)),
                      Cell("m4_benchmark.table_m4", row, "OWA.ours", on),
                      Cell("m4_benchmark.table_m4", row, "OWA.ref", orr),
                      Cell("m4_benchmark.table_m4", row, "OWA.gap", on - orr)]
            lines.append(f"  {block:<8} {len(common):>6}   {fmean(sn):8.4f} {fmean(sr):8.4f}   "
                         f"{fmean(mn):8.4f} {fmean(mr):8.4f}   {on:7.4f} {orr:7.4f} {on - orr:+7.4f}")
        lines.append("")
    write_text(ctx.out / "table_m4.txt", "\n".join(lines))
    return cells
