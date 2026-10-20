"""M4 estimation-objective campaigns of August 2026: mse, huber, mae, ridge
under `initialization = :zeroed` (600 s cap) and `:innovations` (no cap), on the monthly,
quarterly, yearly and weekly M4 series.

Rebuilds, from the per-series campaign files and the stored auto.arima/Naive2 baselines:

* `icarai.table1` / `icarai.table2`: Tables 1 and 2 of these campaigns' replication package
  (`make_tables.py`): total-horizon MASE/sMAPE/OWA and OWA by horizon block, over the series
  common to all eight cells and both baselines;
* `forecasttester.A` / `forecasttester.B`: the same campaigns as tabulated by
  `tabelas_paper.py` (per block MASE/sMAPE/OWA, each initialization over its own common set).

HISTORICAL: these campaigns ran on feature-branch commits (87f7bfb, 6a11698, 5b2ec6b) that are
not part of a release, with mixed commits inside some cells. They are superseded by the
controlled `m4_objectives` experiment; see REPRODUCIBILITY_STATUS.md.
"""

from __future__ import annotations

from .common import (BLOCK_NAMES, Cell, Context, fmean, owa, read_csv, status_ok, write_text)

TITLE = "M4 objective campaigns, Aug 2026 (historical): Tables 1-2 and tabelas_paper A/B"
RAW = "raw/m4-objectives-icarai-2026-08"
FREQS = [("monthly", 48000), ("quarterly", 24000), ("yearly", 23000), ("weekly", 359)]
OBJECTIVES = ["mse", "huber", "mae", "ridge"]


def campaign_file(objective: str, init: str, freq: str) -> str:
    # The mse/:innovations cell of monthly/quarterly/yearly IS the requireTerms=true arm of
    # the guard experiment (obj_mse_innov_<freq>.csv was a byte copy of req_true_<freq>.csv).
    if init == "zeroed":
        return f"{RAW}/campaigns/cel_{objective}_zeroed_{freq}.csv.gz"
    if objective == "mse" and freq != "weekly":
        return f"{RAW}/campaigns/req_true_{freq}.csv.gz"
    return f"{RAW}/campaigns/obj_{objective}_innov_{freq}.csv.gz"


def _cells_present(init: str) -> list[str]:
    return [o for o in OBJECTIVES if not (init == "zeroed" and o == "ridge")]


INPUTS = sorted({campaign_file(o, i, f) for f, _ in FREQS for i in ("zeroed", "innovations")
                 for o in _cells_present(i)}
                | {f"{RAW}/baselines/{f}_{m}_{b}.csv.gz" for f, _ in FREQS
                   for m in ("autoarima_R", "Naive") for b in BLOCK_NAMES})


def read_cell(ctx: Context, rel: str, expected: int):
    rows = read_csv(ctx.path(rel))
    if len(rows) < expected:            # incomplete cells are dropped, never averaged
        return None
    out = {}
    for x in rows:
        if not status_ok(x["status"]):
            continue
        try:
            out[int(x["series"])] = {b: (float(x[f"mase_{b}"]), float(x[f"smape_{b}"])) for b in BLOCK_NAMES}
        except (ValueError, KeyError):
            pass
    return out


def read_baseline(ctx: Context, freq: str, model: str, block: str) -> dict[int, tuple[float, float]]:
    return {int(x["series"]): (float(x["MASE"]), float(x["sMAPE"]))
            for x in read_csv(ctx.path(f"{RAW}/baselines/{freq}_{model}_{block}.csv.gz"))}


def generate(ctx: Context) -> list[Cell]:
    cells: list[Cell] = []
    t1_lines, t2_lines, ft_lines = [], [], []
    for freq, expected in FREQS:
        fam = {init: {o: read_cell(ctx, campaign_file(o, init, freq), expected) for o in _cells_present(init)}
               for init in ("zeroed", "innovations")}
        rb = {b: read_baseline(ctx, freq, "autoarima_R", b) for b in BLOCK_NAMES}
        nb = {b: read_baseline(ctx, freq, "Naive", b) for b in BLOCK_NAMES}

        # ---- Tables 1 and 2 of the campaigns' package: one common set across all eight cells -----------------
        ids = set(rb["total"]) & set(nb["total"])
        for d in (d for fm in fam.values() for d in fm.values() if d is not None):
            ids &= set(d)
        ids = sorted(ids)
        base = {b: (fmean(nb[b][i][0] for i in ids), fmean(nb[b][i][1] for i in ids)) for b in BLOCK_NAMES}
        ref = {b: (fmean(rb[b][i][0] for i in ids), fmean(rb[b][i][1] for i in ids)) for b in BLOCK_NAMES}
        o_ref = {b: owa(ref[b][0], base[b][0], ref[b][1], base[b][1]) for b in BLOCK_NAMES}
        for o in OBJECTIVES:
            parts = []
            for init in ("zeroed", "innovations"):
                d = fam[init].get(o)
                if d is None:
                    parts.append("n/a")
                    continue
                for b in BLOCK_NAMES:
                    m = fmean(d[i][b][0] for i in ids)
                    s = fmean(d[i][b][1] for i in ids)
                    oo = owa(m, base[b][0], s, base[b][1])
                    cells.append(Cell("icarai.table2", f"{freq}|{o}", f"{init}.{b}", oo))
                    if b == "total":
                        cells += [Cell("icarai.table1", f"{freq}|{o}", f"{init}.MASE", m),
                                  Cell("icarai.table1", f"{freq}|{o}", f"{init}.sMAPE", s),
                                  Cell("icarai.table1", f"{freq}|{o}", f"{init}.OWA", oo),
                                  Cell("icarai.table1", f"{freq}|{o}", f"{init}.dOWA", oo - o_ref["total"])]
                        parts.append(f"{m:9.4f} {s:8.4f} {oo:8.4f} {oo - o_ref['total']:+7.4f}")
            t1_lines.append(f"{freq:<10} {o:<7} {' | '.join(parts)}")
        cells += [Cell("icarai.table1", f"{freq}|auto.arima", "MASE", ref["total"][0]),
                  Cell("icarai.table1", f"{freq}|auto.arima", "sMAPE", ref["total"][1]),
                  Cell("icarai.table1", f"{freq}|auto.arima", "OWA", o_ref["total"]),
                  Cell("icarai.table1", f"{freq}|Naive2", "MASE", base["total"][0]),
                  Cell("icarai.table1", f"{freq}|Naive2", "sMAPE", base["total"][1]),
                  Cell("icarai.table1", f"{freq}|Naive2", "n", len(ids), 0)]
        for b in BLOCK_NAMES:
            cells.append(Cell("icarai.table2", f"{freq}|auto.arima", b, o_ref[b]))
        t1_lines.append(f"{'':<10} {'auto.arima':<7} {ref['total'][0]:9.4f} {ref['total'][1]:8.4f} "
                        f"{o_ref['total']:8.4f}   Naive2 {base['total'][0]:.4f} {base['total'][1]:.4f} (n={len(ids)})")

        # ---- tabelas_paper.py: each initialization over its own common set -----------------
        for section, init in (("A", "zeroed"), ("B", "innovations")):
            present = {o: d for o, d in fam[init].items() if d is not None}
            if not present:
                continue
            sid = set(rb["total"]) & set(nb["total"])
            for d in present.values():
                sid &= set(d)
            sid = sorted(sid)
            nv = {b: (fmean(nb[b][i][0] for i in sid), fmean(nb[b][i][1] for i in sid)) for b in BLOCK_NAMES}
            rr = {b: (fmean(rb[b][i][0] for i in sid), fmean(rb[b][i][1] for i in sid)) for b in BLOCK_NAMES}
            tab = f"forecasttester.{section}"
            cells.append(Cell(tab, f"{freq}", "n", len(sid), 0))
            for o, d in present.items():
                for b in BLOCK_NAMES:
                    m = fmean(d[i][b][0] for i in sid)
                    s = fmean(d[i][b][1] for i in sid)
                    oo = owa(m, nv[b][0], s, nv[b][1])
                    o_r = owa(rr[b][0], nv[b][0], rr[b][1], nv[b][1])
                    cells += [Cell(tab, f"{freq}|{o}|{b}", "MASE", m), Cell(tab, f"{freq}|{o}|{b}", "sMAPE", s),
                              Cell(tab, f"{freq}|{o}|{b}", "OWA", oo), Cell(tab, f"{freq}|{o}|{b}", "dOWA", oo - o_r)]
                    ft_lines.append(f"{section} {freq:<10} {o:<6} {b:<7} {m:8.4f} {s:8.4f} {oo:8.4f} {oo - o_r:+8.4f}")
            for b in BLOCK_NAMES:
                o_r = owa(rr[b][0], nv[b][0], rr[b][1], nv[b][1])
                cells += [Cell(tab, f"{freq}|auto.arima|{b}", "MASE", rr[b][0]),
                          Cell(tab, f"{freq}|auto.arima|{b}", "sMAPE", rr[b][1]),
                          Cell(tab, f"{freq}|auto.arima|{b}", "OWA", o_r),
                          Cell(tab, f"{freq}|Naive|{b}", "MASE", nv[b][0]),
                          Cell(tab, f"{freq}|Naive|{b}", "sMAPE", nv[b][1])]

        # Table 2, text form
        for o in OBJECTIVES:
            vals = [f"{c.value:.4f}" for c in cells if c.table == "icarai.table2" and c.row == f"{freq}|{o}"]
            t2_lines.append(f"{freq:<10} {o:<7} {' '.join(vals) or 'n/a'}")

    write_text(ctx.out / "table1_horizon_total.txt",
               "TABLE 1 - total horizon: MASE sMAPE OWA dOWA (:zeroed | :innovations)\n" + "\n".join(t1_lines))
    write_text(ctx.out / "table2_owa_by_block.txt",
               "TABLE 2 - OWA by block (short medium long total; :zeroed then :innovations)\n" + "\n".join(t2_lines))
    write_text(ctx.out / "tabelas_paper_A_B.txt",
               "section freq cell block MASE sMAPE OWA dOWA_vs_auto.arima\n" + "\n".join(ft_lines))
    return cells
