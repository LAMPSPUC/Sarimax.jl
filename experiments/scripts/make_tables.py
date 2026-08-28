# Regenerates the manuscript tables from results/raw and results/baselines.
#
# Reads only the gzipped campaign outputs in this package; nothing is refitted. Writes
# tables/table1_horizon_total.txt and tables/table2_owa_by_block.txt.
#
# OWA is the M4 definition: the mean of (MASE / MASE_naive2) and (sMAPE / sMAPE_naive2),
# each averaged over the SAME series set. The baseline files under results/baselines were
# produced on a different machine (see REPRODUCE.md, "Inputs not produced here").
import csv, gzip, os, pathlib, statistics as st, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
BASE = ROOT / "results" / "baselines"
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)

BLOCKS = ["short", "medium", "long", "total"]
FREQS = [("monthly", 48000), ("quarterly", 24000), ("yearly", 23000), ("weekly", 359)]
OBJECTIVES = ["mse", "huber", "mae", "ridge"]
owa = lambda m, mb, s, sb: 0.5 * (m / mb + s / sb)


def campaign_file(objective, initialization, freq):
    """Maps a table cell to the campaign that produced it.

    The mse/:innovations cell for monthly, quarterly and yearly was NOT a separate run: it
    is the `requireTerms = true` arm of the guard experiment, whose configuration is
    identical to that cell. The historical files named obj_mse_innov_<freq>.csv were
    byte-for-byte copies of req_true_<freq>.csv; this package keeps the original under its
    own name instead of duplicating it. Weekly did have its own mse run.
    """
    if initialization == "zeroed":
        return RAW / f"cel_{objective}_zeroed_{freq}.csv.gz"
    if objective == "mse" and freq != "weekly":
        return RAW / f"req_true_{freq}.csv.gz"
    return RAW / f"obj_{objective}_innov_{freq}.csv.gz"


def read_baseline(freq, method, block):
    p = BASE / f"{freq}_{method}_{block}.csv.gz"
    with gzip.open(p, "rt", newline="") as f:
        return {int(x["series"]): (float(x["MASE"]), float(x["sMAPE"]))
                for x in csv.DictReader(f)}


def read_cell(path, expected):
    """Returns None when the cell is absent or incomplete.

    An incomplete cell is dropped rather than averaged: the subset that exists is "the
    first N series", which is not a random sample of M4, so its mean is not comparable
    with a mean over the population.
    """
    if not path.exists():
        return None
    with gzip.open(path, "rt", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < expected:
        return None
    d = {}
    for x in rows:
        if not x["status"].strip('"').startswith("OK"):
            continue
        try:
            d[int(x["series"])] = {b: (float(x[f"mase_{b}"]), float(x[f"smape_{b}"]))
                                   for b in BLOCKS}
        except ValueError:
            pass
    return d


def common_ids(freq, expected, families):
    r = read_baseline(freq, "autoarima_R", "total")
    n = read_baseline(freq, "Naive", "total")
    ids = set(r) & set(n)
    for fam in families.values():
        for d in fam.values():
            if d is not None:
                ids &= set(d)
    return sorted(ids)


def load(freq, expected):
    return {"z": {o: read_cell(campaign_file(o, "zeroed", freq), expected) for o in OBJECTIVES},
            "i": {o: read_cell(campaign_file(o, "innovations", freq), expected) for o in OBJECTIVES}}


def table1():
    out = ["TABLE 1 - TOTAL HORIZON: MASE, sMAPE and OWA under both initializations", "",
           f"{'freq':<10} {'objective':<7} {'| :zeroed (objective isolated)':<34} "
           f"{'| :innovations (production)':<34}",
           f"{'':<10} {'':<7} {'MASE':>9} {'sMAPE':>8} {'OWA':>8} {'dOWA':>7} "
           f"{'MASE':>9} {'sMAPE':>8} {'OWA':>8} {'dOWA':>7}", "-" * 96]
    for freq, expected in FREQS:
        fam = load(freq, expected)
        ids = common_ids(freq, expected, fam)
        rb = {b: read_baseline(freq, "autoarima_R", b) for b in BLOCKS}
        nb = {b: read_baseline(freq, "Naive", b) for b in BLOCKS}
        rm = st.fmean(rb["total"][i][0] for i in ids)
        rs = st.fmean(rb["total"][i][1] for i in ids)
        bm = st.fmean(nb["total"][i][0] for i in ids)
        bs = st.fmean(nb["total"][i][1] for i in ids)
        oR = owa(rm, bm, rs, bs)
        for o in OBJECTIVES:
            cols = []
            for k in ("z", "i"):
                d = fam[k][o]
                if d is None:
                    cols.append(f"{'n/a':>9} {'n/a':>8} {'n/a':>8} {'n/a':>7}")
                    continue
                m = st.fmean(d[i]["total"][0] for i in ids)
                s = st.fmean(d[i]["total"][1] for i in ids)
                oo = owa(m, bm, s, bs)
                cols.append(f"{m:>9.4f} {s:>8.4f} {oo:>8.4f} {oo - oR:>+7.4f}")
            out.append(f"{freq if o == 'mse' else '':<10} {o:<7} {cols[0]} {cols[1]}")
        out.append(f"{'':<10} {'auto.arima':<7} {rm:>9.4f} {rs:>8.4f} {oR:>8.4f}")
        out.append(f"{'':<10} {'Naive2':<7} {bm:>9.4f} {bs:>8.4f} {1.0:>8.4f}   (n={len(ids)})")
        out.append("")
    return "\n".join(out)


def table2():
    out = ["TABLE 2 - OWA BY HORIZON BLOCK", "",
           f"{'freq':<10} {'objective':<7} {'|':<1} {':zeroed':^35} {'|':<1} {':innovations':^35}",
           f"{'':<10} {'':<7}   {'short':>8} {'medium':>8} {'long':>8} {'total':>8}   "
           f"{'short':>8} {'medium':>8} {'long':>8} {'total':>8}", "-" * 96]
    for freq, expected in FREQS:
        fam = load(freq, expected)
        ids = common_ids(freq, expected, fam)
        rb = {b: read_baseline(freq, "autoarima_R", b) for b in BLOCKS}
        nb = {b: read_baseline(freq, "Naive", b) for b in BLOCKS}
        base = {b: (st.fmean(nb[b][i][0] for i in ids), st.fmean(nb[b][i][1] for i in ids))
                for b in BLOCKS}
        for o in OBJECTIVES:
            cols = []
            for k in ("z", "i"):
                d = fam[k][o]
                if d is None:
                    cols.append(" ".join([f"{'n/a':>8}"] * 4))
                    continue
                vs = []
                for b in BLOCKS:
                    m = st.fmean(d[i][b][0] for i in ids)
                    s = st.fmean(d[i][b][1] for i in ids)
                    vs.append(f"{owa(m, base[b][0], s, base[b][1]):>8.4f}")
                cols.append(" ".join(vs))
            out.append(f"{freq if o == 'mse' else '':<10} {o:<7}   {cols[0]}   {cols[1]}")
        vs = []
        for b in BLOCKS:
            rm = st.fmean(rb[b][i][0] for i in ids)
            rs = st.fmean(rb[b][i][1] for i in ids)
            vs.append(f"{owa(rm, base[b][0], rs, base[b][1]):>8.4f}")
        out.append(f"{'':<10} {'auto.arima':<7}   {' '.join(vs)}")
        out.append("")
    return "\n".join(out)


t1, t2 = table1(), table2()
# The explicit newline is not cosmetic. On Windows the default would write CRLF, the file
# would stop matching SHA256SUMS, and merely regenerating the tables would make the package
# fail its own integrity check.
(OUT / "table1_horizon_total.txt").write_text(t1, encoding="utf-8", newline="\n")
(OUT / "table2_owa_by_block.txt").write_text(t2, encoding="utf-8", newline="\n")
print(t1)
print()
print(t2)
