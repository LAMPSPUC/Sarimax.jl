"""Recompute Table 5 (OWA by objective) and Table 6 (paired MASE differences) of the v2
manuscript from the ICARAI raw files, and compare with the printed values.

    python -I s_table5_check.py      (from paper_results_revision/scripts)

Cells: mse = req_true_<freq> (monthly, quarterly, yearly; the historical obj_mse_innov_* files
were byte copies of them, ICARAI REPRODUCE.md section 3) and obj_mse_innov_weekly; huber, mae,
ridge = obj_<obj>_innov_<freq>. Common successfully fitted series within each frequency.
Quantiles: numpy linear (type 7).
"""
import gzip, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "_replication_icarai/experiments/results"
FREQS = ["monthly", "quarterly", "yearly", "weekly"]
OBJS = ["mse", "huber", "mae", "ridge"]
PAPER5 = {  # Table 5: Reference, MSE, Huber, MAE, Ridge
    "monthly": (0.9044, 0.9040, 0.9099, 0.9157, 0.9028),
    "quarterly": (0.8976, 0.9052, 0.8987, 0.8959, 0.9005),
    "yearly": (0.8913, 0.8915, 0.8855, 0.8846, 0.8852),
    "weekly": (0.9266, 0.9429, 1.0730, 1.2657, 0.9721),
}
PAPER5_N = {"monthly": 47999, "quarterly": 24000, "yearly": 23000, "weekly": 359}
PAPER6 = {  # Table 6: N, mean, q05, median, q95 of MASE(objective) - MASE(mse)
    ("monthly", "huber"): (47999, 0.009, -0.262, 0.000, 0.285),
    ("monthly", "mae"): (47999, 0.016, -0.356, 0.000, 0.394),
    ("monthly", "ridge"): (47999, 0.003, -0.230, 0.000, 0.251),
    ("quarterly", "huber"): (24000, -0.004, -0.349, 0.000, 0.318),
    ("quarterly", "mae"): (24000, -0.007, -0.486, 0.000, 0.440),
    ("quarterly", "ridge"): (24000, -0.002, -0.320, 0.000, 0.325),
    ("yearly", "huber"): (23000, -0.016, -0.916, 0.000, 0.707),
    ("yearly", "mae"): (23000, -0.014, -1.469, 0.000, 1.281),
    ("yearly", "ridge"): (23000, -0.016, -0.988, 0.000, 0.880),
    ("weekly", "huber"): (359, 0.493, -0.665, 0.000, 0.812),
    ("weekly", "mae"): (359, 1.172, -0.650, 0.005, 2.065),
    ("weekly", "ridge"): (359, 0.068, -0.507, 0.000, 0.690),
}


def raw(name):
    with gzip.open(R / "raw" / f"{name}.csv.gz", "rt", encoding="utf-8") as fh:
        return pd.read_csv(fh)


def base(freq, kind):
    with gzip.open(R / "baselines" / f"{freq}_{kind}_total.csv.gz", "rt") as fh:
        return pd.read_csv(fh).set_index("series")


def cell(freq, obj):
    name = ("obj_mse_innov_weekly" if freq == "weekly" else f"req_true_{freq}") if obj == "mse" \
        else f"obj_{obj}_innov_{freq}"
    d = raw(name)
    return name, d[d.status.eq("OK")].set_index("series")


def main():
    out5, out6 = [], []
    for f in FREQS:
        nv, ar = base(f, "Naive"), base(f, "autoarima_R")
        cells = {o: cell(f, o) for o in OBJS}
        ids = set(nv.index) & set(ar.index)
        for o in OBJS:
            ids &= set(cells[o][1].index)
        ids = sorted(ids)
        S, M = nv.loc[ids, "sMAPE"].mean(), nv.loc[ids, "MASE"].mean()
        owa = lambda sm, ms: 0.5 * (sm.loc[ids].mean() / S + ms.loc[ids].mean() / M)
        vals = [owa(ar.sMAPE, ar.MASE)] + [owa(cells[o][1].smape_total, cells[o][1].mase_total) for o in OBJS]
        out5.append([f, len(ids), PAPER5_N[f]] + [round(v, 4) for v in vals] + list(PAPER5[f]))
        for o in OBJS[1:]:
            if (f, o) not in PAPER6 and not (f == "weekly" and o == "ridge"):
                continue
            dlt = (cells[o][1].loc[ids, "mase_total"] - cells["mse"][1].loc[ids, "mase_total"]).to_numpy()
            q = np.quantile(dlt, [0.05, 0.5, 0.95])
            out6.append([f, o, len(dlt), round(dlt.mean(), 3), round(q[0], 3), round(q[1], 3), round(q[2], 3),
                         *PAPER6.get((f, o), ("-",) * 5)])
        commits = {o: sorted(cells[o][1].sarimax_commit.astype(str).unique()) for o in OBJS}
        print(f"{f}: files {[cells[o][0] for o in OBJS]} commits {commits}")
    t5 = pd.DataFrame(out5, columns=["freq", "N", "N_paper", "ref", "mse", "huber", "mae", "ridge",
                                     "ref_p", "mse_p", "huber_p", "mae_p", "ridge_p"])
    t5["matches"] = [all(abs(r[c] - r[c + "_p"]) <= 1e-4 for c in ["ref", "mse", "huber", "mae", "ridge"]) and r.N == r.N_paper
                     for _, r in t5.iterrows()]
    print(t5.to_string(index=False))
    t6 = pd.DataFrame(out6, columns=["freq", "obj", "N", "mean", "q05", "median", "q95",
                                     "N_p", "mean_p", "q05_p", "median_p", "q95_p"])
    t6["matches"] = [r.N_p != "-" and all(abs(r[c] - r[c + "_p"]) <= 1e-3 for c in ["mean", "q05", "median", "q95"]) and r.N == r.N_p
                     for _, r in t6.iterrows()]
    print(t6.to_string(index=False))
    t5.to_csv(ROOT / "tables" / "table5_reproduction.csv", index=False)
    t6.to_csv(ROOT / "tables" / "table6_reproduction.csv", index=False)


if __name__ == "__main__":
    main()
