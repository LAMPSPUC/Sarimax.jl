"""Tables from the GAVEA raw archive (brutos_gavea.zip, Davi Valladao, shared 2026-10-16).

    python -I w_gavea_tables.py      (from paper_results_revision/scripts)

Reads _replication_gavea_brutos/conteudo/ (SHA256SUMS verified 177/177) and the M4 references in
ForecastTester/Results_autoarima_R/<freq>/{autoarima_R,Naive}/total.csv. Nothing is refitted.
OWA = 0.5 (mean sMAPE / mean sMAPE_Naive2 + mean MASE / mean MASE_Naive2) over the series that every
arm of the comparison fitted successfully. Writes tables/gavea_*.csv.
"""
import pathlib, re
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
G = ROOT / "_replication_gavea_brutos" / "conteudo"
REF = ROOT.parent / "Results_autoarima_R"
OUT = ROOT / "tables"
ORD = ["p", "d", "q", "P", "D", "Q"]


def ref(freq, kind):
    return pd.read_csv(REF / freq / kind / "total.csv").set_index("series")[["sMAPE", "MASE"]]


def load(pattern):
    fs = sorted(f for f in G.glob(pattern) if not f.name.endswith("_prev.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return d, [f.name for f in fs]


# The stable campaign of hourly and quarterly (commit dev@fc2c482+PR26) wrote MASE with the NON-seasonal
# in-sample naive (lag 1) as denominator; M4, campaign A, the requireTerms campaign, the monthly stable
# campaign and the 2x2x2x2 census use the seasonal naive (lag s). Verified on 400/400 series per frequency
# (audit, 2026-10-17). MAE is unchanged, so the M4 MASE is recovered exactly: MASE * den(lag 1) / den(lag s).
DATA = ROOT.parent / "datasets"
SEAS = {"hourly": 24, "quarterly": 4}
_den = {}


def m4_mase(freq, d):
    s = SEAS[freq]
    if freq not in _den:
        tr = pd.read_csv(DATA / f"{freq.capitalize()}-train.csv")
        vals = {}
        for i in range(len(tr)):
            y = tr.iloc[i, 1:].dropna().to_numpy(float)
            vals[i + 1] = np.mean(np.abs(y[1:] - y[:-1])) / np.mean(np.abs(y[s:] - y[:-s]))
        _den[freq] = pd.Series(vals)
    d = d.copy()
    d["mase_file"] = d["mase"]
    d["mase"] = d["mase"] * d["sid"].map(_den[freq]).to_numpy()
    return d


def owa(sm, ms, nv):
    return 0.5 * (sm.mean() / nv.sMAPE.mean() + ms.mean() / nv.MASE.mean())


def ok(d):
    s = d["status"].astype(str)
    return d[s.isin(["LOCALLY_SOLVED", "OK", "ALMOST_LOCALLY_SOLVED"])]


def compare(freq, arms, extra_note=""):
    """arms: dict label -> dataframe with sid, smape, mase, status. Pairs on common ok sids."""
    nv, ar = ref(freq, "Naive"), ref(freq, "autoarima_R")
    clean = {}
    for k, d in arms.items():
        d = d.drop_duplicates("sid", keep="last")
        clean[k] = ok(d).set_index("sid")
    finite = lambda r: set(r.index[np.isfinite(r[['sMAPE', 'MASE']]).all(axis=1)])
    ids = finite(nv) & finite(ar)          # series with a non-finite reference are excluded from every arm
    for d in clean.values():
        d = d[np.isfinite(d[['smape', 'mase']]).all(axis=1)]
        ids &= set(d.index)
    ids = sorted(ids)
    rows = [dict(freq=freq, arm="auto.arima (reference)", N=len(ids), smape=ar.loc[ids, "sMAPE"].mean(),
                 mase=ar.loc[ids, "MASE"].mean(), owa=owa(ar.loc[ids, "sMAPE"], ar.loc[ids, "MASE"], nv.loc[ids]))]
    for k, d in clean.items():
        rows.append(dict(freq=freq, arm=k, N=len(ids), smape=d.loc[ids, "smape"].mean(), mase=d.loc[ids, "mase"].mean(),
                         owa=owa(d.loc[ids, "smape"], d.loc[ids, "mase"], nv.loc[ids])))
    out = pd.DataFrame(rows)
    status = {k: arms[k]["status"].astype(str).value_counts().to_dict() for k in arms}
    nrows = {k: len(arms[k].drop_duplicates("sid")) for k in arms}
    return out, ids, clean, status, nrows


def paired(clean, ids, a, b):
    da = clean[a].loc[ids, "mase"] - clean[b].loc[ids, "mase"]
    ds = clean[a].loc[ids, "smape"] - clean[b].loc[ids, "smape"]
    same = (clean[a].loc[ids, ORD].values == clean[b].loc[ids, ORD].values).all(axis=1)
    q = np.quantile(da, [0.05, 0.5, 0.95])
    return dict(pair=f"{a} - {b}", N=len(ids), mean_dMASE=da.mean(), q05=q[0], median=q[1], q95=q[2],
                pct_lower_MASE=100 * (da < -1e-4).mean(), pct_tie=100 * (da.abs() <= 1e-4).mean(),
                pct_lower_sMAPE=100 * (ds < -1e-4).mean(), pct_same_order=100 * same.mean())


summary, pairs, validity = [], [], []

# ---- stable vs mse: hourly, quarterly, daily -------------------------------------------------
for freq in ("hourly", "quarterly", "daily"):
    m, fm = load(f"stable_{freq}/{freq}_mse_*.csv")
    s, fs = load(f"stable_{freq}/{freq}_stable_*.csv")
    if freq in SEAS:
        m, s = m4_mase(freq, m), m4_mase(freq, s)
    out, ids, clean, st, nr = compare(freq, {"mse": m, "stable (alpha=0.5)": s})
    out["campaign"] = "stable"; out["commit"] = "; ".join(sorted(set(m.commit) | set(s.commit)))
    summary.append(out)
    pairs.append(dict(freq=freq, **paired(clean, ids, "stable (alpha=0.5)", "mse")))
    cens = pd.read_csv(G / "stable_daily/censurados.csv") if freq == "daily" else None
    validity.append(dict(freq=freq, campaign="stable", rows_mse=nr["mse"], rows_stable=nr["stable (alpha=0.5)"],
                         status_mse=st["mse"], status_stable=st["stable (alpha=0.5)"], paired_N=len(ids),
                         censored=(cens.motivo.value_counts().to_dict() if cens is not None else {})))

# ---- stable monthly (aa68d57, 5,000-series sample; alpha=0.5 partial) ---------------------------
m, _ = load("stable_monthly/stm4_mse_0.9_*.csv")
s9, _ = load("stable_monthly/stm4_stable_0.9_*.csv")
s5, _ = load("stable_monthly/stm4_stable_0.5_*.csv")
for arms, tag in (({"mse": m, "stable (alpha=0.9)": s9}, "mse vs stable 0.9"),
                  ({"mse": m, "stable (alpha=0.9)": s9, "stable (alpha=0.5)": s5}, "three arms, alpha=0.5 subset")):
    out, ids, clean, st, nr = compare("monthly", arms)
    out["campaign"] = f"stable monthly: {tag}"; out["commit"] = "aa68d57"
    summary.append(out)
    for k in arms:
        if k != "mse":
            pairs.append(dict(freq=f"monthly ({tag})", **paired(clean, ids, k, "mse")))
    validity.append(dict(freq="monthly", campaign=f"stable: {tag}", **{f"rows_{k}": v for k, v in nr.items()},
                         **{f"status_{k}": v for k, v in st.items()}, paired_N=len(ids)))

# ---- stable weekly and yearly: campaign F of host RECREIO (3d7f883), already in _replication/ ----------
F = pd.read_csv(ROOT / "_replication/experiments/results/raw/stable_weekly_yearly.csv", comment="#")
F = F.rename(columns={"series": "sid"})
F["status"] = F["status"].astype(str).str.replace("^OK.*$", "OK", regex=True)
for freq in ("weekly", "yearly"):
    f = F[F.freq == freq]
    arms = {"mse": f[f.arm == "mse"], "stable (alpha=0.9)": f[f.arm == "stable_090"], "stable (alpha=0.5)": f[f.arm == "stable_050"]}
    out, ids, clean, st, nr = compare(freq, arms)
    out["campaign"] = "stable (campaign F, RECREIO)"; out["commit"] = "3d7f883"
    summary.append(out)
    for k in ("stable (alpha=0.9)", "stable (alpha=0.5)"):
        pairs.append(dict(freq=freq, **paired(clean, ids, k, "mse")))
    validity.append(dict(freq=freq, campaign="stable (campaign F)", **{f"rows_{k}": v for k, v in nr.items()},
                         **{f"status_{k}": v for k, v in st.items()}, paired_N=len(ids)))

# ---- requireTermsWhenOverDifferenced = true (v1.0.0) vs false (the stable campaigns' mse arm) ----
for freq in ("hourly", "quarterly", "daily"):
    t, _ = load(f"requireterms/{freq}_mse_*.csv")
    f, _ = load(f"stable_{freq}/{freq}_mse_*.csv")
    if freq in SEAS:
        f = m4_mase(freq, f)
    out, ids, clean, st, nr = compare(freq, {"mse, guard off (stable campaign arm)": f, "mse, guard on": t})
    out["campaign"] = "requireTerms"; out["commit"] = "; ".join(sorted(set(t.commit) | set(f.commit)))
    summary.append(out)
    pairs.append(dict(freq=f"{freq} (guard)", **paired(clean, ids, "mse, guard on", "mse, guard off (stable campaign arm)")))
    validity.append(dict(freq=freq, campaign="requireTerms", rows_on=nr["mse, guard on"], rows_off=nr["mse, guard off (stable campaign arm)"],
                         status_on=st["mse, guard on"], paired_N=len(ids)))

# ---- 2x2x2x2 census, monthly, 4e8cf11 ---------------------------------------------------------
cells = sorted({m.group(1) for p in (G / "censo_2x2x2x2").glob("*.csv")
                for m in [re.match(r"([a-z0-9]+)_\d+\.csv$", p.name)] if m})
arms = {}
for c in cells:
    d, _ = load(f"censo_2x2x2x2/{c}_[0-9]*.csv")
    d = d.loc[:, ~d.columns.duplicated()]
    arms[c] = d
out, ids, clean, st, nr = compare("monthly", arms)
out["campaign"] = "2x2x2x2 census"; out["commit"] = "4e8cf11"
summary.append(out)
validity.append(dict(freq="monthly", campaign="2x2x2x2", **{f"rows_{k}": v for k, v in nr.items()},
                     **{f"status_{k}": v for k, v in st.items()}, paired_N=len(ids)))
fb = {c: int((arms[c]["fallback"].astype(str) != "0").sum()) for c in cells if "fallback" in arms[c]}

S = pd.concat(summary, ignore_index=True)
P = pd.DataFrame(pairs)
V = pd.DataFrame(validity)
S.to_csv(OUT / "gavea_owa_summary.csv", index=False)
P.to_csv(OUT / "gavea_paired_differences.csv", index=False)
V.to_csv(OUT / "gavea_validity.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(S[["campaign", "freq", "arm", "N", "smape", "mase", "owa"]].round(4).to_string(index=False))
print(); print(P.round(4).to_string(index=False))
print(); print(V.to_string(index=False))
print("\n2x2x2x2 fallback counts (fallback != 0):", fb)
