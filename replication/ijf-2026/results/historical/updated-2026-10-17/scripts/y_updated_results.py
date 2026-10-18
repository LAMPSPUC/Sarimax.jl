"""Updated results, 2026-10-17, in the SAME schemas as the earlier tables (Part C, L, M, P):

  <stem>_summary.csv                 frequency, arm, N, sMAPE, MASE, OWA, OWA_autoarima_same_series,
                                     diffs vs the baseline arm         (schema of objective_stable_innovations.csv)
  <stem>_distribution_quantiles.csv  frequency, metric, method, N, mean, q01..q99
                                     (schema of objective_distribution_quantiles_full.csv)
  <stem>_paired_differences.csv      frequency, objective, baseline, N, <metric>_{mean_diff, trimmed_mean_diff_1_99,
                                     median_diff, q01..q99, pct_lower, pct_tie, pct_higher, pct_same_order}
                                     (schema of objective_distribution_summary.csv)
  <stem>_tail_severity.csv           frequency, metric, method, N, q95, q99, es05, es01, exceed_<base>_q95, exceed_ratio
                                     (schema of objective_tail_severity.csv)
  <stem>_union_tail.csv              frequency, metric, method, N_union, mean_in_U, ... (schema of objective_union_tail.csv)
  <stem>_*.tex                       compact LaTeX in the same layout as the earlier .tex files

Stems:
  stable_objective        mse vs stable (CVaR, alpha 0.5 / 0.9), six M4 frequencies
  admissibility_grid      stationary x invertible x MA normalization, monthly (2x2x2x2 census)
  require_terms_guard     requireTermsWhenOverDifferenced off vs on, hourly / quarterly / daily

Inputs: _replication_gavea_brutos/conteudo (brutos_gavea.zip, SHA256SUMS 177/177), campaign F
(_replication/experiments/results/raw/stable_weekly_yearly.csv), Results_autoarima_R/<freq>/{autoarima_R,Naive}.
The stable hourly/quarterly MASE is converted to the M4 (seasonal-naive) denominator exactly; see
reports/gavea_archive_audit.md section 1. Paired on series that every arm fitted and whose
references are finite. Tie tolerance, quantile method: common.py (1e-4, numpy linear).

    python -I y_updated_results.py      (from paper_results_revision/scripts)
"""
import pathlib
import numpy as np, pandas as pd
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C

ROOT = pathlib.Path(__file__).resolve().parents[1]
G = ROOT / "_replication_gavea_brutos" / "conteudo"
REF = ROOT.parent / "Results_autoarima_R"
DATA = ROOT.parent / "datasets"
TAB = ROOT / "tables"
ORD = ["p", "d", "q", "P", "D", "Q"]
SEAS = {"hourly": 24, "quarterly": 4}
METRICS = {"smape": "sMAPE", "mase": "MASE"}


def load(pattern):
    fs = sorted(f for f in G.glob(pattern) if not f.name.endswith("_prev.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


_ratio = {}


def m4_mase(freq, d):
    """MASE written with the lag-1 naive -> MASE with the seasonal naive (exact: MAE unchanged)."""
    s = SEAS[freq]
    if freq not in _ratio:
        tr = pd.read_csv(DATA / f"{freq.capitalize()}-train.csv")
        _ratio[freq] = pd.Series({i + 1: np.mean(np.abs(np.diff(y))) / np.mean(np.abs(y[s:] - y[:-s]))
                                  for i, y in enumerate(tr.iloc[:, 1:].to_numpy(float))
                                  for y in [y[~np.isnan(y)]]})
    d = d.copy()
    d["mase"] = d["mase"] * d["sid"].map(_ratio[freq]).to_numpy()
    return d


def ok(d):
    d = d.drop_duplicates("sid", keep="last")
    d = d[d["status"].astype(str).isin(["LOCALLY_SOLVED", "OK", "ALMOST_LOCALLY_SOLVED"])]
    return d[np.isfinite(d[["smape", "mase"]]).all(axis=1)].set_index("sid")


def ref(freq, kind):
    r = pd.read_csv(REF / freq / kind / "total.csv").set_index("series").rename(columns={"sMAPE": "smape", "MASE": "mase"})
    return r[np.isfinite(r[["smape", "mase"]]).all(axis=1)]


def wide(freq, arms):
    """arms: ordered dict label -> raw dataframe. Adds auto.arima and Naive2. Returns the paired frame."""
    nv, ar = ref(freq, "Naive"), ref(freq, "autoarima_R")
    A = {k: ok(v) for k, v in arms.items()}
    ids = set(nv.index) & set(ar.index)
    for v in A.values():
        ids &= set(v.index)
    ids = sorted(ids)
    W = pd.DataFrame(index=pd.Index(ids, name="series"))
    for k, v in A.items():
        for m in METRICS:
            W[f"{m}|{k}"] = v.loc[ids, m].to_numpy()
        W[f"order|{k}"] = v.loc[ids, ORD].astype(int).astype(str).agg(",".join, axis=1).to_numpy()
    for m in METRICS:
        W[f"{m}|auto.arima"] = ar.loc[ids, m].to_numpy()
        W[f"{m}|Naive2"] = nv.loc[ids, m].to_numpy()
    return W


def owa(W, k):
    return 0.5 * (W[f"smape|{k}"].mean() / W["smape|Naive2"].mean() + W[f"mase|{k}"].mean() / W["mase|Naive2"].mean())


def summary(freq, W, arms, base):
    rows = []
    for k in arms:
        r = {"frequency": freq, "arm": k, "N": len(W), "sMAPE": W[f"smape|{k}"].mean(), "MASE": W[f"mase|{k}"].mean(),
             "OWA": owa(W, k), "OWA_autoarima_same_series": owa(W, "auto.arima")}
        for m, lab in METRICS.items():
            d = W[f"{m}|{k}"] - W[f"{m}|{base}"]
            r.update({f"{lab}_mean_diff_vs_{base}": d.mean() if k != base else np.nan,
                      f"{lab}_median_diff_vs_{base}": d.median() if k != base else np.nan,
                      f"{lab}_pct_lower_vs_{base}": 100 * (d < -C.TIE_TOL).mean() if k != base else np.nan,
                      f"{lab}_pct_higher_vs_{base}": 100 * (d > C.TIE_TOL).mean() if k != base else np.nan})
        rows.append(r)
    return rows


def quantiles(freq, W, methods):
    return [{"frequency": freq, "metric": lab, "method": k, **C.quantile_row(W[f"{m}|{k}"].to_numpy())}
            for m, lab in METRICS.items() for k in methods]


def paired(freq, W, arms, base):
    rows = []
    for b in (base, "auto.arima"):
        for k in list(arms) + ["auto.arima"]:
            if k == b:
                continue
            rec = {"frequency": freq, "objective": k, "baseline": b, "N": len(W)}
            for m, lab in METRICS.items():
                x = (W[f"{m}|{k}"] - W[f"{m}|{b}"]).to_numpy()
                q = np.quantile(x, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], method=C.QUANTILE_METHOD)
                same = (100 * np.mean(W[f"order|{k}"] == W[f"order|{b}"])
                        if f"order|{k}" in W and f"order|{b}" in W else np.nan)
                rec.update({f"{lab}_mean_diff": x.mean(), f"{lab}_trimmed_mean_diff_1_99": x[(x >= q[0]) & (x <= q[-1])].mean(),
                            f"{lab}_median_diff": q[3], f"{lab}_q01": q[0], f"{lab}_q05": q[1], f"{lab}_q25": q[2],
                            f"{lab}_q75": q[4], f"{lab}_q95": q[5], f"{lab}_q99": q[6],
                            f"{lab}_pct_lower": 100 * np.mean(x < -C.TIE_TOL), f"{lab}_pct_tie": 100 * np.mean(np.abs(x) <= C.TIE_TOL),
                            f"{lab}_pct_higher": 100 * np.mean(x > C.TIE_TOL), f"{lab}_pct_same_order": same})
            rows.append(rec)
    return rows


def es(x, a):
    t = np.quantile(x, 1 - a, method=C.QUANTILE_METHOD)
    return float(x[x >= t].mean())


def top_set(x, frac=0.05):
    k = int(np.ceil(frac * x.size))
    return np.argsort(-x, kind="stable")[:k]


def tails(freq, W, methods, base):
    sev, uni = [], []
    for m, lab in METRICS.items():
        b = W[f"{m}|{base}"].to_numpy(float)
        aa = W[f"{m}|auto.arima"].to_numpy(float)
        mask = np.zeros(b.size, bool)
        mask[np.union1d(top_set(b), top_set(aa))] = True
        t95 = float(np.quantile(b, 0.95, method=C.QUANTILE_METHOD))
        for k in methods:
            x = W[f"{m}|{k}"].to_numpy(float)
            d = x - b
            sev.append({"frequency": freq, "metric": lab, "method": k, "N": int(x.size),
                        "q95": float(np.quantile(x, 0.95, method=C.QUANTILE_METHOD)),
                        "q99": float(np.quantile(x, 0.99, method=C.QUANTILE_METHOD)), "es05": es(x, 0.05), "es01": es(x, 0.01),
                        f"exceed_{base}_q95": float(np.mean(x > t95)), f"exceed_ratio_vs_{base}": float(np.mean(x > t95) / 0.05)})
            rec = {"frequency": freq, "metric": lab, "method": k, "N_union": int(mask.sum()),
                   "mean_in_U": float(x[mask].mean()), "median_in_U": float(np.median(x[mask])),
                   "mean_outside_U": float(x[~mask].mean()), "mean_all": float(x.mean())}
            if k != base:
                rec.update({"median_diff_in_U": float(np.median(d[mask])), "mean_diff_in_U": float(d[mask].mean()),
                            "pct_lower_in_U": 100 * float(np.mean(d[mask] < -C.TIE_TOL)),
                            "pct_higher_in_U": 100 * float(np.mean(d[mask] > C.TIE_TOL)),
                            "median_diff_outside_U": float(np.median(d[~mask])), "mean_diff_outside_U": float(d[~mask].mean()),
                            "pct_lower_outside_U": 100 * float(np.mean(d[~mask] < -C.TIE_TOL)),
                            "pct_higher_outside_U": 100 * float(np.mean(d[~mask] > C.TIE_TOL))})
            uni.append(rec)
    return sev, uni


def num(v):
    return f"{int(v):,}".replace(",", "{,}")


def latex_summary(S, arms_label, base):
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             rf"Freq. & Arm & $N$ & sMAPE & MASE & OWA & $\Delta$OWA vs {arms_label.get(base, base)} \\", r"\midrule"]
    last = None
    for _, r in S.iterrows():
        bo = S[(S.frequency == r.frequency) & (S.arm == base) & (S.N == r.N)].OWA
        d = "--" if r.arm == base or bo.empty else f"{r.OWA - bo.iloc[0]:+.4f}"
        if last is not None and r.frequency != last:
            lines.append(r"\addlinespace")
        lines.append(" & ".join([r.frequency.capitalize() if r.frequency != last else "", arms_label.get(r.arm, r.arm), num(r.N),
                                 f"{r.sMAPE:.3f}", f"{r.MASE:.3f}", f"{r.OWA:.4f}", d]) + r" \\")
        last = r.frequency
    lines += [r"\midrule", r"\multicolumn{7}{l}{\footnotesize auto.arima OWA on the same series: " +
              "; ".join(f"{f} {v:.4f}" for f, v in S.groupby('frequency', sort=False).OWA_autoarima_same_series.first().items()) + r"} \\",
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_paired(P, base, arms_label):
    d = P[P.baseline == base]
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             rf" & & & \multicolumn{{4}}{{c}}{{MASE, $\Delta$ vs {arms_label.get(base, base)}}} & \multicolumn{{2}}{{c}}{{sMAPE, $\Delta$ vs {arms_label.get(base, base)}}} \\",
             r"\cmidrule(lr){4-7}\cmidrule(lr){8-9}",
             r"Freq. & Method & $N$ & Median & $q_{05}$ & $q_{95}$ & \% lower & Median & \% lower \\", r"\midrule"]
    last = None
    for _, r in d.iterrows():
        if last is not None and r.frequency != last:
            lines.append(r"\addlinespace")
        lines.append(" & ".join([r.frequency.capitalize() if r.frequency != last else "", arms_label.get(r.objective, r.objective), num(r.N),
                                 f"{r.MASE_median_diff:.3f}", f"{r.MASE_q05:.3f}", f"{r.MASE_q95:.3f}", f"{r.MASE_pct_lower:.1f}",
                                 f"{r.sMAPE_median_diff:.3f}", f"{r.sMAPE_pct_lower:.1f}"]) + r" \\")
        last = r.frequency
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_union(U, Sv, base, arms_label):
    lines = [r"\begin{tabular}{lllrrrrrr}", r"\toprule",
             r" & & & \multicolumn{4}{c}{inside $U$} & \multicolumn{2}{c}{whole sample} \\",
             r"\cmidrule(lr){4-7}\cmidrule(lr){8-9}",
             r"Freq. & Metric & Objective & mean & median $\Delta$ & \% lower & \% higher & $q_{99}$ & $\mathrm{ES}_{5\%}$ \\",
             r"\midrule"]
    last = None
    for _, r in U.iterrows():
        s = Sv[(Sv.frequency == r.frequency) & (Sv.metric == r.metric) & (Sv.method == r.method)].iloc[0]
        key = (r.frequency, r.metric)
        if last is not None and r.frequency != last[0]:
            lines.append(r"\addlinespace")
        isb = r.method == base
        lines.append(" & ".join([r.frequency.capitalize() if last is None or r.frequency != last[0] else "",
                                 r.metric if last != key else "", arms_label.get(r.method, r.method), f"{r.mean_in_U:.3g}",
                                 "--" if isb else f"{r.median_diff_in_U:.3g}", "--" if isb else f"{r.pct_lower_in_U:.1f}\\%",
                                 "--" if isb else f"{r.pct_higher_in_U:.1f}\\%", f"{s.q99:.3g}", f"{s.es05:.3g}"]) + r" \\")
        last = key
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit(stem, blocks, base, label):
    S, Q, P, Sv, U = (pd.DataFrame(sum((b[i] for b in blocks), [])) for i in range(5))
    S.to_csv(TAB / f"{stem}_summary.csv", index=False)
    Q.to_csv(TAB / f"{stem}_distribution_quantiles.csv", index=False)
    P.to_csv(TAB / f"{stem}_paired_differences.csv", index=False)
    Sv.to_csv(TAB / f"{stem}_tail_severity.csv", index=False)
    U.to_csv(TAB / f"{stem}_union_tail.csv", index=False)
    (TAB / f"{stem}_summary.tex").write_text(latex_summary(S, label, base), encoding="utf-8")
    (TAB / f"{stem}_paired_differences.tex").write_text(latex_paired(P, base, label), encoding="utf-8")
    (TAB / f"{stem}_union_tail.tex").write_text(latex_union(U, Sv, base, label), encoding="utf-8")
    print(f"\n=== {stem}"); print(S[["frequency", "arm", "N", "sMAPE", "MASE", "OWA", "OWA_autoarima_same_series"]].round(4).to_string(index=False))


def block(freq, arms, base):
    W = wide(freq, arms)
    methods = list(arms) + ["auto.arima"]
    sev, uni = tails(freq, W, methods, base)
    return (summary(freq, W, list(arms), base), quantiles(freq, W, methods + ["Naive2"]), paired(freq, W, arms, base), sev, uni)


# ---- stable objective, six frequencies ---------------------------------------------------------
F = pd.read_csv(ROOT / "_replication/experiments/results/raw/stable_weekly_yearly.csv", comment="#").rename(columns={"series": "sid"})
F["status"] = F["status"].astype(str).str.replace("^OK.*$", "OK", regex=True)
B = []
for freq in ("yearly",):
    f = F[F.freq == freq]
    B.append(block(freq, {"mse": f[f.arm == "mse"], "stable_050": f[f.arm == "stable_050"], "stable_090": f[f.arm == "stable_090"]}, "mse"))
q_m, q_s = load("stable_quarterly/quarterly_mse_*.csv"), load("stable_quarterly/quarterly_stable_*.csv")
B.append(block("quarterly", {"mse": m4_mase("quarterly", q_m), "stable_050": m4_mase("quarterly", q_s)}, "mse"))
mm, m9, m5 = load("stable_monthly/stm4_mse_0.9_*.csv"), load("stable_monthly/stm4_stable_0.9_*.csv"), load("stable_monthly/stm4_stable_0.5_*.csv")
B.append(block("monthly", {"mse": mm, "stable_090": m9}, "mse"))
b5 = block("monthly", {"mse": mm, "stable_050": m5}, "mse")
for part in b5:
    for r in part:
        r["frequency"] = "monthly (alpha=0.5 subset)"
B.append(b5)
f = F[F.freq == "weekly"]
B.append(block("weekly", {"mse": f[f.arm == "mse"], "stable_050": f[f.arm == "stable_050"], "stable_090": f[f.arm == "stable_090"]}, "mse"))
B.append(block("daily", {"mse": load("stable_daily/daily_mse_*.csv"), "stable_050": load("stable_daily/daily_stable_*.csv")}, "mse"))
B.append(block("hourly", {"mse": m4_mase("hourly", load("stable_hourly/hourly_mse_*.csv")),
                          "stable_050": m4_mase("hourly", load("stable_hourly/hourly_stable_*.csv"))}, "mse"))
emit("stable_objective", B, "mse", {"mse": "MSE", "stable_050": r"stable $\alpha=0.5$", "stable_090": r"stable $\alpha=0.9$"})

# ---- admissibility x normalization grid (Section 5.5), baseline = stationary, normalization on ----
cells = ["gs", "gff", "gi", "gsi", "fs", "f1", "fi", "fsi"]
lab = {"gs": "stat., normalized", "gff": "free, normalized", "gi": "inv., normalized", "gsi": "stat.+inv., normalized",
       "fs": "stat., not normalized", "f1": "free, not normalized", "fi": "inv., not normalized", "fsi": "stat.+inv., not normalized"}
emit("admissibility_grid", [block("monthly", {c: load(f"censo_2x2x2x2/{c}_[0-9]*.csv") for c in cells}, "gs")], "gs", lab)

# ---- over-differencing guard -------------------------------------------------------------------
G3 = []
for freq in ("hourly", "quarterly", "daily"):
    off = load(f"stable_{freq}/{freq}_mse_*.csv")
    if freq in SEAS:
        off = m4_mase(freq, off)
    G3.append(block(freq, {"guard_off": off, "guard_on": load(f"requireterms/{freq}_mse_*.csv")}, "guard_off"))
emit("require_terms_guard", G3, "guard_off", {"guard_off": "MSE, guard off", "guard_on": "MSE, guard on"})
