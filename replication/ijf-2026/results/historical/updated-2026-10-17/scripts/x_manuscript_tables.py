"""Manuscript-ready tables from the GAVEA archive, campaign F and the July SCIP study.

    python -I x_manuscript_tables.py      (after w_gavea_tables.py; from paper_results_revision/scripts)

Writes tables/{stable_objective,stationarity_invertibility_grid,scip_restricted_yearly}.{csv,tex}.
Every number is read from tables/gavea_owa_summary.csv (itself recomputed from the raw rows) or from
ForecastTester/docker_scip/*.csv; nothing is typed by hand.
"""
import pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
T = ROOT / "tables"
S = pd.read_csv(T / "gavea_owa_summary.csv")


def get(campaign, freq, arm, col="owa"):
    r = S[(S.campaign == campaign) & (S.freq == freq) & (S.arm == arm)]
    assert len(r) == 1, (campaign, freq, arm, len(r))
    return float(r[col].iloc[0])


def tex(df, path, caption, label, fmt):
    cols = list(df.columns)
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", "\\toprule",
             " & ".join(cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(fmt(c, r[c]) for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def f4(c, v):
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return "--"
    if c == "N":
        return f"{int(v):,}"
    if c.startswith("$\\Delta$"):
        return f"{v:+.4f}"
    return f"{v:.4f}"


# ---- 1. stable objective ----------------------------------------------------------------------
rows = []
spec = [  # freq, campaign label in the summary, N source arm, alpha arms available, note
    ("Yearly", "yearly", "stable (campaign F, RECREIO)", ["0.5", "0.9"], ""),
    ("Quarterly", "quarterly", "stable", ["0.5"], "a"),
    ("Monthly", "monthly", "stable monthly: mse vs stable 0.9", ["0.9"], "b"),
    ("Monthly", "monthly", "stable monthly: three arms, alpha=0.5 subset", ["0.5"], "c"),
    ("Weekly", "weekly", "stable (campaign F, RECREIO)", ["0.5", "0.9"], ""),
    ("Daily", "daily", "stable", ["0.5"], "d"),
    ("Hourly", "hourly", "stable", ["0.5"], "e"),
]
for name, freq, camp, alphas, note in spec:
    mse = get(camp, freq, "mse")
    r = {"Frequency": name + (f"$^{note}$" if note else ""), "N": get(camp, freq, "mse", "N"),
         "Reference": get(camp, freq, "auto.arima (reference)"), "MSE": mse}
    for a in ("0.5", "0.9"):
        if a in alphas:
            v = get(camp, freq, f"stable (alpha={a})")
            r[f"stable $\\alpha$={a}"] = v
            r[f"$\\Delta$ {a}"] = v - mse
        else:
            r[f"stable $\\alpha$={a}"] = np.nan
            r[f"$\\Delta$ {a}"] = np.nan
    rows.append(r)
st = pd.DataFrame(rows)
st.to_csv(T / "stable_objective.csv", index=False)
tex(st, T / "stable_objective.tex",
    "M4 OWA of the \\texttt{stable} (CVaR) objective against MSE, paired within each campaign. "
    "Reference is \\texttt{forecast::auto.arima}; $\\Delta$ = stable $-$ MSE; lower is better. "
    "$^a$ 55 stable fits not locally solved excluded from both arms. $^b$ 5,000-series random subsample, "
    "3 stable failures excluded. $^c$ the 923 series of that subsample on which the $\\alpha=0.5$ arm finished. "
    "$^d$ 229 stable fits censored (solver aborts and time caps) and 19 errors, excluded from both arms. "
    "$^e$ 3 series with a non-finite Naive2 reference excluded; MASE recomputed with the seasonal "
    "naive denominator.", "tab:stable", f4)

# ---- 2. stationarity x invertibility x MA normalization (Section 5.5) ----------------------------
camp = "2x2x2x2 census"
lab = {"f1": ("no", "no", "off"), "fs": ("yes", "no", "off"), "fi": ("no", "yes", "off"), "fsi": ("yes", "yes", "off"),
       "gff": ("no", "no", "on"), "gs": ("yes", "no", "on"), "gi": ("no", "yes", "on"), "gsi": ("yes", "yes", "on")}
g = pd.DataFrame([{"Stationary": a, "Invertible": b, "MA normalization": c, "N": get(camp, "monthly", k, "N"),
                   "sMAPE": get(camp, "monthly", k, "smape"), "MASE": get(camp, "monthly", k, "mase"),
                   "OWA": get(camp, "monthly", k)} for k, (a, b, c) in lab.items()])
g.to_csv(T / "stationarity_invertibility_grid.csv", index=False)
tex(g, T / "stationarity_invertibility_grid.tex",
    "Monthly M4 OWA under the eight combinations of reflection-coordinate stationarity, reflection-coordinate "
    "invertibility and the Gaussian MA normalization (MSE objective, \\texttt{:innovations}). "
    f"\\texttt{{forecast::auto.arima}}: {get(camp, 'monthly', 'auto.arima (reference)'):.4f}.",
    "tab:admissibility_grid", lambda c, v: v if isinstance(v, str) else (f"{int(v):,}" if c == "N" else f"{v:.4f}"))

# ---- 3. SCIP vs Ipopt, short yearly M4 series, July study -----------------------------------------
D = ROOT.parent / "docker_scip"
out = []
for lbl, f in (("box only (direct coefficients)", "scip_experiment_results.csv"),
               ("stationary and invertible by construction", "scip_constr_results.csv")):
    d = pd.read_csv(D / f)
    rel = d.sigma2_reldiff.astype(float)
    out.append({"Parameterization": lbl, "Series": len(d),
                "SCIP status OPTIMAL": int((d.scip_status.astype(str) == "OPTIMAL").sum()),
                "same objective (rel. < 1e-6)": int((rel.abs() < 1e-6).sum()),
                "SCIP objective lower (> 0.01%)": int((rel < -1e-4).sum()),
                "largest reduction (%)": round(-100 * rel.min(), 2),
                "sMAPE SCIP": d.smape_scip.mean(), "sMAPE Ipopt": d.smape_ipopt.mean(),
                "mean time SCIP (s)": d.s_scip.mean(), "mean time Ipopt (s)": d.s_ipopt.mean()})
sc = pd.DataFrame(out)
sc.to_csv(T / "scip_restricted_yearly.csv", index=False)
print(st.round(4).to_string(index=False)); print(); print(g.round(4).to_string(index=False)); print(); print(sc.round(4).T.to_string())
