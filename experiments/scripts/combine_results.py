#!/usr/bin/env python
"""Combine JSONL raw run records into processed CSV + manuscript LaTeX tables.

Usage: python combine_results.py [block ...]
Blocks: validation forecasting architecture solver  (default: all available)
No fabrication: missing values render as '--'; failed runs are shown with status.
"""
import os, sys, json, glob
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "results", "raw"))
PROC = os.path.normpath(os.path.join(HERE, "..", "results", "processed"))
TAB = os.path.normpath(os.path.join(HERE, "..", "tables"))
os.makedirs(PROC, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


def load_jsonl(block):
    rows = []
    for fp in sorted(glob.glob(os.path.join(RAW, block, "*_results.jsonl"))):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def first_coef(est, *keys):
    """Return first available coefficient by trying list/dict keys."""
    if not isinstance(est, dict):
        return None
    for k in keys:
        v = est.get(k)
        if isinstance(v, list) and v:
            return v[0]
        if isinstance(v, (int, float)):
            return v
    return None


def tex_escape(s):
    s = str(s)
    # ASCII-ize a few unicode symbols that appear in labels (pdfLaTeX-safe)
    for a, b in (("θ", "theta"), ("φ", "phi"), ("ε", "eps"),
                 ("→", "->"), ("²", "2")):
        s = s.replace(a, b)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def latex_table(df, caption, label, colfmt=None):
    cols = list(df.columns)
    if colfmt is None or len(colfmt) != len(cols):
        colfmt = "l" * len(cols)
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{colfmt}}}", r"\hline"]
    lines.append(" & ".join(tex_escape(c) for c in cols) + r" \\")
    lines.append(r"\hline")
    for _, r in df.iterrows():
        lines.append(" & ".join(tex_escape(v) for v in r.values) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_validation():
    rows = load_jsonl("validation")
    if not rows:
        print("validation: no records"); return
    recs = []
    for r in rows:
        est = r.get("estimates", {})
        # statsmodels keys: ar.L1, ma.L1; julia: phi/theta; R: ar1/ma1
        phi = first_coef(est, "phi") or est.get("ar.L1") or est.get("ar1")
        theta = first_coef(est, "theta") or est.get("ma.L1") or est.get("ma1")
        recs.append({
            "Implementation": r["implementation"], "Dataset": r["dataset"],
            "Order": r["order"], "phi1": fmt(phi), "theta1": fmt(theta),
            "logLik": fmt(r.get("loglike"), 2), "AIC": fmt(r.get("aic"), 2),
            "BIC": fmt(r.get("bic"), 2), "RSS": fmt(r.get("rss"), 2),
            "runtime_s": fmt(r.get("runtime_s"), 3), "status": r.get("status"),
        })
    df = pd.DataFrame(recs).sort_values(["Dataset", "Order", "Implementation"])
    df.to_csv(os.path.join(TAB, "table_validation.csv"), index=False)
    tex = latex_table(
        df, "Validation against established implementations under comparable "
            "SARIMA/SARIMAX specifications.", "tab:validation_implementations",
        colfmt="lllrrrrrrrl")
    open(os.path.join(TAB, "table_validation.tex"), "w").write(tex)
    print("wrote table_validation.{csv,tex}  rows:", len(df))


def perfit(r, nd=3):
    """Mean seconds per refit = total runtime / number of rolling origins."""
    rt, no = r.get("runtime_s"), r.get("n_origins")
    if rt is None or not no:
        return "--"
    return fmt(rt / no, nd)


def build_forecasting():
    rows = load_jsonl("forecasting")
    if not rows:
        print("forecasting: no records"); return
    recs = [{
        "Implementation": r["implementation"], "Task": r["dataset"],
        "Order": r.get("order", "--"),
        "Protocol": r.get("protocol", "split"),
        "Origins": r.get("n_origins", "--"), "H": r.get("horizon", "--"),
        "MAE": fmt(r.get("mae")), "RMSE": fmt(r.get("rmse")),
        "sMAPE": fmt(r.get("smape")), "MASE": fmt(r.get("mase")),
        "s/fit": perfit(r), "status": r.get("status"),
    } for r in rows]
    df = pd.DataFrame(recs).drop_duplicates(
        subset=["Implementation", "Task", "Order"]).sort_values(
        ["Task", "Implementation", "Order"])
    df.to_csv(os.path.join(TAB, "table_forecasting.csv"), index=False)
    tex = latex_table(df, "Out-of-sample rolling-origin forecasting accuracy "
                      "(mean over origins).", "tab:forecast_oos",
                      colfmt="llllrrrrrrrl")
    open(os.path.join(TAB, "table_forecasting.tex"), "w").write(tex)
    print("wrote table_forecasting.{csv,tex}  rows:", len(df))


def build_architecture():
    rows = load_jsonl("architecture")
    if not rows:
        print("architecture: no records"); return
    def admis(r):
        if "stationary" not in r and "invertible" not in r:
            return "--"
        return ("stat=" + ("Y" if r.get("stationary") else "N") +
                ",inv=" + ("Y" if r.get("invertible") else "N"))

    recs = [{
        "Experiment": r.get("experiment", r.get("dataset")),
        "Setting": r.get("setting", "--"),
        "Order": r.get("order", "--"),
        "CoefNorm": fmt(r.get("coef_norm")),
        "RSS": fmt(r.get("rss"), 2), "ResidMAE": fmt(r.get("resid_mae")),
        "IC(aicc)": fmt(r.get("ic"), 2), "Admissible": admis(r),
        "Detail": r.get("detail", "--"),
        "runtime_s": fmt(r.get("runtime_s"), 3), "status": r.get("status"),
    } for r in rows]
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(TAB, "table_architecture_checks.csv"), index=False)
    tex = latex_table(df, "Architecture and extensibility checks under alternative "
                          "estimation settings.", "tab:architecture_extensibility",
                      colfmt="lllrrrrllrl")
    open(os.path.join(TAB, "table_architecture_checks.tex"), "w").write(tex)
    print("wrote table_architecture_checks.{csv,tex}  rows:", len(df))


def build_solver():
    rows = [r for r in load_jsonl("solver")
            if r.get("experiment") not in ("scip_scaling", "global_value")]
    if not rows:
        print("solver: no records"); return
    def detail(r):
        if r.get("n_distinct_optima") is not None:
            return f"{r.get('n_converged','?')} conv, {r['n_distinct_optima']} optima"
        if r.get("rel_gap") is not None:
            return f"gap={fmt(r['rel_gap'])}"
        if "warning_emitted" in r:
            return "warning emitted" if r["warning_emitted"] else "no warning"
        return "--"

    recs = [{
        "Level": r.get("implementation", "--"),
        "Model": r["dataset"], "Solver": r.get("solver"),
        "Obj": r.get("objective", "--"),
        "Setting": r.get("setting", "--"),
        "ObjValue": fmt(r.get("obj_value")),
        "ObjSpread": fmt(r.get("obj_spread")),
        "Detail": detail(r),
        "Termination": r.get("termination", "--"),
        "runtime_s": fmt(r.get("runtime_s"), 2), "status": r.get("status"),
    } for r in rows]
    df = pd.DataFrame(recs).sort_values(["Level", "Model", "Solver", "Obj"])
    df.to_csv(os.path.join(TAB, "table_solver_comparison.csv"), index=False)
    tex = latex_table(df, "Solver comparison and nonconvexity diagnostics on small "
                      "MA-containing instances.", "tab:solver_comparison",
                      colfmt="lllllrrllrl")
    open(os.path.join(TAB, "table_solver_comparison.tex"), "w").write(tex)
    print("wrote table_solver_comparison.{csv,tex}  rows:", len(df))


def build_scaling():
    rows = [r for r in load_jsonl("solver") if r.get("experiment") == "scip_scaling"]
    if not rows:
        print("scaling: no records"); return
    rows.sort(key=lambda r: r["n"])
    def gapstr(g):
        if g is None:
            return "n/a"
        if g == 0.0:
            return "0"
        return f"{g:.1e}"
    recs = [{
        "T": r["n"],
        "Termination": r.get("termination", "--"),
        "Certified": "yes" if r.get("certified") else "no",
        "RelGap": gapstr(r.get("rel_gap")),
        "Time (s)": fmt(r.get("runtime_s"), 1),
        "Obj": fmt(r.get("obj_value")),
        "BruteForce": ("agree" if r.get("brute_force_agrees") else "--"),
    } for r in rows]
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(TAB, "table_solver_scaling.csv"), index=False)
    tex = latex_table(df, "Scalability of exact open-source global certification (direct SCIP) "
                      "for the MA(1) least-squares problem as the sample size grows.",
                      "tab:solver_scaling", colfmt="rllrrrl")
    open(os.path.join(TAB, "table_solver_scaling.tex"), "w").write(tex)
    print("wrote table_solver_scaling.{csv,tex}  rows:", len(df))


def build_global_value():
    rows = [r for r in load_jsonl("solver") if r.get("experiment") == "global_value"]
    if not rows:
        print("global_value: no records"); return
    rows.sort(key=lambda r: (r["dataset"], r["solver"]))
    recs = [{
        "Instance": r["dataset"],
        "Solver": r["solver"],
        "Certified": "yes" if r.get("certified") else "no",
        "Objective": fmt(r.get("obj_value")),
        "phi": fmt(r.get("phi")) if r.get("phi") is not None else "--",
        "theta": fmt(r.get("theta")),
        "RMSE_oos": fmt(r.get("rmse_oos")),
        "MAE_oos": fmt(r.get("mae_oos")),
        "Time (s)": fmt(r.get("runtime_s"), 1),
    } for r in rows]
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(TAB, "table_global_value.csv"), index=False)
    tex = latex_table(df, "Local (Ipopt) versus certified global (SCIP) estimation of the same "
                      "specification: in-sample objective, coefficients, and out-of-sample "
                      "forecast errors over a held-out window.",
                      "tab:global_value", colfmt="llllllrrr")
    open(os.path.join(TAB, "table_global_value.tex"), "w").write(tex)
    print("wrote table_global_value.{csv,tex}  rows:", len(df))


def build_energy():
    rows = [r for r in load_jsonl("energy") if r.get("block") == "energy_forecasting"]
    if not rows:
        print("energy: no records"); return
    recs = [{
        "Dataset": r["dataset"], "Model": r.get("order", "--"),
        "Implementation": r["implementation"],
        "RMSE": fmt(r.get("rmse"), 2), "MAE": fmt(r.get("mae"), 2),
        "MASE": fmt(r.get("mase")), "s/fit": perfit(r),
    } for r in rows]
    df = pd.DataFrame(recs).sort_values(["Dataset", "Implementation"])
    df.to_csv(os.path.join(TAB, "table_energy_forecasting.csv"), index=False)
    tex = latex_table(df, "Rolling-origin forecasting on the PJME daily electricity-demand series "
                      "(external validation; mean over origins).", "tab:energy_forecasting",
                      colfmt="lllrrrr")
    open(os.path.join(TAB, "table_energy_forecasting.tex"), "w").write(tex)
    print("wrote table_energy_forecasting.{csv,tex}  rows:", len(df))


BUILDERS = {"validation": build_validation, "forecasting": build_forecasting,
            "architecture": build_architecture, "solver": build_solver,
            "scaling": build_scaling, "global_value": build_global_value,
            "energy": build_energy}

if __name__ == "__main__":
    blocks = sys.argv[1:] or list(BUILDERS)
    for b in blocks:
        BUILDERS[b]()
