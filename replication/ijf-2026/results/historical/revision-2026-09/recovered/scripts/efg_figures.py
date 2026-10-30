"""Figures and tables for Parts E, F and G, from the CSVs written by efg_julia_experiments.jl,
plus the formalization of the existing multistart campaign (campaign C).

Nothing is fitted here. Every number comes from ../data/*.csv or the replication package.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

TAUS = (0.1, 0.5, 0.9)


# ---- E. quantile ----------------------------------------------------------------------------
def part_e():
    import matplotlib.pyplot as plt
    C.setup_matplotlib()
    P = pd.read_csv(C.DATA / "quantile_example_paths.csv")
    S = pd.read_csv(C.DATA / "quantile_example_summary.csv")
    last_train = P.loc[P["split"] == "train", "t"].max()
    window = 72                                              # training points shown
    fig, ax = plt.subplots(figsize=(6.9, 2.8))
    obs = P[P["tau"] == TAUS[0]].sort_values("t")
    obs = obs[obs["t"] > last_train - window]
    ax.plot(obs["t"], obs["y"], color=C.INK2, lw=0.7, marker="o", ms=1.6, label="observed")
    for i, tau in enumerate(TAUS):
        d = P[(P["tau"] == tau) & (P["t"] > last_train - window)].sort_values("t")
        tr, te = d[d["split"] == "train"], d[d["split"] == "test"]
        ax.plot(tr["t"], tr["path"], lw=1.1, color=C.PALETTE[i], ls=C.LINESTYLES[i],
                label=rf"$\tau={tau}$")
        ax.plot(te["t"], te["path"], lw=1.4, color=C.PALETTE[i], ls=C.LINESTYLES[i])
    ax.axvline(last_train + 0.5, color=C.INK, lw=0.6, ls=":")
    ax.text(last_train + 1.2, ax.get_ylim()[1], "forecast", va="top", fontsize=7, color=C.INK2)
    ax.text(last_train - 0.5, ax.get_ylim()[1], "in-sample fit", va="top", ha="right",
            fontsize=7, color=C.INK2)
    ax.set_xlabel("t"); ax.set_ylabel("$y_t$")
    ax.legend(frameon=False, ncol=4, loc="lower left", fontsize=7)
    C.save(fig, "quantile_controlled_example")
    plt.close(fig)

    # table: coverage and holdout pinball matrix (diagonal should be the column minimum)
    pin_cols = [f"holdout_pinball_at_{t}" for t in TAUS]
    colmin = {c: S[c].min() for c in pin_cols}
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Empirical coverage} & \multicolumn{3}{c}{Holdout pinball loss, evaluated at} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Fitted $\tau$ & in-sample & $h=1$ & $h=1..24$ & $0.1$ & $0.5$ & $0.9$ \\", r"\midrule"]
    for _, r in S.iterrows():
        cells = [f"{r['tau_fit']:.1f}", f"{r['coverage_in_sample']:.3f}",
                 f"{r['coverage_holdout_h1']:.3f}", f"{r['coverage_holdout_h1to24']:.3f}"]
        for c in pin_cols:
            v = f"{r[c]:.4f}"
            cells.append(rf"\textbf{{{v}}}" if np.isclose(r[c], colmin[c]) else v)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(S, "quantile_controlled_example", "\n".join(lines) + "\n")


# ---- F. weighted penalty -------------------------------------------------------------------
def part_f():
    import matplotlib.pyplot as plt
    C.setup_matplotlib()
    P = pd.read_csv(C.DATA / "weighted_penalty_path.csv")
    T = pd.read_csv(C.DATA / "weighted_penalty_table.csv")
    meta = (C.DATA / "weighted_penalty_meta.txt").read_text()
    Lstar = float(meta.split("=")[1].split()[0])
    names = [("scalar", r"(a) common $\lambda_j=L$"),
             ("adaptive", r"(b) adaptive $\lambda_j=L/|\tilde\beta_j|$"),
             ("known_active_unpenalized", r"(c) $\lambda_j=0$ on $x_1$-$x_3$, $L$ on $x_4$-$x_6$")]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5), sharey=True)
    for ax, (s, title) in zip(axes, names):
        d = P[P["scheme"] == s]
        for j in range(1, 7):
            dj = d[d["j"] == j].sort_values("L")
            if j <= 3:
                ax.plot(dj["L"], dj["beta_hat"], color=C.PALETTE[j - 1], ls=C.LINESTYLES[j - 1],
                        lw=1.2, label=f"$x_{j}$ ($\\beta={dj['beta_true'].iat[0]:g}$)")
                ax.axhline(dj["beta_true"].iat[0], color=C.PALETTE[j - 1], lw=0.5, ls=":")
            else:
                ax.plot(dj["L"], dj["beta_hat"], color="#9a9893", lw=0.9,
                        label="$x_4$-$x_6$ ($\\beta=0$)" if j == 4 else None)
        ax.axvline(Lstar, color=C.INK, lw=0.6, ls="--")
        ax.set_xscale("log"); ax.set_xlabel("$L$")
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.axhline(0, color=C.GRID, lw=0.5, zorder=0)
    axes[0].set_ylabel(r"$\hat\beta_j$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.07))
    fig.subplots_adjust(wspace=0.08, bottom=0.2)
    C.save(fig, "weighted_regularization_demo")
    plt.close(fig)

    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{common $\lambda$} & \multicolumn{2}{c}{adaptive $\lambda_j$} & \multicolumn{2}{c}{$\lambda_j=0$ on active} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"Coef. & $\beta$ & $\tilde\beta$ (no penalty) & $\lambda_j$ & $\hat\beta_j$ & $\lambda_j$ & $\hat\beta_j$ & $\lambda_j$ & $\hat\beta_j$ \\",
             r"\midrule"]
    zf = lambda v: "0" if abs(v) < 1e-6 else f"{v:.3f}"
    for _, r in T.iterrows():
        lines.append(" & ".join([
            f"${r['coef'][0]}_{r['coef'][1:]}$", f"{r['beta_true']:g}", f"{r['beta_unpenalized']:.3f}",
            f"{r['lambda_scalar']:.1f}", zf(r["beta_scalar"]),
            f"{r['lambda_adaptive']:.1f}", zf(r["beta_adaptive"]),
            f"{r['lambda_known_active_unpenalized']:.1f}", zf(r["beta_known_active_unpenalized"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(T, "weighted_regularization_demo", "\n".join(lines) + "\n")


# ---- G. multistart ---------------------------------------------------------------------------
REL_TOL = 1e-6


def cluster(vals, rel_tol=REL_TOL):
    """Group objective values that agree within rel_tol of the best one, greedily from the
    lowest. Returns cluster labels in increasing order of objective."""
    order = np.argsort(vals)
    lab = np.empty(len(vals), int)
    c, anchor = -1, None
    for i in order:
        if anchor is None or (vals[i] - anchor) > rel_tol * max(abs(anchor), 1e-12):
            c += 1; anchor = vals[i]
        lab[i] = c
    return lab


def part_g():
    K = pd.read_csv(C.DATA / "multistart_kstart.csv")
    K["coef"] = K["coefficients"].astype(str).apply(
        lambda s: np.array([float(x) for x in s.split(";") if x not in ("", "nan")]))
    rows = []
    for (case, init), d in K.groupby(["case", "initialization"], sort=False):
        ok = d["solver_status"].isin(["LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED"])
        v = d["objective"].to_numpy(float)
        lab = cluster(v)
        best = v.min()
        at_best = lab == 0
        coefs = np.vstack(d["coef"].to_numpy())
        spread_best = float(np.max(np.ptp(coefs[at_best], axis=0))) if at_best.sum() > 1 else 0.0
        zero_row = d["start_kind"].eq("zero").to_numpy()
        zero_gap = float((v[zero_row][0] - best) / abs(best)) if zero_row.any() else np.nan
        rows.append({
            "case": case, "initialization": init, "starts": len(d),
            "converged": int(ok.sum()),
            "distinct_objective_values": int(lab.max() + 1),
            "share_at_best": float(at_best.mean()),
            "rel_range_objective": float((v.max() - best) / abs(best)),
            "zero_start_rel_gap": zero_gap,
            "max_coef_spread_within_best": spread_best,
            "median_iterations": float(pd.to_numeric(d["iterations"], errors="coerce").median()),
        })
    G = pd.DataFrame(rows)
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Case & Init. & Conv. & Distinct & At best & Rel. range & Zero-start gap & Coef. spread \\",
             r"\midrule"]
    last = None
    for _, r in G.iterrows():
        if last is not None and r["case"] != last:
            lines.append(r"\addlinespace")
        sci = lambda x: "0" if x == 0 else f"{x:.1e}"
        lines.append(" & ".join([
            r["case"] if r["case"] != last else "", r"\texttt{:" + r["initialization"] + "}",
            f"{r['converged']}/{r['starts']}", str(r["distinct_objective_values"]),
            f"{100*r['share_at_best']:.0f}\\%", sci(r["rel_range_objective"]),
            sci(r["zero_start_rel_gap"]), sci(r["max_coef_spread_within_best"])]) + r" \\")
        last = r["case"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(G, "multistart_kstart", "\n".join(lines) + "\n")

    # existing campaign C, formalized: deterministic {zero, CSS} multistart vs zero start
    M = pd.read_csv(C.RAW / "multistart_random.csv", comment="#")
    M = M[M["status"].astype(str).str.startswith("OK")]
    b = M[M["cell"] == "base"].set_index("series"); m = M[M["cell"] == "multi"].set_index("series")
    ids = sorted(set(b.index) & set(m.index))
    rec = {"series": len(ids)}
    for key, lab in (("smape", "sMAPE"), ("mase", "MASE")):
        x = m.loc[ids, key].to_numpy() - b.loc[ids, key].to_numpy()
        rec.update({f"{lab}_mean_diff": x.mean(), f"{lab}_median_diff": np.median(x),
                    f"{lab}_q05": np.quantile(x, 0.05), f"{lab}_q95": np.quantile(x, 0.95),
                    f"{lab}_pct_lower": 100 * np.mean(x < -C.TIE_TOL),
                    f"{lab}_pct_unchanged": 100 * np.mean(np.abs(x) <= C.TIE_TOL),
                    f"{lab}_pct_higher": 100 * np.mean(x > C.TIE_TOL)})
    rec["pct_multistart_beat_zero_flag"] = 100 * m.loc[ids, "multistart_beat_zero"].mean()
    rec["pct_same_order"] = 100 * np.mean(
        (b.loc[ids, ["p", "d", "q", "P", "D", "Q"]].to_numpy() ==
         m.loc[ids, ["p", "d", "q", "P", "D", "Q"]].to_numpy()).all(axis=1))
    rec["censored_base_ge120s"] = int((b.loc[ids, "seconds"] >= 120).sum())
    rec["censored_multi_ge120s"] = int((m.loc[ids, "seconds"] >= 120).sum())
    Cdf = pd.DataFrame([rec])
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r" & sMAPE & MASE \\", r"\midrule",
             f"Series & \\multicolumn{{2}}{{c}}{{{rec['series']}}} \\\\"]
    for k, lab in (("mean_diff", "Mean $\\Delta$"), ("median_diff", "Median $\\Delta$"),
                   ("q05", "$q_{05}$"), ("q95", "$q_{95}$"), ("pct_lower", r"\% lower"),
                   ("pct_unchanged", r"\% unchanged"), ("pct_higher", r"\% higher")):
        lines.append(f"{lab} & {rec['sMAPE_' + k]:.4f} & {rec['MASE_' + k]:.4f} \\\\")
    lines += [f"Same selected order & \\multicolumn{{2}}{{c}}{{{rec['pct_same_order']:.1f}\\%}} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    C.write_table(Cdf, "multistart_campaign_c", "\n".join(lines) + "\n")
    pd.set_option("display.width", 220)
    print(G.round(8).to_string(index=False))
    print(Cdf.round(4).T.to_string())


CAPTIONS = r"""% ---- Figure: quantile controlled example ------------------------------------------------------
\caption{Functional demonstration of the quantile (pinball) estimation criterion. An AR(1)
series with right-skewed innovations ($e_t=\mathrm{Exp}(1)-1$) is fitted with the same
SARIMAX specification (AR(1) with intercept) at $\tau=0.1$, $0.5$ and $0.9$; the last 72 of
216 training points and the 24-step forecast are shown. The fitted conditional paths are
ordered as the levels are. This illustrates the behaviour of an asymmetric estimation
criterion; it is not a calibrated probabilistic forecast, and the multi-step paths are point
forecasts from the fitted recursion, not predictive quantiles.}
\label{fig:quantile-example}

% ---- Table: quantile controlled example ------------------------------------------------------
\caption{Quantile criterion over 200 replications of the design in
Figure~\ref{fig:quantile-example}. Coverage is the share of observations at or below the fitted
path: in-sample (one-step), at the first holdout step, and over the 24-step holdout. The
pinball loss $\rho_\tau(e)=\tau\max(e,0)+(1-\tau)\max(-e,0)$ is averaged over the holdout and
the replications, and evaluated at each of the three levels; the bold entry is the smallest in
its column. Coverage at $h=1$ targets $\tau$; coverage over $h=1..24$ is not expected to, since
the multi-step path is not a predictive quantile.}
\label{tab:quantile-example}

% ---- Figure/table: weighted regularization ---------------------------------------------------
\caption{Coefficient-specific penalty weights on a regression with AR(1) errors
(\texttt{exogDynamics = :regression\_errors}), $n=200$, six standardized regressors of which
$x_1$--$x_3$ are active ($\beta=2,-1.5,1$) and $x_4$--$x_6$ are null. Lasso penalty
($\alpha=1$) on the exogenous block only, \texttt{initialization = :zeroed}. (a) one common
strength $L$; (b) adaptive weights $\lambda_j=L/|\tilde\beta_j|$ from the unpenalized fit, supplied
by the user; (c) zero weight on $x_1$--$x_3$ and $L$ on $x_4$--$x_6$. Dotted lines: true values.
The dashed line marks $L^\ast$, the smallest grid value at which the common-strength lasso sets
all three null coefficients to zero, chosen from panel (a) alone; the table reports the three
fits at $L^\ast$. This demonstrates the interface; it is not an accuracy comparison.}
\label{fig:weighted-regularization}

% ---- Table: multistart K-start diagnostic ----------------------------------------------------
\caption{Sensitivity of the local solver (Ipopt) to the starting point: for each case, the zero
start and 29 random starts drawn in reflection coordinates (so every start is stationary and
invertible), objective \texttt{mse}, under two initialization conventions. ``Distinct'' is the
number of final objective values that differ by more than a relative $10^{-6}$; ``At best'' the
share of starts reaching the lowest; ``Rel. range'' the spread of final objective values
relative to the lowest; ``Zero-start gap'' how far the default zero start ends above the lowest;
``Coef. spread'' the largest coordinate range of the coefficients among the starts that reach
the lowest objective (non-zero indicates a flat or non-identified optimum). This characterises
local sensitivity only; it does not certify global optimality, which the global-solver results
address.}
\label{tab:multistart-kstart}
"""


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["E", "F", "G"]
    if "E" in which: part_e()
    if "F" in which: part_f()
    if "G" in which: part_g()
    (C.TAB / "efg_captions.tex").write_text(CAPTIONS, encoding="utf-8")
