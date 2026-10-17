"""Per-series distributions behind Table 7 (estimation objectives under `:innovations`): the
October 2026 evidence pipeline, `original-scripts/m4-objective-distributions-2026-10`.

Rebuilds, from the per-series datasets that the pipeline's step 2 wrote and its steps 3-5 read
(`derived/m4-objective-distributions-2026-10/perseries_{archived,latest}.csv.gz`):

* `distributions.reproduction` (s3_reproduce.py): N, mean sMAPE, mean MASE and OWA per
  frequency x method for both candidate sets, against the published Table 7 values, which s3
  parsed from ForecastTester `tabelas_paper.txt` (section B, total horizon; match = same N and
  every difference within 1e-4);
* `distributions.4a_marginal`, `4b_paired`, `4c_tail`, `4c_tail_ratios`, `4d_union_tail`
  (s4_compile.py), for every set whose totals all matched (as in the original run: `archived`;
  `latest` did not reproduce and is not compiled);
* the four figures of s5_figures.py for each compiled set (ECDF of the per-series difference
  to MSE, survival functions from the median; MASE and sMAPE), as PDF and PNG. Figures are
  produced, not compared.

`distributions.perseries_vs_raw` checks step 2 itself: every per-series row against the raw
campaign row it was built from (series, status, orders, stored metrics, MASE denominator,
commit, file) and against the stored auto.arima/Naive2 values, within 1e-6; the largest
|recomputed - stored| sMAPE and MASE (the `recompute_check` quantities); and, when the
official M4 test files are present, sMAPE and MASE recomputed from the stored forecast vectors
as s2 did. These cells have no historical reference.

Every definition is the original one (t7lib.py, reproduced below): numpy means, quantiles by
`np.quantile(x, q, method="linear")` (Hyndman-Fan type 7), the trimmed mean, expected
shortfall, worst sets and shares of t7lib, and per frequency one series set for everything:
auto.arima ids n Naive2 ids n series with status OK in all four objective cells.

HISTORICAL: the archived cells ran on feature-branch commits (87f7bfb, 6a11698, 5b2ec6b), with
mixed commits inside some cells; `latest` replaces huber and ridge with the v1.0.0 re-run.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import re

import numpy as np

from .common import Cell, Context, open_text, read_csv

TITLE = "M4 objective distributions, Oct 2026 (Table 7 evidence): reproduction, tables 4a-4d, figures"
DERIVED = "derived/m4-objective-distributions-2026-10"
PUBLISHED = "historical-tables/forecasttester-2026-08/tabelas_paper.txt"
PUBLISHED_SECTION = "B) CONFIGURACAO DE PRODUCAO"
RAW = "raw/m4-objectives-icarai-2026-08"
RERUN = "raw/m4-objectives-rerun-a1ebc0d-2026-08"

# ---- config.py of the original pipeline -----------------------------------------------------
FREQS = ["monthly", "quarterly", "yearly", "weekly"]
HORIZON = {"monthly": 18, "quarterly": 8, "yearly": 6, "weekly": 13}
OBJECTIVES = ["mse", "huber", "mae", "ridge"]
ALTERNATIVES = ["huber", "mae", "ridge"]
BASELINE = "mse"
SETS = ["archived", "latest"]
TIE_MAIN = 1e-4
TIE_EXTRA = [1e-6, 1e-2]
QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
TOL = 1e-4                                       # s3_reproduce.py
METRICS = ["mase", "smape"]                      # s4_compile.py
METHODS = OBJECTIVES + ["auto.arima", "naive2"]
ORDER = ("p", "d", "q", "P", "D", "Q")

RAW_TOL = 1e-6        # agreement of a per-series value with the raw file it came from
DECIMALS = 6          # float cells: the historical CSVs print full float precision (repr)
TABLE_KEYS = {
    "reproduction": ("set", "frequency", "method"),
    "4a_marginal": ("set", "frequency", "metric", "method"),
    "4b_paired": ("set", "frequency", "metric", "objective", "reference"),
    "4c_tail": ("set", "frequency", "metric", "method"),
    "4c_tail_ratios": ("set", "frequency", "metric", "method", "reference"),
    "4d_union_tail": ("set", "frequency", "metric", "method"),
    "perseries_vs_raw": ("set", "frequency", "objective"),
}
REPRODUCTION_COLS = ["set", "frequency", "method", "N", "N_published", "smape", "smape_published",
                     "smape_diff", "mase", "mase_published", "mase_diff", "owa", "owa_published",
                     "owa_diff", "matches"]


def source_name(objective: str, freq: str, which: str) -> str:
    """The cell file step 2 read (config.cell_file), as recorded in the `source_file` column."""
    if which == "latest" and objective in ("huber", "ridge"):
        return f"obj_{objective}_innov_{freq}__v1.0.0.csv"
    return f"obj_{objective}_innov_{freq}.csv"


def raw_file(objective: str, freq: str, which: str) -> str:
    """The bundle copy of that cell file. obj_mse_innov_<freq>.csv of monthly, quarterly and
    yearly was a byte copy of req_true_<freq>.csv (s1 inventory); the bundle keeps the latter."""
    if which == "latest" and objective in ("huber", "ridge"):
        return f"{RERUN}/obj_{objective}_innov_{freq}__v1.0.0.csv.gz"
    if objective == "mse" and freq != "weekly":
        return f"{RAW}/campaigns/req_true_{freq}.csv.gz"
    return f"{RAW}/campaigns/obj_{objective}_innov_{freq}.csv.gz"


def baseline_file(freq: str, method: str) -> str:
    return f"{RAW}/baselines/{freq}_{method}_total.csv.gz"


INPUTS = ([f"{DERIVED}/perseries_{w}.csv.gz" for w in SETS] + [PUBLISHED]
          + sorted({raw_file(o, f, w) for w in SETS for f in FREQS for o in OBJECTIVES})
          + [baseline_file(f, m) for f in FREQS for m in ("autoarima_R", "Naive")])


# ---- t7lib.py, unchanged ---------------------------------------------------------------------
def strip(v):
    return v.strip().strip('"') if isinstance(v, str) else v


def num(v):
    try:
        x = float(v)
        return x
    except (TypeError, ValueError):
        return math.nan


def smape_series(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float)
    return float(np.mean(200.0 * np.abs(f - a) / (np.abs(f) + np.abs(a) + 1e-12)))


def mase_series(f, a, den):
    f, a = np.asarray(f, float), np.asarray(a, float)
    den = den if den > 0 else 1e-9
    return float(np.mean(np.abs(f - a)) / den)


def owa(smape, mase, smape_n2, mase_n2):
    return 0.5 * np.mean(smape) / np.mean(smape_n2) + 0.5 * np.mean(mase) / np.mean(mase_n2)


def quantiles(x, qs=None):
    qs = QUANTILES if qs is None else qs
    return np.quantile(np.asarray(x, float), qs, method="linear")


def trimmed_mean(x, prop=0.01):
    """Mean after removing floor(prop*n) observations from each end (scipy.stats.trim_mean)."""
    x = np.sort(np.asarray(x, float))
    k = int(math.floor(prop * len(x)))
    return float(np.mean(x[k:len(x) - k])) if len(x) - 2 * k > 0 else float("nan")


def expected_shortfall(x, a):
    """Mean of the values strictly above the (1 - a) quantile."""
    x = np.asarray(x, float)
    q = np.quantile(x, 1 - a, method="linear")
    tail = x[x > q]
    return float(np.mean(tail)) if len(tail) else float(q)


def worst_set(ids, x, frac=0.05):
    """The ceil(frac*n) series with the largest value; ties broken by series id."""
    k = int(math.ceil(frac * len(ids)))
    order = sorted(range(len(ids)), key=lambda i: (-x[i], ids[i]))
    return {ids[i] for i in order[:k]}


def share_top(absd, frac):
    """Fraction of sum|d| carried by the ceil(frac*n) largest |d|."""
    absd = np.sort(np.asarray(absd, float))[::-1]
    tot = absd.sum()
    k = int(math.ceil(frac * len(absd)))
    return float(absd[:k].sum() / tot) if tot > 0 else float("nan")


# ---- s4_compile.py helpers, unchanged --------------------------------------------------------
def pct(mask):
    return 100.0 * float(np.mean(mask))


def paired(d, tie):
    return pct(d < -tie), pct(np.abs(d) < tie), pct(d > tie)


# ---- step 3: the published values -------------------------------------------------------------
def parse_published(ctx: Context) -> dict:
    """s3_reproduce.parse_published: {(frequency, method): {N, mase, smape, owa}} from the
    total-horizon rows of section B of tabelas_paper.txt."""
    txt = ctx.path(PUBLISHED).read_text(encoding="utf-8", errors="replace").splitlines()
    out, freq, n, cell = {}, None, None, None
    in_section = False
    for line in txt:
        m = re.match(r"^(\S\))\s.*\((\w+), n=(\d+)\)\s*$", line)
        if m:
            in_section = line.startswith(PUBLISHED_SECTION)
            freq, n, cell = m.group(2), int(m.group(3)), None
            continue
        if not in_section or not line.strip() or line.startswith("celula") or line.startswith("---"):
            continue
        parts = line.split()
        if parts[0] in ("short", "medium", "long", "total"):
            block, vals = parts[0], parts[1:]
        else:
            cell, block, vals = parts[0], parts[1], parts[2:]
        if block != "total":
            continue
        name = {"auto.arima": "auto.arima", "Naive": "naive2"}.get(cell, cell)
        out[(freq, name)] = {"N": n, "mase": float(vals[0]), "smape": float(vals[1]),
                             "owa": float(vals[2])}
    return out


# ---- the per-series datasets ------------------------------------------------------------------
class FreqData:
    """One frequency of a per-series dataset, reduced to the fields steps 3 and 4 read."""

    __slots__ = ("ok", "ref", "base")

    def __init__(self):
        self.ok: dict[str, dict] = {}   # objective -> {series: (smape, mase, "p,d,q,P,D,Q")}, status OK
        self.ref: set[int] = set()      # series with auto.arima and Naive2 values
        self.base: dict[int, tuple] = {}  # series -> auto.arima/Naive2 text of its OK mse row


def load_set(ctx: Context, which: str, check: "RawCheck") -> dict[str, FreqData]:
    """One pass over derived/perseries_<which>.csv.gz. As in s3/s4, a later OK row of the same
    series and objective replaces an earlier one. Every row is also handed to `check`."""
    data: dict[str, FreqData] = {}
    with open_text(ctx.path(f"{DERIVED}/perseries_{which}.csv.gz")) as fh:
        rd = csv.reader(fh)
        col = {k: i for i, k in enumerate(next(rd))}
        check.columns(col)
        i_f, i_s, i_o, i_st = col["frequency"], col["series"], col["objective"], col["status"]
        i_sm, i_ma = col["smape"], col["mase"]
        i_ord = [col[c] for c in ORDER]
        i_ref = [col[c] for c in ("autoarima_smape", "autoarima_mase", "naive2_smape", "naive2_mase")]
        for r in rd:
            if not r:
                continue
            check.row(r)
            fd = data.get(r[i_f])
            if fd is None:
                fd = data[r[i_f]] = FreqData()
            sid = int(r[i_s])
            if r[i_st] == "OK":
                # orders are compared as text, as s4 compared tuples of the text fields
                fd.ok.setdefault(r[i_o], {})[sid] = (float(r[i_sm]), float(r[i_ma]),
                                                     ",".join(r[i] for i in i_ord))
                if r[i_o] == BASELINE:
                    fd.base[sid] = tuple(r[i] for i in i_ref)
            if r[i_ref[0]] not in ("", "nan") and r[i_ref[2]] not in ("", "nan"):
                fd.ref.add(sid)
    check.finish()
    return data


def matrix(fd: FreqData):
    """s4_compile.matrix (and the arrays of s3_reproduce): {method: {metric: array}},
    {objective: orders}, ids, on the generator's series set, in series order."""
    ids = set(fd.ref)
    for o in OBJECTIVES:
        ids &= set(fd.ok.get(o, {}))
    ids = sorted(ids)
    M, orders = {}, {}
    for o in OBJECTIVES:
        cell = fd.ok[o]
        M[o] = {"mase": np.array([cell[i][1] for i in ids]),
                "smape": np.array([cell[i][0] for i in ids])}
        orders[o] = [cell[i][2] for i in ids]
    b = fd.base
    M["auto.arima"] = {"mase": np.array([float(b[i][1]) for i in ids]),
                       "smape": np.array([float(b[i][0]) for i in ids])}
    M["naive2"] = {"mase": np.array([float(b[i][3]) for i in ids]),
                   "smape": np.array([float(b[i][2]) for i in ids])}
    return M, orders, ids


# ---- step 3: reproduction of the published totals --------------------------------------------
def reproduction(which: str, mats: dict, pub: dict) -> tuple[list[dict], bool]:
    """s3_reproduce.main for one set: its rows of reproduction.csv and whether all match."""
    rows, all_ok = [], True
    for freq in FREQS:
        M, _, ids = mats[freq]
        n2s, n2m = M["naive2"]["smape"], M["naive2"]["mase"]
        for name in METHODS:
            s, m = M[name]["smape"], M[name]["mase"]
            o = owa(s, m, n2s, n2m)
            p = pub.get((freq, name))
            rec = {"set": which, "frequency": freq, "method": name, "N": len(ids),
                   "N_published": p["N"] if p else "not found",
                   "smape": float(np.mean(s)), "smape_published": p["smape"] if p else "not found",
                   "mase": float(np.mean(m)), "mase_published": p["mase"] if p else "not found",
                   "owa": float(o), "owa_published": p["owa"] if p else "not found"}
            if p:
                rec["smape_diff"] = rec["smape"] - p["smape"]
                rec["mase_diff"] = rec["mase"] - p["mase"]
                rec["owa_diff"] = rec["owa"] - p["owa"]
                match = (rec["N"] == p["N"] and abs(rec["smape_diff"]) <= TOL
                         and abs(rec["mase_diff"]) <= TOL and abs(rec["owa_diff"]) <= TOL)
            else:
                rec["smape_diff"] = rec["mase_diff"] = rec["owa_diff"] = "n/a"
                match = False
            rec["matches"] = "yes" if match else "no"
            all_ok &= match
            rows.append({c: rec[c] for c in REPRODUCTION_COLS})
    return rows, all_ok


# ---- step 4: distributional compilation -------------------------------------------------------
def compile_set(which: str, mats: dict, t: dict[str, list]) -> None:
    """s4_compile.main for one set: appends its rows to tables 4a, 4b, 4c, 4c_ratios, 4d."""
    for freq in FREQS:
        M, orders, ids = mats[freq]
        n = len(ids)
        for met in METRICS:
            # ---- 4a marginal distributions
            for m in METHODS:
                x = M[m][met]
                q = quantiles(x)
                t["4a_marginal"].append({"set": which, "frequency": freq, "metric": met, "method": m,
                                         "N": n, "mean": float(np.mean(x)),
                                         **{f"q{int(round(p * 100)):02d}": float(v) for p, v in zip(QUANTILES, q)}})
            # ---- 4b paired differences
            pairs = [(o, ref) for o in ALTERNATIVES for ref in (BASELINE, "auto.arima")]
            pairs.append((BASELINE, "auto.arima"))
            for o, ref in pairs:
                d = M[o][met] - M[ref][met]
                q = quantiles(d)
                lo, ti, hi = paired(d, TIE_MAIN)
                same = (pct(np.array([a == b for a, b in zip(orders[o], orders[ref])]))
                        if ref in orders else "not recorded")
                rec = {"set": which, "frequency": freq, "metric": met, "objective": o,
                       "reference": ref, "N": n, "mean": float(np.mean(d)),
                       "trimmed_mean_1_99": trimmed_mean(d, 0.01),
                       "median": float(np.median(d)),
                       **{f"q{int(round(p * 100)):02d}": float(v) for p, v in zip(QUANTILES, q)},
                       "pct_lower": lo, "pct_tie_1e-4": ti, "pct_higher": hi}
                for tie in TIE_EXTRA:
                    rec[f"pct_tie_{tie:.0e}"] = pct(np.abs(d) < tie)
                rec["pct_same_order"] = same
                rec["share_sum_absdelta_top1pct"] = share_top(np.abs(d), 0.01)
                rec["share_sum_absdelta_top5pct"] = share_top(np.abs(d), 0.05)
                t["4b_paired"].append(rec)
            # ---- 4c tail of each method
            for m in METHODS:
                x = M[m][met]
                t["4c_tail"].append({"set": which, "frequency": freq, "metric": met, "method": m, "N": n,
                                     "q95": float(np.quantile(x, 0.95, method="linear")),
                                     "q99": float(np.quantile(x, 0.99, method="linear")),
                                     "ES5": expected_shortfall(x, 0.05),
                                     "ES1": expected_shortfall(x, 0.01)})
            base = M[BASELINE][met]
            q95b = np.quantile(base, 0.95, method="linear")
            for o in ALTERNATIVES + ["auto.arima"]:
                x = M[o][met]
                t["4c_tail_ratios"].append({"set": which, "frequency": freq, "metric": met, "method": o,
                                            "reference": BASELINE, "N": n,
                                            "P_gt_ref_q95_over_0.05": float(np.mean(x > q95b)) / 0.05,
                                            "pct_method_gt_2x_ref": pct(x > 2 * base),
                                            "pct_ref_gt_2x_method": pct(base > 2 * x)})
            # ---- 4d union tail
            wm = worst_set(ids, M[BASELINE][met], 0.05)
            wa = worst_set(ids, M["auto.arima"][met], 0.05)
            U = wm | wa
            inU = np.array([i in U for i in ids])
            for m in METHODS:
                x = M[m][met]
                rec = {"set": which, "frequency": freq, "metric": met, "method": m,
                       "U_size": len(U), "pct_U_in_both_worst_sets": 100.0 * len(wm & wa) / len(U),
                       "naive2_mean_in_U": float(np.mean(M["naive2"][met][inU])),
                       "mean_in_U": float(np.mean(x[inU])), "mean_outside_U": float(np.mean(x[~inU]))}
                if m != BASELINE and m != "naive2":
                    d = (x - base)[inU]
                    lo, _, hi = paired(d, TIE_MAIN)
                    rec.update({"median_delta_vs_mse_in_U": float(np.median(d)),
                                "mean_delta_vs_mse_in_U": float(np.mean(d)),
                                "pct_lower_vs_mse_in_U": lo, "pct_higher_vs_mse_in_U": hi})
                else:
                    rec.update({"median_delta_vs_mse_in_U": "n/a", "mean_delta_vs_mse_in_U": "n/a",
                                "pct_lower_vs_mse_in_U": "n/a", "pct_higher_vs_mse_in_U": "n/a"})
                t["4d_union_tail"].append(rec)


# ---- step 5: figures --------------------------------------------------------------------------
COLORS = {"mse": "#1f2937", "huber": "#2563eb", "mae": "#d97706", "ridge": "#059669",
          "auto.arima": "#9333ea"}
LABEL = {"mase": "MASE", "smape": "sMAPE"}


def figures(ctx: Context, which: str, mats: dict) -> list[str]:
    """s5_figures.main for one compiled set, unchanged except for the output folder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctx.figures.mkdir(parents=True, exist_ok=True)
    made = []
    with plt.style.context("default"):      # independent of rcParams set by other groups
        for met in METRICS:
            # ---- ECDF of delta vs MSE
            fig, axes = plt.subplots(1, len(FREQS), figsize=(4.2 * len(FREQS), 3.6), squeeze=False)
            for ax, f in zip(axes[0], FREQS):
                M = mats[f]
                for o in ALTERNATIVES:
                    d = np.sort(M[o][met] - M[BASELINE][met])
                    y = np.arange(1, len(d) + 1) / len(d)
                    ax.step(d, y, where="post", color=COLORS[o], lw=1.2, label=o)
                ax.set_xscale("symlog", linthresh=TIE_MAIN)
                # one labelled tick every two decades on each side, so the negative side
                # does not overprint
                lo, hi = ax.get_xlim()
                dec = [10.0 ** k for k in range(-4, 4, 2)]
                ticks = [-v for v in reversed(dec) if -v >= lo] + [0.0] + [v for v in dec if v <= hi]
                ax.set_xticks(ticks)
                ax.tick_params(axis="x", labelsize=7, labelrotation=45)
                ax.axvline(0, color="#9ca3af", lw=0.8)
                ax.set_title(f"{f} (N={len(M[BASELINE][met])})")
                ax.set_xlabel(f"{LABEL[met]}(objective) - {LABEL[met]}(MSE)")
                ax.grid(alpha=0.3)
            axes[0][0].set_ylabel("ECDF")
            axes[0][-1].legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                p = ctx.figures / f"{which}_ecdf_delta_vs_mse_{met}.{ext}"
                fig.savefig(p, dpi=150)
                made.append(p.name)
            plt.close(fig)

            # ---- survival from the median, log-log
            fig, axes = plt.subplots(1, len(FREQS), figsize=(4.2 * len(FREQS), 3.6), squeeze=False)
            for ax, f in zip(axes[0], FREQS):
                M = mats[f]
                for m in [BASELINE] + ALTERNATIVES + ["auto.arima"]:
                    x = np.sort(M[m][met])
                    n = len(x)
                    med = np.median(x)
                    t = x[x >= med]
                    t = t[t > 0]
                    # P(X > t) = 1 - #{x <= t} / n, read off the sorted sample
                    s = 1.0 - np.searchsorted(x, t, side="right") / n
                    keep = s > 0
                    ax.step(t[keep], s[keep], where="post", color=COLORS[m], lw=1.2, label=m)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"{f} (N={len(M[BASELINE][met])})")
                ax.set_xlabel(f"t ({LABEL[met]})")
                ax.grid(alpha=0.3, which="both")
            axes[0][0].set_ylabel(f"P({LABEL[met]} > t)")
            axes[0][-1].legend(loc="lower left", fontsize=8)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                p = ctx.figures / f"{which}_survival_{met}.{ext}"
                fig.savefig(p, dpi=150)
                made.append(p.name)
            plt.close(fig)
    return made


# ---- step 2, checked: per-series rows against the raw files -----------------------------------
def _open_cell(path):
    """A campaign file opened as t7lib.read_cell opens one (UTF-8 with replacement)."""
    return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace", newline="")


def _same(x: float, y: float) -> bool:
    return (math.isnan(x) and math.isnan(y)) or abs(x - y) <= RAW_TOL


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class RawCheck:
    """s2_perseries.py checked backwards. Each per-series row is compared with the raw row it
    was built from (s2 wrote the rows of each cell contiguously and in file order, so the two
    files are read in lockstep), raw rows being mapped by position onto the header as
    t7lib.read_cell does, and with the stored auto.arima/Naive2 values (NaN when absent)."""

    def __init__(self, ctx: Context, which: str):
        self.ctx, self.which = ctx, which
        self.col: dict[str, int] = {}          # per-series columns
        self.rc: dict[str, int] = {}           # raw columns of the current cell file
        self.commit_col = self.fcol = None
        self.key = self.raw = self.handle = self.g = None
        self.groups: dict[tuple[str, str], dict] = {}
        self.baselines: dict[str, dict] = {}
        self.tests: dict[str, dict] = {}

    def columns(self, col: dict[str, int]) -> None:
        self.col = col

    # -- raw cell files
    def _switch(self, key: tuple[str, str]) -> None:
        self._close()
        freq, objective = key
        rel = raw_file(objective, freq, self.which)
        self.handle = _open_cell(self.ctx.path(rel))
        rd = csv.reader(self.handle)
        header = next(rd)
        self.rc = {k: i for i, k in enumerate(header)}
        # s2: the commit column is `sarimax_commit`, else `commit`, else "not recorded"
        self.commit_col = self.rc.get("sarimax_commit", self.rc.get("commit"))
        self.fcol = self.rc["forecast"] if "forecast" in self.rc else self.rc["previsao"]
        width = len(header)

        def rows():
            for r in rd:
                if r:
                    yield [strip(v) for v in (r + [""] * (width - len(r)))[:width]]

        self.raw = rows()
        self.key = key
        self.g = self.groups.setdefault(key, {
            "rows": 0, "raw_rows": 0, "rows_agree": 0, "baseline_rows_agree": 0, "ok_rows": 0,
            "stored_metric_nan": 0, "max_abs_diff_smape": None, "max_abs_diff_mase": None,
            "m4_rows_agree": 0, "m4_max_abs_diff": None, "raw_file": rel})

    def _close(self) -> None:
        if self.raw is not None:
            self.g["raw_rows"] += sum(1 for _ in self.raw)     # raw rows left unmatched
            self.handle.close()
            self.raw = self.handle = None

    def _baseline(self, freq: str, method: str) -> dict:
        key = f"{freq}/{method}"
        if key not in self.baselines:      # s2 read_ref: {series: (MASE, sMAPE)}, by column name
            self.baselines[key] = {int(x["series"]): (num(x["MASE"]), num(x["sMAPE"]))
                                   for x in read_csv(self.ctx.path(baseline_file(freq, method)))}
        return self.baselines[key]

    def _test(self, freq: str) -> dict:
        if freq not in self.tests:         # s2 read_test: the row index is the series id
            out = {}
            with open(self.ctx.m4 / f"{freq.capitalize()}-test.csv", newline="", encoding="utf-8") as f:
                rd = csv.reader(f)
                next(rd)
                for i, r in enumerate(rd, 1):
                    out[i] = [float(v) for v in r[1:] if v not in ("", "NA")]
            self.tests[freq] = out
        return self.tests[freq]

    @staticmethod
    def _max(old, new):
        return new if old is None else max(old, new)

    # -- one per-series row
    def row(self, d: list[str]) -> None:
        c, which = self.col, self.which
        freq, objective = d[c["frequency"]], d[c["objective"]]
        if (freq, objective) != self.key:
            self._switch((freq, objective))
        g = self.g
        g["rows"] += 1
        status = d[c["status"]]
        ok = status == "OK"
        g["ok_rows"] += ok
        sid = _int(d[c["series"]])

        ar = self._baseline(freq, "autoarima_R").get(sid, (math.nan, math.nan))
        nv = self._baseline(freq, "Naive").get(sid, (math.nan, math.nan))
        g["baseline_rows_agree"] += (_same(num(d[c["autoarima_smape"]]), ar[1])
                                     and _same(num(d[c["autoarima_mase"]]), ar[0])
                                     and _same(num(d[c["naive2_smape"]]), nv[1])
                                     and _same(num(d[c["naive2_mase"]]), nv[0]))

        r = next(self.raw, None)
        if r is None:                       # a per-series row without a raw row
            return
        g["raw_rows"] += 1
        rc = self.rc
        smape_old, mase_old = num(r[rc["smape_total"]]), num(r[rc["mase_total"]])
        agree = (sid is not None and sid == _int(r[rc["series"]])
                 and status == r[rc["status"]].split(":")[0]
                 and all(d[c[k]] == r[rc[k]] for k in ORDER)
                 and _same(num(d[c["smape_stored"]]), smape_old)
                 and _same(num(d[c["mase_stored"]]), mase_old)
                 and _same(num(d[c["mase_den"]]), num(r[rc["mase_den"]]))
                 and d[c["sarimax_commit"]] == (r[self.commit_col] if self.commit_col is not None
                                                else "not recorded")
                 and d[c["source_file"]] == source_name(objective, freq, which)
                 and (d[c["smape"]] != "") == ok and (d[c["mase"]] != "") == ok)
        g["rows_agree"] += agree
        if not ok:
            return

        # recomputed (per-series file) against stored (raw file), as in recompute_check
        smape_new, mase_new = num(d[c["smape"]]), num(d[c["mase"]])
        if math.isnan(smape_old) or math.isnan(mase_old):
            g["stored_metric_nan"] += 1
        else:
            g["max_abs_diff_smape"] = self._max(g["max_abs_diff_smape"], abs(smape_new - smape_old))
            g["max_abs_diff_mase"] = self._max(g["max_abs_diff_mase"], abs(mase_new - mase_old))

        # with the M4 test values: sMAPE and MASE from the stored forecast vector, as s2 did
        if self.ctx.m4 is not None:
            a = self._test(freq).get(sid)
            try:
                f = [float(x) for x in r[self.fcol].split(";") if x != ""]
            except ValueError:
                f = None
            if a is None or f is None:
                return
            h = min(HORIZON[freq], len(a), len(f))
            den = num(r[rc["mase_den"]])
            diff = max(abs(smape_series(f[:h], a[:h]) - smape_new),
                       abs(mase_series(f[:h], a[:h], den) - mase_new))
            g["m4_max_abs_diff"] = self._max(g["m4_max_abs_diff"], diff)
            g["m4_rows_agree"] += diff <= RAW_TOL

    def finish(self) -> None:
        self._close()

    def table(self) -> list[dict]:
        """Rows of perseries_vs_raw: one per cell file, then the total of the set."""
        m4 = self.ctx.m4 is not None
        out = []
        for (freq, objective), g in self.groups.items():
            out.append(self._record(freq, objective, g, m4))
        tot = {k: sum(g[k] for g in self.groups.values())
               for k in ("rows", "raw_rows", "rows_agree", "baseline_rows_agree", "ok_rows",
                         "stored_metric_nan", "m4_rows_agree")}
        for k in ("max_abs_diff_smape", "max_abs_diff_mase", "m4_max_abs_diff"):
            vals = [g[k] for g in self.groups.values() if g[k] is not None]
            tot[k] = max(vals) if vals else None
        tot["raw_file"] = f"{len(self.groups)} files"
        out.append(self._record("all", "all", tot, m4))
        return out

    def _record(self, freq, objective, g, m4) -> dict:
        def val(x):
            return "n/a" if x is None else x

        rec = {"set": self.which, "frequency": freq, "objective": objective,
               "rows": g["rows"], "raw_rows": g["raw_rows"], "rows_agree": g["rows_agree"],
               "baseline_rows_agree": g["baseline_rows_agree"], "ok_rows": g["ok_rows"],
               "stored_metric_nan": g["stored_metric_nan"],
               "max_abs_diff_smape": val(g["max_abs_diff_smape"]),
               "max_abs_diff_mase": val(g["max_abs_diff_mase"])}
        good = (g["rows"] == g["raw_rows"] == g["rows_agree"] == g["baseline_rows_agree"]
                and all(g[k] is None or g[k] <= RAW_TOL for k in ("max_abs_diff_smape", "max_abs_diff_mase")))
        if m4:
            rec["m4_rows_agree"] = g["m4_rows_agree"]
            rec["m4_max_abs_diff"] = val(g["m4_max_abs_diff"])
            good = good and g["m4_rows_agree"] == g["ok_rows"]
        rec["all_agree"] = "yes" if good else "no"
        rec["raw_file"] = g["raw_file"]
        return rec


# ---- the group --------------------------------------------------------------------------------
def _write_csv(path, rows: list[dict], cols: list[str] | None = None) -> None:
    """As the original scripts wrote their tables (csv.DictWriter, floats at full precision)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols or list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _cells(name: str, rows: list[dict], skip=()) -> list[Cell]:
    keys = TABLE_KEYS[name]
    out = []
    for rec in rows:
        row = "|".join(str(rec[k]) for k in keys)
        for col, v in rec.items():
            if col in keys or col in skip:
                continue
            if isinstance(v, (float, np.floating)):
                out.append(Cell(f"distributions.{name}", row, col, float(v), DECIMALS))
            elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                out.append(Cell(f"distributions.{name}", row, col, int(v), 0))
            else:
                out.append(Cell(f"distributions.{name}", row, col, str(v), None))
    return out


def generate(ctx: Context) -> list[Cell]:
    pub = parse_published(ctx)
    tables: dict[str, list[dict]] = {name: [] for name in TABLE_KEYS}
    status, figs = {}, []
    for which in SETS:
        check = RawCheck(ctx, which)
        data = load_set(ctx, which, check)
        tables["perseries_vs_raw"] += check.table()
        mats = {freq: matrix(data[freq]) for freq in FREQS}
        del data, check
        rows, status[which] = reproduction(which, mats, pub)
        tables["reproduction"] += rows
        if status[which]:                   # s4/s5 run only on sets whose totals reproduced
            compile_set(which, mats, tables)
            figs += figures(ctx, which, {f: mats[f][0] for f in FREQS})

    _write_csv(ctx.out / "reproduction.csv", tables["reproduction"], REPRODUCTION_COLS)
    for name in ("4a_marginal", "4b_paired", "4c_tail", "4c_tail_ratios", "4d_union_tail",
                 "perseries_vs_raw"):
        if tables[name]:
            _write_csv(ctx.out / f"{name}.csv", tables[name])
    compiled = [s for s, ok in status.items() if ok]
    (ctx.out / "compile_status.json").write_text(json.dumps(
        {"tolerance": TOL, "published_source": PUBLISHED, "published_section": PUBLISHED_SECTION,
         "all_rows_match": status, "compiled_sets": compiled,
         "not_compiled_because_not_reproducing": [s for s in status if s not in compiled],
         "figures": figs}, indent=2) + "\n", encoding="utf-8", newline="\n")

    cells: list[Cell] = []
    for name, rows in tables.items():
        cells += _cells(name, rows, skip=("raw_file",) if name == "perseries_vs_raw" else ())
    return cells
