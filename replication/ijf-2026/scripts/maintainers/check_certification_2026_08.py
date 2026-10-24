"""Check the values derived from the August 2026 certification batch and the Section 5.5 records.

results/historical/certification-2026-08/received/ holds what the collaborator who ran the
August 2026 SCIP certification batch (Table S5, Section 5.6) and the development measurements of
Table S2 and Section 5.5 supplied on 2026-10-21: the per-instance raw outputs of the batch are
lost, and the material is a RECONSTRUCTION from the record of the session that ran it (tables
printed to the console, the harness rebuilt from the patches applied in that session), with the
source of every row declared. See results/historical/certification-2026-08/PROVENANCE.md.

This script verifies, without the artifact bundle:

* every received file against received/SHA256SUMS;
* that the consolidated per-instance table (tabela_scip_lotes1-2_RECONSTRUIDA.csv) agrees with
  the per-case tables recovered for each batch (status, time, certified gap, Ipopt - SCIP);
* that the medians and the wall-time reduction the recovered outputs print follow from their own
  per-seed rows and totals;
* that manuscript_sources.csv, the values the manuscript mapping compares the printed numbers
  with, follows from those files (`--write` regenerates it).

It re-runs nothing: the records cannot be reproduced from this archive alone. Exit 0 when
everything agrees, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import statistics
import sys
from pathlib import Path

REPL = Path(__file__).resolve().parents[2]
D = REPL / "results" / "historical" / "certification-2026-08"
REC = D / "received"
AGG = REC / "resultados_agregados"
SOURCES = D / "manuscript_sources.csv"

# Rows of Table S5 and the instances of the batch each one counts. The batch's record does not
# say which instances the table counts; this is the assignment by the harness's own cells
# (casos() in harness/scip_prog_lote*.jl). Where another assignment gives the same counts, it is
# listed too and the counts are checked under both.
S5 = [
    ("nonseasonal_grid", "cell A (non-seasonal (1,0,1), T in {30, 60, 120, 240}, 3 draws each)",
     [range(1, 13)]),
    ("extended_T480", "cell E, T = 480 (ids 34 and 35 did not finish and are in neither column)",
     [range(33, 36)]),
    ("extended_T960", "cell E, T = 960", [range(36, 39)]),
    ("seasonal_grid", "cell B (seasonal (1,0,1)(P,0,Q)_12, T in {36, 60})", [range(13, 25)]),
    ("diagnostics", "cell C, ids 25-30 (the harness's adversarial cell, with the three M4 series "
     "28-30, which failed the harness's noise-floor assertion); the same counts follow from ids "
     "25-27 with the seasonal cancellation cases 39, 41, 43 of cell F",
     [range(25, 31), [25, 26, 27, 39, 41, 43]]),
    ("free_presample", "ids 31, 32 (cell D, :innovations, T = 60) and 46 (cell G, :innovations, "
     "T = 120); id 45 (cell G, T = 60) in place of 31 or 32 gives the same counts",
     [[31, 32, 46], [32, 45, 46]]),
]
SEASONAL = range(13, 25)


def rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def text(name: str) -> str:
    return (AGG / name).read_text(encoding="utf-8")


def ids(spec) -> str:
    v = list(spec)
    return f"{v[0]}-{v[-1]}" if v == list(range(v[0], v[-1] + 1)) and len(v) > 2 else ", ".join(map(str, v))


def check_sums(problems: list[str]) -> int:
    n = 0
    for line in (REC / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        n += 1
        p = REC / name.strip()
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            problems.append(f"received/{name.strip()}: missing or SHA-256 differs")
    return n


def check_consolidated(table: dict[int, dict], problems: list[str]) -> int:
    """The consolidated table against the per-case tables recovered for each batch."""
    n = 0
    lote1 = [l for l in text("lote1_tabela_32_casos_21-08_1945.txt").splitlines()
             if re.match(r"\| \d+ \|", l)]
    for l in lote1:
        c = [x.strip() for x in l.strip("|").split("|")]
        r = table[int(c[0])]
        got = (r["cel"], r["rotulo"], r["modo"], r["T"], r["status"], r["t_s"], r["gap_cert"],
               r["ipopt_menos_scip"], r["piso_ok"])
        n += 1
        if got != tuple(c[1:10]):
            problems.append(f"consolidated table id {c[0]} differs from the batch 1 table: {got} vs {c[1:10]}")
    lote2 = [l.split(",") for l in text("lote2_parcial_12de14_22-08_1041.txt").splitlines()
             if re.match(r"\d+,", l)]
    for c in lote2:
        r = table[int(c[0])]
        n += 1
        if (r["cel"], r["rotulo"], r["T"], r["status"], r["t_s"], r["gap_cert"]) != tuple(c[1:7]):
            problems.append(f"consolidated table id {c[0]} differs from the batch 2 table")
    if len(lote1) != 32 or len(lote2) != 12:
        problems.append(f"expected 32 batch 1 rows and 12 batch 2 rows, found {len(lote1)} and {len(lote2)}")
    return n


def derive(problems: list[str]) -> list[dict]:
    table = {int(r["id"]): r for r in rows(REC / "tabela_scip_lotes1-2_RECONSTRUIDA.csv")}
    out: list[dict] = []

    def put(quantity: str, value, instances: str, rule: str) -> None:
        out.append({"quantity": quantity, "value": value if isinstance(value, str) else repr(value),
                    "instances": instances, "rule": rule})

    # ---- Table S5: counts of certified (OPTIMAL) and time-limited instances per row
    for key, desc, specs in S5:
        counts = {(sum(table[i]["status"] == "OPTIMAL" for i in s),
                   sum(table[i]["status"] == "TIME_LIMIT" for i in s)) for s in specs}
        if len(counts) != 1:
            problems.append(f"Table S5 {key}: the listed assignments give different counts {counts}")
        cert, tl = sorted(counts)[0]
        inst = " or ".join(ids(s) for s in specs)
        put(f"s5_{key}_certified", cert, inst, f"OPTIMAL among {desc}")
        put(f"s5_{key}_time_limit", tl, inst, f"TIME_LIMIT among {desc}")
    put("s5_extended_T480_unfinished", sum(table[i]["status"].startswith("MORTO") for i in (34, 35)),
        "34, 35", "killed by an external 900 s clock; no SCIP result")
    # by the harness's cells (the table as organized in the submitted Supplementary Material)
    outcome = {"O": "OPTIMAL", "T": "TIME_LIMIT", "X": "MORTO"}
    for cell in sorted({r["cel"] for r in table.values()}) + ["all"]:
        members = sorted(i for i, r in table.items() if cell in ("all", r["cel"]))
        for code, status in outcome.items():
            put(f"s5_cell_{cell}_{code}", sum(table[i]["status"].startswith(status) for i in members),
                ids(members), f"{status} among cell {cell} (X: killed by the external clock, no result)")

    # ---- Section 5.6: seasonal grid (cell B)
    b = [table[i] for i in SEASONAL]
    if {r["cel"] for r in b} != {"B"}:
        problems.append("ids 13-24 are not cell B")
    diff = [float(r["ipopt_menos_scip"]) for r in b]
    gaps = [float(r["gap_cert"]) for r in b]
    put("s56_seasonal_ipopt_minus_scip_max", max(diff), ids(SEASONAL),
        "largest Ipopt - SCIP objective difference in cell B (absolute; the objectives of cell B "
        "were not recorded)")
    put("s56_seasonal_gap_min", min(gaps), ids(SEASONAL), "smallest certified relative gap in cell B "
        "((SCIP primal - dual) / |primal|, all TIME_LIMIT at 600 s)")
    put("s56_seasonal_gap_max", max(gaps), ids(SEASONAL), "largest certified relative gap in cell B")
    put("s56_seasonal_gap_median", statistics.median(gaps), ids(SEASONAL),
        "median certified relative gap in cell B (12 values: mean of the 6th and 7th; the 6th alone "
        f"is {statistics.median_low(gaps)!r})")

    # ---- Section 5.6: near-cancellation instance and the two certificates
    for key, i, what in (("near_cancellation", 26, "cell C, (1,0,1)(1,0,1)_12, Phi = 0.700, Theta = -0.699, T = 60, TIME_LIMIT"),
                         ("certified_1", 44, "cell F, regular cancellation with slack 5e-2, :zeroed, OPTIMAL"),
                         ("certified_2", 45, "cell G, regular cancellation, :innovations, OPTIMAL")):
        r = table[i]
        ip, sc = float(r["obj_ipopt"]), float(r["obj_scip"])
        put(f"s56_{key}_ipopt", ip, str(i), f"Ipopt objective, {what}")
        put(f"s56_{key}_scip", sc, str(i), f"SCIP objective, {what}")
        if key != "near_cancellation":
            put(f"s56_{key}_local_excess", (ip - sc) / sc, str(i), "(Ipopt - SCIP) / SCIP")
    c26 = text("lote1_caso26_21-08_1945.txt")
    if "obj Ipopt : 46.660913" not in c26 or "obj SCIP  : 45.314976" not in c26:
        problems.append("case 26: the recovered case report does not show the objectives of the table")

    # ---- Table S2: exogenous-convention simulation (3 DGPs x 20 replications), per-DGP aggregate
    h = text("h_discriminador_3dgp_20-08_1350.txt")
    for dgp in ("erros_arima", "ma_puro", "armax"):
        m = re.search(rf"^{dgp}\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)", h, re.M)
        if not m:
            problems.append(f"Table S2: no row for {dgp}")
            continue
        for col, v in zip(("rmse_julia", "rmse_r", "dlog"), m.groups()):
            put(f"s2_{dgp}_{col}", v, "seeds 9000+s, s = 1..20",
                "aggregate printed by diag_ler.jl (per-replication outputs lost)")

    # ---- Section 5.5: initialization diagnostic (10 seeds) and the paired 1,500-series gate
    ab, cd = text("g_10sementes_bracos_AB_21-08_0005.txt"), text("g_10sementes_bracos_CD_21-08_0109.txt")
    rows_ab = [m.groups() for m in re.finditer(r"^\s+(\d+)\s+\[[^]]*\]\s+\[[^]]*\]\s+([\d.]+)\s+([\d.]+)\s*$", ab, re.M)]
    rows_cd = [m.groups() for m in re.finditer(r"^\s+(\d+)\s+\[[^]]*\]\s+\[[^]]*\]\s+([\d.]+)\s+([\d.]+)\s+(sim|NAO)\s*$", cd, re.M)]
    seeds = {"A": [float(r[1]) for r in rows_ab], "B": [float(r[2]) for r in rows_ab],
             "C": [float(r[1]) for r in rows_cd], "D": [float(r[2]) for r in rows_cd]}
    med = {"A": re.search(r"distancia mediana ao R: sem ([\d.]+)", ab),
           "B": re.search(r"distancia mediana ao R: sem [\d.]+ -> com ([\d.]+)", ab),
           "C": re.search(r"mediana:\s+C ([\d.]+)", cd), "D": re.search(r"mediana:\s+C [\d.]+\s+D ([\d.]+)", cd)}
    if any(len(v) != 10 for v in seeds.values()) or not all(med.values()):
        problems.append("Section 5.5 initialization: 10 seeds per arm and the printed medians expected")
    else:
        for name in "ABCD":
            if abs(statistics.median(seeds[name]) - float(med[name].group(1))) > 0.5e-4 + 1e-12:
                problems.append(f"Section 5.5 initialization arm {name}: median of the seeds "
                                f"{statistics.median(seeds[name])} != printed {med[name].group(1)}")
        put("s55_init_distance_free_presample", med["A"].group(1), "MersenneTwister(500+r), r = 1..10",
            "arm A (:penalized, free pre-sample, no cross term): median distance to R arima(method=\"ML\")")
        put("s55_init_distance_free_presample_cross_term", med["B"].group(1), "MersenneTwister(500+r), r = 1..10",
            "arm B (:penalized, free pre-sample, with the cross term): median distance to R arima(method=\"ML\")")
        put("s55_init_distance_innovations", med["C"].group(1), "MersenneTwister(500+r), r = 1..10",
            "arm C (:innovations, no determinant): median distance to R arima(method=\"ML\")")
        put("s55_init_distance_innovations_determinant", med["D"].group(1), "MersenneTwister(500+r), r = 1..10",
            "arm D (:innovations, with the determinant term): median distance to R arima(method=\"ML\")")
    g = text("g_gate_1500_21-08_1104.txt")
    arms = {m.group(1): m.groups() for m in re.finditer(
        r"^(:penalized|:innovations)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+(\d+)", g, re.M)}
    tot = re.search(r"tempo total: (\d+)s vs (\d+)s", g)
    if set(arms) != {":penalized", ":innovations"} or not tot:
        problems.append("Section 5.5 gate: both arms and the total times expected")
    else:
        wp, wi = float(arms[":penalized"][3]), float(arms[":innovations"][3])
        tp, ti = map(float, tot.groups())
        if abs((1 - wi / wp) - (1 - ti / tp)) > 1e-3:
            problems.append("Section 5.5 gate: per-series and total wall times disagree")
        put("s55_gate_itercap_penalized", arms[":penalized"][4], "1,500 paired M4 series, commit f90e554",
            "share of model-selection searches whose largest candidate iteration count (iterMax) reached the "
            "200-iteration limit, :penalized (aggregate; per-series outputs lost)")
        put("s55_gate_itercap_innovations", arms[":innovations"][4], "1,500 paired M4 series, commit f90e554",
            "same, :innovations")
        for arm, key in ((":penalized", "penalized"), (":innovations", "innovations")):
            put(f"s55_gate_{key}_smape", arms[arm][1], "1,500 paired M4 series, commit f90e554",
                f"mean sMAPE, {arm} (aggregate)")
            put(f"s55_gate_{key}_mase", arms[arm][2], "1,500 paired M4 series, commit f90e554",
                f"mean MASE, {arm} (aggregate)")
        put("s55_gate_wall_reduction", 1 - wi / wp, "1,500 paired M4 series, commit f90e554",
            f"1 - wall(:innovations) / wall(:penalized) per series ({wi} s / {wp} s; totals {ti:.0f} s / {tp:.0f} s)")

    # ---- Section 5.5: likelihood-ranking diagnostic (two separate measurements)
    pq = re.search(r"n=(\d+) ajustes pareados", text("i_paridade_aicc_1099_16-08_0406.txt"))
    dist = dict((m.group(1), (m.group(2), m.group(3))) for m in re.finditer(
        r"seasonalPresamplePrior=(false|true)\s+n=(\d+)\s+distancia CSS->ML: mediana ([\d.]+)",
        text("i_distancia_css_ml_16-08_0434.txt")))
    if not pq or set(dist) != {"false", "true"}:
        problems.append("Section 5.5 ranking: the AICc-parity count and both distance lines expected")
    else:
        put("s55_rank_aicc_parity_fits", pq.group(1), "first 100 series of the rand400 sample",
            "fits of the AICc-parity comparison with R on the P x Q grid (aicq.jl / aicq.R)")
        put("s55_rank_distance_prior_off", dist["false"][1], f"n = {dist['false'][0]}",
            "mixdist.jl, seasonalPresamplePrior = false: median CSS -> ML log-likelihood distance")
        put("s55_rank_distance_prior_on", dist["true"][1], f"n = {dist['true'][0]}",
            "same, seasonalPresamplePrior = true, same revision (2026-08-16); not paired with the line above")
        put("s55_rank_distance_prior_off_n", dist["false"][0], "mixdist.jl", "fits behind the median with the flag off")
        put("s55_rank_distance_prior_on_n", dist["true"][0], "mixdist.jl", "fits behind the median with the flag on")
    return out


def render(out: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, ["quantity", "value", "instances", "rule"], lineterminator="\n")
    w.writeheader()
    w.writerows(out)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="regenerate manuscript_sources.csv")
    args = ap.parse_args()
    problems: list[str] = []
    listed = check_sums(problems)
    table = {int(r["id"]): r for r in rows(REC / "tabela_scip_lotes1-2_RECONSTRUIDA.csv")}
    if sorted(table) != list(range(1, 47)):
        problems.append("the consolidated table does not hold ids 1-46")
    matched = check_consolidated(table, problems)
    out = derive(problems)
    new = render(out)
    if args.write:
        SOURCES.write_text(new, encoding="utf-8", newline="\n")
    elif not SOURCES.exists() or SOURCES.read_text(encoding="utf-8") != new:
        problems.append(f"{SOURCES.name} does not follow from the received files (run with --write)")
    for p in problems:
        print("FAIL", p)
    print(f"received files: {listed} checked; consolidated table: {matched} rows matched with the "
          f"per-batch tables; {len(out)} derived values: " + ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
