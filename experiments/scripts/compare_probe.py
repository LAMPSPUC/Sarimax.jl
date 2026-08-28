# Compares the output of probe_reproduction.jl against the stored campaign outputs,
# cell by cell. Reads results/reproduction_probe_dev.csv by default; point PROBE at
# another probe run to check a different package build.
import csv, gzip, os, pathlib, statistics as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
PROBE = ROOT / "results" / "reproduction_probe_dev.csv"

# cell -> (campaign file, campaign commit)
CELLS = {
    ("mse", "zeroed"): ("cel_mse_zeroed_monthly", "87f7bfb"),
    ("huber", "zeroed"): ("cel_huber_zeroed_monthly", "87f7bfb"),
    ("mae", "zeroed"): ("cel_mae_zeroed_monthly", "87f7bfb"),
    ("mse", "innovations"): ("req_true_monthly", "87f7bfb (first 38280 rows)"),
    ("huber", "innovations"): ("obj_huber_innov_monthly", "5b2ec6b"),
    ("mae", "innovations"): ("obj_mae_innov_monthly", "5b2ec6b"),
    ("ridge", "innovations"): ("obj_ridge_innov_monthly", "5b2ec6b"),
}

probe = {}
for x in csv.DictReader(open(PROBE, newline="")):
    key = (x["objective"].strip('"'), x["initialization"].strip('"'))
    probe.setdefault(key, {})[int(x["series"])] = (
        x["status"].strip('"'),
        None if x["smape_total"] in ("NaN", "") else float(x["smape_total"]),
        tuple(int(x[k]) for k in ("p", "d", "q", "P", "D", "Q")),
    )

print(f"{'cell':22s} {'campaign commit':28s} {'n':>4s} {'identical':>10s} {'order diff':>11s} {'errors':>7s}  verdict")
for key, (fname, commit) in CELLS.items():
    p = RAW / (fname + ".csv.gz")
    hist = {}
    with gzip.open(p, "rt", newline="") as f:
        for x in csv.DictReader(f):
            i = int(x["series"])
            if i > 200:
                continue
            if not x["status"].strip('"').startswith("OK"):
                continue
            hist[i] = (float(x["smape_total"]),
                       tuple(int(x[k]) for k in ("p", "d", "q", "P", "D", "Q")))
    pr = probe.get(key, {})
    ids = sorted(set(hist) & {i for i, v in pr.items() if v[0] == "OK"})
    errs = sum(1 for i, v in pr.items() if v[0] != "OK")
    if not ids:
        print(f"{key[0]+'/'+key[1]:22s} {commit:28s} {0:>4d} {'-':>10s} {'-':>11s} {errs:>7d}  NO DATA")
        continue
    same = sum(1 for i in ids if abs(hist[i][0] - pr[i][1]) < 1e-9)
    ordd = sum(1 for i in ids if hist[i][1] != pr[i][2])
    verdict = "YES (bit-identical)" if same == len(ids) else f"NO ({len(ids)-same} differ)"
    print(f"{key[0]+'/'+key[1]:22s} {commit:28s} {len(ids):>4d} {same:>10d} {ordd:>11d} {errs:>7d}  {verdict}")
    if same != len(ids):
        d = [(abs(hist[i][0] - pr[i][1]), i) for i in ids]
        d.sort(reverse=True)
        mh = st.fmean(hist[i][0] for i in ids)
        mp = st.fmean(pr[i][1] for i in ids)
        print(f"{'':22s} mean sMAPE hist={mh:.4f} probe={mp:.4f} delta={mp-mh:+.4f}; "
              f"largest |diff| {d[0][0]:.4f} at series {d[0][1]}")
    if errs:
        msgs = {}
        for i, v in pr.items():
            if v[0] != "OK":
                msgs[v[0][:70]] = msgs.get(v[0][:70], 0) + 1
        for m, c in sorted(msgs.items(), key=lambda t: -t[1])[:2]:
            print(f"{'':22s} error x{c}: {m}")
