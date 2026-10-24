import csv, sys, math
from collections import defaultdict
old = list(csv.DictReader(open(sys.argv[1])))
new = list(csv.DictReader(open(sys.argv[2])))
key = lambda r: (r["dgp"], r["nreg"], r["d_true"], r["seed"], r["semantics"])
o = {key(r): r for r in old}; n = {key(r): r for r in new}
assert o.keys() == n.keys()
g = defaultdict(lambda: defaultdict(int))
for k in sorted(o):
    a, b = o[k], n[k]
    grp = (a["dgp"], a["nreg"], a["d_true"], a["semantics"])
    s = g[grp]
    s["cases"] += 1
    s["fallback_old"] += a["fallback"] == "true"
    s["fallback_new"] += b["fallback"] == "true"
    s["order_changed"] += a["order"] != b["order"]
    s["fc_changed"] += abs(float(a["fc12"]) - float(b["fc12"])) > 1e-8
    s["rmse_better"] += float(b["rmse12"]) < float(a["rmse12"]) - 1e-12
    s["rmse_worse"] += float(b["rmse12"]) > float(a["rmse12"]) + 1e-12
    s["sum_rmse_old"] += float(a["rmse12"]); s["sum_rmse_new"] += float(b["rmse12"])
print("dgp,nreg,d,semantics,cases,css_fallback_v100,css_fallback_fixed,order_changed,forecast_changed,rmse12_lower,rmse12_higher,mean_rmse12_v100,mean_rmse12_fixed")
for grp, s in sorted(g.items()):
    c = s["cases"]
    print(",".join(map(str, grp)) + f",{c},{s['fallback_old']},{s['fallback_new']},{s['order_changed']},{s['fc_changed']},{s['rmse_better']},{s['rmse_worse']},{s['sum_rmse_old']/c:.4f},{s['sum_rmse_new']/c:.4f}")
# semantics preference: which semantics has the lower AICc per case, old vs new
pref = defaultdict(lambda: [0, 0, 0])
for k in o:
    if k[4] != "armax": continue
    k2 = k[:4] + ("regression_errors",)
    for i, src in enumerate((o, n)):
        if float(src[k]["aicc"]) < float(src[k2]["aicc"]): pref[(k[0], k[1], k[2], ("v100", "fixed")[i])][0] += 1
        else: pref[(k[0], k[1], k[2], ("v100", "fixed")[i])][1] += 1
print("\ndgp,nreg,d,code,aicc_prefers_armax,aicc_prefers_regression_errors")
for kk, v in sorted(pref.items()):
    print(",".join(map(str, kk)) + f",{v[0]},{v[1]}")
